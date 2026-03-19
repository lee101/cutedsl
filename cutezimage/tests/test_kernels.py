"""Tests for cutezimage Triton kernels."""

import pytest
import torch


# Skip all if no CUDA
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for Triton kernels",
)


class TestRMSNorm:
    def _reference_rms_norm(self, x, weight, eps):
        variance = x.float().pow(2).mean(-1, keepdim=True)
        normed = x * torch.rsqrt(variance + eps)
        if weight is not None:
            normed = normed.to(weight.dtype) if weight.dtype != normed.dtype else normed
            normed = weight * normed
        return normed

    @pytest.mark.parametrize("shape", [(4, 256), (2, 32, 256), (1, 64, 3840)])
    def test_matches_reference(self, shape):
        from cutezimage.triton_kernels.rms_norm import rms_norm

        torch.manual_seed(42)
        x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(shape[-1], device="cuda", dtype=torch.bfloat16)

        ref = self._reference_rms_norm(x, weight, 1e-5)
        out = rms_norm(x, weight, eps=1e-5)

        assert out.shape == ref.shape
        max_err = (out.float() - ref.float()).abs().max().item()
        assert max_err < 1e-2, f"Max error {max_err} >= 1e-2"

    def test_no_weight(self):
        from cutezimage.triton_kernels.rms_norm import rms_norm

        torch.manual_seed(42)
        x = torch.randn(4, 256, device="cuda", dtype=torch.float32)

        ref = self._reference_rms_norm(x, None, 1e-5)
        out = rms_norm(x, None, eps=1e-5)

        max_err = (out - ref).abs().max().item()
        assert max_err < 1e-5, f"Max error {max_err}"


class TestFusedSiLUGate:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_matches_reference(self, dtype):
        from cutezimage.triton_kernels.fused_silu_gate_ffn import fused_silu_gate

        torch.manual_seed(42)
        x1 = torch.randn(2, 16, 512, device="cuda", dtype=dtype)
        x3 = torch.randn(2, 16, 512, device="cuda", dtype=dtype)

        ref = torch.nn.functional.silu(x1) * x3
        out = fused_silu_gate(x1, x3)

        assert out.shape == ref.shape
        max_err = (out.float() - ref.float()).abs().max().item()
        atol = 1e-5 if dtype == torch.float32 else 0.05
        assert max_err < atol, f"Max error {max_err} for {dtype}"

    def test_small_tensors(self):
        from cutezimage.triton_kernels.fused_silu_gate_ffn import fused_silu_gate

        torch.manual_seed(42)
        x1 = torch.randn(2, 8, 64, device="cuda", dtype=torch.float32)
        x3 = torch.randn(2, 8, 64, device="cuda", dtype=torch.float32)

        ref = torch.nn.functional.silu(x1) * x3
        out = fused_silu_gate(x1, x3)

        max_err = (out - ref).abs().max().item()
        assert max_err < 1e-5, f"Max error {max_err}"


class TestFusedSiLUGateFFN:
    def test_full_ffn(self):
        from cutezimage.triton_kernels.fused_silu_gate_ffn import fused_silu_gate_ffn

        torch.manual_seed(42)
        dim, hidden = 128, 256
        # Use float32 for the full FFN test since bf16 matmul accumulation
        # amplifies rounding errors through the triple matmul chain
        x = torch.randn(2, 8, dim, device="cuda", dtype=torch.float32) * 0.1
        w1 = torch.randn(hidden, dim, device="cuda", dtype=torch.float32) * 0.1
        w2 = torch.randn(dim, hidden, device="cuda", dtype=torch.float32) * 0.1
        w3 = torch.randn(hidden, dim, device="cuda", dtype=torch.float32) * 0.1

        # Reference
        x1_ref = torch.nn.functional.linear(x, w1)
        x3_ref = torch.nn.functional.linear(x, w3)
        gated_ref = torch.nn.functional.silu(x1_ref) * x3_ref
        ref = torch.nn.functional.linear(gated_ref, w2)

        out = fused_silu_gate_ffn(x, w1, w2, w3)

        assert out.shape == ref.shape
        max_err = (out.float() - ref.float()).abs().max().item()
        assert max_err < 1e-4, f"Max error {max_err}"


class TestFusedAdaLNNorm:
    def test_matches_reference(self):
        from cutezimage.triton_kernels.fused_adaln_norm import fused_adaln_rms_norm

        torch.manual_seed(42)
        B, S, D = 2, 16, 256
        x = torch.randn(B, S, D, device="cuda", dtype=torch.bfloat16)
        scale = torch.randn(B, D, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(D, device="cuda", dtype=torch.bfloat16)

        # Reference
        variance = x.float().pow(2).mean(-1, keepdim=True)
        normed = x * torch.rsqrt(variance + 1e-5)
        normed = normed.to(weight.dtype) * weight
        ref = normed * scale.unsqueeze(1)

        out = fused_adaln_rms_norm(x, scale, weight, eps=1e-5)

        assert out.shape == ref.shape
        max_err = (out.float() - ref.float()).abs().max().item()
        assert max_err < 0.05, f"Max error {max_err}"


class TestRoPEComplex:
    def _reference_rope(self, x, freqs_cis):
        with torch.amp.autocast("cuda", enabled=False):
            x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
            if freqs_cis.ndim < x_complex.ndim:
                freqs_cis = freqs_cis.unsqueeze(2)
            x_out = torch.view_as_real(x_complex * freqs_cis).flatten(x_complex.ndim - 1)
            return x_out.type_as(x)

    def test_matches_reference(self):
        from cutezimage.triton_kernels.rope_complex import apply_rope_complex

        torch.manual_seed(42)
        B, S, H, D = 2, 16, 4, 64
        x = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
        # Create complex freqs
        freqs = torch.randn(1, S, 1, D // 2, device="cuda", dtype=torch.float32)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # unit magnitude, random phase

        ref = self._reference_rope(x, freqs_cis)
        out = apply_rope_complex(x, freqs_cis)

        assert out.shape == ref.shape
        max_err = (out.float() - ref.float()).abs().max().item()
        assert max_err < 0.02, f"Max error {max_err}"
