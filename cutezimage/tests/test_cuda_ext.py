"""Tests for the compiled CUDA C extension (cutezimage._C).

Skipped when CUDA is unavailable or the extension has not been built yet.
Run after building with:
    python cutezimage/csrc/setup_cuda.py build_ext --inplace
"""

from __future__ import annotations

import pytest
import torch

# Try to import the C extension; skip entire module if absent
try:
    from cutezimage import _C as cute_c
    _HAS_C_EXT = True
except ImportError:
    _HAS_C_EXT = False

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(not _HAS_C_EXT, reason="cutezimage._C not built (run setup_cuda.py)"),
]


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------

def _ref_rms_norm(x: torch.Tensor, w: torch.Tensor | None, eps: float) -> torch.Tensor:
    variance = x.float().pow(2).mean(-1, keepdim=True)
    normed = x.float() * torch.rsqrt(variance + eps)
    if w is not None:
        normed = normed * w.float()
    return normed.to(x.dtype)


def _ref_silu_gate(x1: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(x1) * x3


def _ref_qk_norm(q, k, qw, kw, eps):
    def _rms(x, w):
        var = x.float().pow(2).mean(-1, keepdim=True)
        n = x.float() * torch.rsqrt(var + eps)
        return (n * w.float()).to(x.dtype)
    return _rms(q, qw), _rms(k, kw)


# ---------------------------------------------------------------------------
# RMS Norm
# ---------------------------------------------------------------------------

class TestCExtRMSNorm:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("shape", [(4, 256), (2, 16, 3840), (1, 128, 3840)])
    def test_matches_reference(self, dtype, shape):
        torch.manual_seed(0)
        x = torch.randn(shape, device="cuda", dtype=dtype)
        w = torch.randn(shape[-1], device="cuda", dtype=dtype)

        ref = _ref_rms_norm(x, w, 1e-5)
        out = cute_c.rms_norm(x, w, 1e-5)

        assert out.shape == ref.shape
        assert out.dtype == dtype
        atol = 1e-2 if dtype != torch.float32 else 1e-5
        assert (out.float() - ref.float()).abs().max().item() < atol

    def test_no_weight(self):
        torch.manual_seed(1)
        x = torch.randn(8, 512, device="cuda", dtype=torch.float32)
        ref = _ref_rms_norm(x, None, 1e-5)
        out = cute_c.rms_norm(x, None, 1e-5)
        assert (out - ref).abs().max().item() < 1e-5

    def test_non_contiguous_raises(self):
        x = torch.randn(4, 256, device="cuda", dtype=torch.float32)
        with pytest.raises(RuntimeError):
            cute_c.rms_norm(x.t(), None, 1e-5)

    @pytest.mark.parametrize("N", [128, 256, 512, 1024, 3840, 10240])
    def test_various_widths(self, N):
        x = torch.randn(8, N, device="cuda", dtype=torch.bfloat16)
        w = torch.ones(N, device="cuda", dtype=torch.bfloat16)
        out = cute_c.rms_norm(x, w, 1e-5)
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# Fused SiLU Gate
# ---------------------------------------------------------------------------

class TestCExtFusedSiLUGate:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_matches_reference(self, dtype):
        torch.manual_seed(2)
        x1 = torch.randn(4, 16, 10240, device="cuda", dtype=dtype)
        x3 = torch.randn_like(x1)

        ref = _ref_silu_gate(x1, x3)
        out = cute_c.fused_silu_gate(x1, x3)

        assert out.shape == ref.shape
        assert out.dtype == dtype
        atol = 0.05 if dtype != torch.float32 else 1e-5
        assert (out.float() - ref.float()).abs().max().item() < atol

    def test_1d_input(self):
        torch.manual_seed(3)
        x1 = torch.randn(10240, device="cuda", dtype=torch.float32)
        x3 = torch.randn_like(x1)
        ref = _ref_silu_gate(x1, x3)
        out = cute_c.fused_silu_gate(x1, x3)
        assert (out - ref).abs().max().item() < 1e-5

    def test_odd_size(self):
        # Odd N forces the tail-element path
        torch.manual_seed(4)
        x1 = torch.randn(4, 257, device="cuda", dtype=torch.bfloat16)
        x3 = torch.randn_like(x1)
        ref = _ref_silu_gate(x1, x3)
        out = cute_c.fused_silu_gate(x1, x3)
        assert (out.float() - ref.float()).abs().max().item() < 0.05


# ---------------------------------------------------------------------------
# Fused QK Norm
# ---------------------------------------------------------------------------

class TestCExtFusedQKNorm:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_matches_reference(self, dtype):
        torch.manual_seed(5)
        B, S, H, D = 2, 32, 30, 128
        q  = torch.randn(B, S, H, D, device="cuda", dtype=dtype)
        k  = torch.randn_like(q)
        qw = torch.randn(D, device="cuda", dtype=dtype)
        kw = torch.randn(D, device="cuda", dtype=dtype)

        q_ref, k_ref = _ref_qk_norm(q, k, qw, kw, 1e-5)
        q_out, k_out = cute_c.fused_qk_norm(q, k, qw, kw, 1e-5)

        assert q_out.shape == q.shape
        assert k_out.shape == k.shape
        atol = 0.05 if dtype != torch.float32 else 1e-4
        assert (q_out.float() - q_ref.float()).abs().max().item() < atol
        assert (k_out.float() - k_ref.float()).abs().max().item() < atol

    def test_head_dim_64(self):
        torch.manual_seed(6)
        B, S, H, D = 1, 16, 8, 64
        q  = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32)
        k  = torch.randn_like(q)
        qw = torch.ones(D, device="cuda", dtype=torch.float32)
        kw = torch.ones(D, device="cuda", dtype=torch.float32)
        q_out, k_out = cute_c.fused_qk_norm(q, k, qw, kw, 1e-5)
        assert q_out.shape == q.shape


# ---------------------------------------------------------------------------
# Consistency: C extension vs Triton (when both available)
# ---------------------------------------------------------------------------

class TestCExtVsTriton:
    def _skip_if_no_triton(self):
        try:
            import triton  # noqa: F401
        except ImportError:
            pytest.skip("triton not installed")

    def test_rms_norm_vs_triton(self):
        self._skip_if_no_triton()
        from cutezimage.triton_kernels.rms_norm import rms_norm as triton_rms_norm

        torch.manual_seed(7)
        x = torch.randn(16, 3840, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(3840, device="cuda", dtype=torch.bfloat16)

        t_out = triton_rms_norm(x, w, 1e-5)
        c_out = cute_c.rms_norm(x, w, 1e-5)

        max_err = (t_out.float() - c_out.float()).abs().max().item()
        assert max_err < 0.02, f"Triton vs C extension max error: {max_err}"

    def test_silu_gate_vs_triton(self):
        self._skip_if_no_triton()
        from cutezimage.triton_kernels.fused_silu_gate_ffn import fused_silu_gate as triton_sg

        torch.manual_seed(8)
        x1 = torch.randn(8, 10240, device="cuda", dtype=torch.bfloat16)
        x3 = torch.randn_like(x1)

        t_out = triton_sg(x1, x3)
        c_out = cute_c.fused_silu_gate(x1, x3)

        max_err = (t_out.float() - c_out.float()).abs().max().item()
        assert max_err < 0.02, f"Triton vs C extension max error: {max_err}"
