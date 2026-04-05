"""Benchmark tests for cutezimage Triton kernels.

Each test validates correctness AND measures Triton vs PyTorch reference timing.
Run with: pytest test_kernel_benchmarks.py -v -s  (need -s to see timing output)

These are unit-level microbenchmarks — they isolate each kernel at production
tensor sizes (dim=3840, heads=30, head_dim=128) to identify regression or
improvement from kernel changes.
"""

import pytest
import time
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for Triton kernel benchmarks",
)

# Production Z-Image sizes
DIM = 3840
HEADS = 30
HEAD_DIM = DIM // HEADS  # 128
HIDDEN_DIM = 10240
BATCH = 1
SEQ_LEN = 2048
WARMUP = 10
RUNS = 50


def _bench(fn, warmup=WARMUP, runs=RUNS):
    """Benchmark a callable, return avg_ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    avg = sum(times) / len(times)
    return avg


class TestRMSNormBenchmark:
    """Benchmark Triton RMS norm vs PyTorch reference at production size."""

    def _ref_rms_norm(self, x, weight, eps=1e-5):
        variance = x.float().pow(2).mean(-1, keepdim=True)
        normed = x * torch.rsqrt(variance + eps)
        return (normed * weight).to(x.dtype)

    def test_rms_norm_perf(self):
        from cutezimage.triton_kernels.rms_norm import rms_norm

        torch.manual_seed(42)
        x = torch.randn(BATCH, SEQ_LEN, DIM, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(DIM, device="cuda", dtype=torch.bfloat16)

        # Correctness
        ref = self._ref_rms_norm(x, w)
        out = rms_norm(x, w, eps=1e-5)
        max_err = (out.float() - ref.float()).abs().max().item()
        assert max_err < 0.05, f"Max error {max_err}"

        # Benchmark
        ref_ms = _bench(lambda: self._ref_rms_norm(x, w))
        triton_ms = _bench(lambda: rms_norm(x, w, eps=1e-5))
        speedup = ref_ms / triton_ms

        print(f"\n  RMS Norm ({BATCH}x{SEQ_LEN}x{DIM} bf16):")
        print(f"    PyTorch: {ref_ms:.3f} ms")
        print(f"    Triton:  {triton_ms:.3f} ms")
        print(f"    Speedup: {speedup:.2f}x")


class TestFusedSiLUGateBenchmark:
    """Benchmark fused SiLU gate vs separate PyTorch ops."""

    def test_silu_gate_perf(self):
        from cutezimage.triton_kernels.fused_silu_gate_ffn import fused_silu_gate

        torch.manual_seed(42)
        x1 = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM, device="cuda", dtype=torch.bfloat16)
        x3 = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM, device="cuda", dtype=torch.bfloat16)

        ref_fn = lambda: F.silu(x1) * x3
        triton_fn = lambda: fused_silu_gate(x1, x3)

        # Correctness
        ref = ref_fn()
        out = triton_fn()
        max_err = (out.float() - ref.float()).abs().max().item()
        assert max_err < 0.1, f"Max error {max_err}"

        ref_ms = _bench(ref_fn)
        triton_ms = _bench(triton_fn)
        speedup = ref_ms / triton_ms

        print(f"\n  Fused SiLU Gate ({BATCH}x{SEQ_LEN}x{HIDDEN_DIM} bf16):")
        print(f"    PyTorch: {ref_ms:.3f} ms")
        print(f"    Triton:  {triton_ms:.3f} ms")
        print(f"    Speedup: {speedup:.2f}x")

    def test_full_ffn_perf(self):
        from cutezimage.triton_kernels.fused_silu_gate_ffn import fused_silu_gate_ffn

        torch.manual_seed(42)
        x = torch.randn(BATCH, SEQ_LEN, DIM, device="cuda", dtype=torch.bfloat16)
        w1 = torch.randn(HIDDEN_DIM, DIM, device="cuda", dtype=torch.bfloat16)
        w2 = torch.randn(DIM, HIDDEN_DIM, device="cuda", dtype=torch.bfloat16)
        w3 = torch.randn(HIDDEN_DIM, DIM, device="cuda", dtype=torch.bfloat16)

        def ref_fn():
            x1 = F.linear(x, w1)
            x3 = F.linear(x, w3)
            gated = F.silu(x1) * x3
            return F.linear(gated, w2)

        triton_fn = lambda: fused_silu_gate_ffn(x, w1, w2, w3)

        # Correctness
        ref = ref_fn()
        out = triton_fn()
        max_err = (out.float() - ref.float()).abs().max().item()
        assert max_err < 1.0, f"Max error {max_err}"  # bf16 matmul accumulation

        ref_ms = _bench(ref_fn)
        triton_ms = _bench(triton_fn)
        speedup = ref_ms / triton_ms

        print(f"\n  Full FFN ({BATCH}x{SEQ_LEN}x{DIM}->{HIDDEN_DIM} bf16):")
        print(f"    PyTorch: {ref_ms:.3f} ms")
        print(f"    Triton:  {triton_ms:.3f} ms")
        print(f"    Speedup: {speedup:.2f}x")


class TestFusedAdaLNNormBenchmark:
    """Benchmark fused AdaLN + RMS norm vs separate ops."""

    def test_adaln_norm_perf(self):
        from cutezimage.triton_kernels.fused_adaln_norm import fused_adaln_rms_norm

        torch.manual_seed(42)
        x = torch.randn(BATCH, SEQ_LEN, DIM, device="cuda", dtype=torch.bfloat16)
        scale = torch.randn(BATCH, DIM, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(DIM, device="cuda", dtype=torch.bfloat16)

        def ref_fn():
            variance = x.float().pow(2).mean(-1, keepdim=True)
            normed = x * torch.rsqrt(variance + 1e-5)
            normed = normed.to(weight.dtype) * weight
            return normed * scale.unsqueeze(1)

        triton_fn = lambda: fused_adaln_rms_norm(x, scale, weight, eps=1e-5)

        # Correctness
        ref = ref_fn()
        out = triton_fn()
        max_err = (out.float() - ref.float()).abs().max().item()
        assert max_err < 0.1, f"Max error {max_err}"

        ref_ms = _bench(ref_fn)
        triton_ms = _bench(triton_fn)
        speedup = ref_ms / triton_ms

        print(f"\n  Fused AdaLN Norm ({BATCH}x{SEQ_LEN}x{DIM} bf16):")
        print(f"    PyTorch: {ref_ms:.3f} ms")
        print(f"    Triton:  {triton_ms:.3f} ms")
        print(f"    Speedup: {speedup:.2f}x")


class TestFusedQKNormBenchmark:
    """Benchmark fused QK norm vs separate per-head norms."""

    def _ref_per_head_rms_norm(self, x, weight, eps=1e-5):
        variance = x.float().pow(2).mean(-1, keepdim=True)
        normed = x * torch.rsqrt(variance + eps)
        return (normed * weight).to(x.dtype)

    def test_qk_norm_perf(self):
        from cutezimage.triton_kernels.fused_qkv_norm_rope import fused_qk_norm

        torch.manual_seed(42)
        q = torch.randn(BATCH, SEQ_LEN, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(BATCH, SEQ_LEN, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        qw = torch.randn(HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        kw = torch.randn(HEAD_DIM, device="cuda", dtype=torch.bfloat16)

        def ref_fn():
            q_n = self._ref_per_head_rms_norm(q, qw)
            k_n = self._ref_per_head_rms_norm(k, kw)
            return q_n, k_n

        triton_fn = lambda: fused_qk_norm(q, k, qw, kw, eps=1e-5)

        # Correctness
        q_ref, k_ref = ref_fn()
        q_out, k_out = triton_fn()
        q_err = (q_out.float() - q_ref.float()).abs().max().item()
        k_err = (k_out.float() - k_ref.float()).abs().max().item()
        assert q_err < 0.05, f"Q max error {q_err}"
        assert k_err < 0.05, f"K max error {k_err}"

        ref_ms = _bench(ref_fn)
        triton_ms = _bench(triton_fn)
        speedup = ref_ms / triton_ms

        print(f"\n  Fused QK Norm ({BATCH}x{SEQ_LEN}x{HEADS}x{HEAD_DIM} bf16):")
        print(f"    PyTorch: {ref_ms:.3f} ms")
        print(f"    Triton:  {triton_ms:.3f} ms")
        print(f"    Speedup: {speedup:.2f}x")


class TestRoPEComplexBenchmark:
    """Benchmark fused RoPE vs PyTorch complex multiply."""

    def _ref_rope(self, x, freqs_cis):
        with torch.amp.autocast("cuda", enabled=False):
            x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
            if freqs_cis.ndim < x_complex.ndim:
                freqs_cis = freqs_cis.unsqueeze(2)
            x_out = torch.view_as_real(x_complex * freqs_cis).flatten(x_complex.ndim - 1)
            return x_out.type_as(x)

    def test_rope_perf(self):
        from cutezimage.triton_kernels.rope_complex import apply_rope_complex

        torch.manual_seed(42)
        x = torch.randn(BATCH, SEQ_LEN, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        freqs = torch.randn(1, SEQ_LEN, HEAD_DIM // 2, device="cuda", dtype=torch.float32)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

        ref_fn = lambda: self._ref_rope(x, freqs_cis)
        triton_fn = lambda: apply_rope_complex(x, freqs_cis)

        # Correctness
        ref = ref_fn()
        out = triton_fn()
        max_err = (out.float() - ref.float()).abs().max().item()
        assert max_err < 0.05, f"Max error {max_err}"

        ref_ms = _bench(ref_fn)
        triton_ms = _bench(triton_fn)
        speedup = ref_ms / triton_ms

        print(f"\n  RoPE Complex ({BATCH}x{SEQ_LEN}x{HEADS}x{HEAD_DIM} bf16):")
        print(f"    PyTorch: {ref_ms:.3f} ms")
        print(f"    Triton:  {triton_ms:.3f} ms")
        print(f"    Speedup: {speedup:.2f}x")


class TestFusedQKNormRoPEBenchmark:
    """Benchmark the fused QK norm + RoPE kernel vs PyTorch reference."""

    def _ref_per_head_rms_norm(self, x, weight, eps=1e-5):
        variance = x.float().pow(2).mean(-1, keepdim=True)
        normed = x * torch.rsqrt(variance + eps)
        return (normed * weight).to(x.dtype)

    def _ref_rope(self, x, freqs_cis):
        with torch.amp.autocast("cuda", enabled=False):
            x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
            if freqs_cis.ndim < x_complex.ndim:
                freqs_cis = freqs_cis.unsqueeze(2)
            x_out = torch.view_as_real(x_complex * freqs_cis).flatten(x_complex.ndim - 1)
            return x_out.type_as(x)

    def test_fused_qk_norm_rope_perf(self):
        """Measures the fused QK norm + RoPE kernel (single launch)."""
        from cutezimage.triton_kernels.fused_qkv_norm_rope import fused_qk_norm, fused_qk_norm_rope

        torch.manual_seed(42)
        q = torch.randn(BATCH, SEQ_LEN, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(BATCH, SEQ_LEN, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        qw = torch.randn(HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        kw = torch.randn(HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        freqs = torch.randn(1, SEQ_LEN, HEAD_DIM // 2, device="cuda", dtype=torch.float32)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

        def ref_fn():
            q_n = self._ref_per_head_rms_norm(q, qw)
            k_n = self._ref_per_head_rms_norm(k, kw)
            q_r = self._ref_rope(q_n, freqs_cis)
            k_r = self._ref_rope(k_n, freqs_cis)
            return q_r, k_r

        norm_only_fn = lambda: fused_qk_norm(q, k, qw, kw, eps=1e-5)
        fused_fn = lambda: fused_qk_norm_rope(q, k, qw, kw, freqs_cis, eps=1e-5)

        # Correctness against PyTorch reference
        q_ref, k_ref = ref_fn()
        q_fused, k_fused = fused_fn()
        q_err = (q_fused.float() - q_ref.float()).abs().max().item()
        k_err = (k_fused.float() - k_ref.float()).abs().max().item()
        assert q_err < 0.1, f"Fused Q max error {q_err}"
        assert k_err < 0.1, f"Fused K max error {k_err}"

        ref_ms = _bench(ref_fn)
        norm_ms = _bench(norm_only_fn)
        fused_ms = _bench(fused_fn)

        print(f"\n  Fused QK Norm + RoPE ({BATCH}x{SEQ_LEN}x{HEADS}x{HEAD_DIM} bf16):")
        print(f"    PyTorch ref (norm+rope): {ref_ms:.3f} ms")
        print(f"    Triton (norm only):      {norm_ms:.3f} ms")
        print(f"    Triton (fused norm+rope):{fused_ms:.3f} ms")
        print(f"    Speedup vs PyTorch:      {ref_ms / fused_ms:.2f}x")
        print(f"    Overhead vs norm-only:   {fused_ms / norm_ms:.2f}x (RoPE is ~free)")
