#!/usr/bin/env python3
"""Benchmark: Fused QKV+Norm+RoPE kernel vs separate operations.

Compares the fused SM120 kernel against the current separate-operation
pipeline used in AcceleratedZImageTransformerBlock._apply_attention:
  1. QKV projection  (single packed linear)
  2. Unflatten into heads
  3. Per-head RMS norm on Q and K
  4. Complex RoPE on Q and K

Usage:
    python -m cutezimage.benchmark_fused_qkv [--batch 2] [--seq 1024]
"""

from __future__ import annotations

import argparse
import sys
import time

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Try to import Triton kernels -- graceful fallback for CPU-only envs
# ---------------------------------------------------------------------------

_HAS_TRITON = False
try:
    from cutezimage.triton_kernels.fused_qkv_rope_sm120 import (
        fused_qkv_rope_sm120,
        fused_qkv_rope_sm120_cublas,
    )
    from cutezimage.triton_kernels.fused_qkv_norm_rope import fused_qk_norm
    from cutezimage.triton_kernels.rope_complex import apply_rope_complex
    _HAS_TRITON = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Reference (separate ops) implementation
# ---------------------------------------------------------------------------

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() / rms * self.weight.float()).to(x.dtype)


def reference_qkv_norm_rope(
    x: torch.Tensor,
    qkv_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    freqs_cis: torch.Tensor,
    n_heads: int,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Separate operations matching the current pipeline."""
    B, S, D = x.shape
    head_dim = D // n_heads

    # 1. QKV projection
    qkv = F.linear(x, qkv_weight)
    q, k, v = torch.split(qkv, [D, D, D], dim=-1)

    # 2. Unflatten
    q = q.unflatten(-1, (n_heads, head_dim))
    k = k.unflatten(-1, (n_heads, head_dim))
    v = v.unflatten(-1, (n_heads, head_dim))

    # 3. Per-head RMS norm
    def rms_norm_fn(t, w):
        rms = torch.sqrt(t.float().pow(2).mean(-1, keepdim=True) + eps)
        return (t.float() / rms * w.float()).to(t.dtype)

    q = rms_norm_fn(q, q_norm_weight)
    k = rms_norm_fn(k, k_norm_weight)

    # 4. Complex RoPE
    qk = torch.cat([q, k], dim=2)
    x_complex = torch.view_as_complex(qk.float().reshape(*qk.shape[:-1], -1, 2))
    # freqs_cis should be complex, shape (1, S, 1, head_dim//2)
    x_out = torch.view_as_real(x_complex * freqs_cis).flatten(3)
    qk = x_out.to(q.dtype)
    q, k = torch.split(qk, [n_heads, n_heads], dim=2)

    return q, k, v


def reference_triton_separate(
    x: torch.Tensor,
    qkv_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    freqs_cis: torch.Tensor,
    n_heads: int,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Separate Triton kernels (current accelerated path)."""
    B, S, D = x.shape
    head_dim = D // n_heads

    # 1. QKV projection (cuBLAS)
    qkv = F.linear(x, qkv_weight)
    q, k, v = torch.split(qkv, [D, D, D], dim=-1)

    # 2. Unflatten
    q = q.unflatten(-1, (n_heads, head_dim))
    k = k.unflatten(-1, (n_heads, head_dim))
    v = v.unflatten(-1, (n_heads, head_dim))

    # 3. Triton QK norm
    q, k = fused_qk_norm(q, k, q_norm_weight, k_norm_weight, eps=eps)

    # 4. Triton Complex RoPE
    qk = torch.cat([q, k], dim=2)
    qk = apply_rope_complex(qk, freqs_cis)
    q, k = torch.split(qk, [n_heads, n_heads], dim=2)

    return q, k, v


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------

def cuda_timer(fn, warmup: int = 10, iters: int = 100) -> float:
    """Time a function using CUDA events, return median ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    return times[len(times) // 2]  # median


def time_component(fn, warmup: int = 10, iters: int = 100) -> float:
    """Time a single component."""
    return cuda_timer(fn, warmup, iters)


# ---------------------------------------------------------------------------
# Build test data
# ---------------------------------------------------------------------------

def make_test_data(B: int, S: int, D: int = 3840, n_heads: int = 30, dtype=torch.bfloat16):
    device = "cuda"
    head_dim = D // n_heads

    x = torch.randn(B, S, D, device=device, dtype=dtype)
    qkv_weight = torch.randn(3 * D, D, device=device, dtype=dtype) * 0.02
    q_norm_w = torch.ones(head_dim, device=device, dtype=dtype)
    k_norm_w = torch.ones(head_dim, device=device, dtype=dtype)

    # Complex RoPE frequencies
    half_hd = head_dim // 2
    theta = 256.0
    freqs = 1.0 / (theta ** (torch.arange(0, half_hd, device=device).float() / half_hd))
    t = torch.arange(S, device=device).float()
    freqs_outer = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs_outer), freqs_outer)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)  # (1, S, 1, half_hd)

    return x, qkv_weight, q_norm_w, k_norm_w, freqs_cis


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------

def check_correctness(B: int, S: int, D: int = 3840, n_heads: int = 30):
    print("=" * 60)
    print("Correctness check")
    print("=" * 60)

    x, qkv_weight, q_norm_w, k_norm_w, freqs_cis = make_test_data(B, S, D, n_heads)

    # Reference
    q_ref, k_ref, v_ref = reference_qkv_norm_rope(
        x, qkv_weight, q_norm_w, k_norm_w, freqs_cis, n_heads,
    )

    # Fused Triton GEMM
    q_fused, k_fused, v_fused = fused_qkv_rope_sm120(
        x, qkv_weight, q_norm_w, k_norm_w, freqs_cis, n_heads=n_heads,
    )

    # Fused cuBLAS + Triton epilogue
    q_cublas, k_cublas, v_cublas = fused_qkv_rope_sm120_cublas(
        x, qkv_weight, q_norm_w, k_norm_w, freqs_cis, n_heads=n_heads,
    )

    def check(name, ref, test):
        maxdiff = (ref.float() - test.float()).abs().max().item()
        reldiff = maxdiff / (ref.float().abs().max().item() + 1e-8)
        status = "PASS" if reldiff < 0.02 else "FAIL"  # bf16 tolerance
        print(f"  {name}: max_abs={maxdiff:.6f}, rel={reldiff:.6f}  [{status}]")

    print("vs fused_qkv_rope_sm120 (Triton GEMM):")
    check("Q", q_ref, q_fused)
    check("K", k_ref, k_fused)
    check("V", v_ref, v_fused)

    print("vs fused_qkv_rope_sm120_cublas (cuBLAS GEMM):")
    check("Q", q_ref, q_cublas)
    check("K", k_ref, k_cublas)
    check("V", v_ref, v_cublas)
    print()


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark(B: int, S: int, D: int = 3840, n_heads: int = 30, iters: int = 100):
    print("=" * 60)
    print(f"Benchmark: B={B}, S={S}, D={D}, n_heads={n_heads}")
    print(f"  M = B*S = {B*S}, head_dim = {D // n_heads}")
    print("=" * 60)

    x, qkv_weight, q_norm_w, k_norm_w, freqs_cis = make_test_data(B, S, D, n_heads)
    head_dim = D // n_heads

    # --------------- Component timings (separate ops) ---------------
    print("\n--- Separate operation timings ---")

    # QKV projection only
    def fn_qkv_proj():
        return F.linear(x, qkv_weight)

    t_proj = time_component(fn_qkv_proj, iters=iters)
    print(f"  QKV projection (cuBLAS):     {t_proj:.3f} ms")

    # Prepare data for norm/rope timing
    qkv = F.linear(x, qkv_weight)
    q_raw, k_raw, v_raw = torch.split(qkv, [D, D, D], dim=-1)
    q_4d = q_raw.unflatten(-1, (n_heads, head_dim)).contiguous()
    k_4d = k_raw.unflatten(-1, (n_heads, head_dim)).contiguous()

    if _HAS_TRITON:
        # QK Norm (Triton)
        def fn_qk_norm():
            return fused_qk_norm(q_4d, k_4d, q_norm_w, k_norm_w, eps=1e-5)

        t_norm = time_component(fn_qk_norm, iters=iters)
        print(f"  QK norm (Triton fused):      {t_norm:.3f} ms")

        # RoPE (Triton)
        q_normed, k_normed = fused_qk_norm(q_4d, k_4d, q_norm_w, k_norm_w, eps=1e-5)
        qk_cat = torch.cat([q_normed, k_normed], dim=2).contiguous()

        def fn_rope():
            return apply_rope_complex(qk_cat, freqs_cis)

        t_rope = time_component(fn_rope, iters=iters)
        print(f"  RoPE (Triton complex):       {t_rope:.3f} ms")

        # Total separate (Triton)
        def fn_separate_triton():
            return reference_triton_separate(
                x, qkv_weight, q_norm_w, k_norm_w, freqs_cis, n_heads,
            )

        t_sep_triton = time_component(fn_separate_triton, iters=iters)
        print(f"  Total separate (Triton):     {t_sep_triton:.3f} ms")
    else:
        t_norm = t_rope = t_sep_triton = float("nan")
        print("  (Triton not available, skipping Triton separate timings)")

    # QK Norm (PyTorch reference)
    def fn_qk_norm_pt():
        rms_q = torch.sqrt(q_4d.float().pow(2).mean(-1, keepdim=True) + 1e-5)
        rms_k = torch.sqrt(k_4d.float().pow(2).mean(-1, keepdim=True) + 1e-5)
        return (q_4d.float() / rms_q * q_norm_w.float()).to(q_4d.dtype), \
               (k_4d.float() / rms_k * k_norm_w.float()).to(k_4d.dtype)

    t_norm_pt = time_component(fn_qk_norm_pt, iters=iters)
    print(f"  QK norm (PyTorch):           {t_norm_pt:.3f} ms")

    # Total separate (PyTorch)
    def fn_separate_pytorch():
        return reference_qkv_norm_rope(
            x, qkv_weight, q_norm_w, k_norm_w, freqs_cis, n_heads,
        )

    t_sep_pytorch = time_component(fn_separate_pytorch, iters=iters)
    print(f"  Total separate (PyTorch):    {t_sep_pytorch:.3f} ms")

    # --------------- Fused kernel timings ---------------
    print("\n--- Fused kernel timings ---")

    if _HAS_TRITON:
        # Fused Triton GEMM variant
        def fn_fused_triton():
            return fused_qkv_rope_sm120(
                x, qkv_weight, q_norm_w, k_norm_w, freqs_cis, n_heads=n_heads,
            )

        t_fused_triton = time_component(fn_fused_triton, iters=iters)
        print(f"  Fused (Triton GEMM):         {t_fused_triton:.3f} ms")

        # Fused cuBLAS + Triton epilogue variant
        def fn_fused_cublas():
            return fused_qkv_rope_sm120_cublas(
                x, qkv_weight, q_norm_w, k_norm_w, freqs_cis, n_heads=n_heads,
            )

        t_fused_cublas = time_component(fn_fused_cublas, iters=iters)
        print(f"  Fused (cuBLAS + epilogue):   {t_fused_cublas:.3f} ms")
    else:
        t_fused_triton = t_fused_cublas = float("nan")
        print("  (Triton not available, skipping fused kernel timings)")

    # --------------- Summary ---------------
    print("\n--- Summary ---")
    print(f"  {'Operation':<35} {'Time (ms)':>10}")
    print(f"  {'-'*35} {'-'*10}")
    print(f"  {'QKV projection (cuBLAS)':<35} {t_proj:>10.3f}")
    if _HAS_TRITON:
        print(f"  {'QK norm (Triton)':<35} {t_norm:>10.3f}")
        print(f"  {'RoPE (Triton)':<35} {t_rope:>10.3f}")
        print(f"  {'Proj + Norm + RoPE sum':<35} {t_proj + t_norm + t_rope:>10.3f}")
        print(f"  {'Total separate (Triton e2e)':<35} {t_sep_triton:>10.3f}")
    print(f"  {'QK norm (PyTorch)':<35} {t_norm_pt:>10.3f}")
    print(f"  {'Total separate (PyTorch e2e)':<35} {t_sep_pytorch:>10.3f}")
    if _HAS_TRITON:
        print(f"  {'Fused Triton GEMM':<35} {t_fused_triton:>10.3f}")
        print(f"  {'Fused cuBLAS + epilogue':<35} {t_fused_cublas:>10.3f}")
        print()
        if t_sep_triton > 0:
            speedup_triton = t_sep_triton / t_fused_triton
            speedup_cublas = t_sep_triton / t_fused_cublas
            print(f"  Speedup (Triton GEMM fused vs separate Triton):    {speedup_triton:.2f}x")
            print(f"  Speedup (cuBLAS fused vs separate Triton):         {speedup_cublas:.2f}x")
        speedup_pt_triton = t_sep_pytorch / t_fused_triton
        speedup_pt_cublas = t_sep_pytorch / t_fused_cublas
        print(f"  Speedup (Triton GEMM fused vs separate PyTorch):   {speedup_pt_triton:.2f}x")
        print(f"  Speedup (cuBLAS fused vs separate PyTorch):        {speedup_pt_cublas:.2f}x")
    print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark fused QKV+Norm+RoPE kernel")
    parser.add_argument("--batch", type=int, default=2, help="Batch size")
    parser.add_argument("--seq", type=int, default=1024, help="Sequence length")
    parser.add_argument("--dim", type=int, default=3840, help="Model dimension")
    parser.add_argument("--n-heads", type=int, default=30, help="Number of heads")
    parser.add_argument("--iters", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--skip-correctness", action="store_true", help="Skip correctness check")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available. This benchmark requires a GPU.")
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    if _HAS_TRITON:
        import triton
        print(f"Triton version: {triton.__version__}")
    else:
        print("Triton: NOT AVAILABLE (only PyTorch baselines will run)")
    print()

    if not args.skip_correctness and _HAS_TRITON:
        check_correctness(args.batch, min(args.seq, 256), args.dim, args.n_heads)

    # Run benchmark at multiple sequence lengths
    for seq_len in [256, 512, args.seq]:
        if seq_len > args.seq:
            continue
        benchmark(args.batch, seq_len, args.dim, args.n_heads, args.iters)


if __name__ == "__main__":
    main()
