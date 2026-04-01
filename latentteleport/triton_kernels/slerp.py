"""Fused Triton kernel for spherical linear interpolation (SLERP)."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _slerp_kernel(
    z0_ptr, z1_ptr, out_ptr,
    t: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    z0 = tl.load(z0_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    z1 = tl.load(z1_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # Compute norms
    z0_sq = tl.sum(z0 * z0)
    z1_sq = tl.sum(z1 * z1)
    z0_norm = tl.sqrt(z0_sq + 1e-8)
    z1_norm = tl.sqrt(z1_sq + 1e-8)

    # Normalized dot product
    dot = tl.sum((z0 / z0_norm) * (z1 / z1_norm))
    dot = tl.minimum(tl.maximum(dot, -1.0), 1.0)

    omega = tl.math.acos(dot)
    so = tl.math.sin(omega)

    # Fallback to lerp if vectors are nearly parallel
    use_lerp = so < 1e-6
    w0 = tl.where(use_lerp, 1.0 - t, tl.math.sin((1.0 - t) * omega) / (so + 1e-8))
    w1 = tl.where(use_lerp, t, tl.math.sin(t * omega) / (so + 1e-8))

    result = w0 * z0 + w1 * z1
    tl.store(out_ptr + offs, result, mask=mask)


def triton_slerp(z0: torch.Tensor, z1: torch.Tensor, t: float) -> torch.Tensor:
    """SLERP using Triton kernel. Falls back to PyTorch on CPU."""
    if not z0.is_cuda:
        from latentteleport.combiner import slerp
        return slerp(z0, z1, t)

    flat0 = z0.reshape(-1).contiguous()
    flat1 = z1.reshape(-1).contiguous()
    out = torch.empty_like(flat0)
    N = flat0.numel()
    BLOCK = 1024
    grid = ((N + BLOCK - 1) // BLOCK,)
    _slerp_kernel[grid](flat0, flat1, out, t, N, BLOCK)
    return out.reshape(z0.shape).to(z0.dtype)
