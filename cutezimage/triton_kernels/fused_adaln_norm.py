from __future__ import annotations

"""Fused AdaLN + RMS Norm kernel for Z-Image transformer blocks.

Z-Image uses adaptive layer normalization (AdaLN) for timestep conditioning:
    scale, gate = chunk(linear(adaln_input), 2)
    scale = 1 + scale
    gate = tanh(gate)
    normed = rms_norm(x) * scale

This kernel fuses RMS normalization with the scale multiplication into a
single kernel, avoiding materializing the intermediate normalized tensor.

For the modulated (refiner) blocks, it also handles per-token modulation
where noisy and clean tokens get different scale/gate values.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_adaln_rms_norm_kernel(
    X_ptr,
    Scale_ptr,
    W_ptr,
    Y_ptr,
    stride_xb,
    stride_xs,
    stride_xd,
    stride_sb,
    stride_sd,
    stride_yb,
    stride_ys,
    stride_yd,
    S,
    D,
    eps,
    HAS_WEIGHT: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fused AdaLN + RMS norm: out = rms_norm(x) * w * scale.

    Grid: (B * S,)
    Each program handles one (batch, seq) position.
    Scale is broadcast from (B, D) to (B, S, D).
    """
    pid = tl.program_id(0)
    batch_idx = pid // S
    seq_idx = pid % S

    x_base = X_ptr + batch_idx * stride_xb + seq_idx * stride_xs
    y_base = Y_ptr + batch_idx * stride_yb + seq_idx * stride_ys
    s_base = Scale_ptr + batch_idx * stride_sb

    d_offs = tl.arange(0, BLOCK_D)
    mask = d_offs < D

    # Load input row
    x = tl.load(x_base + d_offs * stride_xd, mask=mask, other=0.0)

    # RMS norm in FP32
    x_fp32 = x.to(tl.float32)
    sq_sum = tl.sum(x_fp32 * x_fp32, axis=0)
    rrms = tl.rsqrt(sq_sum / D + eps)
    normed = x_fp32 * rrms

    # Apply weight if present
    if HAS_WEIGHT:
        w = tl.load(W_ptr + d_offs, mask=mask, other=1.0)
        normed = normed.to(x.dtype) * w
    else:
        normed = normed.to(x.dtype)

    # Apply scale (1 + scale from AdaLN)
    scale = tl.load(s_base + d_offs * stride_sd, mask=mask, other=0.0)
    out = normed * scale

    tl.store(y_base + d_offs * stride_yd, out, mask=mask)


def fused_adaln_rms_norm(
    x: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Fused AdaLN modulated RMS normalization.

    Computes: rms_norm(x, weight) * scale

    where scale is typically (1 + adaln_modulation_output) broadcast
    from (B, D) to (B, S, D).

    Args:
        x: Input tensor, shape (B, S, D).
        scale: Modulation scale, shape (B, D) - broadcast over S.
        weight: Optional RMS norm weight, shape (D,).
        eps: Epsilon for numerical stability.

    Returns:
        Modulated normalized tensor, shape (B, S, D).
    """
    B, S, D = x.shape
    assert scale.shape == (B, D), f"scale shape {scale.shape} != ({B}, {D})"

    x = x.contiguous()
    scale = scale.contiguous()

    y = torch.empty_like(x)

    BLOCK_D = triton.next_power_of_2(D)
    has_weight = weight is not None

    if has_weight:
        weight = weight.contiguous()

    grid = (B * S,)

    _fused_adaln_rms_norm_kernel[grid](
        x,
        scale,
        weight if has_weight else x,  # dummy when no weight
        y,
        x.stride(0), x.stride(1), x.stride(2),
        scale.stride(0), scale.stride(1),
        y.stride(0), y.stride(1), y.stride(2),
        S, D, eps,
        HAS_WEIGHT=has_weight,
        BLOCK_D=BLOCK_D,
    )

    return y
