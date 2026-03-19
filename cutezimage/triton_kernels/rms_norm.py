"""Triton RMS LayerNorm kernel for Z-Image transformer.

Z-Image uses standard RMS norm (with optional affine weight):
    out = weight * (x * rsqrt(mean(x^2) + eps))

Variance computed in FP32 for numerical stability.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_fwd_kernel(
    X_ptr,
    W_ptr,
    Y_ptr,
    stride_x_row,
    stride_y_row,
    N,
    eps,
    HAS_WEIGHT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    x_row_ptr = X_ptr + row_idx * stride_x_row
    y_row_ptr = Y_ptr + row_idx * stride_y_row

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N

    x = tl.load(x_row_ptr + col_offsets, mask=mask, other=0.0)

    x_fp32 = x.to(tl.float32)
    sq_sum = tl.sum(x_fp32 * x_fp32, axis=0)
    mean_sq = sq_sum / N
    rrms = tl.rsqrt(mean_sq + eps)

    out = x_fp32 * rrms

    if HAS_WEIGHT:
        w = tl.load(W_ptr + col_offsets, mask=mask, other=1.0)
        out = out.to(x.dtype) * w
    else:
        out = out.to(x.dtype)

    tl.store(y_row_ptr + col_offsets, out, mask=mask)


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Apply RMS LayerNorm using a Triton kernel.

    Args:
        x: Input tensor of shape (..., N).
        weight: Optional scale parameter of shape (N,).
        eps: Epsilon for numerical stability.

    Returns:
        Normalized tensor of same shape and dtype as x.
    """
    orig_shape = x.shape
    N = orig_shape[-1]

    x_2d = x.reshape(-1, N).contiguous()
    num_rows = x_2d.shape[0]

    y = torch.empty_like(x_2d)

    BLOCK_SIZE = triton.next_power_of_2(N)
    has_weight = weight is not None

    if has_weight:
        weight = weight.contiguous()

    _rms_norm_fwd_kernel[(num_rows,)](
        x_2d,
        weight if has_weight else x_2d,  # dummy pointer when no weight
        y,
        x_2d.stride(0),
        y.stride(0),
        N,
        eps,
        HAS_WEIGHT=has_weight,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return y.reshape(orig_shape)
