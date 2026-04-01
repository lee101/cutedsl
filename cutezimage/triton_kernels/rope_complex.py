from __future__ import annotations

"""Fused complex-valued RoPE kernel for Z-Image transformer.

Z-Image uses complex-multiplication RoPE:
    x_complex = view_as_complex(x.reshape(..., -1, 2))
    x_out = view_as_real(x_complex * freqs_cis).flatten(...)

This is different from the rotate_half convention used in LLMs.
The complex multiply is: (a+bi)(c+di) = (ac-bd) + (ad+bc)i

This kernel fuses the reshape, complex multiply, and flatten into
a single pass, avoiding intermediate complex tensor allocations.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_complex_fwd_kernel(
    X_ptr,       # input: (B, S, H, D)
    Freq_ptr,    # freqs_cis: (1, S, 1, D//2) as interleaved real/imag
    Out_ptr,
    stride_xb, stride_xs, stride_xh, stride_xd,
    stride_fb, stride_fs, stride_fh, stride_fd,
    stride_ob, stride_os, stride_oh, stride_od,
    S,
    H,
    D: tl.constexpr,        # head dimension (must be even)
    HALF_D: tl.constexpr,   # D // 2
    BLOCK_S: tl.constexpr,
):
    """Apply complex-valued RoPE to a single (batch, head) tile.

    For each pair (x[2i], x[2i+1]) and freq (cos, sin):
        out_real = x_real * cos - x_imag * sin
        out_imag = x_real * sin + x_imag * cos
    """
    pid = tl.program_id(0)
    num_s_blocks = tl.cdiv(S, BLOCK_S)

    pid_b = pid // (H * num_s_blocks)
    remainder = pid % (H * num_s_blocks)
    pid_h = remainder // num_s_blocks
    pid_s = remainder % num_s_blocks

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < S

    d_offs = tl.arange(0, HALF_D)

    # Pointers for real and imaginary parts
    # x is stored as (B, S, H, D) where D pairs are (real, imag)
    x_base = X_ptr + pid_b * stride_xb + pid_h * stride_xh
    o_base = Out_ptr + pid_b * stride_ob + pid_h * stride_oh

    x_real_ptrs = x_base + s_offs[:, None] * stride_xs + (2 * d_offs[None, :]) * stride_xd
    x_imag_ptrs = x_base + s_offs[:, None] * stride_xs + (2 * d_offs[None, :] + 1) * stride_xd

    x_real = tl.load(x_real_ptrs, mask=s_mask[:, None], other=0.0).to(tl.float32)
    x_imag = tl.load(x_imag_ptrs, mask=s_mask[:, None], other=0.0).to(tl.float32)

    # Load freqs_cis - stored as complex (real, imag) pairs
    # freqs shape: (1, S, 1, HALF_D) with real/imag interleaved
    freq_real_ptrs = Freq_ptr + s_offs[:, None] * stride_fs + (2 * d_offs[None, :]) * stride_fd
    freq_imag_ptrs = Freq_ptr + s_offs[:, None] * stride_fs + (2 * d_offs[None, :] + 1) * stride_fd

    f_real = tl.load(freq_real_ptrs, mask=s_mask[:, None], other=1.0).to(tl.float32)
    f_imag = tl.load(freq_imag_ptrs, mask=s_mask[:, None], other=0.0).to(tl.float32)

    # Complex multiply: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    out_real = x_real * f_real - x_imag * f_imag
    out_imag = x_real * f_imag + x_imag * f_real

    out_real_ptrs = o_base + s_offs[:, None] * stride_os + (2 * d_offs[None, :]) * stride_od
    out_imag_ptrs = o_base + s_offs[:, None] * stride_os + (2 * d_offs[None, :] + 1) * stride_od

    tl.store(out_real_ptrs, out_real.to(X_ptr.dtype.element_ty), mask=s_mask[:, None])
    tl.store(out_imag_ptrs, out_imag.to(X_ptr.dtype.element_ty), mask=s_mask[:, None])


def apply_rope_complex(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """Apply complex-valued RoPE using a fused Triton kernel.

    Replaces the diffusers implementation:
        x_complex = view_as_complex(x.float().reshape(..., -1, 2))
        x_out = view_as_real(x_complex * freqs_cis).flatten(3)

    Args:
        x: Input tensor, shape (B, S, H, D) where D is even.
        freqs_cis: Frequency tensor, broadcastable to (B, S, H, D)
            as real-valued pairs (stored interleaved or as complex).

    Returns:
        RoPE-applied tensor, same shape and dtype as x.
    """
    B, S, H, D = x.shape
    assert D % 2 == 0, f"Head dimension must be even, got {D}"
    HALF_D = D // 2

    x = x.contiguous()
    out = torch.empty_like(x)

    # Handle freqs_cis broadcasting
    # It may come as complex-valued or as real pairs
    if freqs_cis.is_complex():
        # Convert complex to interleaved real pairs
        freqs_real = torch.view_as_real(freqs_cis).contiguous()
        # Shape: (..., HALF_D, 2) -> need to flatten last two dims
        freqs_flat = freqs_real.reshape(*freqs_cis.shape[:-1], D).contiguous()
    else:
        freqs_flat = freqs_cis.contiguous()

    # Ensure freqs is 4D and broadcastable
    while freqs_flat.ndim < 4:
        freqs_flat = freqs_flat.unsqueeze(0)

    BLOCK_S = triton.next_power_of_2(min(S, 128))
    num_s_blocks = triton.cdiv(S, BLOCK_S)
    grid = (B * H * num_s_blocks,)

    _rope_complex_fwd_kernel[grid](
        x, freqs_flat, out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        freqs_flat.stride(0), freqs_flat.stride(1), freqs_flat.stride(2), freqs_flat.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        S, H, D=D, HALF_D=HALF_D, BLOCK_S=BLOCK_S,
    )

    return out
