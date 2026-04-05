from __future__ import annotations

"""Fused QK-Norm kernel for Z-Image attention.

Z-Image attention applies per-head RMS norm to Q and K after projection:
    q = norm_q(proj_q(x).unflatten(..., heads, head_dim))
    k = norm_k(proj_k(x).unflatten(..., heads, head_dim))

This kernel fuses both norms into a single kernel launch,
processing one (batch, seq, head) position per program.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_qk_norm_kernel(
    Q_ptr,          # (B, S, H, D)
    K_ptr,          # (B, S, H, D)
    QW_ptr,         # (D,)
    KW_ptr,         # (D,)
    Q_out_ptr,
    K_out_ptr,
    stride_qb, stride_qs, stride_qh, stride_qd,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_qob, stride_qos, stride_qoh, stride_qod,
    stride_kob, stride_kos, stride_koh, stride_kod,
    B, S, H,
    D: tl.constexpr,
    eps: tl.constexpr,
):
    """One program per (batch, seq, head) position."""
    pid = tl.program_id(0)

    # Decompose flat pid -> (b, s, h)
    h_idx = pid % H
    remainder = pid // H
    s_idx = remainder % S
    b_idx = remainder // S

    d_offs = tl.arange(0, D)

    # Load Q
    q_base = Q_ptr + b_idx * stride_qb + s_idx * stride_qs + h_idx * stride_qh
    q = tl.load(q_base + d_offs * stride_qd).to(tl.float32)

    # RMS norm Q
    q_sq = tl.sum(q * q, axis=0)
    q_rrms = tl.rsqrt(q_sq / D + eps)
    qw = tl.load(QW_ptr + d_offs).to(tl.float32)
    q_normed = q * q_rrms * qw

    # Load K
    k_base = K_ptr + b_idx * stride_kb + s_idx * stride_ks + h_idx * stride_kh
    k = tl.load(k_base + d_offs * stride_kd).to(tl.float32)

    # RMS norm K
    k_sq = tl.sum(k * k, axis=0)
    k_rrms = tl.rsqrt(k_sq / D + eps)
    kw = tl.load(KW_ptr + d_offs).to(tl.float32)
    k_normed = k * k_rrms * kw

    # Store
    q_out_base = Q_out_ptr + b_idx * stride_qob + s_idx * stride_qos + h_idx * stride_qoh
    k_out_base = K_out_ptr + b_idx * stride_kob + s_idx * stride_kos + h_idx * stride_koh
    tl.store(q_out_base + d_offs * stride_qod, q_normed.to(Q_ptr.dtype.element_ty))
    tl.store(k_out_base + d_offs * stride_kod, k_normed.to(K_ptr.dtype.element_ty))


def fused_qk_norm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused QK normalization in a single kernel launch.

    Applies per-head RMS norm to Q and K simultaneously.

    Args:
        q: Query tensor, shape (B, S, H, D).
        k: Key tensor, shape (B, S, H, D).
        q_weight: Q norm weight, shape (D,).
        k_weight: K norm weight, shape (D,).
        eps: Norm epsilon.

    Returns:
        Normalized Q and K, same shapes.
    """
    B, S, H, D = q.shape
    assert k.shape == q.shape
    assert D == q_weight.shape[0] == k_weight.shape[0]

    q = q.contiguous()
    k = k.contiguous()
    q_weight = q_weight.contiguous()
    k_weight = k_weight.contiguous()

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    grid = (B * S * H,)

    _fused_qk_norm_kernel[grid](
        q, k, q_weight, k_weight, q_out, k_out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        q_out.stride(0), q_out.stride(1), q_out.stride(2), q_out.stride(3),
        k_out.stride(0), k_out.stride(1), k_out.stride(2), k_out.stride(3),
        B, S, H,
        D=D,
        eps=eps,
    )

    return q_out, k_out


# ---------------------------------------------------------------------------
# Fused QK-Norm + RoPE kernel
# ---------------------------------------------------------------------------
# Combines per-head RMS norm and complex-valued RoPE into a single kernel
# launch, eliminating the intermediate memory round-trip between the two ops.

@triton.jit
def _fused_qk_norm_rope_kernel(
    Q_ptr,          # (B, S, H, D)
    K_ptr,          # (B, S, H, D)
    QW_ptr,         # (D,)
    KW_ptr,         # (D,)
    Freq_ptr,       # (1, S, 1, D) interleaved real/imag pairs
    Q_out_ptr,
    K_out_ptr,
    stride_qb, stride_qs, stride_qh, stride_qd,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_fb, stride_fs, stride_fh, stride_fd,
    stride_qob, stride_qos, stride_qoh, stride_qod,
    stride_kob, stride_kos, stride_koh, stride_kod,
    B, S, H,
    D: tl.constexpr,
    HALF_D: tl.constexpr,
    eps: tl.constexpr,
):
    """One program per (batch, seq, head) — fused RMS norm + complex RoPE.

    Strategy: load full D-dim vector, compute RMS norm, then apply RoPE
    by treating consecutive pairs as (real, imag). We load the full
    D-dimensional head vector once, normalize it, then extract even/odd
    elements for the complex multiply using stride-2 indexing.
    """
    pid = tl.program_id(0)

    h_idx = pid % H
    remainder = pid // H
    s_idx = remainder % S
    b_idx = remainder // S

    # Use HALF_D-based indexing throughout: load pairs as (real, imag)
    pair_offs = tl.arange(0, HALF_D)
    real_idx = 2 * pair_offs       # even indices: 0, 2, 4, ...
    imag_idx = 2 * pair_offs + 1   # odd indices:  1, 3, 5, ...

    # --- Q: load as pairs, compute RMS norm, apply RoPE ---
    q_base = Q_ptr + b_idx * stride_qb + s_idx * stride_qs + h_idx * stride_qh

    # Load even and odd elements separately (stride-2 gather)
    q_r_raw = tl.load(q_base + real_idx * stride_qd).to(tl.float32)
    q_i_raw = tl.load(q_base + imag_idx * stride_qd).to(tl.float32)

    # RMS norm: variance over all D elements = (sum(even^2) + sum(odd^2)) / D
    q_sq_sum = tl.sum(q_r_raw * q_r_raw, axis=0) + tl.sum(q_i_raw * q_i_raw, axis=0)
    q_rrms = tl.rsqrt(q_sq_sum / D + eps)

    # Load norm weights for even/odd positions
    qw_r = tl.load(QW_ptr + real_idx).to(tl.float32)
    qw_i = tl.load(QW_ptr + imag_idx).to(tl.float32)

    # Apply RMS norm
    q_r = q_r_raw * q_rrms * qw_r
    q_i = q_i_raw * q_rrms * qw_i

    # Load freqs (interleaved real/imag pairs)
    freq_base = Freq_ptr + s_idx * stride_fs
    f_real = tl.load(freq_base + real_idx * stride_fd).to(tl.float32)
    f_imag = tl.load(freq_base + imag_idx * stride_fd).to(tl.float32)

    # Complex multiply: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    q_out_r = q_r * f_real - q_i * f_imag
    q_out_i = q_r * f_imag + q_i * f_real

    # Store
    q_out_base = Q_out_ptr + b_idx * stride_qob + s_idx * stride_qos + h_idx * stride_qoh
    tl.store(q_out_base + real_idx * stride_qod, q_out_r.to(Q_ptr.dtype.element_ty))
    tl.store(q_out_base + imag_idx * stride_qod, q_out_i.to(Q_ptr.dtype.element_ty))

    # --- K: same pattern ---
    k_base = K_ptr + b_idx * stride_kb + s_idx * stride_ks + h_idx * stride_kh

    k_r_raw = tl.load(k_base + real_idx * stride_kd).to(tl.float32)
    k_i_raw = tl.load(k_base + imag_idx * stride_kd).to(tl.float32)

    k_sq_sum = tl.sum(k_r_raw * k_r_raw, axis=0) + tl.sum(k_i_raw * k_i_raw, axis=0)
    k_rrms = tl.rsqrt(k_sq_sum / D + eps)

    kw_r = tl.load(KW_ptr + real_idx).to(tl.float32)
    kw_i = tl.load(KW_ptr + imag_idx).to(tl.float32)

    k_r = k_r_raw * k_rrms * kw_r
    k_i = k_i_raw * k_rrms * kw_i

    k_out_r = k_r * f_real - k_i * f_imag
    k_out_i = k_r * f_imag + k_i * f_real

    k_out_base = K_out_ptr + b_idx * stride_kob + s_idx * stride_kos + h_idx * stride_koh
    tl.store(k_out_base + real_idx * stride_kod, k_out_r.to(K_ptr.dtype.element_ty))
    tl.store(k_out_base + imag_idx * stride_kod, k_out_i.to(K_ptr.dtype.element_ty))


def fused_qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    freqs_cis: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused QK normalization + complex RoPE in a single kernel launch.

    Combines per-head RMS norm and complex-valued rotary position embedding,
    eliminating the intermediate memory write/read between the two operations.

    Args:
        q: Query tensor, shape (B, S, H, D) where D is even.
        k: Key tensor, shape (B, S, H, D).
        q_weight: Q norm weight, shape (D,).
        k_weight: K norm weight, shape (D,).
        freqs_cis: Frequency tensor, complex-valued or interleaved real pairs.
        eps: Norm epsilon.

    Returns:
        Normalized and RoPE-applied Q and K, same shapes.
    """
    B, S, H, D = q.shape
    assert k.shape == q.shape
    assert D % 2 == 0
    assert D == q_weight.shape[0] == k_weight.shape[0]
    HALF_D = D // 2

    q = q.contiguous()
    k = k.contiguous()
    q_weight = q_weight.contiguous()
    k_weight = k_weight.contiguous()

    # Handle freqs_cis format — normalize to (1, S, 1, D) to match Q/K layout
    if freqs_cis.is_complex():
        freqs_real = torch.view_as_real(freqs_cis).contiguous()
        # Flatten last two dims: (..., HALF_D, 2) -> (..., D)
        freqs_flat = freqs_real.reshape(*freqs_cis.shape[:-1], D).contiguous()
    else:
        freqs_flat = freqs_cis.contiguous()

    # Reshape to exactly (1, S, 1, D) for correct stride mapping:
    # The kernel indexes freqs as freq_base + s_idx * stride_fs
    # where stride_fs must be the stride along the S dimension.
    if freqs_flat.ndim == 2:
        # (S, D) -> (1, S, 1, D)
        freqs_flat = freqs_flat.unsqueeze(0).unsqueeze(2)
    elif freqs_flat.ndim == 3:
        # (1, S, D) or (B, S, D) -> (*, S, 1, D)
        freqs_flat = freqs_flat.unsqueeze(2)
    # For 4D, assume already (B, S, H, D) or (1, S, 1, D)

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    grid = (B * S * H,)

    _fused_qk_norm_rope_kernel[grid](
        q, k, q_weight, k_weight, freqs_flat, q_out, k_out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        freqs_flat.stride(0), freqs_flat.stride(1), freqs_flat.stride(2), freqs_flat.stride(3),
        q_out.stride(0), q_out.stride(1), q_out.stride(2), q_out.stride(3),
        k_out.stride(0), k_out.stride(1), k_out.stride(2), k_out.stride(3),
        B, S, H,
        D=D,
        HALF_D=HALF_D,
        eps=eps,
    )

    return q_out, k_out
