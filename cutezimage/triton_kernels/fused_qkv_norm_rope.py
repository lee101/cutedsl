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
