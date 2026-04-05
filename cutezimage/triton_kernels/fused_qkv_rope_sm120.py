from __future__ import annotations

"""Fused QKV Projection + QK RMS Norm + Complex RoPE kernel for Z-Image.

Targets SM120 (Blackwell) which has 228KB shared memory per SM, enabling
larger tile sizes for the GEMM and post-projection operations.

The kernel fuses three operations that are otherwise separate kernel launches:
  1. QKV linear projection  (x @ W_qkv^T)
  2. Per-head RMS norm on Q and K
  3. Complex-valued RoPE rotation on Q and K

For Z-Image defaults: D=3840, n_heads=30, head_dim=128.
Packed QKV weight is (3*D, D) = (11520, 3840).

Design:
  - The GEMM is done as a tiled matmul in Triton, computing a (BLOCK_M, 3*D)
    output tile from (BLOCK_M, D) input x (D, 3*D) weight.
  - After GEMM, the output is split into Q, K, V regions.
  - Q and K go through per-head RMS norm + Complex RoPE in registers.
  - V is written directly.

Since D=3840 is large, we use a two-stage approach:
  - Stage 1: GEMM using standard tiled matmul with Triton's dot product
  - Stage 2: Epilogue kernel that applies QK-norm + RoPE on the GEMM output

This avoids needing to hold the full 3*D output in shared memory at once.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Stage 1: Tiled GEMM for QKV projection
# This is a standard matmul: out[m, n] = sum_k x[m, k] * W[n, k]
# where W is stored as (N, K) i.e. (3*D, D), so we compute x @ W^T
# ---------------------------------------------------------------------------

@triton.jit
def _qkv_gemm_kernel(
    X_ptr,          # (M, K) -- input, where M = B*S
    W_ptr,          # (N, K) -- packed QKV weight, N = 3*D
    Out_ptr,        # (M, N) -- output
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Tiled GEMM: Out = X @ W^T."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offs = tl.arange(0, BLOCK_K)

    m_mask = m_offs < M
    n_mask = n_offs < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + k_offs
        k_mask = k_idx < K

        # Load X tile: (BLOCK_M, BLOCK_K)
        x = tl.load(
            X_ptr + m_offs[:, None] * stride_xm + k_idx[None, :] * stride_xk,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )

        # Load W tile: (BLOCK_N, BLOCK_K) -> transpose to (BLOCK_K, BLOCK_N)
        w = tl.load(
            W_ptr + n_offs[None, :] * stride_wn + k_idx[:, None] * stride_wk,
            mask=n_mask[None, :] & k_mask[:, None],
            other=0.0,
        )

        acc += tl.dot(x, w)

    # Store output tile
    out_ptrs = Out_ptr + m_offs[:, None] * stride_om + n_offs[None, :] * stride_on
    tl.store(out_ptrs, acc.to(Out_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :])


# ---------------------------------------------------------------------------
# Stage 2: Fused QK-Norm + Complex RoPE epilogue
# Operates on the GEMM output, applies per-head RMS norm to Q/K regions,
# then applies complex RoPE rotation.
# ---------------------------------------------------------------------------

@triton.jit
def _qk_norm_rope_epilogue_kernel(
    QKV_ptr,        # (M, 3*D) -- GEMM output, M = B*S
    Freq_ptr,       # (1, S, 1, head_dim) as interleaved real/imag
    QW_ptr,         # (head_dim,) -- Q norm weight
    KW_ptr,         # (head_dim,) -- K norm weight
    Q_out_ptr,      # (B, S, H, head_dim)
    K_out_ptr,      # (B, S, H, head_dim)
    V_out_ptr,      # (B, S, H, head_dim)
    B, S, H,
    D: tl.constexpr,          # model dim (3840)
    HEAD_DIM: tl.constexpr,   # 128
    HALF_HD: tl.constexpr,    # 64
    stride_qkv_m, stride_qkv_n,
    stride_fs, stride_fd,
    stride_ob, stride_os, stride_oh, stride_od,
    eps: tl.constexpr,
    HAS_ROPE: tl.constexpr,
):
    """One program per (batch, seq, head).

    Reads Q/K/V from the flat GEMM output, applies norm + RoPE to Q/K,
    and writes all three to their output tensors in (B, S, H, D) layout.
    """
    pid = tl.program_id(0)

    h_idx = pid % H
    remainder = pid // H
    s_idx = remainder % S
    b_idx = remainder // S

    m_idx = b_idx * S + s_idx  # row in the (M, 3*D) GEMM output

    d_offs = tl.arange(0, HEAD_DIM)

    # Q region: columns [h_idx * HEAD_DIM, (h_idx+1) * HEAD_DIM)
    q_col = h_idx * HEAD_DIM + d_offs
    q_ptrs = QKV_ptr + m_idx * stride_qkv_m + q_col * stride_qkv_n
    q = tl.load(q_ptrs).to(tl.float32)

    # K region: columns [D + h_idx * HEAD_DIM, D + (h_idx+1) * HEAD_DIM)
    k_col = D + h_idx * HEAD_DIM + d_offs
    k_ptrs = QKV_ptr + m_idx * stride_qkv_m + k_col * stride_qkv_n
    k = tl.load(k_ptrs).to(tl.float32)

    # V region: columns [2*D + h_idx * HEAD_DIM, 2*D + (h_idx+1) * HEAD_DIM)
    v_col = 2 * D + h_idx * HEAD_DIM + d_offs
    v_ptrs = QKV_ptr + m_idx * stride_qkv_m + v_col * stride_qkv_n
    v = tl.load(v_ptrs)

    # --- Per-head RMS Norm on Q ---
    q_sq = tl.sum(q * q, axis=0)
    q_rrms = tl.rsqrt(q_sq / HEAD_DIM + eps)
    qw = tl.load(QW_ptr + d_offs).to(tl.float32)
    q = q * q_rrms * qw

    # --- Per-head RMS Norm on K ---
    k_sq = tl.sum(k * k, axis=0)
    k_rrms = tl.rsqrt(k_sq / HEAD_DIM + eps)
    kw = tl.load(KW_ptr + d_offs).to(tl.float32)
    k = k * k_rrms * kw

    # --- Complex RoPE on Q and K ---
    if HAS_ROPE:
        pair_offs = tl.arange(0, HALF_HD)

        # Load frequencies for this sequence position
        freq_real_ptrs = Freq_ptr + s_idx * stride_fs + (2 * pair_offs) * stride_fd
        freq_imag_ptrs = Freq_ptr + s_idx * stride_fs + (2 * pair_offs + 1) * stride_fd
        f_real = tl.load(freq_real_ptrs).to(tl.float32)
        f_imag = tl.load(freq_imag_ptrs).to(tl.float32)

        # Store Q to output first, then reload with strided access for RoPE
        # This avoids Triton's inability to gather from register-resident tensors
        q_out_base = Q_out_ptr + b_idx * stride_ob + s_idx * stride_os + h_idx * stride_oh
        tl.store(q_out_base + d_offs * stride_od, q.to(Q_out_ptr.dtype.element_ty))

        # Reload even/odd elements with stride-2 access
        q_re = tl.load(q_out_base + (2 * pair_offs) * stride_od).to(tl.float32)
        q_im = tl.load(q_out_base + (2 * pair_offs + 1) * stride_od).to(tl.float32)

        # Complex multiply: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        q_out_re = q_re * f_real - q_im * f_imag
        q_out_im = q_re * f_imag + q_im * f_real

        # Write interleaved result back
        tl.store(q_out_base + (2 * pair_offs) * stride_od, q_out_re.to(Q_out_ptr.dtype.element_ty))
        tl.store(q_out_base + (2 * pair_offs + 1) * stride_od, q_out_im.to(Q_out_ptr.dtype.element_ty))

        # K: same store-reload-RoPE approach
        k_out_base = K_out_ptr + b_idx * stride_ob + s_idx * stride_os + h_idx * stride_oh
        tl.store(k_out_base + d_offs * stride_od, k.to(K_out_ptr.dtype.element_ty))

        k_re = tl.load(k_out_base + (2 * pair_offs) * stride_od).to(tl.float32)
        k_im = tl.load(k_out_base + (2 * pair_offs + 1) * stride_od).to(tl.float32)

        k_out_re = k_re * f_real - k_im * f_imag
        k_out_im = k_re * f_imag + k_im * f_real

        tl.store(k_out_base + (2 * pair_offs) * stride_od, k_out_re.to(K_out_ptr.dtype.element_ty))
        tl.store(k_out_base + (2 * pair_offs + 1) * stride_od, k_out_im.to(K_out_ptr.dtype.element_ty))

    # --- Write outputs (V always, Q/K only if no RoPE since RoPE path already wrote them) ---
    out_base = b_idx * stride_ob + s_idx * stride_os + h_idx * stride_oh
    v_out_base = V_out_ptr + out_base
    tl.store(v_out_base + d_offs * stride_od, v)

    if not HAS_ROPE:
        q_out_base = Q_out_ptr + out_base
        k_out_base = K_out_ptr + out_base
        tl.store(q_out_base + d_offs * stride_od, q.to(Q_out_ptr.dtype.element_ty))
        tl.store(k_out_base + d_offs * stride_od, k.to(K_out_ptr.dtype.element_ty))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fused_qkv_rope_sm120(
    x: torch.Tensor,
    qkv_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    freqs_cis: torch.Tensor | None = None,
    eps: float = 1e-5,
    n_heads: int = 30,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused QKV projection + QK RMS norm + Complex RoPE.

    Combines three separate kernel launches into two (GEMM + epilogue),
    eliminating intermediate memory traffic for the QKV output.

    Args:
        x: Input tensor, shape (B, S, D) where D=3840.
        qkv_weight: Packed QKV weight, shape (3*D, D). Rows ordered as
            [W_q; W_k; W_v].
        q_norm_weight: Q RMS norm weight, shape (head_dim,).
        k_norm_weight: K RMS norm weight, shape (head_dim,).
        freqs_cis: Complex RoPE frequencies, broadcastable to (1, S, 1, head_dim).
            Can be complex-valued or interleaved real pairs. Pass None to skip RoPE.
        eps: RMS norm epsilon.
        n_heads: Number of attention heads.

    Returns:
        (Q, K, V) each with shape (B, S, n_heads, head_dim), ready for attention.
    """
    assert x.ndim == 3, f"Expected (B, S, D) input, got shape {x.shape}"
    B, S, D = x.shape
    N = 3 * D  # total output dim
    head_dim = D // n_heads
    assert head_dim % 2 == 0, f"Head dim must be even for complex RoPE, got {head_dim}"
    half_hd = head_dim // 2

    assert qkv_weight.shape == (N, D), (
        f"Expected qkv_weight shape ({N}, {D}), got {qkv_weight.shape}"
    )

    x_flat = x.reshape(B * S, D).contiguous()
    qkv_weight = qkv_weight.contiguous()

    # ---- Stage 1: GEMM ----
    M = B * S
    qkv_out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # SM120-aware tile sizes: larger tiles for 228KB shared memory
    # BLOCK_K=128 means each tile uses BLOCK_M*128 + BLOCK_N*128 elements in smem
    # With bf16 (2 bytes): (64*128 + 128*128)*2 = 49152 bytes ~ 48KB per tile pair
    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 128

    grid_gemm = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _qkv_gemm_kernel[grid_gemm](
        x_flat, qkv_weight, qkv_out,
        M, N, D,
        x_flat.stride(0), x_flat.stride(1),
        qkv_weight.stride(0), qkv_weight.stride(1),
        qkv_out.stride(0), qkv_out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    # ---- Stage 2: QK-Norm + RoPE epilogue ----
    Q_out = torch.empty((B, S, n_heads, head_dim), device=x.device, dtype=x.dtype)
    K_out = torch.empty((B, S, n_heads, head_dim), device=x.device, dtype=x.dtype)
    V_out = torch.empty((B, S, n_heads, head_dim), device=x.device, dtype=x.dtype)

    q_norm_weight = q_norm_weight.contiguous()
    k_norm_weight = k_norm_weight.contiguous()

    has_rope = freqs_cis is not None

    if has_rope:
        if freqs_cis.is_complex():
            freqs_real = torch.view_as_real(freqs_cis).contiguous()
            freqs_flat = freqs_real.reshape(*freqs_cis.shape[:-1], head_dim).contiguous()
        else:
            freqs_flat = freqs_cis.contiguous()
        # Ensure 4D
        while freqs_flat.ndim < 4:
            freqs_flat = freqs_flat.unsqueeze(0)
        freq_stride_s = freqs_flat.stride(1)
        freq_stride_d = freqs_flat.stride(3)
    else:
        # Dummy -- won't be accessed since HAS_ROPE=False
        freqs_flat = q_norm_weight  # any valid pointer
        freq_stride_s = 0
        freq_stride_d = 0

    grid_epilogue = (B * S * n_heads,)

    _qk_norm_rope_epilogue_kernel[grid_epilogue](
        qkv_out, freqs_flat,
        q_norm_weight, k_norm_weight,
        Q_out, K_out, V_out,
        B, S, n_heads,
        D=D, HEAD_DIM=head_dim, HALF_HD=half_hd,
        stride_qkv_m=qkv_out.stride(0), stride_qkv_n=qkv_out.stride(1),
        stride_fs=freq_stride_s, stride_fd=freq_stride_d,
        stride_ob=Q_out.stride(0), stride_os=Q_out.stride(1),
        stride_oh=Q_out.stride(2), stride_od=Q_out.stride(3),
        eps=eps,
        HAS_ROPE=has_rope,
    )

    return Q_out, K_out, V_out


# ---------------------------------------------------------------------------
# Convenience: single-GEMM version using torch.mm + fused epilogue
# This variant uses cuBLAS for the GEMM (often faster for large D) and only
# fuses the epilogue (norm + RoPE) in Triton.
# ---------------------------------------------------------------------------

def fused_qkv_rope_sm120_cublas(
    x: torch.Tensor,
    qkv_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    freqs_cis: torch.Tensor | None = None,
    eps: float = 1e-5,
    n_heads: int = 30,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same as fused_qkv_rope_sm120 but uses cuBLAS for the GEMM.

    For large D (3840), cuBLAS is typically faster than a Triton GEMM.
    The fusion benefit comes from the epilogue: we avoid writing 3*D to
    global memory and re-reading it for norm and RoPE.

    In this variant the GEMM result goes to global memory once, then the
    epilogue reads it, applies norm+RoPE, and writes the (B, S, H, D)
    outputs. Net saving: one full read of the QKV buffer vs three separate
    kernel launches that each read a portion.
    """
    assert x.ndim == 3
    B, S, D = x.shape
    N = 3 * D
    head_dim = D // n_heads
    half_hd = head_dim // 2

    # cuBLAS GEMM: x_flat @ W^T
    x_flat = x.reshape(B * S, D)
    qkv_out = torch.mm(x_flat, qkv_weight.t())  # (M, 3*D)

    Q_out = torch.empty((B, S, n_heads, head_dim), device=x.device, dtype=x.dtype)
    K_out = torch.empty((B, S, n_heads, head_dim), device=x.device, dtype=x.dtype)
    V_out = torch.empty((B, S, n_heads, head_dim), device=x.device, dtype=x.dtype)

    q_norm_weight = q_norm_weight.contiguous()
    k_norm_weight = k_norm_weight.contiguous()

    has_rope = freqs_cis is not None

    if has_rope:
        if freqs_cis.is_complex():
            freqs_real = torch.view_as_real(freqs_cis).contiguous()
            freqs_flat = freqs_real.reshape(*freqs_cis.shape[:-1], head_dim).contiguous()
        else:
            freqs_flat = freqs_cis.contiguous()
        while freqs_flat.ndim < 4:
            freqs_flat = freqs_flat.unsqueeze(0)
        freq_stride_s = freqs_flat.stride(1)
        freq_stride_d = freqs_flat.stride(3)
    else:
        freqs_flat = q_norm_weight
        freq_stride_s = 0
        freq_stride_d = 0

    grid = (B * S * n_heads,)

    _qk_norm_rope_epilogue_kernel[grid](
        qkv_out, freqs_flat,
        q_norm_weight, k_norm_weight,
        Q_out, K_out, V_out,
        B, S, n_heads,
        D=D, HEAD_DIM=head_dim, HALF_HD=half_hd,
        stride_qkv_m=qkv_out.stride(0), stride_qkv_n=qkv_out.stride(1),
        stride_fs=freq_stride_s, stride_fd=freq_stride_d,
        stride_ob=Q_out.stride(0), stride_os=Q_out.stride(1),
        stride_oh=Q_out.stride(2), stride_od=Q_out.stride(3),
        eps=eps,
        HAS_ROPE=has_rope,
    )

    return Q_out, K_out, V_out
