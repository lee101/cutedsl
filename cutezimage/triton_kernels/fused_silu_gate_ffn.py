from __future__ import annotations

"""Fused SiLU-gated FFN kernel for Z-Image transformer.

Z-Image FeedForward uses a gated architecture:
    output = w2(silu(w1(x)) * w3(x))

where:
    w1: (dim, hidden_dim)  e.g. (3840, 10240)
    w2: (hidden_dim, dim)
    w3: (dim, hidden_dim)

The 10240-wide intermediate (silu(w1(x)) * w3(x)) is the main allocation target.
This kernel fuses silu + elementwise multiply to avoid a separate pass.

For the full tiled MLP fusion, cuBLAS matmuls are still faster than Triton at
these sizes on modern GPUs, so we fuse just the activation to eliminate the
separate silu kernel launch and gating multiply.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_silu_gate_kernel(
    X1_ptr,  # w1(x): (M, N)
    X3_ptr,  # w3(x): (M, N)
    Out_ptr, # silu(x1) * x3: (M, N)
    M,
    N,
    stride_x1m,
    stride_x1n,
    stride_x3m,
    stride_x3n,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fused silu(x1) * x3 elementwise kernel.

    Grid: (cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N),)
    """
    pid = tl.program_id(0)
    num_blocks_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_blocks_n
    pid_n = pid % num_blocks_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    x1_ptrs = X1_ptr + offs_m[:, None] * stride_x1m + offs_n[None, :] * stride_x1n
    x3_ptrs = X3_ptr + offs_m[:, None] * stride_x3m + offs_n[None, :] * stride_x3n

    x1 = tl.load(x1_ptrs, mask=mask, other=0.0).to(tl.float32)
    x3 = tl.load(x3_ptrs, mask=mask, other=0.0).to(tl.float32)

    # SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    silu_x1 = x1 * tl.sigmoid(x1)
    out = silu_x1 * x3

    out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, out.to(Out_ptr.dtype.element_ty), mask=mask)


def fused_silu_gate(x1: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
    """Compute silu(x1) * x3 with a single fused Triton kernel.

    Eliminates the separate SiLU kernel launch and the intermediate
    allocation for the gating multiply.

    Args:
        x1: Output of w1 projection, shape (*, hidden_dim).
        x3: Output of w3 projection, shape (*, hidden_dim).

    Returns:
        silu(x1) * x3, same shape and dtype as inputs.
    """
    assert x1.shape == x3.shape, f"Shape mismatch: {x1.shape} vs {x3.shape}"
    orig_shape = x1.shape

    x1_2d = x1.reshape(-1, x1.shape[-1]).contiguous()
    x3_2d = x3.reshape(-1, x3.shape[-1]).contiguous()
    M, N = x1_2d.shape

    out = torch.empty_like(x1_2d)

    BLOCK_M = 64
    BLOCK_N = min(1024, triton.next_power_of_2(N))

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    _fused_silu_gate_kernel[grid](
        x1_2d, x3_2d, out,
        M, N,
        x1_2d.stride(0), x1_2d.stride(1),
        x3_2d.stride(0), x3_2d.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    return out.reshape(orig_shape)


def fused_silu_gate_ffn(
    x: torch.Tensor,
    w1_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w3_weight: torch.Tensor,
) -> torch.Tensor:
    """Full gated FFN: w2(silu(w1(x)) * w3(x)).

    Uses cuBLAS for the matmuls (fastest at these sizes) but fuses the
    activation + gating into a single Triton kernel.

    Args:
        x: Input, shape (*, dim).
        w1_weight: Gate projection weight, shape (hidden_dim, dim).
        w2_weight: Output projection weight, shape (dim, hidden_dim).
        w3_weight: Up projection weight, shape (hidden_dim, dim).

    Returns:
        Output, shape (*, dim).
    """
    import torch.nn.functional as F

    x1 = F.linear(x, w1_weight)  # (*, hidden_dim)
    x3 = F.linear(x, w3_weight)  # (*, hidden_dim)
    gated = fused_silu_gate(x1, x3)  # fused silu(x1) * x3
    return F.linear(gated, w2_weight)  # (*, dim)
