"""Fused adaLN-Zero modulation and gated residual for Cosmos/Anima blocks.

Reference (diffusers `CosmosAdaLayerNormZero`)::

    normed = layer_norm(x, eps=1e-6)              # no affine
    y = normed * (1 + scale) + shift              # scale/shift are [B, D]
    x = x + gate * block(y)                       # gate is [B, D]

Each Anima block runs the pair three times, so the reference does 12 full passes
over the [B, S, D] activation per block. The kernels below do 1 pass each.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _adaln_modulate_kernel(
    x_ptr, shift_ptr, scale_ptr, y_ptr,
    rows_per_batch, dim,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // rows_per_batch
    offs = tl.arange(0, BLOCK)
    mask = offs < dim

    x = tl.load(x_ptr + row * dim + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / dim
    centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) / dim
    rstd = 1.0 / tl.sqrt(var + eps)

    shift = tl.load(shift_ptr + batch * dim + offs, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + batch * dim + offs, mask=mask, other=0.0).to(tl.float32)
    y = centered * rstd * (1.0 + scale) + shift
    tl.store(y_ptr + row * dim + offs, y.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _gated_residual_kernel(
    x_ptr, delta_ptr, gate_ptr, out_ptr,
    rows_per_batch, dim,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // rows_per_batch
    offs = tl.arange(0, BLOCK)
    mask = offs < dim

    x = tl.load(x_ptr + row * dim + offs, mask=mask, other=0.0).to(tl.float32)
    delta = tl.load(delta_ptr + row * dim + offs, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + batch * dim + offs, mask=mask, other=0.0).to(tl.float32)
    out = x + gate * delta
    tl.store(out_ptr + row * dim + offs, out.to(out_ptr.dtype.element_ty), mask=mask)


def _launch_shape(hidden_states: torch.Tensor):
    batch, seq_len, dim = hidden_states.shape
    if dim > 65536:
        raise ValueError("fused adaLN kernels require dim <= 65536")
    return batch * seq_len, seq_len, dim, triton.next_power_of_2(dim)


def adaln_modulate(
    hidden_states: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """LayerNorm (no affine, fp32 stats) then `* (1 + scale) + shift` in one pass."""
    hidden_states = hidden_states.contiguous()
    rows, rows_per_batch, dim, block = _launch_shape(hidden_states)
    out = torch.empty_like(hidden_states)
    _adaln_modulate_kernel[(rows,)](
        hidden_states, shift.contiguous(), scale.contiguous(), out,
        rows_per_batch, dim, eps, BLOCK=block, num_warps=8,
    )
    return out


def gated_residual(hidden_states: torch.Tensor, delta: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """`hidden_states + gate * delta` with a [B, D] gate broadcast over the sequence."""
    hidden_states = hidden_states.contiguous()
    rows, rows_per_batch, dim, block = _launch_shape(hidden_states)
    out = torch.empty_like(hidden_states)
    _gated_residual_kernel[(rows,)](
        hidden_states, delta.contiguous(), gate.contiguous(), out,
        rows_per_batch, dim, BLOCK=block, num_warps=8,
    )
    return out


def reference_adaln_modulate(
    hidden_states: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    normed = torch.nn.functional.layer_norm(hidden_states, (hidden_states.shape[-1],), eps=eps)
    return normed * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def reference_gated_residual(
    hidden_states: torch.Tensor, delta: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    return hidden_states + gate.unsqueeze(1) * delta
