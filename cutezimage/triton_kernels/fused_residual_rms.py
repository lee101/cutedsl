from __future__ import annotations

"""Fused output RMSNorm, optional AdaLN gate, and residual update.

Z-Image ends each attention and FFN branch with::

    residual + gate * rms_norm(branch, weight)

Keeping this as one kernel avoids materializing the normalized branch and
removes the separate gate multiply and residual-add launches.  The kernel is
inference-only; model dispatch retains the PyTorch path while gradients are
enabled.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_residual_rms_kernel(
    Residual_ptr,
    Branch_ptr,
    Weight_ptr,
    Gate_ptr,
    Out_ptr,
    stride_rb,
    stride_rs,
    stride_rd,
    stride_bb,
    stride_bs,
    stride_bd,
    stride_gb,
    stride_gs,
    stride_gd,
    stride_ob,
    stride_os,
    stride_od,
    S,
    D,
    eps,
    HAS_GATE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // S
    seq_idx = pid % S

    d_offs = tl.arange(0, BLOCK_D)
    mask = d_offs < D

    residual_base = Residual_ptr + batch_idx * stride_rb + seq_idx * stride_rs
    branch_base = Branch_ptr + batch_idx * stride_bb + seq_idx * stride_bs
    out_base = Out_ptr + batch_idx * stride_ob + seq_idx * stride_os

    branch = tl.load(branch_base + d_offs * stride_bd, mask=mask, other=0.0)
    branch_fp32 = branch.to(tl.float32)
    sq_sum = tl.sum(branch_fp32 * branch_fp32, axis=0)
    rrms = tl.rsqrt(sq_sum / D + eps)

    # Match the existing RMSNorm path's cast before the learned weight.
    normalized = (branch_fp32 * rrms).to(branch.dtype)
    weight = tl.load(Weight_ptr + d_offs, mask=mask, other=1.0)
    update = (normalized * weight).to(branch.dtype)

    if HAS_GATE:
        gate_base = Gate_ptr + batch_idx * stride_gb + seq_idx * stride_gs
        gate = tl.load(gate_base + d_offs * stride_gd, mask=mask, other=0.0)
        update = (update * gate).to(branch.dtype)

    residual = tl.load(residual_base + d_offs * stride_rd, mask=mask, other=0.0)
    out = (residual + update).to(residual.dtype)
    tl.store(out_base + d_offs * stride_od, out, mask=mask)


def fused_residual_rms(
    residual: torch.Tensor,
    branch: torch.Tensor,
    weight: torch.Tensor,
    gate: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Compute ``residual + gate * rms_norm(branch, weight)`` in one launch.

    ``gate`` may be absent, global ``(B, 1, D)``, or per-token
    ``(B, S, D)``.  Expanded global gates are accepted without forcing a
    contiguous copy.
    """
    if residual.ndim != 3 or branch.ndim != 3:
        raise ValueError("residual and branch must have shape (B, S, D)")
    if residual.shape != branch.shape:
        raise ValueError(f"residual shape {residual.shape} != branch shape {branch.shape}")
    if not residual.is_cuda or not branch.is_cuda or not weight.is_cuda:
        raise ValueError("fused_residual_rms requires CUDA tensors")
    if residual.device != branch.device or residual.device != weight.device:
        raise ValueError("residual, branch, and weight must share a device")
    if residual.dtype != branch.dtype or residual.dtype != weight.dtype:
        raise ValueError("residual, branch, and weight must share a dtype")

    B, S, D = residual.shape
    if weight.shape != (D,):
        raise ValueError(f"weight shape {weight.shape} != ({D},)")
    if residual.stride(-1) != 1 or branch.stride(-1) != 1 or weight.stride(0) != 1:
        raise ValueError("last dimensions must be contiguous")

    has_gate = gate is not None
    if gate is not None:
        if gate.device != residual.device or gate.dtype != residual.dtype:
            raise ValueError("gate must share residual's device and dtype")
        if gate.ndim != 3 or gate.shape[0] != B or gate.shape[2] != D or gate.shape[1] not in (1, S):
            raise ValueError(f"gate shape {gate.shape} must be (B, 1, D) or (B, S, D)")
        if gate.stride(-1) != 1:
            raise ValueError("gate's last dimension must be contiguous")
        gate_stride_b = gate.stride(0)
        gate_stride_s = 0 if gate.shape[1] == 1 else gate.stride(1)
        gate_stride_d = gate.stride(2)
    else:
        gate = residual  # Dummy pointer for the constexpr-disabled branch.
        gate_stride_b = gate_stride_s = gate_stride_d = 0

    out = torch.empty_like(residual)
    block_d = triton.next_power_of_2(D)
    if block_d > 65536:
        raise ValueError(f"hidden dimension {D} is too large for the fused RMS kernel")

    _fused_residual_rms_kernel[(B * S,)](
        residual,
        branch,
        weight,
        gate,
        out,
        residual.stride(0), residual.stride(1), residual.stride(2),
        branch.stride(0), branch.stride(1), branch.stride(2),
        gate_stride_b, gate_stride_s, gate_stride_d,
        out.stride(0), out.stride(1), out.stride(2),
        S,
        D,
        eps,
        HAS_GATE=has_gate,
        BLOCK_D=block_d,
        num_warps=8 if block_d >= 2048 else 4,
    )
    return out
