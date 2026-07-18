"""Fused multi-LoRA delta computation.

The swapper applies ``Σ_i scale_i · (B_i @ A_i)`` per target module. The stock
path loops the LoRAs, running one ``d×r · r×d`` fp32 GEMM per adapter and one
accumulate. For N adapters on the same module that is N GEMM launches + N adds,
each producing a full ``d×d`` delta — kernel-launch-bound for the small ranks
LoRAs actually use (r=8..64).

Fused path: concatenate the factors along the rank axis and fold the scales
into one side, so N adapters collapse to a SINGLE GEMM:

    B_cat = [B_1 | B_2 | ... | B_N]                 (d, ΣR)
    A_cat = [scale_1·A_1 ; ... ; scale_N·A_N]       (ΣR, d)
    delta = B_cat @ A_cat                            (d, d)  == Σ scale_i·B_i@A_i

One GEMM with a fatter K dimension is far better GPU utilization than N thin
GEMMs, and there is one accumulate instead of N. Also runs the GEMM in bf16
(with fp32 accumulate) which halves memory traffic on the d×d output at
delta-precision that is well within LoRA-merge tolerance.
"""
from __future__ import annotations

import torch


def fused_lora_delta(terms, out_dtype, device, compute_dtype=torch.bfloat16):
    """terms: list of (A[r,d], B[d,r], scale). Returns delta[d,d] in out_dtype,
    or None if there are no non-zero terms."""
    live = [(a, b, s) for (a, b, s) in terms if s != 0.0]
    if not live:
        return None

    b_parts = []
    a_parts = []
    for a, b, s in live:
        a = a.to(device, compute_dtype, non_blocking=True)
        b = b.to(device, compute_dtype, non_blocking=True)
        b_parts.append(b)
        a_parts.append(a * s)  # fold scale into A (cheaper: r×d << d×d)

    b_cat = torch.cat(b_parts, dim=1)   # (d, ΣR)
    a_cat = torch.cat(a_parts, dim=0)   # (ΣR, d)
    # fp32-accumulate matmul; bf16 inputs. torch routes this to tensor cores.
    delta = torch.matmul(b_cat.float(), a_cat.float()) if compute_dtype == torch.float32 \
        else torch.matmul(b_cat, a_cat)
    return delta.to(out_dtype)


def stock_lora_delta(terms, out_dtype, device):
    """Reference: the per-adapter fp32 loop the swapper ships today."""
    delta = None
    for a, b, s in terms:
        if s == 0.0:
            continue
        term = (b.to(device, torch.float32) @ a.to(device, torch.float32)) * s
        delta = term if delta is None else delta.add_(term)
    if delta is None:
        return None
    return delta.to(out_dtype)
