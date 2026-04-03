"""Microbenchmark for Chronos time attention fusion choices.

Compares the current fused-QKV path against the sequential RMSNorm -> Q/K/V
projection path on the same weights and inputs.

Usage:
    python -m cutechronos.benchmark_time_attention --device cuda --dtype bfloat16
"""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn.functional as F

from cutechronos.kernel_backends import rms_layernorm as _rms_layernorm_selected
from cutechronos.kernel_backends import unscaled_attention as _unscaled_attention_selected
from cutechronos.modules._fallbacks import (
    apply_rope as _apply_rope_fallback,
    compute_cos_sin as _compute_cos_sin_fallback,
    rms_layernorm as _rms_layernorm_fallback,
    unscaled_attention as _unscaled_attention_fallback,
)
from cutechronos.modules.time_attention import FusedTimeSelfAttention

try:
    from cutechronos.triton_kernels.fused_layernorm_linear import fused_rms_norm_qkv
    _HAS_FUSED_QKV = True
except (ImportError, ModuleNotFoundError):
    _HAS_FUSED_QKV = False


def _parse_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[key]


def _make_inputs(
    batch_size: int,
    seq_len: int,
    d_model: int,
    num_heads: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden_states = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)
    mask = torch.zeros(batch_size, num_heads, seq_len, seq_len, device=device, dtype=dtype)
    causal = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    mask[:, :, causal] = torch.finfo(dtype).min
    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    return hidden_states, mask, position_ids


def _forward_sequential(
    module: FusedTimeSelfAttention,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    batch_size, seq_len, _ = hidden_states.shape
    normed = _rms_layernorm_selected(hidden_states, module.layer_norm_weight, module.layer_norm_eps)
    q = F.linear(normed, module.q.weight).view(batch_size, seq_len, module.num_heads, module.d_kv).transpose(1, 2)
    k = F.linear(normed, module.k.weight).view(batch_size, seq_len, module.num_heads, module.d_kv).transpose(1, 2)
    v = F.linear(normed, module.v.weight).view(batch_size, seq_len, module.num_heads, module.d_kv).transpose(1, 2)
    cos, sin = _compute_cos_sin_fallback(module.inv_freq, position_ids, q.dtype)
    q, k = _apply_rope_fallback(q, k, cos, sin)
    attn_output = _unscaled_attention_selected(q, k, v, attention_mask)
    attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, module.inner_dim)
    return hidden_states + F.linear(attn_output, module.o.weight)


def _forward_fused_qkv(
    module: FusedTimeSelfAttention,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    batch_size, seq_len, _ = hidden_states.shape
    q, k, v = fused_rms_norm_qkv(
        hidden_states,
        module.layer_norm_weight,
        module.q.weight,
        module.k.weight,
        module.v.weight,
        module.layer_norm_eps,
    )
    q = q.view(batch_size, seq_len, module.num_heads, module.d_kv).transpose(1, 2)
    k = k.view(batch_size, seq_len, module.num_heads, module.d_kv).transpose(1, 2)
    v = v.view(batch_size, seq_len, module.num_heads, module.d_kv).transpose(1, 2)
    cos, sin = _compute_cos_sin_fallback(module.inv_freq, position_ids, q.dtype)
    q, k = _apply_rope_fallback(q, k, cos, sin)
    attn_output = _unscaled_attention_fallback(q, k, v, attention_mask)
    attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, module.inner_dim)
    return hidden_states + F.linear(attn_output, module.o.weight)


def _time_call(fn, warmup: int, runs: int) -> float:
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latencies_ms = []
    for _ in range(runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
    return sum(latencies_ms) / len(latencies_ms)


def _forward_runtime(
    module: FusedTimeSelfAttention,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    return module(hidden_states, attention_mask, position_ids)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Chronos time-attention microbenchmark")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--d-kv", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = _parse_dtype(args.dtype)
    module = FusedTimeSelfAttention(
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_kv=args.d_kv,
    ).eval().to(device=device, dtype=dtype)
    hidden_states, attention_mask, position_ids = _make_inputs(
        args.batch_size,
        args.seq_len,
        args.d_model,
        args.num_heads,
        device,
        dtype,
    )

    result = {
        "device": str(device),
        "dtype": str(dtype),
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "has_fused_qkv_kernel": _HAS_FUSED_QKV,
    }

    with torch.no_grad():
        sequential_out = _forward_sequential(module, hidden_states, attention_mask, position_ids)
        result["sequential_ms"] = _time_call(
            lambda: _forward_sequential(module, hidden_states, attention_mask, position_ids),
            args.warmup,
            args.runs,
        )
        runtime_out = _forward_runtime(module, hidden_states, attention_mask, position_ids)
        result["runtime_ms"] = _time_call(
            lambda: _forward_runtime(module, hidden_states, attention_mask, position_ids),
            args.warmup,
            args.runs,
        )
        result["runtime_max_abs_diff"] = (runtime_out - sequential_out).abs().max().item()

        if device.type == "cuda" and _HAS_FUSED_QKV:
            fused_out = _forward_fused_qkv(module, hidden_states, attention_mask, position_ids)
            result["fused_qkv_ms"] = _time_call(
                lambda: _forward_fused_qkv(module, hidden_states, attention_mask, position_ids),
                args.warmup,
                args.runs,
            )
            result["max_abs_diff"] = (fused_out - sequential_out).abs().max().item()
            result["speedup_vs_sequential"] = result["sequential_ms"] / result["fused_qkv_ms"]
        else:
            result["fused_qkv_ms"] = None
            result["max_abs_diff"] = None
            result["speedup_vs_sequential"] = None

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
