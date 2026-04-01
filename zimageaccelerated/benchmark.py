"""Benchmark baseline vs accelerated Z-Image transformer blocks."""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

from cutezimage.model import CuteZImageTransformerBlock
from zimageaccelerated.model import AcceleratedZImageTransformerBlock


def _parse_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    return mapping[name.lower()]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_call(fn, device: torch.device, warmup: int, runs: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    _sync(device)

    latencies_ms: list[float] = []
    for _ in range(runs):
        _sync(device)
        start = time.perf_counter()
        fn()
        _sync(device)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
    return {
        "avg_ms": sum(latencies_ms) / len(latencies_ms),
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ZImage accelerated block benchmark")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--dim", type=int, default=3840)
    parser.add_argument("--n-heads", type=int, default=30)
    parser.add_argument("--n-kv-heads", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--modulation", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = _parse_dtype(args.dtype)

    baseline = CuteZImageTransformerBlock(
        layer_id=0,
        dim=args.dim,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        modulation=args.modulation,
    ).eval().to(device=device, dtype=dtype)
    accelerated = AcceleratedZImageTransformerBlock.from_cutezimage_block(baseline).eval()

    x = torch.randn(args.batch_size, args.seq_len, args.dim, device=device, dtype=dtype)
    attn_mask = None
    head_dim = args.dim // args.n_heads
    freqs_cis = torch.complex(
        torch.randn(1, args.seq_len, head_dim // 2, device=device, dtype=torch.float32),
        torch.randn(1, args.seq_len, head_dim // 2, device=device, dtype=torch.float32),
    )

    adaln_input = None
    if args.modulation:
        adaln_dim = min(args.dim, 256)
        adaln_input = torch.randn(args.batch_size, adaln_dim, device=device, dtype=dtype)

    with torch.no_grad():
        if args.modulation:
            baseline_out = baseline(x, attn_mask=attn_mask, freqs_cis=freqs_cis, adaln_input=adaln_input)
            accelerated_out = accelerated(x, attn_mask=attn_mask, freqs_cis=freqs_cis, adaln_input=adaln_input)
            baseline_stats = _time_call(
                lambda: baseline(x, attn_mask=attn_mask, freqs_cis=freqs_cis, adaln_input=adaln_input),
                device,
                args.warmup,
                args.runs,
            )
            accelerated_stats = _time_call(
                lambda: accelerated(x, attn_mask=attn_mask, freqs_cis=freqs_cis, adaln_input=adaln_input),
                device,
                args.warmup,
                args.runs,
            )
        else:
            baseline_out = baseline(x, attn_mask=attn_mask, freqs_cis=freqs_cis)
            accelerated_out = accelerated(x, attn_mask=attn_mask, freqs_cis=freqs_cis)
            baseline_stats = _time_call(
                lambda: baseline(x, attn_mask=attn_mask, freqs_cis=freqs_cis),
                device,
                args.warmup,
                args.runs,
            )
            accelerated_stats = _time_call(
                lambda: accelerated(x, attn_mask=attn_mask, freqs_cis=freqs_cis),
                device,
                args.warmup,
                args.runs,
            )

    result = {
        "device": str(device),
        "dtype": str(dtype),
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "dim": args.dim,
        "n_heads": args.n_heads,
        "n_kv_heads": args.n_kv_heads,
        "modulation": args.modulation,
        "triton_rope_enabled": os.environ.get("CUTEZIMAGE_USE_TRITON_ROPE") == "1",
        "baseline": baseline_stats,
        "accelerated": accelerated_stats,
        "max_abs_diff": (baseline_out - accelerated_out).abs().max().item(),
        "speedup_vs_baseline": baseline_stats["avg_ms"] / accelerated_stats["avg_ms"],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
