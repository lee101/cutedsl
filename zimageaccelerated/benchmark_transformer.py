"""Benchmark baseline vs accelerated full Z-Image transformers."""

from __future__ import annotations

import argparse
import json
import time

import torch

from cutezimage.model import CuteZImageConfig, CuteZImageTransformer
from zimageaccelerated.model import AcceleratedZImageTransformer


def _parse_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }[name.lower()]


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


def _make_inputs(
    batch_size: int,
    in_channels: int,
    height: int,
    width: int,
    cap_feat_dim: int,
    device: torch.device,
    dtype: torch.dtype,
):
    x = [torch.randn(in_channels, 1, height, width, device=device, dtype=dtype) for _ in range(batch_size)]
    t = torch.tensor([0.5] * batch_size, device=device, dtype=dtype)
    cap_feats = [torch.randn(77, cap_feat_dim, device=device, dtype=dtype) for _ in range(batch_size)]
    return x, t, cap_feats


def _axes_dims_for_head_dim(head_dim: int) -> list[int]:
    quarter = max(2, (head_dim // 4) // 2 * 2)
    remaining = head_dim - quarter
    half_remaining = (remaining // 2) // 2 * 2
    last = head_dim - quarter - half_remaining
    if last % 2 != 0:
        last -= 1
        half_remaining += 1
    return [quarter, half_remaining, last]


def main() -> None:
    parser = argparse.ArgumentParser(description="ZImage accelerated transformer benchmark")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-refiner-layers", type=int, default=1)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-kv-heads", type=int, default=4)
    parser.add_argument("--in-channels", type=int, default=4)
    parser.add_argument("--cap-feat-dim", type=int, default=128)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = _parse_dtype(args.dtype)
    head_dim = args.dim // args.n_heads
    config = CuteZImageConfig(
        patch_size=2,
        f_patch_size=1,
        in_channels=args.in_channels,
        dim=args.dim,
        n_layers=args.n_layers,
        n_refiner_layers=args.n_refiner_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        cap_feat_dim=args.cap_feat_dim,
        axes_dims=_axes_dims_for_head_dim(head_dim),
        axes_lens=[256, 128, 128],
    )
    baseline = CuteZImageTransformer(config).eval().to(device=device, dtype=dtype)
    accelerated = AcceleratedZImageTransformer.from_cutezimage(baseline).eval()

    x, t, cap_feats = _make_inputs(
        args.batch_size,
        args.in_channels,
        args.height,
        args.width,
        args.cap_feat_dim,
        device,
        dtype,
    )

    with torch.no_grad():
        baseline_out = baseline(x, t, cap_feats, return_dict=False)[0]
        accelerated_out = accelerated(x, t, cap_feats, return_dict=False)[0]
        baseline_stats = _time_call(lambda: baseline(x, t, cap_feats, return_dict=False), device, args.warmup, args.runs)
        accelerated_stats = _time_call(
            lambda: accelerated(x, t, cap_feats, return_dict=False), device, args.warmup, args.runs
        )

    max_abs_diff = max((b - a).abs().max().item() for b, a in zip(baseline_out, accelerated_out))
    result = {
        "device": str(device),
        "dtype": str(dtype),
        "batch_size": args.batch_size,
        "height": args.height,
        "width": args.width,
        "dim": args.dim,
        "n_layers": args.n_layers,
        "n_refiner_layers": args.n_refiner_layers,
        "n_heads": args.n_heads,
        "n_kv_heads": args.n_kv_heads,
        "baseline": baseline_stats,
        "accelerated": accelerated_stats,
        "max_abs_diff": max_abs_diff,
        "speedup_vs_baseline": baseline_stats["avg_ms"] / accelerated_stats["avg_ms"],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
