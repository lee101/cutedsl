"""Microbenchmarks for hot CuteChronos inference/compilation helpers.

These benchmarks compare the current optimized helpers against equivalent
baseline implementations that mirror the older allocation-heavy code paths.
They do not require model weights.

Example:
    python scripts/benchmark_inference_kernels.py --device cpu --json results.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from cutechronos.model import CuteChronos2Config, CuteChronos2Model
from cutechronos.pipeline import CuteChronos2Pipeline, _left_pad_and_cat_2d


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_us(fn: Callable[[], Any], *, device: torch.device, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    _synchronize(device)

    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        _synchronize(device)
        samples.append((time.perf_counter() - start) * 1_000_000.0)
    return statistics.median(samples)


def _baseline_group_mask(
    group_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    floating_type: torch.dtype,
) -> torch.Tensor:
    group_mask = group_ids[:, None] == group_ids[None, :]
    group_time_mask = torch.einsum(
        "qb, bt -> qbt",
        group_mask.to(floating_type),
        attention_mask.to(floating_type),
    )
    group_time_mask = group_time_mask.permute(2, 0, 1).unsqueeze(1)
    return (1.0 - group_time_mask) * torch.finfo(floating_type).min


def _baseline_left_pad_and_cat_2d(tensors: list[torch.Tensor]) -> torch.Tensor:
    max_len = max(t.shape[-1] for t in tensors)
    padded = []
    for tensor in tensors:
        if tensor.shape[-1] < max_len:
            pad = torch.full(
                (tensor.shape[0], max_len - tensor.shape[-1]),
                float("nan"),
                dtype=tensor.dtype,
                device=tensor.device,
            )
            tensor = torch.cat([pad, tensor], dim=-1)
        padded.append(tensor)
    return torch.cat(padded, dim=0)


def _baseline_left_pad_and_stack_rows(rows: list[torch.Tensor]) -> torch.Tensor:
    max_len = max(row.shape[-1] for row in rows)
    padded = []
    for row in rows:
        pad_len = max_len - row.shape[-1]
        if pad_len > 0:
            pad = torch.full((pad_len,), float("nan"), dtype=row.dtype, device=row.device)
            row = torch.cat([pad, row])
        padded.append(row)
    return torch.stack(padded)


def _baseline_position_ids(model: CuteChronos2Model, seq_length: int) -> torch.Tensor:
    return torch.arange(seq_length, dtype=torch.long, device=model._param_device()).unsqueeze(0)


def _record(name: str, baseline_us: float, optimized_us: float, extra: dict[str, Any]) -> dict[str, Any]:
    speedup = baseline_us / max(optimized_us, 1e-12)
    return {
        "name": name,
        "baseline_us": baseline_us,
        "optimized_us": optimized_us,
        "speedup": speedup,
        **extra,
    }


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    results: list[dict[str, Any]] = []

    for batch in args.batch_sizes:
        seq = args.sequence_length
        group_ids = (torch.arange(batch, device=device) // args.group_size).long()
        attention_mask = (torch.rand(batch, seq, device=device) > 0.1).to(dtype)
        expected = _baseline_group_mask(group_ids, attention_mask, dtype)
        actual = CuteChronos2Model._construct_and_invert_group_time_mask(group_ids, attention_mask, dtype)
        torch.testing.assert_close(actual, expected)

        baseline_us = _time_us(
            lambda: _baseline_group_mask(group_ids, attention_mask, dtype),
            device=device,
            warmup=args.warmup,
            repeat=args.repeat,
        )
        optimized_us = _time_us(
            lambda: CuteChronos2Model._construct_and_invert_group_time_mask(group_ids, attention_mask, dtype),
            device=device,
            warmup=args.warmup,
            repeat=args.repeat,
        )
        results.append(
            _record(
                "group_time_mask",
                baseline_us,
                optimized_us,
                {"batch": batch, "sequence_length": seq, "dtype": str(dtype)},
            )
        )

    model = CuteChronos2Model(CuteChronos2Config(num_layers=1, d_model=32, d_kv=8, d_ff=64, num_heads=4)).to(device)
    for seq in args.position_lengths:
        torch.testing.assert_close(model._get_position_ids(seq), _baseline_position_ids(model, seq))
        baseline_us = _time_us(
            lambda: _baseline_position_ids(model, seq),
            device=device,
            warmup=args.warmup,
            repeat=args.repeat,
        )
        optimized_us = _time_us(
            lambda: model._get_position_ids(seq),
            device=device,
            warmup=args.warmup,
            repeat=args.repeat,
        )
        results.append(
            _record(
                "position_ids",
                baseline_us,
                optimized_us,
                {"sequence_length": seq, "dtype": "torch.long"},
            )
        )

    row_lengths = [args.max_row_length - (idx % args.length_jitter) for idx in range(args.rows)]
    rows = [torch.randn(length, device=device) for length in row_lengths]
    tensors_2d = [torch.randn(args.rows_per_task, length, device=device) for length in row_lengths[:: args.rows_per_task]]

    torch.testing.assert_close(
        CuteChronos2Pipeline._left_pad_and_stack_rows(rows),
        _baseline_left_pad_and_stack_rows(rows),
        equal_nan=True,
    )
    baseline_us = _time_us(
        lambda: _baseline_left_pad_and_stack_rows(rows),
        device=device,
        warmup=args.warmup,
        repeat=args.repeat,
    )
    optimized_us = _time_us(
        lambda: CuteChronos2Pipeline._left_pad_and_stack_rows(rows),
        device=device,
        warmup=args.warmup,
        repeat=args.repeat,
    )
    results.append(
        _record(
            "left_pad_and_stack_rows",
            baseline_us,
            optimized_us,
            {"rows": len(rows), "max_row_length": max(row_lengths)},
        )
    )

    torch.testing.assert_close(
        _left_pad_and_cat_2d(tensors_2d),
        _baseline_left_pad_and_cat_2d(tensors_2d),
        equal_nan=True,
    )
    baseline_us = _time_us(
        lambda: _baseline_left_pad_and_cat_2d(tensors_2d),
        device=device,
        warmup=args.warmup,
        repeat=args.repeat,
    )
    optimized_us = _time_us(
        lambda: _left_pad_and_cat_2d(tensors_2d),
        device=device,
        warmup=args.warmup,
        repeat=args.repeat,
    )
    results.append(
        _record(
            "left_pad_and_cat_2d",
            baseline_us,
            optimized_us,
            {"tasks": len(tensors_2d), "rows": sum(t.shape[0] for t in tensors_2d)},
        )
    )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CuteChronos inference helper hot paths")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 64, 256])
    parser.add_argument("--sequence-length", type=int, default=34)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--position-lengths", type=int, nargs="+", default=[18, 34, 66])
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--rows-per-task", type=int, default=4)
    parser.add_argument("--max-row-length", type=int, default=512)
    parser.add_argument("--length-jitter", type=int, default=37)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    results = run(args)
    for result in results:
        print(
            f"{result['name']:<26} "
            f"baseline={result['baseline_us']:>9.2f}us "
            f"optimized={result['optimized_us']:>9.2f}us "
            f"speedup={result['speedup']:>5.2f}x"
        )

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
