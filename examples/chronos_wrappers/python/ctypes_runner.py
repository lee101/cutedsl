from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cutechronos import foreign


def parse_csv_floats(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def compute_mae(pred: list[float], actual: list[float]) -> float:
    diffs = [
        abs(p - a)
        for p, a in zip(pred, actual)
        if p == p and a == a
    ]
    if not diffs:
        return 0.0
    return sum(diffs) / len(diffs)


def compute_mape_pct(pred: list[float], actual: list[float]) -> float | None:
    diffs = [
        abs((p - a) / a) * 100.0
        for p, a in zip(pred, actual)
        if p == p and a == a and abs(a) > 1e-12
    ]
    if not diffs:
        return None
    return sum(diffs) / len(diffs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="amazon/chronos-2")
    parser.add_argument("--backend", default="cute")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--compile-mode", default="")
    parser.add_argument("--prediction-length", type=int, default=3)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--context", required=True)
    parser.add_argument("--actual", required=True)
    args = parser.parse_args()

    context = parse_csv_floats(args.context)
    actual = parse_csv_floats(args.actual)

    handle = foreign.init_pipeline(
        model_id=args.model_id,
        backend=args.backend,
        device=args.device,
        dtype_name=args.dtype,
        compile_mode=args.compile_mode or None,
    )

    try:
        for _ in range(args.warmup):
            foreign.predict_median(handle, context, args.prediction_length)

        total_outer = 0.0
        total_inner = 0.0
        forecast: list[float] = []
        for _ in range(args.runs):
            start = time.perf_counter()
            forecast, inner_ms = foreign.predict_median(handle, context, args.prediction_length)
            total_outer += (time.perf_counter() - start) * 1000.0
            total_inner += inner_ms

        result = {
            "language": "python_wrapper",
            "backend": args.backend,
            "model_id": args.model_id,
            "device": args.device,
            "prediction_length": args.prediction_length,
            "runs": args.runs,
            "warmup": args.warmup,
            "avg_outer_latency_ms": total_outer / args.runs,
            "avg_inner_latency_ms": total_inner / args.runs,
            "mae": compute_mae(forecast, actual),
            "mape_pct": compute_mape_pct(forecast, actual),
            "forecast": forecast,
            "actual": actual,
        }
        print(json.dumps(result, indent=2))
    finally:
        foreign.destroy_pipeline(handle)


if __name__ == "__main__":
    main()
