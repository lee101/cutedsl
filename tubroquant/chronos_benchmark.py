"""Benchmark Chronos forecasting with optional TurboQuant attention hooks."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def benchmark_pipeline(
    pipeline,
    contexts: list[torch.Tensor],
    actuals: list[torch.Tensor],
    prediction_length: int,
    n_warmup: int,
    n_runs: int,
    *,
    compute_mae,
    median_from_predictions,
) -> dict[str, float]:
    for _ in range(n_warmup):
        for ctx in contexts:
            pipeline.predict(ctx, prediction_length=prediction_length, limit_prediction_length=False)

    latencies: list[float] = []
    maes: list[float] = []
    for _ in range(n_runs):
        for ctx, actual in zip(contexts, actuals, strict=False):
            t0 = time.perf_counter()
            preds = pipeline.predict(ctx, prediction_length=prediction_length, limit_prediction_length=False)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            maes.append(compute_mae(median_from_predictions(preds, pipeline.quantiles), actual))

    summary = {
        "avg_latency_ms": float(sum(latencies) / max(len(latencies), 1)),
        "avg_mae": float(sum(maes) / max(len(maes), 1)),
    }
    if getattr(pipeline.model, "get_tubroquant_summary", None) is not None:
        summary["tubroquant"] = pipeline.model.get_tubroquant_summary()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Chronos benchmark with TurboQuant-style K/V compression")
    parser.add_argument("--model-id", default="amazon/chronos-bolt-base")
    parser.add_argument("--data-dir", default="../stock-prediction/trainingdata")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSD", "ETHUSD", "TSLA"])
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--prediction-length", type=int, default=30)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--rotation", default="hadamard", choices=["hadamard", "qr"])
    parser.add_argument("--key-mode", default="prod", choices=["mse", "prod"])
    parser.add_argument("--value-mode", default="mse", choices=["mse", "prod"])
    parser.add_argument("--n-warmup", type=int, default=1)
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--output", default="tubroquant_benchmark.json")
    args = parser.parse_args()

    from cutechronos.benchmark import compute_mae, load_series, median_from_predictions
    from cutechronos.pipeline import CuteChronos2Pipeline

    data_dir = Path(args.data_dir)
    contexts: list[torch.Tensor] = []
    actuals: list[torch.Tensor] = []
    found_symbols: list[str] = []

    for symbol in args.symbols:
        csv_path = data_dir / f"{symbol}.csv"
        if not csv_path.exists():
            continue
        ctx, act = load_series(str(csv_path), args.context_length, args.prediction_length)
        contexts.append(ctx)
        actuals.append(act)
        found_symbols.append(symbol)

    if not contexts:
        raise FileNotFoundError(f"No symbol CSVs found under {data_dir}")

    baseline = CuteChronos2Pipeline.from_pretrained(args.model_id, device="cpu", dtype=torch.float32, use_cute=True)
    quantized = CuteChronos2Pipeline.from_pretrained(args.model_id, device="cpu", dtype=torch.float32, use_cute=True)
    quantized.model.enable_turboquant_kv(
        bits=args.bits,
        key_mode=args.key_mode,
        value_mode=args.value_mode,
        rotation=args.rotation,
    )

    results = {
        "symbols": found_symbols,
        "baseline": benchmark_pipeline(
            baseline,
            contexts,
            actuals,
            prediction_length=args.prediction_length,
            n_warmup=args.n_warmup,
            n_runs=args.n_runs,
            compute_mae=compute_mae,
            median_from_predictions=median_from_predictions,
        ),
        "tubroquant": benchmark_pipeline(
            quantized,
            contexts,
            actuals,
            prediction_length=args.prediction_length,
            n_warmup=args.n_warmup,
            n_runs=args.n_runs,
            compute_mae=compute_mae,
            median_from_predictions=median_from_predictions,
        ),
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
