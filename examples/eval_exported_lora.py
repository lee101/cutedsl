from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import sys
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STOCK_REPO = Path("/home/lee/code/stock")
DEFAULT_DATA_ROOT = DEFAULT_STOCK_REPO / "trainingdatahourly" / "crypto"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cutechronos import foreign


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an exported LoRA checkpoint on Cute and/or stock backends.")
    parser.add_argument("--model-id", required=True, help="Path to finetuned-ckpt or model id.")
    parser.add_argument("--backends", nargs="+", default=["cute", "original"])
    parser.add_argument("--stock-repo", default=str(DEFAULT_STOCK_REPO))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--symbol", default="BTCUSD")
    parser.add_argument("--preaug", default="percent_change")
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--prediction-length", type=int, default=24)
    parser.add_argument("--quantile", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--stride", type=int, default=24)
    parser.add_argument("--warmup-windows", type=int, default=1)
    parser.add_argument("--output-json", default="examples/lora_frontier/export_eval.json")
    return parser.parse_args()


def load_augmentation(stock_repo: Path, name: str):
    stock_repo_str = str(stock_repo)
    if stock_repo_str not in sys.path:
        sys.path.insert(0, stock_repo_str)
    module = importlib.import_module("preaug_sweeps.augmentations.strategies")
    registry = getattr(module, "AUGMENTATION_REGISTRY")
    if name not in registry:
        raise SystemExit(f"Unknown preaug '{name}'. Available: {sorted(registry)}")
    return registry[name]


def load_close_series(csv_path: Path) -> list[float]:
    values: list[float] = []
    with csv_path.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            value = float(row["close"])
            if math.isfinite(value):
                values.append(value)
    return values


def build_windows(values: list[float], context_length: int, prediction_length: int, windows: int, stride: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    cutoff = len(values) - prediction_length
    while cutoff - context_length >= 0 and len(result) < windows:
        result.append(
            {
                "cutoff": cutoff,
                "context": values[cutoff - context_length : cutoff],
                "actual": values[cutoff : cutoff + prediction_length],
            }
        )
        cutoff -= stride
    result.reverse()
    return result


def compute_mae(prediction: np.ndarray, actual: np.ndarray) -> float | None:
    mask = np.isfinite(prediction) & np.isfinite(actual)
    if not mask.any():
        return None
    return float(np.mean(np.abs(prediction[mask] - actual[mask])))


def compute_mape(prediction: np.ndarray, actual: np.ndarray) -> float | None:
    mask = np.isfinite(prediction) & np.isfinite(actual) & (np.abs(actual) > 1e-12)
    if not mask.any():
        return None
    return float(np.mean(np.abs((prediction[mask] - actual[mask]) / actual[mask])) * 100.0)


def safe_mean(values: list[float]) -> float | None:
    clean = [value for value in values if value is not None and math.isfinite(value)]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def safe_median(values: list[float]) -> float | None:
    clean = [value for value in values if value is not None and math.isfinite(value)]
    if not clean:
        return None
    return float(median(clean))


def format_metric(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.6f}"


def evaluate_backend(
    *,
    backend: str,
    model_id: str,
    windows: list[dict[str, Any]],
    augmentation_factory,
    prediction_length: int,
    quantile: float,
    device: str,
    dtype: str,
    warmup_windows: int,
) -> dict[str, Any]:
    handle = foreign.init_pipeline(
        model_id=model_id,
        backend=backend,
        device=device,
        dtype_name=dtype,
        compile_mode=None,
    )
    try:
        mape_values: list[float] = []
        mae_values: list[float] = []
        latency_values: list[float] = []
        sample_windows: list[dict[str, Any]] = []

        for index, window in enumerate(windows):
            context_df = pd.DataFrame({"close": window["context"]})
            augmentation = augmentation_factory()
            transformed_df = augmentation.transform_dataframe(context_df)
            transformed_context = transformed_df["close"].astype(float).tolist()

            forecast_raw, latency_ms = foreign.predict_quantile(
                handle,
                transformed_context,
                prediction_length,
                quantile,
            )
            restored = augmentation.inverse_transform_predictions(
                np.asarray(forecast_raw, dtype=float).reshape(-1, 1),
                context_df,
                columns=["close"],
            )[:, 0].astype(float)
            actual = np.asarray(window["actual"], dtype=float)
            mae = compute_mae(restored, actual)
            mape = compute_mape(restored, actual)

            if index >= warmup_windows:
                if mae is not None:
                    mae_values.append(mae)
                if mape is not None:
                    mape_values.append(mape)
                latency_values.append(latency_ms)

            if len(sample_windows) < 2:
                sample_windows.append(
                    {
                        "cutoff": int(window["cutoff"]),
                        "actual": [float(value) for value in actual.tolist()],
                        "forecast": [float(value) for value in restored.tolist()],
                        "mae": mae,
                        "mape_pct": mape,
                    }
                )

        return {
            "backend": backend,
            "mean_mae": safe_mean(mae_values),
            "median_mae": safe_median(mae_values),
            "mean_mape_pct": safe_mean(mape_values),
            "median_mape_pct": safe_median(mape_values),
            "avg_latency_ms": safe_mean(latency_values),
            "sample_windows": sample_windows,
        }
    finally:
        foreign.destroy_pipeline(handle)


def main() -> None:
    args = parse_args()
    stock_repo = Path(args.stock_repo)
    augmentation_factory = load_augmentation(stock_repo, args.preaug)

    env_path = os.environ.get("CHRONOS_FORECASTING_SRC")
    if not env_path:
        os.environ["CHRONOS_FORECASTING_SRC"] = str(stock_repo / "chronos-forecasting" / "src")

    data_path = Path(args.data_root) / f"{args.symbol}.csv"
    values = load_close_series(data_path)
    windows = build_windows(values, args.context_length, args.prediction_length, args.windows, args.stride)
    if len(windows) <= args.warmup_windows:
        raise SystemExit("Not enough windows for evaluation.")

    results = []
    for backend in args.backends:
        print(f"Evaluating backend={backend} model={args.model_id}")
        result = evaluate_backend(
            backend=backend,
            model_id=args.model_id,
            windows=windows,
            augmentation_factory=augmentation_factory,
            prediction_length=args.prediction_length,
            quantile=args.quantile,
            device=args.device,
            dtype=args.dtype,
            warmup_windows=args.warmup_windows,
        )
        results.append(result)
        print(
            f"  backend={backend:<8} "
            f"mean_mape%={format_metric(result['mean_mape_pct'])} "
            f"mean_mae={format_metric(result['mean_mae'])} "
            f"latency_ms={format_metric(result['avg_latency_ms'])}"
        )

    output_path = REPO_ROOT / args.output_json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "model_id": args.model_id,
                "symbol": args.symbol,
                "preaug": args.preaug,
                "context_length": args.context_length,
                "prediction_length": args.prediction_length,
                "quantile": args.quantile,
                "windows": len(windows),
                "warmup_windows": args.warmup_windows,
                "results": results,
            },
            indent=2,
        )
    )
    print(f"Saved export evaluation to {output_path}")


if __name__ == "__main__":
    main()
