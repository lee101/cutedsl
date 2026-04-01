from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Callable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STOCK_REPO = Path("/home/lee/code/stock")
DEFAULT_STOCK_DATA = DEFAULT_STOCK_REPO / "trainingdatadailybinance"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cutechronos import foreign


@dataclass(frozen=True)
class StrategySpec:
    base_name: str
    label: str
    params: dict[str, Any]
    factory: Callable[..., Any]


@dataclass(frozen=True)
class InferenceSpec:
    kind: str
    label: str
    strides: tuple[int, ...] = (1,)
    aggregation: str = "single"
    trim: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep CuteChronos pre-augmentation and inference knobs.")
    parser.add_argument("--model-id", default="amazon/chronos-2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtypes", nargs="+", default=["bfloat16"])
    parser.add_argument(
        "--compile-modes",
        nargs="+",
        default=["", "reduce-overhead"],
        help="Use empty string for eager.",
    )
    parser.add_argument("--stock-repo", default=str(DEFAULT_STOCK_REPO))
    parser.add_argument("--stock-data-dir", default=str(DEFAULT_STOCK_DATA))
    parser.add_argument("--symbols", nargs="+", default=["BTCUSD", "ETHUSD", "SOLUSD"])
    parser.add_argument("--context-lengths", nargs="+", type=int, default=[128, 256, 512, 1024])
    parser.add_argument("--prediction-length", type=int, default=3)
    parser.add_argument("--windows", type=int, default=16)
    parser.add_argument("--stride", type=int, default=7)
    parser.add_argument("--warmup-windows", type=int, default=1)
    parser.add_argument("--strategies", nargs="+", default=[
        "baseline",
        "percent_change",
        "log_returns",
        "differencing",
        "detrending",
        "robust_scaling",
        "minmax_standard",
        "rolling_norm",
    ])
    parser.add_argument("--differencing-orders", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--rolling-windows", nargs="+", type=int, default=[20, 64])
    parser.add_argument("--quantiles", nargs="+", type=float, default=[0.3, 0.5, 0.7])
    parser.add_argument("--inference-strategies", nargs="+", default=["single", "dilation"])
    parser.add_argument(
        "--dilation-stride-sets",
        nargs="+",
        default=["1,2", "1,2,4", "1,2,4,8"],
        help="Comma-separated stride sets for dilation ensembles.",
    )
    parser.add_argument("--ensemble-aggregations", nargs="+", default=["trimmed_mean", "median"])
    parser.add_argument("--trim-values", nargs="+", type=int, default=[1])
    parser.add_argument("--selection-metric", choices=["mean_mape_pct", "median_mape_pct", "mean_mae"], default="mean_mape_pct")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-json", default="examples/chronos_wrappers/tuning_results.json")
    parser.add_argument("--output-csv", default="examples/chronos_wrappers/tuning_results.csv")
    return parser.parse_args()


def load_augmentation_registry(stock_repo: Path) -> dict[str, type]:
    if stock_repo.exists():
        stock_repo_str = str(stock_repo)
        if stock_repo_str not in sys.path:
            sys.path.insert(0, stock_repo_str)

    try:
        module = importlib.import_module("preaug_sweeps.augmentations.strategies")
    except ImportError as exc:
        raise SystemExit(
            "Could not import stock preaugmentation strategies. "
            f"Checked stock repo at {stock_repo}."
        ) from exc

    return getattr(module, "AUGMENTATION_REGISTRY")


def compile_mode_label(raw: str) -> str:
    return raw if raw else "eager"


def load_close_series(csv_path: Path) -> list[float]:
    values: list[float] = []
    with csv_path.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            value = float(row["close"])
            if math.isfinite(value):
                values.append(value)
    return values


def build_windows(
    values: list[float],
    *,
    context_length: int,
    prediction_length: int,
    max_windows: int,
    stride: int,
) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    cutoff = len(values) - prediction_length
    while cutoff - context_length >= 0 and len(windows) < max_windows:
        context = values[cutoff - context_length : cutoff]
        actual = values[cutoff : cutoff + prediction_length]
        windows.append(
            {
                "cutoff": cutoff,
                "context": context,
                "actual": actual,
            }
        )
        cutoff -= stride
    windows.reverse()
    return windows


def expand_strategy_specs(args: argparse.Namespace, registry: dict[str, type]) -> list[StrategySpec]:
    specs: list[StrategySpec] = []
    for name in args.strategies:
        if name not in registry:
            raise SystemExit(f"Unknown strategy '{name}'. Available: {sorted(registry)}")

        factory = registry[name]
        if name == "differencing":
            for order in args.differencing_orders:
                spec = StrategySpec(
                    base_name=name,
                    label=f"differencing_order{order}",
                    params={"order": order},
                    factory=factory,
                )
                specs.append(spec)
            continue

        if name == "rolling_norm":
            for window_size in args.rolling_windows:
                spec = StrategySpec(
                    base_name=name,
                    label=f"rolling_norm_w{window_size}",
                    params={"window_size": window_size},
                    factory=factory,
                )
                specs.append(spec)
            continue

        specs.append(
            StrategySpec(
                base_name=name,
                label=name,
                params={},
                factory=factory,
            )
        )
    return specs


def expand_inference_specs(args: argparse.Namespace) -> list[InferenceSpec]:
    specs: list[InferenceSpec] = []
    for name in args.inference_strategies:
        if name == "single":
            specs.append(InferenceSpec(kind="single", label="single"))
            continue
        if name != "dilation":
            raise SystemExit(f"Unknown inference strategy '{name}'. Supported: single, dilation")

        for raw in args.dilation_stride_sets:
            strides = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
            if not strides:
                continue
            for aggregation in args.ensemble_aggregations:
                if aggregation == "trimmed_mean":
                    for trim in args.trim_values:
                        stride_label = "_".join(str(value) for value in strides)
                        specs.append(
                            InferenceSpec(
                                kind="dilation",
                                label=f"dilation_s{stride_label}_{aggregation}_t{trim}",
                                strides=strides,
                                aggregation=aggregation,
                                trim=trim,
                            )
                        )
                else:
                    stride_label = "_".join(str(value) for value in strides)
                    specs.append(
                        InferenceSpec(
                            kind="dilation",
                            label=f"dilation_s{stride_label}_{aggregation}",
                            strides=strides,
                            aggregation=aggregation,
                            trim=0,
                        )
                    )
    return specs


def safe_mean(values: list[float]) -> float | None:
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def safe_median(values: list[float]) -> float | None:
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return None
    return float(median(clean))


def compute_mae(prediction: np.ndarray, actual: np.ndarray) -> float | None:
    mask = np.isfinite(prediction) & np.isfinite(actual)
    if not mask.any():
        return None
    return float(np.mean(np.abs(prediction[mask] - actual[mask])))


def compute_mape_pct(prediction: np.ndarray, actual: np.ndarray) -> float | None:
    mask = np.isfinite(prediction) & np.isfinite(actual) & (np.abs(actual) > 1e-12)
    if not mask.any():
        return None
    return float(np.mean(np.abs((prediction[mask] - actual[mask]) / actual[mask])) * 100.0)


def build_context_variants(context: list[float], inference: InferenceSpec, *, target_points: int) -> list[list[float]]:
    if inference.kind == "single":
        return [context]

    n_time = len(context)
    variants: list[list[float]] = []
    for stride in inference.strides:
        indices = np.arange(n_time - 1, -1, -stride)[::-1]
        if len(indices) > target_points:
            indices = indices[-target_points:]
        if len(indices) < 4:
            continue
        variant = [context[int(index)] for index in indices.tolist()]
        variants.append(variant)
    return variants or [context]


def aggregate_forecasts(forecasts: list[np.ndarray], inference: InferenceSpec) -> np.ndarray:
    if len(forecasts) == 1 or inference.aggregation == "single":
        return forecasts[0]

    stacked = np.stack(forecasts, axis=0)
    if inference.aggregation == "mean":
        return np.mean(stacked, axis=0)
    if inference.aggregation == "median":
        return np.median(stacked, axis=0)
    if inference.aggregation == "trimmed_mean":
        trim = max(0, inference.trim)
        if trim > 0 and stacked.shape[0] > 2 * trim:
            ordered = np.sort(stacked, axis=0)
            trimmed = ordered[trim:-trim]
        else:
            trimmed = stacked
        return np.mean(trimmed, axis=0)
    raise ValueError(f"Unsupported aggregation '{inference.aggregation}'")


def forecast_augmented(
    handle: int,
    *,
    context: list[float],
    prediction_length: int,
    quantile: float,
    strategy: StrategySpec,
    inference: InferenceSpec,
    target_points: int,
) -> tuple[np.ndarray, float]:
    total_latency_ms = 0.0
    forecasts: list[np.ndarray] = []
    for variant in build_context_variants(context, inference, target_points=target_points):
        context_df = pd.DataFrame({"close": variant})
        augmentation = strategy.factory(**strategy.params)
        transformed_df = augmentation.transform_dataframe(context_df)
        transformed_context = transformed_df["close"].astype(float).tolist()
        transformed_forecast, latency_ms = foreign.predict_quantile(
            handle,
            transformed_context,
            prediction_length,
            quantile,
        )
        total_latency_ms += latency_ms
        transformed_array = np.asarray(transformed_forecast, dtype=float).reshape(-1, 1)
        restored = augmentation.inverse_transform_predictions(
            transformed_array,
            context_df,
            columns=["close"],
        )
        forecasts.append(restored[:, 0].astype(float))
    return aggregate_forecasts(forecasts, inference), total_latency_ms


def evaluate_symbol_config(
    handle: int,
    *,
    symbol: str,
    windows: list[dict[str, Any]],
    prediction_length: int,
    quantile: float,
    strategy: StrategySpec,
    inference: InferenceSpec,
    warmup_windows: int,
    target_points: int,
) -> dict[str, Any]:
    mae_values: list[float] = []
    mape_values: list[float] = []
    latency_values: list[float] = []
    sample_windows: list[dict[str, Any]] = []

    for index, window in enumerate(windows):
        forecast, latency_ms = forecast_augmented(
            handle,
            context=window["context"],
            prediction_length=prediction_length,
            quantile=quantile,
            strategy=strategy,
            inference=inference,
            target_points=target_points,
        )
        if index < warmup_windows:
            continue

        actual = np.asarray(window["actual"], dtype=float)
        mae = compute_mae(forecast, actual)
        mape_pct = compute_mape_pct(forecast, actual)
        if mae is not None:
            mae_values.append(mae)
        if mape_pct is not None:
            mape_values.append(mape_pct)
        latency_values.append(latency_ms)

        if len(sample_windows) < 2:
            sample_windows.append(
                {
                    "cutoff": int(window["cutoff"]),
                    "actual": [float(value) for value in actual.tolist()],
                    "forecast": [float(value) for value in forecast.tolist()],
                    "mae": mae,
                    "mape_pct": mape_pct,
                }
            )

    return {
        "symbol": symbol,
        "evaluated_windows": max(0, len(windows) - warmup_windows),
        "mean_mae": safe_mean(mae_values),
        "median_mae": safe_median(mae_values),
        "mean_mape_pct": safe_mean(mape_values),
        "median_mape_pct": safe_median(mape_values),
        "avg_latency_ms": safe_mean(latency_values),
        "sample_windows": sample_windows,
    }


def selection_value(row: dict[str, Any], metric: str) -> float:
    value = row.get(metric)
    if value is None:
        return math.inf
    return float(value)


def format_value(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}"


def main() -> None:
    args = parse_args()
    registry = load_augmentation_registry(Path(args.stock_repo))
    strategy_specs = expand_strategy_specs(args, registry)
    inference_specs = expand_inference_specs(args)

    symbol_series: dict[str, list[float]] = {}
    data_dir = Path(args.stock_data_dir)
    for symbol in args.symbols:
        csv_path = data_dir / f"{symbol}.csv"
        if not csv_path.exists():
            print(f"Skipping {symbol}: missing {csv_path}")
            continue
        values = load_close_series(csv_path)
        if values:
            symbol_series[symbol] = values

    if not symbol_series:
        raise SystemExit("No usable symbol CSVs found for tuning.")

    rows: list[dict[str, Any]] = []

    for dtype in args.dtypes:
        for compile_mode in args.compile_modes:
            mode_label = compile_mode_label(compile_mode)
            print(f"\nLoading CuteChronos backend: dtype={dtype} compile_mode={mode_label}")
            handle = foreign.init_pipeline(
                model_id=args.model_id,
                backend="cute",
                device=args.device,
                dtype_name=dtype,
                compile_mode=compile_mode or None,
            )
            try:
                for context_length in args.context_lengths:
                    symbol_windows: dict[str, list[dict[str, Any]]] = {}
                    for symbol, values in symbol_series.items():
                        windows = build_windows(
                            values,
                            context_length=context_length,
                            prediction_length=args.prediction_length,
                            max_windows=args.windows,
                            stride=args.stride,
                        )
                        if len(windows) <= args.warmup_windows:
                            continue
                        symbol_windows[symbol] = windows

                    if not symbol_windows:
                        print(f"Skipping context_length={context_length}: not enough history.")
                        continue

                    for strategy in strategy_specs:
                        for inference in inference_specs:
                            for quantile in args.quantiles:
                                for symbol, windows in symbol_windows.items():
                                    base_row = {
                                        "symbol": symbol,
                                        "dtype": dtype,
                                        "compile_mode": mode_label,
                                        "context_length": context_length,
                                        "prediction_length": args.prediction_length,
                                        "strategy": strategy.label,
                                        "strategy_base": strategy.base_name,
                                        "strategy_params": strategy.params,
                                        "inference_strategy": inference.label,
                                        "quantile": quantile,
                                    }
                                    try:
                                        result = evaluate_symbol_config(
                                            handle,
                                            symbol=symbol,
                                            windows=windows,
                                            prediction_length=args.prediction_length,
                                            quantile=quantile,
                                            strategy=strategy,
                                            inference=inference,
                                            warmup_windows=args.warmup_windows,
                                            target_points=context_length,
                                        )
                                    except Exception as exc:
                                        row = {
                                            **base_row,
                                            "evaluated_windows": 0,
                                            "mean_mae": None,
                                            "median_mae": None,
                                            "mean_mape_pct": None,
                                            "median_mape_pct": None,
                                            "avg_latency_ms": None,
                                            "sample_windows": [],
                                            "error": repr(exc),
                                        }
                                        rows.append(row)
                                        print(
                                            f"{symbol:<8} "
                                            f"ctx={context_length:<4} "
                                            f"strategy={strategy.label:<20} "
                                            f"inference={inference.label:<28} "
                                            f"q={quantile:<4.1f} "
                                            f"compile={mode_label:<15} "
                                            f"ERROR={row['error']}"
                                        )
                                        continue

                                    row = {
                                        **base_row,
                                        **result,
                                        "error": None,
                                    }
                                    rows.append(row)
                                    print(
                                        f"{symbol:<8} "
                                        f"ctx={context_length:<4} "
                                        f"strategy={strategy.label:<20} "
                                        f"inference={inference.label:<28} "
                                        f"q={quantile:<4.1f} "
                                        f"compile={mode_label:<15} "
                                        f"MAPE%={format_value(row['mean_mape_pct']):>10} "
                                        f"MAE={format_value(row['mean_mae']):>10}"
                                    )
            finally:
                foreign.destroy_pipeline(handle)

    if not rows:
        raise SystemExit("No tuning rows were produced.")

    by_symbol: dict[str, list[dict[str, Any]]] = {}
    by_config: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_symbol.setdefault(row["symbol"], []).append(row)
        config_key = json.dumps(
            {
                "dtype": row["dtype"],
                "compile_mode": row["compile_mode"],
                "context_length": row["context_length"],
                "prediction_length": row["prediction_length"],
                "strategy": row["strategy"],
                "inference_strategy": row["inference_strategy"],
                "quantile": row["quantile"],
            },
            sort_keys=True,
        )
        by_config.setdefault(config_key, []).append(row)

    symbol_summary: dict[str, Any] = {}
    for symbol, symbol_rows in sorted(by_symbol.items()):
        ranked = sorted(symbol_rows, key=lambda row: selection_value(row, args.selection_metric))
        symbol_summary[symbol] = {
            "best": ranked[0],
            "top_k": ranked[: args.top_k],
        }

    global_ranking: list[dict[str, Any]] = []
    for config_key, config_rows in by_config.items():
        representative = json.loads(config_key)
        global_ranking.append(
            {
                **representative,
                "symbols": [row["symbol"] for row in config_rows],
                "mean_mape_pct": safe_mean([row["mean_mape_pct"] for row in config_rows if row["mean_mape_pct"] is not None]),
                "median_mape_pct": safe_mean([row["median_mape_pct"] for row in config_rows if row["median_mape_pct"] is not None]),
                "mean_mae": safe_mean([row["mean_mae"] for row in config_rows if row["mean_mae"] is not None]),
                "avg_latency_ms": safe_mean([row["avg_latency_ms"] for row in config_rows if row["avg_latency_ms"] is not None]),
                "error_count": sum(1 for row in config_rows if row.get("error")),
            }
        )
    global_ranking.sort(key=lambda row: selection_value(row, args.selection_metric))

    output_json = REPO_ROOT / args.output_json
    output_csv = REPO_ROOT / args.output_csv
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_id": args.model_id,
        "device": args.device,
        "selection_metric": args.selection_metric,
        "search_space": {
            "dtypes": args.dtypes,
            "compile_modes": [compile_mode_label(value) for value in args.compile_modes],
            "context_lengths": args.context_lengths,
            "prediction_length": args.prediction_length,
            "strategies": [spec.label for spec in strategy_specs],
            "inference_strategies": [spec.label for spec in inference_specs],
            "quantiles": args.quantiles,
            "windows": args.windows,
            "stride": args.stride,
            "warmup_windows": args.warmup_windows,
        },
        "per_symbol": symbol_summary,
        "global_top_k": global_ranking[: args.top_k],
        "rows": rows,
    }
    output_json.write_text(json.dumps(payload, indent=2))

    with output_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "symbol",
                "dtype",
                "compile_mode",
                "context_length",
                "prediction_length",
                "strategy",
                "strategy_base",
                "inference_strategy",
                "quantile",
                "evaluated_windows",
                "mean_mae",
                "median_mae",
                "mean_mape_pct",
                "median_mape_pct",
                "avg_latency_ms",
                "error",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: row.get(key)
                    for key in writer.fieldnames
                }
            )

    print("\nBest configs by symbol:")
    for symbol, summary in symbol_summary.items():
        best = summary["best"]
        print(
            f"{symbol:<8} "
            f"ctx={best['context_length']:<4} "
            f"strategy={best['strategy']:<20} "
            f"inference={best['inference_strategy']:<28} "
            f"q={best['quantile']:<4.1f} "
            f"dtype={best['dtype']:<8} "
            f"compile={best['compile_mode']:<15} "
            f"MAPE%={format_value(best['mean_mape_pct']):>10} "
            f"MAE={format_value(best['mean_mae']):>10}"
        )

    print("\nGlobal top configs:")
    for row in global_ranking[: args.top_k]:
        print(
            f"ctx={row['context_length']:<4} "
            f"strategy={row['strategy']:<20} "
            f"inference={row['inference_strategy']:<28} "
            f"q={row['quantile']:<4.1f} "
            f"dtype={row['dtype']:<8} "
            f"compile={row['compile_mode']:<15} "
            f"MAPE%={format_value(row['mean_mape_pct']):>10} "
            f"MAE={format_value(row['mean_mae']):>10} "
            f"latency={format_value(row['avg_latency_ms']):>10}"
        )

    print(f"\nSaved tuning JSON to {output_json}")
    print(f"Saved tuning CSV to {output_csv}")


if __name__ == "__main__":
    main()
