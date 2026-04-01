from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STOCK_REPO = Path("/home/lee/code/stock")
DEFAULT_DATA_ROOT = DEFAULT_STOCK_REPO / "trainingdatahourly" / "crypto"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cutechronos.pipeline import CuteChronos2Pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare univariate vs paired multivariate forecasting for hourly crypto symbols."
    )
    parser.add_argument("--model-id", default="amazon/chronos-2")
    parser.add_argument("--stock-repo", default=str(DEFAULT_STOCK_REPO))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--target-symbols", nargs="+", default=["SOLUSDT", "ETHUSDT"])
    parser.add_argument("--partner-symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSD", "BTCUSD"])
    parser.add_argument("--prediction-length", type=int, default=24)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--context-lengths", nargs="+", type=int, default=None)
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--stride", type=int, default=24)
    parser.add_argument("--strides", nargs="+", type=int, default=None)
    parser.add_argument("--partner-pool-sizes", nargs="+", type=int, default=[1])
    parser.add_argument("--warmup-windows", type=int, default=1)
    parser.add_argument("--quantile-level", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--backends", nargs="+", default=["cute", "original"])
    parser.add_argument("--cross-learning-options", nargs="+", default=["off", "on"])
    parser.add_argument("--max-multivariate-runs", type=int, default=0, help="0 means all partner/context/stride combinations.")
    parser.add_argument("--output-csv", default="examples/multivariate_pair_eval/results.csv")
    parser.add_argument("--output-md", default="examples/multivariate_pair_eval/results.md")
    parser.add_argument("--top-per-target", type=int, default=3)
    parser.add_argument("--output-json", default="examples/multivariate_pair_eval/results.json")
    return parser.parse_args()


def resolve_context_lengths(args: argparse.Namespace) -> list[int]:
    values = args.context_lengths if args.context_lengths else [args.context_length]
    return [value for value in values if value > 0]


def resolve_strides(args: argparse.Namespace) -> list[int]:
    values = args.strides if args.strides else [args.stride]
    return [value for value in values if value > 0]


def resolve_cross_learning_options(raw_values: list[str]) -> list[bool]:
    mapping = {
        "off": False,
        "false": False,
        "0": False,
        "no": False,
        "on": True,
        "true": True,
        "1": True,
        "yes": True,
    }
    resolved: list[bool] = []
    for value in raw_values:
        key = value.strip().lower()
        if key not in mapping:
            raise ValueError(f"Unsupported cross-learning option: {value}")
        resolved.append(mapping[key])
    return list(dict.fromkeys(resolved))


def build_partner_sets(target_symbol: str, partner_symbols: list[str], partner_pool_sizes: list[int]) -> list[tuple[str, ...]]:
    deduped = []
    seen = set()
    for symbol in partner_symbols:
        if symbol == target_symbol or symbol in seen:
            continue
        deduped.append(symbol)
        seen.add(symbol)

    partner_sets: list[tuple[str, ...]] = []
    for pool_size in partner_pool_sizes:
        if pool_size <= 0:
            continue
        for combo in itertools.combinations(deduped, pool_size):
            partner_sets.append(combo)
    return partner_sets


def select_frontier(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = [
        row
        for row in results
        if row.get("mean_mape_pct") is not None
        and row.get("avg_latency_ms") is not None
        and math.isfinite(float(row["mean_mape_pct"]))
        and math.isfinite(float(row["avg_latency_ms"]))
    ]
    valid.sort(key=lambda row: (float(row["avg_latency_ms"]), float(row["mean_mape_pct"])))

    frontier: list[dict[str, Any]] = []
    best_seen = math.inf
    for row in valid:
        metric = float(row["mean_mape_pct"])
        if metric < best_seen:
            frontier.append(row)
            best_seen = metric
    return frontier


def valid_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in results
        if row.get("status") == "ok"
        and row.get("mean_mape_pct") is not None
        and row.get("avg_latency_ms") is not None
        and math.isfinite(float(row["mean_mape_pct"]))
        and math.isfinite(float(row["avg_latency_ms"]))
    ]


def rank_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = list(valid_results(results))
    ranked.sort(key=lambda row: (float(row["mean_mape_pct"]), float(row["avg_latency_ms"])))
    return ranked


def best_results_by_target(results: list[dict[str, Any]], top_per_target: int) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rank_results(results):
        target = str(row["target_symbol"])
        grouped.setdefault(target, [])
        if len(grouped[target]) < max(1, top_per_target):
            grouped[target].append(row)
    return grouped


def improvement_vs_univariate(best_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baselines: dict[tuple[str, str], float] = {}
    for row in valid_results(best_rows):
        if row.get("mode") != "univariate":
            continue
        key = (str(row["target_symbol"]), str(row["backend"]))
        baselines[key] = min(baselines.get(key, math.inf), float(row["mean_mape_pct"]))

    enriched: list[dict[str, Any]] = []
    for row in rank_results(best_rows):
        item = dict(row)
        baseline = baselines.get((str(row["target_symbol"]), str(row["backend"])))
        if baseline is not None:
            item["delta_vs_univariate_mape_pct"] = float(row["mean_mape_pct"]) - baseline
        enriched.append(item)
    return enriched


def partners_label(row: dict[str, Any]) -> str:
    partners = row.get("partner_symbols") or []
    return ",".join(str(partner) for partner in partners) if partners else "-"


def render_markdown_summary(
    results: list[dict[str, Any]],
    frontier: list[dict[str, Any]],
    model_id: str,
    prediction_length: int,
    quantile_level: float,
    top_per_target: int,
) -> str:
    ranked = improvement_vs_univariate(results)
    by_target = best_results_by_target(ranked, top_per_target)
    unsupported = [row for row in results if row.get("status") == "unsupported"]

    lines = [
        "# Multivariate Pair Eval Summary",
        "",
        f"- model_id: `{model_id}`",
        f"- prediction_length: `{prediction_length}`",
        f"- quantile_level: `{quantile_level}`",
        f"- valid_runs: `{len(valid_results(results))}`",
        f"- unsupported_runs: `{len(unsupported)}`",
        "",
        "## Best Per Target",
        "",
    ]

    for target in sorted(by_target):
        lines.append(f"### {target}")
        lines.append("")
        lines.append("| rank | backend | mode | partners | ctx | stride | cross_learning | mape_pct | latency_ms | delta_vs_univariate |")
        lines.append("| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: |")
        for idx, row in enumerate(by_target[target], start=1):
            delta = row.get("delta_vs_univariate_mape_pct")
            delta_text = f"{float(delta):+.3f}" if delta is not None else "n/a"
            lines.append(
                f"| {idx} | {row['backend']} | {row['mode']} | {partners_label(row)} | "
                f"{row['context_length']} | {row['stride']} | {row['cross_learning']} | "
                f"{float(row['mean_mape_pct']):.3f} | {float(row['avg_latency_ms']):.2f} | {delta_text} |"
            )
        lines.append("")

    lines.append("## Frontier")
    lines.append("")
    if frontier:
        lines.append("| target | backend | mode | partners | ctx | stride | cross_learning | mape_pct | latency_ms |")
        lines.append("| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: |")
        for row in frontier[:20]:
            lines.append(
                f"| {row['target_symbol']} | {row['backend']} | {row['mode']} | {partners_label(row)} | "
                f"{row['context_length']} | {row['stride']} | {row['cross_learning']} | "
                f"{float(row['mean_mape_pct']):.3f} | {float(row['avg_latency_ms']):.2f} |"
            )
    else:
        lines.append("No valid frontier rows.")
    lines.append("")

    if unsupported:
        lines.append("## Unsupported")
        lines.append("")
        lines.append("| target | backend | mode | partners | cross_learning | error |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for row in unsupported[:20]:
            lines.append(
                f"| {row['target_symbol']} | {row['backend']} | {row['mode']} | {partners_label(row)} | "
                f"{row['cross_learning']} | {row.get('error', '')} |"
            )
        lines.append("")

    return "\n".join(lines)


def maybe_add_local_chronos_checkout(stock_repo: Path) -> None:
    env_path = os.environ.get("CHRONOS_FORECASTING_SRC")
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(stock_repo / "chronos-forecasting" / "src")
    for candidate in candidates:
        if candidate.exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return


def load_close_frame(csv_path: Path, symbol: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, usecols=["timestamp", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["timestamp", "close"]).sort_values("timestamp")
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    return df.rename(columns={"close": symbol}).reset_index(drop=True)


def build_windows_1d(values: np.ndarray, context_length: int, prediction_length: int, windows: int, stride: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    cutoff = len(values) - prediction_length
    while cutoff - context_length >= 0 and len(result) < windows:
        result.append(
            {
                "cutoff": cutoff,
                "context": values[cutoff - context_length : cutoff].astype(float),
                "actual": values[cutoff : cutoff + prediction_length].astype(float),
            }
        )
        cutoff -= stride
    result.reverse()
    return result


def build_windows_2d(values: np.ndarray, context_length: int, prediction_length: int, windows: int, stride: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    cutoff = values.shape[1] - prediction_length
    while cutoff - context_length >= 0 and len(result) < windows:
        result.append(
            {
                "cutoff": cutoff,
                "context": values[:, cutoff - context_length : cutoff].astype(float),
                "actual": values[0, cutoff : cutoff + prediction_length].astype(float),
            }
        )
        cutoff -= stride
    result.reverse()
    return result


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


def compute_mape(prediction: np.ndarray, actual: np.ndarray) -> float | None:
    mask = np.isfinite(prediction) & np.isfinite(actual) & (np.abs(actual) > 1e-12)
    if not mask.any():
        return None
    return float(np.mean(np.abs((prediction[mask] - actual[mask]) / actual[mask])) * 100.0)


def init_backend(model_id: str, backend: str, device: str, dtype: str, stock_repo: Path):
    dtype_obj = getattr(torch, dtype)
    if backend == "cute":
        return CuteChronos2Pipeline.from_pretrained(model_id, device=device, dtype=dtype_obj, use_cute=True)
    if backend == "original":
        maybe_add_local_chronos_checkout(stock_repo)
        from chronos.chronos2 import Chronos2Pipeline

        pipe = Chronos2Pipeline.from_pretrained(model_id, dtype=dtype_obj)
        pipe.model = pipe.model.to(device)
        return pipe
    raise ValueError(f"Unsupported backend {backend}")


def predict_quantiles_with_options(
    pipeline: Any,
    context: Any,
    prediction_length: int,
    quantile_levels: list[float],
    cross_learning: bool,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    try:
        return pipeline.predict_quantiles(
            context,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
            cross_learning=cross_learning,
        )
    except TypeError:
        try:
            predictions = pipeline.predict(
                context,
                prediction_length=prediction_length,
                cross_learning=cross_learning,
                limit_prediction_length=False,
            )
        except TypeError as exc:
            if cross_learning:
                raise RuntimeError(
                    f"{type(pipeline).__name__} does not support cross_learning in this configuration"
                ) from exc
            predictions = pipeline.predict(
                context,
                prediction_length=prediction_length,
                limit_prediction_length=False,
            )
        training_quantiles = list(getattr(pipeline, "quantiles"))
        indices = [training_quantiles.index(level) for level in quantile_levels]
        quantiles = [prediction.permute(0, 2, 1)[..., indices] for prediction in predictions]
        median_idx = training_quantiles.index(0.5)
        mean = [prediction.permute(0, 2, 1)[..., median_idx] for prediction in predictions]
        return quantiles, mean


def run_univariate(
    pipeline: Any,
    backend: str,
    windows: list[dict[str, Any]],
    prediction_length: int,
    quantile_level: float,
    warmup_windows: int,
    cross_learning: bool,
) -> dict[str, Any]:
    mape_values: list[float] = []
    latency_values: list[float] = []
    for index, window in enumerate(windows):
        context = torch.tensor(window["context"], dtype=torch.float32)
        start = time.perf_counter()
        if backend == "cute":
            quantiles, _ = predict_quantiles_with_options(
                pipeline,
                context,
                prediction_length,
                [quantile_level],
                cross_learning,
            )
        else:
            quantiles, _ = predict_quantiles_with_options(
                pipeline,
                [context],
                prediction_length,
                [quantile_level],
                cross_learning,
            )
        latency_values.append((time.perf_counter() - start) * 1000.0)
        forecast = quantiles[0][0, :, 0].detach().cpu().numpy()
        if index >= warmup_windows:
            mape = compute_mape(forecast, window["actual"])
            if mape is not None:
                mape_values.append(mape)
    return {
        "mean_mape_pct": safe_mean(mape_values),
        "median_mape_pct": safe_median(mape_values),
        "avg_latency_ms": safe_mean(latency_values[warmup_windows:]),
    }


def run_multivariate(
    pipeline: Any,
    backend: str,
    windows: list[dict[str, Any]],
    prediction_length: int,
    quantile_level: float,
    warmup_windows: int,
    cross_learning: bool,
) -> dict[str, Any]:
    mape_values: list[float] = []
    latency_values: list[float] = []
    for index, window in enumerate(windows):
        context = torch.tensor(window["context"][None, :, :], dtype=torch.float32)
        start = time.perf_counter()
        quantiles, _ = predict_quantiles_with_options(
            pipeline,
            context,
            prediction_length,
            [quantile_level],
            cross_learning,
        )
        latency_values.append((time.perf_counter() - start) * 1000.0)
        forecast = quantiles[0][0, :, 0].detach().cpu().numpy()
        if index >= warmup_windows:
            mape = compute_mape(forecast, window["actual"])
            if mape is not None:
                mape_values.append(mape)
    return {
        "mean_mape_pct": safe_mean(mape_values),
        "median_mape_pct": safe_median(mape_values),
        "avg_latency_ms": safe_mean(latency_values[warmup_windows:]),
    }


def main() -> None:
    args = parse_args()
    stock_repo = Path(args.stock_repo)
    data_root = Path(args.data_root)
    context_lengths = resolve_context_lengths(args)
    strides = resolve_strides(args)
    cross_learning_options = resolve_cross_learning_options(args.cross_learning_options)

    pipelines = {
        backend: init_backend(args.model_id, backend, args.device, args.dtype, stock_repo)
        for backend in args.backends
    }

    results: list[dict[str, Any]] = []
    for target_symbol in args.target_symbols:
        target_path = data_root / f"{target_symbol}.csv"
        target_frame = load_close_frame(target_path, target_symbol)
        target_values = target_frame[target_symbol].to_numpy(dtype=float)
        for context_length, stride in itertools.product(context_lengths, strides):
            univariate_windows = build_windows_1d(
                target_values,
                context_length,
                args.prediction_length,
                args.windows,
                stride,
            )
            if len(univariate_windows) <= args.warmup_windows:
                continue

            for backend, pipeline in pipelines.items():
                for cross_learning in cross_learning_options:
                    row = {
                        "target_symbol": target_symbol,
                        "partner_symbols": [],
                        "partner_count": 0,
                        "backend": backend,
                        "mode": "univariate",
                        "context_length": context_length,
                        "stride": stride,
                        "cross_learning": cross_learning,
                        "status": "ok",
                    }
                    try:
                        baseline = run_univariate(
                            pipeline,
                            backend,
                            univariate_windows,
                            args.prediction_length,
                            args.quantile_level,
                            args.warmup_windows,
                            cross_learning,
                        )
                        row.update(baseline)
                    except RuntimeError as exc:
                        row.update(
                            {
                                "mean_mape_pct": None,
                                "median_mape_pct": None,
                                "avg_latency_ms": None,
                                "status": "unsupported",
                                "error": str(exc),
                            }
                        )
                    results.append(row)

        multivariate_runs = 0
        for partner_set in build_partner_sets(target_symbol, args.partner_symbols, args.partner_pool_sizes):
            partner_frames = []
            missing_partner = False
            for partner_symbol in partner_set:
                partner_path = data_root / f"{partner_symbol}.csv"
                if not partner_path.exists():
                    missing_partner = True
                    break
                partner_frames.append(load_close_frame(partner_path, partner_symbol))
            if missing_partner:
                continue

            merged = target_frame
            for partner_frame in partner_frames:
                merged = merged.merge(partner_frame, on="timestamp", how="inner")

            for context_length, stride in itertools.product(context_lengths, strides):
                if len(merged) < context_length + args.prediction_length + args.warmup_windows + 2:
                    continue
                values = merged[[target_symbol, *partner_set]].to_numpy(dtype=float).T
                windows = build_windows_2d(
                    values,
                    context_length,
                    args.prediction_length,
                    args.windows,
                    stride,
                )
                if len(windows) <= args.warmup_windows:
                    continue
                for backend, pipeline in pipelines.items():
                    for cross_learning in cross_learning_options:
                        row = {
                            "target_symbol": target_symbol,
                            "partner_symbols": list(partner_set),
                            "partner_count": len(partner_set),
                            "backend": backend,
                            "mode": "multivariate",
                            "context_length": context_length,
                            "stride": stride,
                            "cross_learning": cross_learning,
                            "status": "ok",
                        }
                        try:
                            paired = run_multivariate(
                                pipeline,
                                backend,
                                windows,
                                args.prediction_length,
                                args.quantile_level,
                                args.warmup_windows,
                                cross_learning,
                            )
                            row.update(paired)
                        except RuntimeError as exc:
                            row.update(
                                {
                                    "mean_mape_pct": None,
                                    "median_mape_pct": None,
                                    "avg_latency_ms": None,
                                    "status": "unsupported",
                                    "error": str(exc),
                                }
                            )
                        results.append(row)
                        multivariate_runs += 1
                        if args.max_multivariate_runs > 0 and multivariate_runs >= args.max_multivariate_runs:
                            break
                    if args.max_multivariate_runs > 0 and multivariate_runs >= args.max_multivariate_runs:
                        break
                if args.max_multivariate_runs > 0 and multivariate_runs >= args.max_multivariate_runs:
                    break
            if args.max_multivariate_runs > 0 and multivariate_runs >= args.max_multivariate_runs:
                break

    output_path = REPO_ROOT / args.output_json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = REPO_ROOT / args.output_csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frontier = select_frontier(results)
    payload = {
        "model_id": args.model_id,
        "context_lengths": context_lengths,
        "prediction_length": args.prediction_length,
        "quantile_level": args.quantile_level,
        "strides": strides,
        "cross_learning_options": cross_learning_options,
        "results": results,
        "frontier": frontier,
    }
    output_path.write_text(json.dumps(payload, indent=2))
    if results:
        fieldnames = [
            "target_symbol",
            "partner_symbols",
            "partner_count",
            "backend",
            "mode",
            "context_length",
            "stride",
            "cross_learning",
            "status",
            "mean_mape_pct",
            "median_mape_pct",
            "avg_latency_ms",
        ]
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                serialized = dict(row)
                serialized["partner_symbols"] = ",".join(serialized.get("partner_symbols", []))
                writer.writerow({key: serialized.get(key) for key in fieldnames})
    print(f"Saved pair evaluation to {output_path}")
    print(f"Saved CSV summary to {csv_path}")
    if frontier:
        print("Top frontier candidates:")
        for row in frontier[:10]:
            partners = ",".join(row["partner_symbols"]) if row["partner_symbols"] else "-"
            print(
                f"  target={row['target_symbol']} backend={row['backend']} mode={row['mode']} "
                f"partners={partners} ctx={row['context_length']} stride={row['stride']} "
                f"cross_learning={row['cross_learning']} "
                f"mape={row['mean_mape_pct']:.3f} latency_ms={row['avg_latency_ms']:.2f}"
            )


if __name__ == "__main__":
    main()
