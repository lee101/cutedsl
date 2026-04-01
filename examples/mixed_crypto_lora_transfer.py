from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STOCK_REPO = Path("/home/lee/code/stock")
DEFAULT_DATA_ROOT = DEFAULT_STOCK_REPO / "trainingdatahourly" / "crypto"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "examples" / "mixed_crypto_lora"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a mixed-symbol crypto LoRA base, then evaluate transfer on one or more target symbols."
        )
    )
    parser.add_argument("--model-id", default="amazon/chronos-2")
    parser.add_argument("--stock-repo", default=str(DEFAULT_STOCK_REPO))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--train-symbols", nargs="+", default=[])
    parser.add_argument("--eval-symbols", nargs="+", default=["SOLUSDT", "ETHUSDT"])
    parser.add_argument("--exclude-symbols", nargs="+", default=[])
    parser.add_argument("--min-history", type=int, default=10000)
    parser.add_argument("--max-train-symbols", type=int, default=24)
    parser.add_argument("--preaug", default="percent_change")
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--prediction-length", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--val-hours", type=int, default=168)
    parser.add_argument("--test-hours", type=int, default=168)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--run-prefix", default="mixed_crypto_lora")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output-json", default="examples/mixed_crypto_lora/transfer_report.json")
    return parser.parse_args()


def import_stock_modules(stock_repo: Path) -> dict[str, Any]:
    stock_repo_str = str(stock_repo)
    if stock_repo_str not in sys.path:
        sys.path.insert(0, stock_repo_str)
    train_mod = __import__("train_crypto_lora_sweep", fromlist=["dummy"])
    trainer_mod = __import__("chronos2_trainer", fromlist=["dummy"])
    preaug_mod = __import__("preaug", fromlist=["dummy"])
    return {
        "load_hourly_frame": getattr(train_mod, "load_hourly_frame"),
        "split_data": getattr(train_mod, "split_data"),
        "compute_consistency_metrics": getattr(train_mod, "compute_consistency_metrics"),
        "get_augmentation": getattr(preaug_mod, "get_augmentation"),
        "_load_pipeline": getattr(trainer_mod, "_load_pipeline"),
        "_fit_pipeline": getattr(trainer_mod, "_fit_pipeline"),
        "_save_pipeline": getattr(trainer_mod, "_save_pipeline"),
    }


def discover_symbols(
    data_root: Path,
    *,
    explicit: list[str],
    exclude: set[str],
    min_history: int,
    max_train_symbols: int,
) -> list[str]:
    if explicit:
        return [symbol for symbol in explicit if symbol not in exclude]

    rows: list[tuple[str, int]] = []
    for path in sorted(data_root.glob("*.csv")):
        if path.name in {"summary.csv", "download_summary.csv"}:
            continue
        count = 0
        with path.open(newline="", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                try:
                    value = float(row["close"])
                except Exception:
                    continue
                if math.isfinite(value):
                    count += 1
        if count >= min_history and path.stem not in exclude:
            rows.append((path.stem, count))

    rows.sort(key=lambda item: (-item[1], item[0]))
    symbols = [symbol for symbol, _ in rows]
    if max_train_symbols and max_train_symbols > 0:
        symbols = symbols[:max_train_symbols]
    return symbols


def build_inputs(
    symbols: list[str],
    *,
    data_root: Path,
    target_cols: list[str],
    preaug_name: str,
    val_hours: int,
    test_hours: int,
    load_hourly_frame,
    split_data,
    get_augmentation,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    train_inputs: list[dict[str, Any]] = []
    val_inputs: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for symbol in symbols:
        df = load_hourly_frame(data_root / f"{symbol}.csv")
        train_df, val_df, _ = split_data(df, val_hours, test_hours)
        augmentation = get_augmentation(preaug_name)
        train_aug = augmentation.transform_dataframe(train_df.copy())
        val_aug = augmentation.transform_dataframe(val_df.copy())
        train_inputs.append({"target": train_aug[target_cols].to_numpy(dtype=np.float32).T})
        val_inputs.append({"target": val_aug[target_cols].to_numpy(dtype=np.float32).T})
        counts[symbol] = len(df)
    return train_inputs, val_inputs, counts


def evaluate_symbols(
    pipeline: Any,
    *,
    symbols: list[str],
    data_root: Path,
    context_length: int,
    prediction_length: int,
    preaug_name: str,
    val_hours: int,
    test_hours: int,
    load_hourly_frame,
    split_data,
    compute_consistency_metrics,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for symbol in symbols:
        df = load_hourly_frame(data_root / f"{symbol}.csv")
        train_df, val_df, test_df = split_data(df, val_hours, test_hours)
        full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        val_start = len(train_df)
        val_end = len(train_df) + len(val_df)
        test_start = val_end
        test_end = len(full_df)
        val_metrics = compute_consistency_metrics(
            pipeline,
            full_df,
            context_length,
            prediction_length,
            val_start,
            val_end,
            preaug_name,
        )
        test_metrics = compute_consistency_metrics(
            pipeline,
            full_df,
            context_length,
            prediction_length,
            test_start,
            test_end,
            preaug_name,
        )
        results[symbol] = {
            "val": asdict(val_metrics),
            "test": asdict(test_metrics),
        }
    return results


def mean_metric(results: dict[str, Any], split: str, metric: str) -> float | None:
    values = []
    for payload in results.values():
        value = payload.get(split, {}).get(metric)
        if value is None:
            continue
        try:
            value = float(value)
        except Exception:
            continue
        if math.isfinite(value):
            values.append(value)
    if not values:
        return None
    return float(sum(values) / len(values))


@dataclass
class FakeConfig:
    context_length: int
    prediction_length: int
    batch_size: int
    learning_rate: float
    num_steps: int
    finetune_mode: str
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_targets: tuple[str, ...]
    merge_lora: bool = True


def main() -> None:
    args = parse_args()
    stock_repo = Path(args.stock_repo)
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    mods = import_stock_modules(stock_repo)
    target_cols = ["open", "high", "low", "close"]

    eval_symbols = [symbol.upper() for symbol in args.eval_symbols]
    exclude = {symbol.upper() for symbol in args.exclude_symbols} | set(eval_symbols)
    train_symbols = discover_symbols(
        data_root,
        explicit=[symbol.upper() for symbol in args.train_symbols],
        exclude=exclude,
        min_history=args.min_history,
        max_train_symbols=args.max_train_symbols,
    )
    if not train_symbols:
        raise SystemExit("No train symbols selected for mixed transfer run.")

    print(f"Training symbols ({len(train_symbols)}): {' '.join(train_symbols)}")
    print(f"Eval symbols ({len(eval_symbols)}): {' '.join(eval_symbols)}")

    train_inputs, val_inputs, train_counts = build_inputs(
        train_symbols,
        data_root=data_root,
        target_cols=target_cols,
        preaug_name=args.preaug,
        val_hours=args.val_hours,
        test_hours=args.test_hours,
        load_hourly_frame=mods["load_hourly_frame"],
        split_data=mods["split_data"],
        get_augmentation=mods["get_augmentation"],
    )

    base_pipeline = mods["_load_pipeline"](args.model_id, args.device, args.dtype)
    base_eval = evaluate_symbols(
        base_pipeline,
        symbols=eval_symbols,
        data_root=data_root,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        preaug_name=args.preaug,
        val_hours=args.val_hours,
        test_hours=args.test_hours,
        load_hourly_frame=mods["load_hourly_frame"],
        split_data=mods["split_data"],
        compute_consistency_metrics=mods["compute_consistency_metrics"],
    )

    run_name = (
        f"{args.run_prefix}_pool{len(train_symbols)}_{args.preaug}"
        f"_ctx{args.context_length}_lr{args.learning_rate:.0e}"
        f"_steps{args.num_steps}_r{args.lora_r}_{int(time.time())}"
    )
    output_dir = output_root / run_name
    cfg = FakeConfig(
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        finetune_mode="lora",
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_targets=("q", "k", "v", "o"),
    )

    start = time.perf_counter()
    finetuned = mods["_fit_pipeline"](base_pipeline, train_inputs, val_inputs, cfg, output_dir)
    train_seconds = time.perf_counter() - start
    ckpt_path = mods["_save_pipeline"](finetuned, output_dir, "finetuned-ckpt")

    transfer_eval = evaluate_symbols(
        finetuned,
        symbols=eval_symbols,
        data_root=data_root,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        preaug_name=args.preaug,
        val_hours=args.val_hours,
        test_hours=args.test_hours,
        load_hourly_frame=mods["load_hourly_frame"],
        split_data=mods["split_data"],
        compute_consistency_metrics=mods["compute_consistency_metrics"],
    )

    payload = {
        "model_id": args.model_id,
        "data_root": str(data_root),
        "train_symbols": train_symbols,
        "eval_symbols": eval_symbols,
        "preaug": args.preaug,
        "context_length": args.context_length,
        "prediction_length": args.prediction_length,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_steps": args.num_steps,
        "lora_r": args.lora_r,
        "train_seconds": train_seconds,
        "train_symbol_rows": train_counts,
        "output_dir": str(output_dir),
        "checkpoint_path": str(ckpt_path),
        "base_eval": base_eval,
        "transfer_eval": transfer_eval,
        "summary": {
            "base_val_mean_mape_pct": mean_metric(base_eval, "val", "mae_percent_mean"),
            "base_test_mean_mape_pct": mean_metric(base_eval, "test", "mae_percent_mean"),
            "transfer_val_mean_mape_pct": mean_metric(transfer_eval, "val", "mae_percent_mean"),
            "transfer_test_mean_mape_pct": mean_metric(transfer_eval, "test", "mae_percent_mean"),
        },
    }

    output_path = REPO_ROOT / args.output_json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))

    print(json.dumps(payload["summary"], indent=2))
    print(f"Saved mixed transfer report to {output_path}")
    print(f"Saved mixed checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
