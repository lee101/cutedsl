from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STOCK_REPO = Path("/home/lee/code/stock")
DEFAULT_PYTHON = DEFAULT_STOCK_REPO / ".venv" / "bin" / "python"
DEFAULT_DATA_ROOT = DEFAULT_STOCK_REPO / "trainingdatahourly" / "crypto"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "examples" / "crypto_hourly_frontier"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage a broad hourly-crypto CuteChronos search: "
            "base tuning across all symbols, optional dilation rescue, "
            "then LoRA frontier runs on laggards."
        )
    )
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--stock-repo", default=str(DEFAULT_STOCK_REPO))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--symbols", nargs="+", default=[])
    parser.add_argument("--min-history", type=int, default=5000)
    parser.add_argument("--max-symbols", type=int, default=0, help="0 means all discovered symbols.")

    parser.add_argument("--prediction-length", type=int, default=24)
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--stride", type=int, default=24)
    parser.add_argument("--warmup-windows", type=int, default=1)

    parser.add_argument("--base-context-lengths", nargs="+", type=int, default=[128, 256])
    parser.add_argument("--base-strategies", nargs="+", default=["percent_change", "log_returns", "baseline"])
    parser.add_argument("--base-quantiles", nargs="+", type=float, default=[0.5, 0.7])
    parser.add_argument("--base-dtypes", nargs="+", default=["bfloat16"])
    parser.add_argument("--base-compile-modes", nargs="+", default=["", "reduce-overhead"])

    parser.add_argument("--run-dilation-pass", action="store_true")
    parser.add_argument("--dilation-threshold", type=float, default=4.0)
    parser.add_argument("--dilation-top-n", type=int, default=12)
    parser.add_argument("--dilation-context-lengths", nargs="+", type=int, default=[128, 256])
    parser.add_argument("--dilation-strategies", nargs="+", default=["percent_change", "log_returns"])
    parser.add_argument("--dilation-quantiles", nargs="+", type=float, default=[0.5, 0.7])
    parser.add_argument("--dilation-stride-sets", nargs="+", default=["1,2,4"])
    parser.add_argument("--dilation-aggregations", nargs="+", default=["median"])
    parser.add_argument("--dilation-trim-values", nargs="+", type=int, default=[1])

    parser.add_argument("--run-lora-pass", action="store_true")
    parser.add_argument("--lora-threshold", type=float, default=4.0)
    parser.add_argument("--lora-top-n", type=int, default=8)
    parser.add_argument("--lora-preaugs", nargs="+", default=["percent_change", "log_returns"])
    parser.add_argument("--lora-context-lengths", nargs="+", type=int, default=[128])
    parser.add_argument("--lora-learning-rates", nargs="+", type=float, default=[5e-5, 1e-4])
    parser.add_argument("--lora-num-steps", nargs="+", type=int, default=[100, 250])
    parser.add_argument("--lora-ranks", nargs="+", type=int, default=[16])
    parser.add_argument("--lora-batch-size", type=int, default=32)
    parser.add_argument(
        "--lora-max-runs-per-symbol",
        type=int,
        default=4,
        help="Bound the per-symbol LoRA grid. 0 means all configs.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def discover_symbols(data_root: Path, explicit: list[str], min_history: int, max_symbols: int) -> list[str]:
    if explicit:
        return explicit[: max_symbols or None]

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
        if count >= min_history:
            rows.append((path.stem, count))

    rows.sort(key=lambda item: (-item[1], item[0]))
    symbols = [symbol for symbol, _ in rows]
    if max_symbols and max_symbols > 0:
        symbols = symbols[:max_symbols]
    return symbols


def make_env(stock_repo: Path) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_parts = [str(REPO_ROOT), str(stock_repo), env.get("PYTHONPATH", "")]
    env["PYTHONPATH"] = ":".join(part for part in pythonpath_parts if part)
    return env


def format_cmd(command: list[str]) -> str:
    return " ".join(command)


def run_command(
    command: list[str],
    *,
    env: dict[str, str],
    cwd: Path,
    dry_run: bool,
) -> None:
    print(f"\n$ {format_cmd(command)}")
    if dry_run:
        return
    subprocess.run(command, check=True, cwd=cwd, env=env)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def metric_value(row: dict[str, Any] | None, metric: str = "mean_mape_pct") -> float:
    if not row:
        return math.inf
    value = row.get(metric)
    if value is None:
        return math.inf
    try:
        return float(value)
    except Exception:
        return math.inf


def choose_best_rows(*payloads: dict[str, Any]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        per_symbol = payload.get("per_symbol") or {}
        for symbol, summary in per_symbol.items():
            best = summary.get("best")
            if symbol not in merged or metric_value(best) < metric_value(merged[symbol]):
                merged[symbol] = best
    return merged


def select_laggards(best_rows: dict[str, dict[str, Any]], threshold: float, top_n: int) -> list[str]:
    laggards = [
        (symbol, row)
        for symbol, row in best_rows.items()
        if metric_value(row) > threshold
    ]
    laggards.sort(key=lambda item: metric_value(item[1]), reverse=True)
    symbols = [symbol for symbol, _ in laggards]
    if top_n and top_n > 0:
        symbols = symbols[:top_n]
    return symbols


def collect_lora_bests(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best_by_symbol: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        symbol = str(row["symbol"])
        current = best_by_symbol.get(symbol)
        score = row.get("val_mae_percent")
        if score is None:
            continue
        if current is None or float(score) < float(current["val_mae_percent"]):
            best_by_symbol[symbol] = row
    return best_by_symbol


def summarize(best_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    values = [metric_value(row) for row in best_rows.values() if math.isfinite(metric_value(row))]
    values.sort()
    if not values:
        return {"count": 0, "mean_mape_pct": None, "median_mape_pct": None}
    return {
        "count": len(values),
        "mean_mape_pct": sum(values) / len(values),
        "median_mape_pct": values[len(values) // 2],
        "best_mape_pct": values[0],
        "worst_mape_pct": values[-1],
    }


def main() -> None:
    args = parse_args()
    stock_repo = Path(args.stock_repo)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    symbols = discover_symbols(data_root, args.symbols, args.min_history, args.max_symbols)
    if not symbols:
        raise SystemExit(f"No usable symbols found under {data_root}")

    print(f"Discovered {len(symbols)} symbols")
    print(" ".join(symbols))

    env = make_env(stock_repo)
    base_json = output_dir / "base_tuning.json"
    base_csv = output_dir / "base_tuning.csv"
    base_command = [
        args.python,
        "examples/chronos_wrappers/tune_cutechronos.py",
        "--stock-repo",
        str(stock_repo),
        "--stock-data-dir",
        str(data_root),
        "--symbols",
        *symbols,
        "--context-lengths",
        *(str(value) for value in args.base_context_lengths),
        "--prediction-length",
        str(args.prediction_length),
        "--windows",
        str(args.windows),
        "--stride",
        str(args.stride),
        "--warmup-windows",
        str(args.warmup_windows),
        "--strategies",
        *args.base_strategies,
        "--quantiles",
        *(str(value) for value in args.base_quantiles),
        "--dtypes",
        *args.base_dtypes,
        "--compile-modes",
        *args.base_compile_modes,
        "--inference-strategies",
        "single",
        "--top-k",
        "12",
        "--output-json",
        str(base_json.relative_to(REPO_ROOT)),
        "--output-csv",
        str(base_csv.relative_to(REPO_ROOT)),
    ]
    run_command(base_command, env=env, cwd=REPO_ROOT, dry_run=args.dry_run)

    if args.dry_run:
        return

    base_payload = load_json(base_json)
    base_best = choose_best_rows(base_payload)
    best_payloads = [base_payload]

    dilation_symbols: list[str] = []
    if args.run_dilation_pass:
        dilation_symbols = select_laggards(base_best, args.dilation_threshold, args.dilation_top_n)
        if dilation_symbols:
            dilation_json = output_dir / "dilation_tuning.json"
            dilation_csv = output_dir / "dilation_tuning.csv"
            dilation_command = [
                args.python,
                "examples/chronos_wrappers/tune_cutechronos.py",
                "--stock-repo",
                str(stock_repo),
                "--stock-data-dir",
                str(data_root),
                "--symbols",
                *dilation_symbols,
                "--context-lengths",
                *(str(value) for value in args.dilation_context_lengths),
                "--prediction-length",
                str(args.prediction_length),
                "--windows",
                str(args.windows),
                "--stride",
                str(args.stride),
                "--warmup-windows",
                str(args.warmup_windows),
                "--strategies",
                *args.dilation_strategies,
                "--quantiles",
                *(str(value) for value in args.dilation_quantiles),
                "--dtypes",
                *args.base_dtypes,
                "--compile-modes",
                *args.base_compile_modes,
                "--inference-strategies",
                "dilation",
                "--dilation-stride-sets",
                *args.dilation_stride_sets,
                "--ensemble-aggregations",
                *args.dilation_aggregations,
                "--trim-values",
                *(str(value) for value in args.dilation_trim_values),
                "--top-k",
                "12",
                "--output-json",
                str(dilation_json.relative_to(REPO_ROOT)),
                "--output-csv",
                str(dilation_csv.relative_to(REPO_ROOT)),
            ]
            run_command(dilation_command, env=env, cwd=REPO_ROOT, dry_run=False)
            best_payloads.append(load_json(dilation_json))

    best_rows = choose_best_rows(*best_payloads)
    lora_rows: list[dict[str, Any]] = []
    lora_symbols: list[str] = []
    if args.run_lora_pass:
        lora_symbols = select_laggards(best_rows, args.lora_threshold, args.lora_top_n)
        for symbol in lora_symbols:
            symbol_slug = symbol.lower()
            frontier_json = output_dir / f"lora_{symbol_slug}.json"
            frontier_csv = output_dir / f"lora_{symbol_slug}.csv"
            command = [
                args.python,
                "examples/lora_frontier.py",
                "--python",
                args.python,
                "--stock-repo",
                str(stock_repo),
                "--data-root",
                str(data_root),
                "--symbols",
                symbol,
                "--preaugs",
                *args.lora_preaugs,
                "--context-lengths",
                *(str(value) for value in args.lora_context_lengths),
                "--learning-rates",
                *(str(value) for value in args.lora_learning_rates),
                "--num-steps",
                *(str(value) for value in args.lora_num_steps),
                "--lora-ranks",
                *(str(value) for value in args.lora_ranks),
                "--prediction-length",
                str(args.prediction_length),
                "--batch-size",
                str(args.lora_batch_size),
                "--max-runs",
                str(args.lora_max_runs_per_symbol),
                "--run-prefix",
                f"cutedsl_hourly_{symbol_slug}",
                "--output-json",
                str(frontier_json.relative_to(REPO_ROOT)),
                "--output-csv",
                str(frontier_csv.relative_to(REPO_ROOT)),
            ]
            run_command(command, env=env, cwd=REPO_ROOT, dry_run=False)
            frontier_payload = load_json(frontier_json)
            lora_rows.extend(frontier_payload.get("rows") or [])

    lora_best = collect_lora_bests(lora_rows)

    final_per_symbol: dict[str, dict[str, Any]] = {}
    for symbol in symbols:
        base_row = best_rows.get(symbol)
        lora_row = lora_best.get(symbol)
        chosen = {
            "best_inference": base_row,
            "best_lora": lora_row,
            "recommended": {
                "source": "lora" if metric_value(lora_row, "val_mae_percent") < metric_value(base_row, "mean_mape_pct") else "inference",
                "row": lora_row if metric_value(lora_row, "val_mae_percent") < metric_value(base_row, "mean_mape_pct") else base_row,
            },
        }
        final_per_symbol[symbol] = chosen

    summary = {
        "symbols": symbols,
        "prediction_length": args.prediction_length,
        "windows": args.windows,
        "stride": args.stride,
        "base_summary": summarize(base_best),
        "final_inference_summary": summarize(best_rows),
        "lora_summary": summarize(
            {
                symbol: {
                    "mean_mape_pct": row.get("val_mae_percent"),
                }
                for symbol, row in lora_best.items()
            }
        ),
        "dilation_symbols": dilation_symbols,
        "lora_symbols": lora_symbols,
        "per_symbol": final_per_symbol,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\nFinal hourly crypto summary:")
    print(json.dumps(summary["base_summary"], indent=2))
    print(json.dumps(summary["final_inference_summary"], indent=2))
    if lora_symbols:
        print(json.dumps(summary["lora_summary"], indent=2))
    print(f"\nSaved summary JSON to {summary_path}")


if __name__ == "__main__":
    sys.exit(main())
