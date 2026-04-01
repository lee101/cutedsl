from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STOCK_REPO = Path("/home/lee/code/stock")
DEFAULT_PYTHON = DEFAULT_STOCK_REPO / ".venv" / "bin" / "python"
DEFAULT_TRAINER = DEFAULT_STOCK_REPO / "train_crypto_lora_sweep.py"
DEFAULT_DATA_ROOT = DEFAULT_STOCK_REPO / "trainingdatahourly" / "crypto"
DEFAULT_OUTPUT_ROOT = DEFAULT_STOCK_REPO / "chronos2_finetuned"
DEFAULT_RESULTS_DIR = REPO_ROOT / "examples" / "lora_frontier_runs"
DEFAULT_EXISTING_RESULTS = DEFAULT_STOCK_REPO / "crypto_lora_sweep"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and rank LoRA training frontier jobs.")
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--stock-repo", default=str(DEFAULT_STOCK_REPO))
    parser.add_argument("--trainer-script", default=str(DEFAULT_TRAINER))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--existing-results-dir", default=str(DEFAULT_EXISTING_RESULTS))
    parser.add_argument("--symbols", nargs="+", default=["BTCUSD"])
    parser.add_argument("--preaugs", nargs="+", default=["percent_change", "log_returns", "baseline", "robust_scaling"])
    parser.add_argument("--context-lengths", nargs="+", type=int, default=[128, 256])
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[5e-5, 1e-4])
    parser.add_argument("--num-steps", nargs="+", type=int, default=[50, 100, 250, 500])
    parser.add_argument("--lora-ranks", nargs="+", type=int, default=[16])
    parser.add_argument("--prediction-length", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-runs", type=int, default=0, help="0 means all generated configs.")
    parser.add_argument("--run-prefix", default="cutedsl_lora_frontier")
    parser.add_argument("--output-json", default="examples/lora_frontier/frontier_results.json")
    parser.add_argument("--output-csv", default="examples/lora_frontier/frontier_results.csv")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_existing_priors(results_dir: Path) -> dict[str, dict[str, float]]:
    stats: dict[str, list[float]] = defaultdict(list)
    if not results_dir.exists():
        return {}

    for path in results_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        config = payload.get("config") or {}
        val = (payload.get("val") or {}).get("mae_percent_mean")
        if val is None or not math.isfinite(float(val)):
            continue
        preaug = str(config.get("preaug", "unknown"))
        stats[preaug].append(float(val))

    summary: dict[str, dict[str, float]] = {}
    for preaug, values in stats.items():
        summary[preaug] = {
            "count": float(len(values)),
            "mean_val_mape": float(sum(values) / len(values)),
            "best_val_mape": float(min(values)),
        }
    return summary


def sort_preaugs(preaugs: list[str], priors: dict[str, dict[str, float]]) -> list[str]:
    def key(name: str) -> tuple[float, float, str]:
        info = priors.get(name)
        if info is None:
            return (math.inf, math.inf, name)
        return (float(info["mean_val_mape"]), float(info["best_val_mape"]), name)

    return sorted(preaugs, key=key)


def build_configs(args: argparse.Namespace, priors: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    preaugs = sort_preaugs(list(args.preaugs), priors)
    configs: list[dict[str, Any]] = []
    for symbol in args.symbols:
        for preaug, context_length, learning_rate, steps, rank in itertools.product(
            preaugs,
            args.context_lengths,
            args.learning_rates,
            args.num_steps,
            args.lora_ranks,
        ):
            configs.append(
                {
                    "symbol": symbol,
                    "preaug": preaug,
                    "context_length": context_length,
                    "learning_rate": learning_rate,
                    "num_steps": steps,
                    "lora_r": rank,
                }
            )

    def heuristic(cfg: dict[str, Any]) -> tuple[float, int, int, float, int]:
        prior = priors.get(cfg["preaug"], {})
        prior_mean = float(prior.get("mean_val_mape", 1e9))
        # Try strong priors and shorter runs first so time-to-good-MAE is visible early.
        return (
            prior_mean,
            int(cfg["context_length"]),
            int(cfg["num_steps"]),
            float(cfg["learning_rate"]),
            int(cfg["lora_r"]),
        )

    configs.sort(key=heuristic)
    if args.max_runs and args.max_runs > 0:
        configs = configs[: args.max_runs]
    return configs


def run_training_job(
    args: argparse.Namespace,
    config: dict[str, Any],
    run_index: int,
    total_runs: int,
) -> dict[str, Any]:
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    run_token = (
        f"{args.run_prefix}_{config['symbol']}_{config['preaug']}"
        f"_ctx{config['context_length']}_lr{config['learning_rate']:.0e}"
        f"_steps{config['num_steps']}_r{config['lora_r']}_{int(time.time())}"
    )

    command = [
        args.python,
        args.trainer_script,
        "--symbol",
        config["symbol"],
        "--data-root",
        args.data_root,
        "--output-root",
        args.output_root,
        "--results-dir",
        args.results_dir,
        "--context-length",
        str(config["context_length"]),
        "--prediction-length",
        str(args.prediction_length),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(config["learning_rate"]),
        "--num-steps",
        str(config["num_steps"]),
        "--lora-r",
        str(config["lora_r"]),
        "--preaug",
        config["preaug"],
        "--run-prefix",
        run_token,
    ]

    print(
        f"[{run_index}/{total_runs}] "
        f"{config['symbol']} preaug={config['preaug']} ctx={config['context_length']} "
        f"lr={config['learning_rate']:.0e} steps={config['num_steps']} r={config['lora_r']}"
    )

    if args.dry_run:
        return {
            **config,
            "run_token": run_token,
            "command": command,
            "train_seconds": None,
            "val_mae_percent": None,
            "test_mae_percent": None,
            "val_mae": None,
            "test_mae": None,
            "result_path": None,
            "status": "dry_run",
        }

    env = os.environ.copy()
    stock_repo = str(Path(args.stock_repo))
    pythonpath_parts = [stock_repo, env.get("PYTHONPATH", "")]
    env["PYTHONPATH"] = ":".join(part for part in pythonpath_parts if part)

    start = time.perf_counter()
    subprocess.run(command, check=True, env=env)
    elapsed = time.perf_counter() - start

    matches = sorted(results_dir.glob(f"{run_token}_*.json"))
    if not matches:
        raise RuntimeError(f"No result JSON found for run prefix {run_token}")
    result_path = matches[-1]
    payload = json.loads(result_path.read_text())
    val = payload.get("val") or {}
    test = payload.get("test") or {}

    return {
        **config,
        "run_token": run_token,
        "command": command,
        "train_seconds": elapsed,
        "val_mae_percent": val.get("mae_percent_mean"),
        "test_mae_percent": test.get("mae_percent_mean"),
        "val_mae": val.get("mae_mean"),
        "test_mae": test.get("mae_mean"),
        "result_path": str(result_path),
        "status": "ok",
    }


def compute_frontier(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = [
        row
        for row in rows
        if row.get("status") == "ok"
        and row.get("train_seconds") is not None
        and row.get("val_mae_percent") is not None
        and math.isfinite(float(row["train_seconds"]))
        and math.isfinite(float(row["val_mae_percent"]))
    ]
    valid.sort(key=lambda row: (float(row["train_seconds"]), float(row["val_mae_percent"])))

    frontier: list[dict[str, Any]] = []
    best_seen = math.inf
    for row in valid:
        metric = float(row["val_mae_percent"])
        if metric < best_seen:
            frontier.append(row)
            best_seen = metric
    return frontier


def compute_time_to_targets(rows: list[dict[str, Any]], targets: list[float]) -> dict[str, dict[str, Any] | None]:
    valid = [
        row
        for row in rows
        if row.get("status") == "ok"
        and row.get("train_seconds") is not None
        and row.get("val_mae_percent") is not None
    ]
    answers: dict[str, dict[str, Any] | None] = {}
    for target in targets:
        hits = [
            row for row in valid
            if float(row["val_mae_percent"]) <= target
        ]
        if not hits:
            answers[f"{target:.3f}"] = None
            continue
        answers[f"{target:.3f}"] = min(hits, key=lambda row: float(row["train_seconds"]))
    return answers


def main() -> None:
    args = parse_args()
    priors = load_existing_priors(Path(args.existing_results_dir))
    configs = build_configs(args, priors)

    print("Prior preaug ranking from existing LoRA results:")
    for name in sort_preaugs(list(args.preaugs), priors):
        info = priors.get(name)
        if info is None:
            print(f"  {name}: no prior data")
            continue
        print(
            f"  {name}: count={int(info['count'])} "
            f"mean_val_mape={info['mean_val_mape']:.6f} "
            f"best_val_mape={info['best_val_mape']:.6f}"
        )

    rows: list[dict[str, Any]] = []
    total_runs = len(configs)
    for index, config in enumerate(configs, start=1):
        row = run_training_job(args, config, index, total_runs)
        rows.append(row)
        if row["status"] == "ok":
            print(
                f"  -> train_seconds={float(row['train_seconds']):.2f} "
                f"val_mae%={float(row['val_mae_percent']):.6f} "
                f"test_mae%={float(row['test_mae_percent']):.6f}"
            )

    frontier = compute_frontier(rows)
    time_to_targets = compute_time_to_targets(rows, targets=[5.0, 3.0, 2.0, 1.5, 1.0])

    payload = {
        "search_space": {
            "symbols": args.symbols,
            "preaugs": args.preaugs,
            "context_lengths": args.context_lengths,
            "learning_rates": args.learning_rates,
            "num_steps": args.num_steps,
            "lora_ranks": args.lora_ranks,
            "prediction_length": args.prediction_length,
            "batch_size": args.batch_size,
        },
        "existing_priors": priors,
        "rows": rows,
        "frontier": frontier,
        "time_to_targets": time_to_targets,
    }

    output_json = REPO_ROOT / args.output_json
    output_csv = REPO_ROOT / args.output_csv
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2))

    with output_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "symbol",
                "preaug",
                "context_length",
                "learning_rate",
                "num_steps",
                "lora_r",
                "train_seconds",
                "val_mae_percent",
                "test_mae_percent",
                "val_mae",
                "test_mae",
                "status",
                "result_path",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})

    print("\nPareto frontier (train_seconds vs val_mae%):")
    for row in frontier:
        print(
            f"  {row['symbol']} preaug={row['preaug']} ctx={row['context_length']} "
            f"lr={row['learning_rate']:.0e} steps={row['num_steps']} r={row['lora_r']} "
            f"train_s={float(row['train_seconds']):.2f} val_mae%={float(row['val_mae_percent']):.6f}"
        )

    print("\nFastest runs to target val_mae%:")
    for target, row in time_to_targets.items():
        if row is None:
            print(f"  <= {target}%: no hit")
            continue
        print(
            f"  <= {target}%: {row['symbol']} preaug={row['preaug']} ctx={row['context_length']} "
            f"lr={row['learning_rate']:.0e} steps={row['num_steps']} r={row['lora_r']} "
            f"train_s={float(row['train_seconds']):.2f} val_mae%={float(row['val_mae_percent']):.6f}"
        )

    print(f"\nSaved frontier JSON to {output_json}")
    print(f"Saved frontier CSV to {output_csv}")


if __name__ == "__main__":
    main()
