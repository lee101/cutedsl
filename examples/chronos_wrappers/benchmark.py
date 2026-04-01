from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = Path(__file__).resolve().parent
DEFAULT_PYTHON = Path("/home/lee/code/stock/.venv/bin/python")
DEFAULT_CHRONOS_SRC = Path("/home/lee/code/stock/chronos-forecasting/src")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-language CuteChronos wrapper benchmark")
    parser.add_argument("--python", default=os.environ.get("CUTECHRONOS_BENCH_PYTHON", str(DEFAULT_PYTHON)))
    parser.add_argument("--chronos-src", default=os.environ.get("CHRONOS_FORECASTING_SRC", str(DEFAULT_CHRONOS_SRC)))
    parser.add_argument("--model-id", default="amazon/chronos-2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--compile-mode", default="")
    parser.add_argument("--prediction-length", type=int, default=3)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--stock-data-dir", default="/home/lee/code/stock/trainingdatadailybinance")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSD", "ETHUSD", "SOLUSD"])
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--output", default="examples/chronos_wrappers/benchmark_results.json")
    return parser.parse_args()


def runtime_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_parts = [str(REPO_ROOT), args.chronos_src]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    merged_pythonpath = ":".join(part for part in pythonpath_parts if part)

    env["PYTHONPATH"] = merged_pythonpath
    env["CUTECHRONOS_PYTHONPATH"] = merged_pythonpath
    env["CHRONOS_FORECASTING_SRC"] = args.chronos_src
    return env


def build_artifacts(args: argparse.Namespace, env: dict[str, str]) -> None:
    subprocess.run(
        ["make", "-C", str(EXAMPLES_ROOT), f"PYTHON={args.python}", f"CHRONOS_SRC={args.chronos_src}"],
        check=True,
        env=env,
    )


def csv_series_case(csv_path: Path, context_length: int, prediction_length: int) -> dict[str, object]:
    closes: list[float] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            closes.append(float(row["close"]))

    total_needed = context_length + prediction_length
    if len(closes) < total_needed:
        raise ValueError(f"{csv_path} has only {len(closes)} rows; need {total_needed}")

    values = closes[-total_needed:]
    return {
        "name": csv_path.stem,
        "context": values[:context_length],
        "actual": values[context_length:],
    }


def benchmark_original_python(args: argparse.Namespace, case: dict[str, object], env: dict[str, str]) -> dict[str, object]:
    script = f"""
import json, os, sys, time
sys.path.insert(0, {str(REPO_ROOT)!r})
sys.path.insert(0, {args.chronos_src!r})
import torch
from chronos import BaseChronosPipeline

context = torch.tensor({case['context']!r}, dtype=torch.float32)
actual = {case['actual']!r}
device = {args.device!r}
device_map = "cuda" if device.startswith("cuda") else device
pipe = BaseChronosPipeline.from_pretrained(
    {args.model_id!r},
    device_map=device_map,
    torch_dtype={args.dtype!r},
)

def sync():
    if device.startswith('cuda') and torch.cuda.is_available():
        torch.cuda.synchronize()

for _ in range({args.warmup}):
    pipe.predict_quantiles([context], prediction_length={args.prediction_length})

latencies = []
forecast = None
for _ in range({args.runs}):
    sync()
    t0 = time.perf_counter()
    _, mean = pipe.predict_quantiles([context], prediction_length={args.prediction_length})
    sync()
    latencies.append((time.perf_counter() - t0) * 1000.0)
    forecast = mean[0][0].tolist()

valid = [
    abs(f - a)
    for f, a in zip(forecast, actual)
    if f == f and a == a
]
mae = (sum(valid) / len(valid)) if valid else None
mape = [
    abs((f - a) / a) * 100.0
    for f, a in zip(forecast, actual)
    if f == f and a == a and abs(a) > 1e-12
]
mape_pct = (sum(mape) / len(mape)) if mape else None
print(json.dumps({{
    "language": "python_original",
    "backend": "original",
    "model_id": {args.model_id!r},
    "device": device,
    "prediction_length": {args.prediction_length},
    "runs": {args.runs},
    "warmup": {args.warmup},
    "avg_outer_latency_ms": sum(latencies) / len(latencies),
    "avg_inner_latency_ms": sum(latencies) / len(latencies),
    "mae": mae,
    "mape_pct": mape_pct,
    "forecast": forecast,
    "actual": actual,
}}, indent=2))
"""
    proc = subprocess.run(
        [args.python, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return json.loads(proc.stdout)


def run_json_command(cmd: list[str], env: dict[str, str]) -> dict[str, object]:
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
    return json.loads(proc.stdout)


def context_csv(values: list[float]) -> str:
    return ",".join(f"{value:.6f}" for value in values)


def benchmark_case(args: argparse.Namespace, case: dict[str, object], env: dict[str, str]) -> dict[str, object]:
    context = case["context"]
    actual = case["actual"]
    context_arg = context_csv(context)
    actual_arg = context_csv(actual)

    base_args = [
        "--model-id", args.model_id,
        "--backend", "cute",
        "--device", args.device,
        "--dtype", args.dtype,
        "--prediction-length", str(args.prediction_length),
        "--runs", str(args.runs),
        "--warmup", str(args.warmup),
        "--context", context_arg,
        "--actual", actual_arg,
    ]
    if args.compile_mode:
        base_args.extend(["--compile-mode", args.compile_mode])

    c_runner = EXAMPLES_ROOT / "build" / "chronos_c_runner"
    go_runner = EXAMPLES_ROOT / "build" / "chronos_go_runner"
    ctypes_runner = EXAMPLES_ROOT / "python" / "ctypes_runner.py"

    results = {
        "case": case["name"],
        "c": run_json_command([str(c_runner), *base_args], env),
        "go": run_json_command([str(go_runner), *base_args], env),
        "python_wrapper": run_json_command([args.python, str(ctypes_runner), *base_args], env),
        "python_original": benchmark_original_python(args, case, env),
    }
    return results


def print_case_summary(case_result: dict[str, object]) -> None:
    def fmt(value: object) -> str:
        if value is None:
            return "n/a"
        return f"{float(value):.6f}"

    print(f"\nCase: {case_result['case']}")
    print(f"{'Runner':<18} {'Outer(ms)':>12} {'Inner(ms)':>12} {'MAE':>12} {'MAPE%':>12}")
    for key in ["c", "go", "python_wrapper", "python_original"]:
        result = case_result[key]
        print(
            f"{result['language']:<18} "
            f"{result['avg_outer_latency_ms']:>12.3f} "
            f"{result['avg_inner_latency_ms']:>12.3f} "
            f"{fmt(result['mae']):>12} "
            f"{fmt(result.get('mape_pct')):>12}"
        )


def main() -> None:
    args = parse_args()
    env = runtime_env(args)
    build_artifacts(args, env)

    cases = [
        {
            "name": "toy_even_sequence",
            "context": [2.0, 4.0, 6.0, 8.0, 12.0],
            "actual": [14.0, 16.0, 18.0],
        }
    ]

    stock_data_dir = Path(args.stock_data_dir)
    for symbol in args.symbols:
        csv_path = stock_data_dir / f"{symbol}.csv"
        if csv_path.exists():
            cases.append(csv_series_case(csv_path, args.context_length, args.prediction_length))

    all_results = {
        "model_id": args.model_id,
        "device": args.device,
        "dtype": args.dtype,
        "compile_mode": args.compile_mode or None,
        "prediction_length": args.prediction_length,
        "runs": args.runs,
        "warmup": args.warmup,
        "cases": [],
    }

    for case in cases:
        case_result = benchmark_case(args, case, env)
        all_results["cases"].append(case_result)
        print_case_summary(case_result)

    output_path = REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
