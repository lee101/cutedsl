"""Remote GPU runner for zimageaccelerated experiments.

This follows the same broad pattern as the sibling TradingJEPA helper:
- acquire or reuse a remote GPU pod
- rsync this repo to the pod
- install the local package
- run a benchmark or arbitrary experiment command
- pull JSON results back locally
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_args, **_kwargs):
        return False


load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[1]
STOCK_ROOT = REPO_ROOT.parent / "stock"

sys.path.insert(0, str(STOCK_ROOT))
PoolPod = object
POOL_AVAILABLE = False
_POOL_IMPORT_ERROR: str | None = None


def _load_pool_support() -> None:
    global PoolPod, POOL_AVAILABLE, _POOL_IMPORT_ERROR
    global get_or_create_pod, load_pool_state, refresh_pod_status, save_pool_state
    global scp_from, ssh_exec, _print_cost_estimate, _require_client, _resolve_gpu

    if POOL_AVAILABLE:
        return
    try:
        from pufferlib_market.gpu_pool_rl import (
            PoolPod as _PoolPod,
            get_or_create_pod as _get_or_create_pod,
            load_pool_state as _load_pool_state,
            refresh_pod_status as _refresh_pod_status,
            save_pool_state as _save_pool_state,
            scp_from as _scp_from,
            ssh_exec as _ssh_exec,
            _print_cost_estimate as __print_cost_estimate,
            _require_client as __require_client,
            _resolve_gpu as __resolve_gpu,
        )
    except Exception as exc:
        _POOL_IMPORT_ERROR = str(exc)
        POOL_AVAILABLE = False
        return

    PoolPod = _PoolPod
    get_or_create_pod = _get_or_create_pod
    load_pool_state = _load_pool_state
    refresh_pod_status = _refresh_pod_status
    save_pool_state = _save_pool_state
    scp_from = _scp_from
    ssh_exec = _ssh_exec
    _print_cost_estimate = __print_cost_estimate
    _require_client = __require_client
    _resolve_gpu = __resolve_gpu
    POOL_AVAILABLE = True


def bootstrap_pod(pod: PoolPod, remote_dir: str = "/workspace/cutedsl") -> None:
    print(f"  bootstrapping {pod.name}...")
    rsync_cmd = [
        "rsync",
        "-az",
        "--delete",
        "--exclude", "__pycache__/",
        "--exclude", ".git/",
        "--exclude", ".pytest_cache/",
        "--exclude", ".venv/",
        "--exclude", "examples/lora_frontier_runs/",
        "-e", f"ssh -o StrictHostKeyChecking=no -p {pod.ssh_port}",
        f"{REPO_ROOT}/",
        f"root@{pod.ssh_host}:{remote_dir}/",
    ]
    result = subprocess.run(rsync_cmd, capture_output=True, text=True, timeout=600)
    if result.returncode not in (0, 23):
        raise RuntimeError(f"rsync failed: {result.stderr.strip()}")

    setup_script = f"""
set -euo pipefail
cd {shlex.quote(remote_dir)}
python -m pip install --upgrade pip >/dev/null 2>&1 || true
python -m pip install uv >/dev/null 2>&1 || true
if command -v uv >/dev/null 2>&1; then
  UV_CACHE_DIR=/tmp/uv-cache uv sync --extra zimage --extra benchmark --extra dev >/tmp/cutedsl_uv.log 2>&1 || (cat /tmp/cutedsl_uv.log && exit 1)
else
  python -m pip install -e ".[zimage,benchmark,dev]" >/tmp/cutedsl_pip.log 2>&1 || (cat /tmp/cutedsl_pip.log && exit 1)
fi
python - <<'PY'
import torch
print("BOOTSTRAP_OK")
print(torch.__version__)
print(torch.cuda.is_available())
PY
"""
    result = ssh_exec(pod, setup_script)
    if "BOOTSTRAP_OK" not in result.stdout:
        raise RuntimeError(f"bootstrap failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")
    print(f"  bootstrap complete on {pod.name}")


def run_remote_command(
    pod: PoolPod,
    *,
    command: str,
    remote_dir: str = "/workspace/cutedsl",
    time_budget: int = 1800,
    output_name: str = "experiment_result.json",
) -> dict:
    wrapped = (
        f"cd {shlex.quote(remote_dir)} && "
        f"timeout {time_budget} bash -lc {shlex.quote(command)} "
        f"|| true"
    )
    print(f"  [{pod.name}] running: {command}")
    result = ssh_exec(pod, wrapped)

    local_out_dir = REPO_ROOT / "zimageaccelerated" / "remote_results"
    local_out_dir.mkdir(parents=True, exist_ok=True)
    local_output = local_out_dir / output_name
    try:
        scp_from(pod, f"{remote_dir}/{output_name}", str(local_output))
    except Exception:
        pass

    return {
        "pod": pod.name,
        "command": command,
        "status": "completed" if result.returncode == 0 else "timeout_or_error",
        "stdout_tail": result.stdout[-4000:] if result.stdout else "",
        "stderr_tail": result.stderr[-4000:] if result.stderr else "",
        "local_output": str(local_output),
    }


def build_block_benchmark_command(args: argparse.Namespace) -> tuple[str, str]:
    output_name = args.output_name or "benchmark_block_result.json"
    command = (
        "python -m zimageaccelerated.benchmark "
        f"--device cuda --dtype {shlex.quote(args.dtype)} "
        f"--batch-size {args.batch_size} "
        f"--seq-len {args.seq_len} "
        f"--dim {args.dim} "
        f"--n-heads {args.n_heads} "
        f"--n-kv-heads {args.n_kv_heads} "
        f"--warmup {args.warmup} "
        f"--runs {args.runs} "
        + ("--modulation " if args.modulation else "")
        + f"> {shlex.quote(output_name)}"
    )
    return command, output_name


def build_transformer_benchmark_command(args: argparse.Namespace) -> tuple[str, str]:
    output_name = args.output_name or "benchmark_transformer_result.json"
    command = (
        "python -m zimageaccelerated.benchmark_transformer "
        f"--device cuda --dtype {shlex.quote(args.dtype)} "
        f"--batch-size {args.batch_size} "
        f"--height {args.height} "
        f"--width {args.width} "
        f"--warmup {args.warmup} "
        f"--runs {args.runs} "
        f"--dim {args.dim} "
        f"--n-layers {args.n_layers} "
        f"--n-refiner-layers {args.n_refiner_layers} "
        f"--n-heads {args.n_heads} "
        f"--n-kv-heads {args.n_kv_heads} "
        f"--in-channels {args.in_channels} "
        f"--cap-feat-dim {args.cap_feat_dim} "
        f"> {shlex.quote(output_name)}"
    )
    return command, output_name


def build_generate_dataset_command(args: argparse.Namespace) -> tuple[str, str]:
    if not args.model_id:
        raise ValueError("--model-id is required for generate-dataset")
    output_name = args.output_name or "generate_dataset_summary.json"
    output_dir = args.dataset_output_dir or "zimageaccelerated/remote_results/generated_dataset"
    command = (
        "python -m zimageaccelerated.generate_dataset "
        f"--model-id {shlex.quote(args.model_id)} "
        f"--transformer {shlex.quote(args.transformer)} "
        f"--device cuda "
        f"--dtype {shlex.quote(args.dtype)} "
        f"--width {args.width} "
        f"--height {args.height} "
        f"--steps {args.steps} "
        f"--guidance-scale {args.guidance_scale} "
        f"--seed-start {args.seed_start} "
        f"--num-seeds {args.num_seeds} "
        f"--num-prompts {args.num_prompts} "
        f"--output-dir {shlex.quote(output_dir)} "
        + (f"--prompts-file {shlex.quote(args.prompts_file)} " if args.prompts_file else "")
        + ("--cpu-offload " if args.cpu_offload else "")
        + f"> {shlex.quote(output_name)}"
    )
    return command, output_name


def cmd_run(args: argparse.Namespace) -> None:
    _load_pool_support()
    if not POOL_AVAILABLE:
        print("ERROR: remote GPU pool support unavailable")
        if _POOL_IMPORT_ERROR:
            print(_POOL_IMPORT_ERROR)
        sys.exit(1)

    client = _require_client()
    state = load_pool_state()
    refresh_pod_status(state, client)

    gpu_type = _resolve_gpu(args.gpu)
    _print_cost_estimate(gpu_type, args.time_budget)

    pod = get_or_create_pod(state, client, gpu_type)
    if not pod.ssh_host:
        client.wait_for_pod(pod.pod_id, timeout=600)

    bootstrap_pod(pod, remote_dir=args.remote_dir)

    pod.status = "busy"
    pod.current_experiment = "zimageaccelerated"
    save_pool_state(state)

    try:
        if args.experiment == "benchmark-block":
            command, output_name = build_block_benchmark_command(args)
        elif args.experiment == "benchmark-transformer":
            command, output_name = build_transformer_benchmark_command(args)
        elif args.experiment == "generate-dataset":
            command, output_name = build_generate_dataset_command(args)
        else:
            output_name = args.output_name or "custom_result.json"
            command = args.command

        metrics = run_remote_command(
            pod,
            command=command,
            remote_dir=args.remote_dir,
            time_budget=args.time_budget,
            output_name=output_name,
        )
    finally:
        pod.status = "ready"
        pod.current_experiment = ""
        save_pool_state(state)

    if args.stop_after:
        client.stop_pod(pod.pod_id)
        pod.status = "stopped"
        save_pool_state(state)

    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Remote GPU experiments for zimageaccelerated")
    sub = parser.add_subparsers(dest="command_name", required=True)

    run_p = sub.add_parser("run")
    run_p.add_argument(
        "--experiment",
        choices=["benchmark-block", "benchmark-transformer", "generate-dataset", "custom"],
        default="benchmark-block",
    )
    run_p.add_argument("--gpu", default="4090")
    run_p.add_argument("--remote-dir", default="/workspace/cutedsl")
    run_p.add_argument("--time-budget", type=int, default=1800)
    run_p.add_argument("--stop-after", action="store_true")
    run_p.add_argument("--output-name", default="")

    run_p.add_argument("--dtype", default="bfloat16")
    run_p.add_argument("--batch-size", type=int, default=1)
    run_p.add_argument("--seq-len", type=int, default=2048)
    run_p.add_argument("--dim", type=int, default=3840)
    run_p.add_argument("--n-heads", type=int, default=30)
    run_p.add_argument("--n-kv-heads", type=int, default=30)
    run_p.add_argument("--warmup", type=int, default=10)
    run_p.add_argument("--runs", type=int, default=50)
    run_p.add_argument("--modulation", action="store_true")

    run_p.add_argument("--height", type=int, default=32)
    run_p.add_argument("--width", type=int, default=32)
    run_p.add_argument("--n-layers", type=int, default=2)
    run_p.add_argument("--n-refiner-layers", type=int, default=1)
    run_p.add_argument("--in-channels", type=int, default=4)
    run_p.add_argument("--cap-feat-dim", type=int, default=128)

    run_p.add_argument("--command", default="python -m zimageaccelerated.benchmark --device cuda > custom_result.json")
    run_p.add_argument("--model-id", default="")
    run_p.add_argument("--transformer", choices=["diffusers", "cute", "accelerated"], default="accelerated")
    run_p.add_argument("--steps", type=int, default=20)
    run_p.add_argument("--guidance-scale", type=float, default=0.0)
    run_p.add_argument("--seed-start", type=int, default=0)
    run_p.add_argument("--num-seeds", type=int, default=1)
    run_p.add_argument("--num-prompts", type=int, default=20)
    run_p.add_argument("--prompts-file", default="")
    run_p.add_argument("--dataset-output-dir", default="")
    run_p.add_argument("--cpu-offload", action="store_true")

    args = parser.parse_args()
    if args.command_name == "run":
        cmd_run(args)


if __name__ == "__main__":
    main()
