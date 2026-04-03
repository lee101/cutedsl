"""Benchmark a pure stable-diffusion.cpp Z-Image path.

This provides a first-class experiment harness for the low-VRAM GGUF backend so
it can be compared against the Python and Triton-based Z-Image paths.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path


def summarize_latencies(latencies_ms: list[float]) -> dict[str, float]:
    if not latencies_ms:
        raise ValueError("latencies_ms must not be empty")
    return {
        "avg_latency_ms": statistics.fmean(latencies_ms),
        "std_latency_ms": statistics.pstdev(latencies_ms) if len(latencies_ms) > 1 else 0.0,
        "min_latency_ms": min(latencies_ms),
        "max_latency_ms": max(latencies_ms),
    }


def build_sdcpp_command(
    *,
    sdcpp_bin: Path,
    diffusion_model: Path,
    vae: Path,
    llm: Path,
    prompt: str,
    output_path: Path,
    width: int,
    height: int,
    steps: int,
    cfg_scale: float,
    seed: int,
    sampling_method: str,
    scheduler: str | None,
    rng: str | None,
    offload_to_cpu: bool,
    diffusion_fa: bool,
    clip_on_cpu: bool,
    vae_on_cpu: bool,
    vae_tiling: bool,
    cache_mode: str | None,
    cache_option: str | None,
    cache_preset: str | None,
    verbose: bool,
    extra_args: list[str],
) -> list[str]:
    command = [
        str(sdcpp_bin),
        "--diffusion-model",
        str(diffusion_model),
        "--vae",
        str(vae),
        "--llm",
        str(llm),
        "-p",
        prompt,
        "-o",
        str(output_path),
        "-H",
        str(height),
        "-W",
        str(width),
        "--steps",
        str(steps),
        "--cfg-scale",
        str(cfg_scale),
        "-s",
        str(seed),
        "--sampling-method",
        sampling_method,
    ]
    if scheduler:
        command.extend(["--scheduler", scheduler])
    if rng:
        command.extend(["--rng", rng])
    if offload_to_cpu:
        command.append("--offload-to-cpu")
    if diffusion_fa:
        command.append("--diffusion-fa")
    if clip_on_cpu:
        command.append("--clip-on-cpu")
    if vae_on_cpu:
        command.append("--vae-on-cpu")
    if vae_tiling:
        command.append("--vae-tiling")
    if cache_mode:
        command.extend(["--cache-mode", cache_mode])
    if cache_option:
        command.extend(["--cache-option", cache_option])
    if cache_preset:
        command.extend(["--cache-preset", cache_preset])
    if verbose:
        command.append("-v")
    command.extend(extra_args)
    return command


def run_command(command: list[str], *, cwd: Path, log_path: Path) -> tuple[float, subprocess.CompletedProcess[str]]:
    start = time.perf_counter()
    proc = subprocess.run(command, cwd=str(cwd), text=True, capture_output=True, check=False)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    log_path.write_text(
        f"$ {' '.join(command)}\n\n"
        f"[exit_code]={proc.returncode}\n"
        f"[elapsed_ms]={elapsed_ms:.3f}\n\n"
        f"[stdout]\n{proc.stdout}\n\n"
        f"[stderr]\n{proc.stderr}\n"
    )
    return elapsed_ms, proc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the stable-diffusion.cpp Z-Image backend")
    parser.add_argument("--sdcpp-bin", default="external/stable-diffusion.cpp/build/bin/sd-cli")
    parser.add_argument("--diffusion-model", default="downloads/models/zimage/z_image_turbo-Q3_K.gguf")
    parser.add_argument("--llm", default="downloads/models/zimage/Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
    parser.add_argument("--vae", default="downloads/models/zimage/ae.safetensors")
    parser.add_argument(
        "--prompt",
        default=(
            "A cinematic, melancholic photograph of a solitary hooded figure walking "
            "through a sprawling, rain-slicked metropolis at night. The city lights are "
            "a chaotic blur of neon orange and cool blue, reflecting on the wet asphalt. "
            "Superimposed text: THE CITY IS A CIRCUIT BOARD, AND I AM A BROKEN TRANSISTOR."
        ),
    )
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling-method", default="euler")
    parser.add_argument("--scheduler", default=None)
    parser.add_argument("--rng", default=None)
    parser.add_argument("--n-warmup", type=int, default=1)
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--output-dir", default="benchmark_images/sdcpp")
    parser.add_argument("--output-name", default="sdcpp.png")
    parser.add_argument("--output", default="benchmark_results/sdcpp_benchmark.json")
    parser.add_argument("--log-name", default="sdcpp.log")
    parser.add_argument("--offload-to-cpu", action="store_true")
    parser.add_argument("--diffusion-fa", action="store_true")
    parser.add_argument("--clip-on-cpu", action="store_true")
    parser.add_argument("--vae-on-cpu", action="store_true")
    parser.add_argument("--vae-tiling", action="store_true")
    parser.add_argument("--cache-mode", default=None)
    parser.add_argument("--cache-option", default=None)
    parser.add_argument("--cache-preset", default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra stable-diffusion.cpp CLI argument. Repeat the flag to append multiple raw args.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved sd-cli command and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent

    sdcpp_bin = (repo_root / args.sdcpp_bin).resolve()
    diffusion_model = (repo_root / args.diffusion_model).resolve()
    llm = (repo_root / args.llm).resolve()
    vae = (repo_root / args.vae).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name
    log_path = output_dir / args.log_name

    command = build_sdcpp_command(
        sdcpp_bin=sdcpp_bin,
        diffusion_model=diffusion_model,
        vae=vae,
        llm=llm,
        prompt=args.prompt,
        output_path=output_path,
        width=args.width,
        height=args.height,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
        sampling_method=args.sampling_method,
        scheduler=args.scheduler,
        rng=args.rng,
        offload_to_cpu=args.offload_to_cpu,
        diffusion_fa=args.diffusion_fa,
        clip_on_cpu=args.clip_on_cpu,
        vae_on_cpu=args.vae_on_cpu,
        vae_tiling=args.vae_tiling,
        cache_mode=args.cache_mode,
        cache_option=args.cache_option,
        cache_preset=args.cache_preset,
        verbose=args.verbose,
        extra_args=args.extra_arg,
    )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "backend": "stable-diffusion.cpp",
                    "cwd": str(repo_root),
                    "command": command,
                    "output_path": str(output_path),
                    "log_path": str(log_path),
                },
                indent=2,
            )
        )
        return

    missing = [path for path in [sdcpp_bin, diffusion_model, llm, vae] if not path.exists()]
    if missing:
        missing_text = "\n".join(f"- {path}" for path in missing)
        raise SystemExit(
            "Missing required stable-diffusion.cpp binary or model assets:\n"
            f"{missing_text}\n"
            "Build the binary with scripts/setup_external_zimage.sh --build-sdcpp and "
            "download the GGUF weights with scripts/download_zimage_lowvram_weights.py."
        )

    for _ in range(args.n_warmup):
        _, proc = run_command(command, cwd=repo_root, log_path=log_path)
        if proc.returncode != 0:
            raise SystemExit(f"stable-diffusion.cpp warmup failed; see {log_path}")

    latencies_ms: list[float] = []
    for _ in range(args.n_runs):
        elapsed_ms, proc = run_command(command, cwd=repo_root, log_path=log_path)
        if proc.returncode != 0:
            raise SystemExit(f"stable-diffusion.cpp benchmark run failed; see {log_path}")
        latencies_ms.append(elapsed_ms)

    if not output_path.exists():
        raise SystemExit(f"stable-diffusion.cpp completed but did not write {output_path}")

    result = {
        "backend": "stable-diffusion.cpp",
        "command": command,
        "cwd": str(repo_root),
        "sdcpp_bin": str(sdcpp_bin),
        "diffusion_model": str(diffusion_model),
        "llm": str(llm),
        "vae": str(vae),
        "prompt": args.prompt,
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "cfg_scale": args.cfg_scale,
        "seed": args.seed,
        "sampling_method": args.sampling_method,
        "scheduler": args.scheduler,
        "rng": args.rng,
        "offload_to_cpu": args.offload_to_cpu,
        "diffusion_fa": args.diffusion_fa,
        "clip_on_cpu": args.clip_on_cpu,
        "vae_on_cpu": args.vae_on_cpu,
        "vae_tiling": args.vae_tiling,
        "cache_mode": args.cache_mode,
        "cache_option": args.cache_option,
        "cache_preset": args.cache_preset,
        "n_warmup": args.n_warmup,
        "n_runs": args.n_runs,
        "output_image": str(output_path),
        "output_image_size_bytes": output_path.stat().st_size,
        "log_path": str(log_path),
        "extra_args": args.extra_arg,
    }
    result.update(summarize_latencies(latencies_ms))

    output_json = (repo_root / args.output).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
