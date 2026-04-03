"""Create a reviewable Z-Image baseline matrix with numbered result folders.

This is intentionally lightweight: it records exact commands, writes manifests,
and can run cheap validation tasks. Heavy image generation is left opt-in so
weights can be staged separately.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


DEFAULT_PROMPT = (
    "A cinematic, melancholic photograph of a solitary hooded figure walking "
    "through a sprawling, rain-slicked metropolis at night. The city lights "
    "are a chaotic blur of neon orange and cool blue, reflecting on the wet "
    "asphalt. Superimposed text: THE CITY IS A CIRCUIT BOARD, AND I AM A "
    "BROKEN TRANSISTOR."
)


def run_capture(cmd: list[str], cwd: Path, log_path: Path) -> int:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )
    log_path.write_text(
        f"$ {' '.join(cmd)}\n\n[exit_code]={proc.returncode}\n\n"
        f"[stdout]\n{proc.stdout}\n\n[stderr]\n{proc.stderr}\n"
    )
    return proc.returncode


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def make_case(dir_path: Path, name: str, description: str, commands: list[list[str]]) -> dict:
    dir_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name,
        "description": description,
        "commands": commands,
    }
    write_json(dir_path / "manifest.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Z-Image review matrix")
    parser.add_argument("--results-dir", default="artifacts/zimage_review")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--model-id", default="Tongyi-MAI/Z-Image-Turbo")
    parser.add_argument("--num-inference-steps", type=int, default=9)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--run-smoke", action="store_true", help="Run cheap validation commands and save logs")
    parser.add_argument("--sdcpp-bin", default="external/stable-diffusion.cpp/build/bin/sd-cli")
    parser.add_argument("--zimage-gguf", default="downloads/models/zimage/z_image_turbo-Q3_K.gguf")
    parser.add_argument("--qwen-gguf", default="downloads/models/zimage/Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
    parser.add_argument("--vae", default="downloads/models/zimage/ae.safetensors")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    results_dir = (root / args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    baseline_dir = results_dir / "01_baseline"
    lemica_dir = results_dir / "02_lemica"
    meancache_dir = results_dir / "03_meancache"
    sdcpp_dir = results_dir / "04_sdcpp"

    manifests = []
    manifests.append(
        make_case(
            baseline_dir,
            "baseline",
            "Vanilla diffusers / CuteDSL Z-Image runs used as the visual reference.",
            [
                [
                    "python",
                    "-m",
                    "cutezimage.benchmark",
                    "--pipeline",
                    "--model-id",
                    args.model_id,
                    "--prompt",
                    args.prompt,
                    "--num-inference-steps",
                    str(args.num_inference_steps),
                    "--width",
                    str(args.width),
                    "--height",
                    str(args.height),
                ]
            ],
        )
    )
    manifests.append(
        make_case(
            lemica_dir,
            "lemica",
            "LeMiCa Z-Image cache schedules. 1/2/3 map to slow/medium/fast review buckets.",
            [
                ["python", "external/LeMiCa/LeMiCa4Z-Image/inference_zimage.py"],
                ["python", "external/LeMiCa/LeMiCa4Z-Image/inference_zimage.py", "--cache", "slow"],
                ["python", "external/LeMiCa/LeMiCa4Z-Image/inference_zimage.py", "--cache", "medium"],
                ["python", "external/LeMiCa/LeMiCa4Z-Image/inference_zimage.py", "--cache", "fast"],
            ],
        )
    )
    manifests.append(
        make_case(
            meancache_dir,
            "meancache",
            "MeanCache Z-Image cache-jvp sweep.",
            [
                ["python", "external/MeanCache/MeanCache4Z-Image/MC_zimage.py"],
                ["python", "external/MeanCache/MeanCache4Z-Image/MC_zimage.py", "--cache-jvp", "25"],
                ["python", "external/MeanCache/MeanCache4Z-Image/MC_zimage.py", "--cache-jvp", "20"],
                ["python", "external/MeanCache/MeanCache4Z-Image/MC_zimage.py", "--cache-jvp", "15"],
                ["python", "external/MeanCache/MeanCache4Z-Image/MC_zimage.py", "--cache-jvp", "13"],
            ],
        )
    )
    manifests.append(
        make_case(
            sdcpp_dir,
            "stable-diffusion.cpp",
            "Low-VRAM GGUF C/C++ inference recipes for Z-Image via the benchmark wrapper.",
            [
                [
                    "python",
                    "-m",
                    "zimageaccelerated.sdcpp_benchmark",
                    "--sdcpp-bin",
                    args.sdcpp_bin,
                    "--diffusion-model",
                    args.zimage_gguf,
                    "--vae",
                    args.vae,
                    "--llm",
                    args.qwen_gguf,
                    "--prompt",
                    args.prompt,
                    "--cfg-scale",
                    "1.0",
                    "--offload-to-cpu",
                    "--diffusion-fa",
                    "--height",
                    str(args.height),
                    "--width",
                    str(args.width),
                ],
                [
                    "python",
                    "-m",
                    "zimageaccelerated.sdcpp_benchmark",
                    "--sdcpp-bin",
                    args.sdcpp_bin,
                    "--diffusion-model",
                    args.zimage_gguf,
                    "--vae",
                    args.vae,
                    "--llm",
                    args.qwen_gguf,
                    "--prompt",
                    args.prompt,
                    "--cfg-scale",
                    "1.0",
                    "--offload-to-cpu",
                    "--diffusion-fa",
                    "--vae-tiling",
                    "--clip-on-cpu",
                    "--height",
                    str(args.height),
                    "--width",
                    str(args.width),
                ],
            ],
        )
    )

    review_index = {
        "prompt": args.prompt,
        "model_id": args.model_id,
        "num_inference_steps": args.num_inference_steps,
        "width": args.width,
        "height": args.height,
        "groups": manifests,
        "notes": {
            "1": "baseline reference bucket",
            "2": "LeMiCa cache schedule bucket",
            "3": "MeanCache average-velocity bucket",
            "4": "stable-diffusion.cpp low-VRAM C/C++ bucket",
        },
    }
    write_json(results_dir / "index.json", review_index)

    if args.run_smoke:
        smoke_jobs = [
            (baseline_dir / "smoke_help.log", ["python", "-m", "cutezimage.benchmark", "--help"]),
            (lemica_dir / "smoke_py_compile.log", ["python", "-m", "py_compile", "external/LeMiCa/LeMiCa4Z-Image/inference_zimage.py"]),
            (meancache_dir / "smoke_py_compile.log", ["python", "-m", "py_compile", "external/MeanCache/MeanCache4Z-Image/MC_zimage.py"]),
        ]
        sdcpp_bin = (root / args.sdcpp_bin).resolve()
        if sdcpp_bin.exists():
            smoke_jobs.append((sdcpp_dir / "smoke_help.log", [str(sdcpp_bin), "--help"]))
        for log_path, cmd in smoke_jobs:
            run_capture(cmd, root, log_path)

    print(results_dir)


if __name__ == "__main__":
    main()
