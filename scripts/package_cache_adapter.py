#!/usr/bin/env python3
"""Assemble a self-contained Hugging Face artifact for the cache adapter."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def run(
    trained_dir: Path,
    evaluated_dir: Path,
    visual_summary: Path,
    output_dir: Path,
    summary_output: Path | None = None,
    figure: Path | None = None,
) -> dict:
    trained = json.loads((trained_dir / "metrics.json").read_text())
    evaluated = json.loads((evaluated_dir / "metrics.json").read_text())
    visual = json.loads(visual_summary.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename in (
        "cache_adapter.safetensors",
        "config.json",
        "schedule_coefficients.json",
    ):
        shutil.copy2(evaluated_dir / filename, output_dir / filename)
    if figure is not None:
        shutil.copy2(figure, output_dir / "cache_procedure_grid.png")

    metrics = {
        "protocol": evaluated["protocol"],
        "model": evaluated["model"],
        "training": trained["training"],
        "inference_timing": evaluated["inference_timing"],
        "test": evaluated["test"],
        "decoded_endpoint": {
            "protocol": visual["protocol"],
            "summary": visual["summary"],
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    test = metrics["test"]["summary"]
    image = metrics["decoded_endpoint"]["summary"]
    timing = metrics["inference_timing"]
    training = metrics["training"]
    content = f"""---
library_name: pytorch
license: apache-2.0
base_model: Tongyi-MAI/Z-Image-Turbo
tags:
- diffusion
- image-generation
- acceleration
- latent-teleportation
---

# Z-Image Latent Teleportation Cache Adapter

Experimental **7,312-parameter** residual head for cache-guided Z-Image Turbo
latent teleportation. It starts from the fitted scalar local-plus-cache gate and
learns a spatial correction from:

1. the scalar-gated base movement;
2. the weighted mean of eight motion-pruned cached residuals; and
3. per-pixel/channel disagreement across those neighbours.

The output projection is zero-initialized, so an untrained head exactly matches
the scalar cache gate.

Implementation, training scripts, and the paper are in
[lee101/cutedsl](https://github.com/lee101/cutedsl).

## Held-out results

The checkpoint was trained on 160 trajectories and evaluated on a disjoint
40-trajectory fold across valid scheduler cells and horizons 1, 2, 4, and 8.

| Method | relL2 |
|---|---:|
| Calibrated local momentum | {test['local']:.4f} |
| Scalar pruned-cache gate | {test['scalar_gate']:.4f} |
| **Learned cache adapter** | **{test['adapter']:.4f}** |

That is a {test['adapter_improvement_vs_local_pct']:.2f}% reduction versus local
momentum and {test['adapter_improvement_vs_scalar_gate_pct']:.2f}% beyond the
scalar cache gate.

For six held-out step-6 to step-14 endpoint decodes:

| Method | PSNR | SSIM |
|---|---:|---:|
| Local momentum | {image['local']['psnr_db']:.2f} dB | {image['local']['ssim']:.4f} |
| Scalar pruned-cache gate | {image['retrieval']['psnr_db']:.2f} dB | {image['retrieval']['ssim']:.4f} |
| **Learned cache adapter** | **{image['adapter']['psnr_db']:.2f} dB** | {image['adapter']['ssim']:.4f} |

![Held-out cache procedure grid](cache_procedure_grid.png)

The 7,312-parameter head takes {timing['median_ms']:.2f} ms per 64x64 latent on
the recorded one-thread resident CPU benchmark. Training took
{training['elapsed_seconds']:.1f} seconds on CPU. An RTX 5090 run was attempted,
but the safety gate deferred it because production workloads left less than
2.5 GiB allocatable VRAM; the checkpoint therefore does not claim a GPU training
time.

## Files

- `cache_adapter.safetensors`: adapter parameters.
- `config.json`: architecture and retrieval protocol.
- `schedule_coefficients.json`: momentum and scalar-gate coefficients by step
  and horizon.
- `metrics.json`: training history, horizon breakdown, timing, and decoded-image
  summary.

## Scope

This is an adapter for the CuteDSL latent-teleportation implementation, not a
standalone image model. It is fitted specifically to Z-Image Turbo at 512px with
the recorded 16-step schedule. A new model, solver, resolution, or schedule
requires new trajectories and refitting. The decoded comparison is an offline
endpoint forecast; a fully branched sampler benchmark remains separate.
"""
    (output_dir / "README.md").write_text(content)
    if summary_output is not None:
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary_output.write_text(
            json.dumps(
                {
                    "huggingface_repo": "lee101/zimage-latent-teleport-cache-adapter",
                    **metrics,
                },
                indent=2,
            )
            + "\n"
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Package cache-adapter weights")
    parser.add_argument("--trained-dir", type=Path, required=True)
    parser.add_argument("--evaluated-dir", type=Path, required=True)
    parser.add_argument("--visual-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    run(
        args.trained_dir,
        args.evaluated_dir,
        args.visual_summary,
        args.output_dir,
        args.summary_output,
        args.figure,
    )
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
