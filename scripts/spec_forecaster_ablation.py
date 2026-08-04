#!/usr/bin/env python3
"""Cross-validated latent-trajectory forecaster ablations.

This analysis is CPU-only and uses the 200 recorded Z-Image trajectories.  It
tests whether cheap schedule calibration improves the fixed Taylor baseline,
without generating new images or calling the denoiser.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

METHODS = (
    "identity",
    "taylor1",
    "average_velocity",
    "taylor2",
    "scaled_momentum",
    "two_delta_fit",
)


def _dot(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left * right).sum(dtype=torch.float64)


def _rel_l2(prediction: torch.Tensor, target: torch.Tensor, anchor: torch.Tensor) -> float:
    movement = (target - anchor).flatten(1).norm(dim=1).clamp_min(1e-8)
    error = (prediction - target).flatten(1).norm(dim=1)
    return float((error / movement).mean())


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": round(statistics.mean(values), 5),
        "std": round(statistics.stdev(values), 5) if len(values) > 1 else 0.0,
    }


def _fit_coefficients(trajectories: torch.Tensor, step: int, draft_k: int) -> tuple[float, float, float]:
    anchor = trajectories[:, step]
    target_move = trajectories[:, step + draft_k] - anchor
    delta = anchor - trajectories[:, step - 1]
    previous_delta = trajectories[:, step - 1] - trajectories[:, step - 2]

    velocity = draft_k * delta
    alpha = _dot(velocity, target_move) / _dot(velocity, velocity).clamp_min(1e-12)

    aa = _dot(delta, delta)
    ab = _dot(delta, previous_delta)
    bb = _dot(previous_delta, previous_delta)
    ay = _dot(delta, target_move)
    by = _dot(previous_delta, target_move)
    determinant = (aa * bb - ab * ab).clamp_min(1e-12)
    first_weight = (ay * bb - by * ab) / determinant
    previous_weight = (by * aa - ay * ab) / determinant
    return float(alpha), float(first_weight), float(previous_weight)


def run(trajectory_dir: Path, folds: int, max_k: int) -> dict:
    files = sorted(trajectory_dir.glob("*.pt"))
    if len(files) < folds:
        raise ValueError(f"need at least {folds} trajectories, found {len(files)}")

    loaded = [
        torch.load(path, map_location="cpu", weights_only=False)["latents"]
        for path in files
    ]
    trajectories = torch.stack(loaded).float()
    del loaded

    n_trajectories, n_steps = trajectories.shape[:2]
    fold_rows: list[dict] = []
    cells: list[dict] = []
    coefficients: list[dict] = []
    excluded: list[dict] = []

    all_indices = torch.arange(n_trajectories)
    for fold in range(folds):
        test_mask = all_indices.remainder(folds) == fold
        train = trajectories[~test_mask]
        test = trajectories[test_mask]

        for draft_k in range(1, max_k + 1):
            by_method = {method: [] for method in METHODS}
            for step in range(2, n_steps - draft_k):
                anchor = test[:, step]
                target = test[:, step + draft_k]
                delta = anchor - test[:, step - 1]
                previous_delta = test[:, step - 1] - test[:, step - 2]
                movement = (target - anchor).flatten(1).norm(dim=1)
                if float(movement.mean()) < 1e-4:
                    excluded.append({"fold": fold, "step": step, "draft_k": draft_k})
                    continue

                alpha, first_weight, previous_weight = _fit_coefficients(
                    train, step, draft_k
                )

                second_difference = delta - previous_delta
                curvature_factor = draft_k * (draft_k + 1) / 2
                predictions = {
                    "identity": anchor,
                    "taylor1": anchor + draft_k * delta,
                    "average_velocity": anchor + draft_k * 0.5 * (delta + previous_delta),
                    "taylor2": anchor + draft_k * delta + curvature_factor * second_difference,
                    "scaled_momentum": anchor + alpha * draft_k * delta,
                    "two_delta_fit": (
                        anchor
                        + first_weight * delta
                        + previous_weight * previous_delta
                    ),
                }

                scores = {
                    method: _rel_l2(prediction, target, anchor)
                    for method, prediction in predictions.items()
                }
                for method, score in scores.items():
                    by_method[method].append(score)
                cells.append(
                    {
                        "fold": fold,
                        "step": step,
                        "draft_k": draft_k,
                        "scores": {method: round(score, 6) for method, score in scores.items()},
                    }
                )
                coefficients.append(
                    {
                        "fold": fold,
                        "step": step,
                        "draft_k": draft_k,
                        "momentum_scale": round(alpha, 6),
                        "first_delta_weight": round(first_weight, 6),
                        "previous_delta_weight": round(previous_weight, 6),
                    }
                )

            fold_rows.append(
                {
                    "fold": fold,
                    "draft_k": draft_k,
                    "n_cells": len(by_method["identity"]),
                    "methods": {
                        method: round(statistics.mean(values), 6)
                        for method, values in by_method.items()
                    },
                }
            )

    summary = {}
    for draft_k in range(1, max_k + 1):
        rows = [row for row in fold_rows if row["draft_k"] == draft_k]
        methods = {
            method: _mean_std([row["methods"][method] for row in rows])
            for method in METHODS
        }
        baseline = methods["taylor1"]["mean"]
        for method in ("scaled_momentum", "two_delta_fit"):
            methods[method]["improvement_vs_taylor_pct"] = round(
                100.0 * (1.0 - methods[method]["mean"] / baseline), 2
            )
        summary[str(draft_k)] = {
            "folds": folds,
            "methods": methods,
        }

    deployment_coefficients = []
    for draft_k in range(1, max_k + 1):
        for step in range(2, n_steps - draft_k):
            movement = (
                trajectories[:, step + draft_k] - trajectories[:, step]
            ).flatten(1).norm(dim=1)
            if float(movement.mean()) < 1e-4:
                continue
            alpha, first_weight, previous_weight = _fit_coefficients(
                trajectories, step, draft_k
            )
            deployment_coefficients.append(
                {
                    "step": step,
                    "draft_k": draft_k,
                    "momentum_scale": round(alpha, 6),
                    "first_delta_weight": round(first_weight, 6),
                    "previous_delta_weight": round(previous_weight, 6),
                }
            )

    return {
        "protocol": {
            "trajectory_dir": str(trajectory_dir),
            "n_trajectories": n_trajectories,
            "n_steps": n_steps,
            "folds": folds,
            "max_draft_k": max_k,
            "fold_assignment": "sorted trajectory index modulo fold count",
            "fit_scope": "one or two scalar coefficients per scheduler step and draft_k, trained on four folds",
            "metric": "mean relL2 per (step, draft_k) cell; cells weighted equally",
        },
        "summary": summary,
        "fold_rows": fold_rows,
        "cells": cells,
        "coefficients": coefficients,
        "deployment_coefficients": deployment_coefficients,
        "excluded_degenerate_cells": excluded,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-validated latent forecaster ablation")
    parser.add_argument(
        "--trajectory-dir",
        type=Path,
        default=Path("/sdb-disk/latentteleport-spec/trajs-16step-512"),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-k", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/speculative/forecaster-ablation.json"),
    )
    args = parser.parse_args()

    result = run(args.trajectory_dir, args.folds, args.max_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.output}")
    for draft_k, row in result["summary"].items():
        methods = row["methods"]
        print(
            f"k={draft_k}: Taylor {methods['taylor1']['mean']:.3f}; "
            f"scaled {methods['scaled_momentum']['mean']:.3f} "
            f"({methods['scaled_momentum']['improvement_vs_taylor_pct']:.1f}% better); "
            f"two-delta {methods['two_delta_fit']['mean']:.3f} "
            f"({methods['two_delta_fit']['improvement_vs_taylor_pct']:.1f}% better)"
        )


if __name__ == "__main__":
    main()
