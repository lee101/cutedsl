#!/usr/bin/env python3
"""Held-out architectures for retrieval-conditioned latent teleportation.

The experiment uses prompt embeddings only for initial retrieval, then optionally
prunes that candidate set by agreement with the query's observed latent motion.
All neighbours come from the training fold.  The measured online-cost proxy
assumes embeddings, compact motion descriptors, and latent deltas are resident.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def _dot(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left * right).sum(dtype=torch.float64)


def _normalize(rows: torch.Tensor) -> torch.Tensor:
    return F.normalize(rows.float(), dim=1, eps=1e-8)


def _motion_descriptor(delta: torch.Tensor, size: int = 8) -> torch.Tensor:
    pooled = F.adaptive_avg_pool2d(delta.float(), (size, size))
    return _normalize(pooled.flatten(1))


def _rel_l2_rows(
    prediction_move: torch.Tensor,
    target_move: torch.Tensor,
) -> torch.Tensor:
    error = (prediction_move - target_move).norm(dim=1)
    movement = target_move.norm(dim=1).clamp_min(1e-8)
    return error / movement


def _neighbor_plan(
    query_embeddings: torch.Tensor,
    query_motion: torch.Tensor,
    bank_embeddings: torch.Tensor,
    bank_motion: torch.Tensor,
    *,
    top_k: int,
    pool_k: int,
    temperature: float,
    motion_weight: float,
    exclude_diagonal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return pruned neighbour indices and normalized attention weights."""
    text_similarity = query_embeddings @ bank_embeddings.T
    if exclude_diagonal:
        if text_similarity.shape[0] != text_similarity.shape[1]:
            raise ValueError("diagonal exclusion requires a square query/bank set")
        text_similarity.fill_diagonal_(-torch.inf)

    candidate_count = min(pool_k, bank_embeddings.shape[0] - int(exclude_diagonal))
    candidate_similarity, candidate_indices = text_similarity.topk(
        candidate_count, dim=1
    )
    candidate_motion = bank_motion[candidate_indices]
    motion_similarity = torch.einsum(
        "qd,qpd->qp", query_motion, candidate_motion
    )
    combined = (
        (1.0 - motion_weight) * candidate_similarity
        + motion_weight * motion_similarity
    )
    keep = min(top_k, candidate_count)
    kept_scores, kept_offsets = combined.topk(keep, dim=1)
    kept_indices = candidate_indices.gather(1, kept_offsets)
    weights = torch.softmax(kept_scores / temperature, dim=1)
    return kept_indices, weights


def _apply_plan(
    bank_moves: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    selected = bank_moves[indices]
    return torch.einsum("qk,qkd->qd", weights, selected)


def _fit_two_input_gate(
    local_move: torch.Tensor,
    neighbor_move: torch.Tensor,
    target_move: torch.Tensor,
) -> tuple[float, float]:
    aa = _dot(local_move, local_move)
    ab = _dot(local_move, neighbor_move)
    bb = _dot(neighbor_move, neighbor_move)
    ay = _dot(local_move, target_move)
    by = _dot(neighbor_move, target_move)
    determinant = aa * bb - ab * ab
    if abs(float(determinant)) < 1e-12:
        return 1.0, 0.0
    local_weight = (ay * bb - by * ab) / determinant
    neighbor_weight = (by * aa - ay * ab) / determinant
    return float(local_weight), float(neighbor_weight)


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": round(statistics.mean(values), 6),
        "std": round(statistics.stdev(values), 6) if len(values) > 1 else 0.0,
    }


def _timing_benchmark(
    embeddings: torch.Tensor,
    latents: torch.Tensor,
    top_ks: tuple[int, ...],
    pool_k: int,
    temperature: float,
    motion_weight: float,
    repeats: int,
) -> dict:
    """Single-query resident-cache CPU timing at a representative cell."""
    torch.set_num_threads(1)
    query_index = 0
    bank_indices = torch.arange(1, embeddings.shape[0])
    step, horizon = 6, 4
    query_embedding = embeddings[query_index : query_index + 1]
    bank_embeddings = embeddings[bank_indices]
    delta = latents[:, step] - latents[:, step - 1]
    query_motion = _motion_descriptor(delta[query_index : query_index + 1])
    bank_motion = _motion_descriptor(delta[bank_indices])
    bank_moves = (
        latents[bank_indices, step + horizon] - latents[bank_indices, step]
    ).flatten(1)

    def measure(callable_) -> float:
        samples = []
        for _ in range(5):
            callable_()
        for _ in range(repeats):
            started = time.perf_counter_ns()
            callable_()
            samples.append((time.perf_counter_ns() - started) / 1e6)
        return round(statistics.median(samples), 4)

    timings = {
        "local_scaled": measure(
            lambda: horizon * delta[query_index : query_index + 1].flatten(1)
        )
    }
    for top_k in top_ks:
        for name, weight in (("text_topk", 0.0), ("motion_pruned", motion_weight)):
            def predict(weight=weight, top_k=top_k):
                plan = _neighbor_plan(
                    query_embedding,
                    query_motion,
                    bank_embeddings,
                    bank_motion,
                    top_k=top_k,
                    pool_k=pool_k,
                    temperature=temperature,
                    motion_weight=weight,
                )
                return _apply_plan(bank_moves, *plan)

            timings[f"{name}_k{top_k}"] = measure(predict)
    dense_weights = torch.softmax(
        (query_embedding @ bank_embeddings.T) / temperature, dim=1
    )
    timings["dense_attention"] = measure(lambda: dense_weights @ bank_moves)
    return {
        "device": "CPU, one PyTorch thread",
        "cell": {"step": step, "horizon": horizon},
        "resident_bank_trajectories": len(bank_indices),
        "repeats": repeats,
        "median_ms_per_query": timings,
    }


def run(
    trajectory_dir: Path,
    folds: int,
    horizons: tuple[int, ...],
    top_ks: tuple[int, ...],
    pool_k: int,
    temperature: float,
    motion_weight: float,
    timing_repeats: int,
) -> dict:
    files = sorted(trajectory_dir.glob("*.pt"))
    if len(files) < folds + 1:
        raise ValueError(f"need at least {folds + 1} trajectories, found {len(files)}")
    records = [torch.load(path, map_location="cpu", weights_only=False) for path in files]
    latents = torch.stack([record["latents"] for record in records]).float()
    embeddings = _normalize(torch.stack([record["text_emb"] for record in records]))
    del records

    n_trajectories, n_steps = latents.shape[:2]
    all_indices = torch.arange(n_trajectories)
    fold_rows: list[dict] = []
    cells: list[dict] = []

    for fold in range(folds):
        test_mask = all_indices.remainder(folds) == fold
        train_indices = all_indices[~test_mask]
        test_indices = all_indices[test_mask]
        train_embeddings = embeddings[train_indices]
        test_embeddings = embeddings[test_indices]

        for step in range(2, n_steps - 1):
            all_delta = latents[:, step] - latents[:, step - 1]
            train_delta = all_delta[train_indices]
            test_delta = all_delta[test_indices]
            train_motion = _motion_descriptor(train_delta)
            test_motion = _motion_descriptor(test_delta)

            plans: dict[tuple[str, int, str], tuple[torch.Tensor, torch.Tensor]] = {}
            for top_k in top_ks:
                for name, weight in (("text", 0.0), ("pruned", motion_weight)):
                    plans[(name, top_k, "train")] = _neighbor_plan(
                        train_embeddings,
                        train_motion,
                        train_embeddings,
                        train_motion,
                        top_k=top_k,
                        pool_k=pool_k,
                        temperature=temperature,
                        motion_weight=weight,
                        exclude_diagonal=True,
                    )
                    plans[(name, top_k, "test")] = _neighbor_plan(
                        test_embeddings,
                        test_motion,
                        train_embeddings,
                        train_motion,
                        top_k=top_k,
                        pool_k=pool_k,
                        temperature=temperature,
                        motion_weight=weight,
                    )

            for horizon in horizons:
                if step + horizon >= n_steps - 1:
                    continue
                train_target = (
                    latents[train_indices, step + horizon]
                    - latents[train_indices, step]
                ).flatten(1)
                test_target = (
                    latents[test_indices, step + horizon]
                    - latents[test_indices, step]
                ).flatten(1)
                train_local = horizon * train_delta.flatten(1)
                test_local = horizon * test_delta.flatten(1)
                alpha = _dot(train_local, train_target) / _dot(
                    train_local, train_local
                ).clamp_min(1e-12)

                scores: dict[str, float] = {
                    "local_scaled": float(
                        _rel_l2_rows(float(alpha) * test_local, test_target).mean()
                    )
                }
                gates = {}
                for top_k in top_ks:
                    test_text = _apply_plan(
                        train_target, *plans[("text", top_k, "test")]
                    )
                    train_pruned = _apply_plan(
                        train_target, *plans[("pruned", top_k, "train")]
                    )
                    test_pruned = _apply_plan(
                        train_target, *plans[("pruned", top_k, "test")]
                    )
                    local_weight, neighbor_weight = _fit_two_input_gate(
                        train_local, train_pruned, train_target
                    )
                    gated = local_weight * test_local + neighbor_weight * test_pruned

                    # A more conservative architecture transports how neighbour
                    # trajectories bend away from the global schedule-calibrated
                    # tangent instead of transporting their absolute movement.
                    train_residual = train_target - float(alpha) * train_local
                    train_neighbor_residual = _apply_plan(
                        train_residual, *plans[("pruned", top_k, "train")]
                    )
                    test_neighbor_residual = _apply_plan(
                        train_residual, *plans[("pruned", top_k, "test")]
                    )
                    residual_local_weight, residual_weight = _fit_two_input_gate(
                        float(alpha) * train_local,
                        train_neighbor_residual,
                        train_target,
                    )
                    residual_prediction = (
                        residual_local_weight * float(alpha) * test_local
                        + residual_weight * test_neighbor_residual
                    )

                    # Neighbours can also predict only schedule speed.  Scalar
                    # transport is nearly free; channel transport adds 16 values
                    # per neighbour while retaining the query's own spatial path.
                    train_scalar = (
                        (train_local * train_target).sum(dim=1)
                        / train_local.square().sum(dim=1).clamp_min(1e-12)
                    )
                    test_scalar = _apply_plan(
                        train_scalar[:, None], *plans[("pruned", top_k, "test")]
                    )
                    scalar_prediction = test_scalar * test_local

                    channels = latents.shape[2]
                    train_local_channel = train_local.reshape(
                        train_local.shape[0], channels, -1
                    )
                    train_target_channel = train_target.reshape(
                        train_target.shape[0], channels, -1
                    )
                    train_channel_scale = (
                        (train_local_channel * train_target_channel).sum(dim=2)
                        / train_local_channel.square().sum(dim=2).clamp_min(1e-12)
                    )
                    train_channel_prediction = (
                        _apply_plan(
                            train_channel_scale,
                            *plans[("pruned", top_k, "train")],
                        )[:, :, None]
                        * train_local_channel
                    ).flatten(1)
                    test_channel_prediction = (
                        _apply_plan(
                            train_channel_scale,
                            *plans[("pruned", top_k, "test")],
                        )[:, :, None]
                        * test_local.reshape(test_local.shape[0], channels, -1)
                    ).flatten(1)
                    channel_local_weight, channel_weight = _fit_two_input_gate(
                        float(alpha) * train_local,
                        train_channel_prediction - float(alpha) * train_local,
                        train_target,
                    )
                    channel_prediction = (
                        channel_local_weight * float(alpha) * test_local
                        + channel_weight
                        * (test_channel_prediction - float(alpha) * test_local)
                    )
                    scores[f"text_topk_k{top_k}"] = float(
                        _rel_l2_rows(test_text, test_target).mean()
                    )
                    scores[f"motion_pruned_k{top_k}"] = float(
                        _rel_l2_rows(test_pruned, test_target).mean()
                    )
                    scores[f"gated_pruned_k{top_k}"] = float(
                        _rel_l2_rows(gated, test_target).mean()
                    )
                    scores[f"gated_residual_k{top_k}"] = float(
                        _rel_l2_rows(residual_prediction, test_target).mean()
                    )
                    scores[f"pruned_scalar_k{top_k}"] = float(
                        _rel_l2_rows(scalar_prediction, test_target).mean()
                    )
                    scores[f"gated_channel_k{top_k}"] = float(
                        _rel_l2_rows(channel_prediction, test_target).mean()
                    )
                    gates[str(top_k)] = {
                        "local_weight": round(local_weight, 6),
                        "neighbor_weight": round(neighbor_weight, 6),
                        "residual_local_weight": round(residual_local_weight, 6),
                        "residual_weight": round(residual_weight, 6),
                        "channel_local_weight": round(channel_local_weight, 6),
                        "channel_weight": round(channel_weight, 6),
                    }
                cells.append(
                    {
                        "fold": fold,
                        "step": step,
                        "horizon": horizon,
                        "scores": {name: round(value, 6) for name, value in scores.items()},
                        "gates": gates,
                    }
                )

        method_names = sorted(cells[-1]["scores"])
        fold_cells = [cell for cell in cells if cell["fold"] == fold]
        fold_rows.append(
            {
                "fold": fold,
                "n_cells": len(fold_cells),
                "methods": {
                    method: round(
                        statistics.mean(cell["scores"][method] for cell in fold_cells),
                        6,
                    )
                    for method in method_names
                },
            }
        )

    methods = {
        method: _mean_std([row["methods"][method] for row in fold_rows])
        for method in fold_rows[0]["methods"]
    }
    baseline = methods["local_scaled"]["mean"]
    for values in methods.values():
        values["change_vs_local_scaled_pct"] = round(
            100.0 * (baseline / values["mean"] - 1.0), 2
        )

    by_horizon = {}
    for horizon in horizons:
        horizon_cells = [cell for cell in cells if cell["horizon"] == horizon]
        by_horizon[str(horizon)] = {
            method: round(
                statistics.mean(cell["scores"][method] for cell in horizon_cells), 6
            )
            for method in methods
        }

    timing = _timing_benchmark(
        embeddings,
        latents,
        top_ks,
        pool_k,
        temperature,
        motion_weight,
        timing_repeats,
    )
    return {
        "protocol": {
            "trajectory_dir": str(trajectory_dir),
            "n_trajectories": n_trajectories,
            "n_steps": n_steps,
            "folds": folds,
            "fold_assignment": "sorted trajectory index modulo fold count",
            "horizons": list(horizons),
            "top_ks": list(top_ks),
            "text_candidate_pool": pool_k,
            "motion_descriptor": "adaptive-average-pool latent delta to 16x8x8, cosine normalized",
            "motion_weight": motion_weight,
            "attention_temperature": temperature,
            "fit_scope": "two scalar gate weights per training fold, step, horizon, and k",
            "metric": "mean relL2 per held-out (step, horizon) cell; cells weighted equally",
        },
        "summary": methods,
        "by_horizon": by_horizon,
        "fold_rows": fold_rows,
        "cells": cells,
        "timing": timing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-validated kNN trajectory ablation")
    parser.add_argument(
        "--trajectory-dir",
        type=Path,
        default=Path("/sdb-disk/latentteleport-spec/trajs-16step-512"),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--horizons", type=int, nargs="+", default=(1, 2, 4, 8))
    parser.add_argument("--top-ks", type=int, nargs="+", default=(1, 2, 4, 8))
    parser.add_argument("--pool-k", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--motion-weight", type=float, default=0.5)
    parser.add_argument("--timing-repeats", type=int, default=30)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/speculative/knn-ablation.json"),
    )
    args = parser.parse_args()
    result = run(
        args.trajectory_dir,
        args.folds,
        tuple(args.horizons),
        tuple(args.top_ks),
        args.pool_k,
        args.temperature,
        args.motion_weight,
        args.timing_repeats,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.output}")
    for method, values in result["summary"].items():
        print(
            f"{method:24s} relL2={values['mean']:.4f} "
            f"change={values['change_vs_local_scaled_pct']:+.1f}%"
        )
    print("resident CPU timing (ms/query):")
    for method, value in result["timing"]["median_ms_per_query"].items():
        print(f"  {method:24s} {value:.4f}")


if __name__ == "__main__":
    main()
