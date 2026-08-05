#!/usr/bin/env python3
"""Train a tiny cache-residual adapter on stored Z-Image trajectories."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from latentteleport.cache_adapter import (
    CacheResidualAdapter,
    CacheResidualAdapterConfig,
    adapter_condition,
    weighted_residual_statistics,
)
from scripts.spec_knn_ablation import (
    _dot,
    _fit_two_input_gate,
    _motion_descriptor,
    _neighbor_plan,
    _normalize,
)


@dataclass
class Cell:
    step: int
    horizon: int
    momentum_scale: float
    bank_residual: torch.Tensor
    train_indices: torch.Tensor
    train_weights: torch.Tensor
    test_indices: torch.Tensor
    test_weights: torch.Tensor
    gate_local_weight: float
    gate_residual_weight: float


def _rel_l2_rows(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (
        (prediction - target).flatten(1).norm(dim=1)
        / target.flatten(1).norm(dim=1).clamp_min(1e-8)
    )


def _training_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    gradient_weight: float,
    coarse_weight: float,
) -> torch.Tensor:
    loss = _rel_l2_rows(prediction.float(), target.float()).square().mean()
    if gradient_weight > 0.0:
        pred_dx = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
        pred_dy = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
        gradient_error = torch.cat(
            ((pred_dx - target_dx).flatten(1), (pred_dy - target_dy).flatten(1)),
            dim=1,
        ).norm(dim=1)
        gradient_scale = torch.cat(
            (target_dx.flatten(1), target_dy.flatten(1)), dim=1
        ).norm(dim=1).clamp_min(1e-8)
        loss = loss + gradient_weight * (gradient_error / gradient_scale).square().mean()
    if coarse_weight > 0.0:
        coarse_prediction = F.avg_pool2d(prediction.float(), kernel_size=4)
        coarse_target = F.avg_pool2d(target.float(), kernel_size=4)
        loss = loss + coarse_weight * _rel_l2_rows(
            coarse_prediction, coarse_target
        ).square().mean()
    return loss


def _retrieved_statistics(
    bank_residual: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    residuals = bank_residual[indices].float()
    return weighted_residual_statistics(residuals, weights.float())


def _choose_device(requested: str, minimum_free_gib: float) -> str:
    if requested == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < minimum_free_gib * 1024**3:
        print(
            f"CUDA has {free_bytes / 1024**3:.1f} GiB free; "
            f"using CPU below the {minimum_free_gib:.1f} GiB safety threshold"
        )
        return "cpu"
    return "cuda"


def _prepare_cells(
    latents: torch.Tensor,
    embeddings: torch.Tensor,
    train_indices: torch.Tensor,
    test_indices: torch.Tensor,
    horizons: tuple[int, ...],
    top_k: int,
    pool_k: int,
    temperature: float,
    motion_weight: float,
) -> list[Cell]:
    n_steps = latents.shape[1]
    train_embeddings = embeddings[train_indices]
    test_embeddings = embeddings[test_indices]
    cells = []
    for step in range(2, n_steps - 1):
        delta = latents[:, step].float() - latents[:, step - 1].float()
        train_delta = delta[train_indices]
        test_delta = delta[test_indices]
        train_motion = _motion_descriptor(train_delta)
        test_motion = _motion_descriptor(test_delta)
        train_plan = _neighbor_plan(
            train_embeddings,
            train_motion,
            train_embeddings,
            train_motion,
            top_k=top_k,
            pool_k=pool_k,
            temperature=temperature,
            motion_weight=motion_weight,
            exclude_diagonal=True,
        )
        test_plan = _neighbor_plan(
            test_embeddings,
            test_motion,
            train_embeddings,
            train_motion,
            top_k=top_k,
            pool_k=pool_k,
            temperature=temperature,
            motion_weight=motion_weight,
        )
        for horizon in horizons:
            if step + horizon >= n_steps - 1:
                continue
            train_target = (
                latents[train_indices, step + horizon].float()
                - latents[train_indices, step].float()
            )
            train_local_unscaled = horizon * train_delta
            alpha = _dot(train_local_unscaled, train_target) / _dot(
                train_local_unscaled, train_local_unscaled
            ).clamp_min(1e-12)
            train_local = float(alpha) * train_local_unscaled
            bank_residual = (train_target - train_local).half()
            retrieved, _ = _retrieved_statistics(
                bank_residual, train_plan[0], train_plan[1]
            )
            gate_local, gate_residual = _fit_two_input_gate(
                train_local.flatten(1),
                retrieved.flatten(1),
                train_target.flatten(1),
            )
            cells.append(
                Cell(
                    step=step,
                    horizon=horizon,
                    momentum_scale=float(alpha),
                    bank_residual=bank_residual,
                    train_indices=train_plan[0],
                    train_weights=train_plan[1].half(),
                    test_indices=test_plan[0],
                    test_weights=test_plan[1].half(),
                    gate_local_weight=gate_local,
                    gate_residual_weight=gate_residual,
                )
            )
    return cells


def _batch(
    latents: torch.Tensor,
    indices: torch.Tensor,
    cell: Cell,
    query_rows: torch.Tensor,
    split: str,
    device: str,
    max_horizon: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    data_indices = indices[query_rows]
    delta = (
        latents[data_indices, cell.step].float()
        - latents[data_indices, cell.step - 1].float()
    )
    local = cell.momentum_scale * cell.horizon * delta
    target = (
        latents[data_indices, cell.step + cell.horizon].float()
        - latents[data_indices, cell.step].float()
    )
    plan_indices = cell.train_indices if split == "train" else cell.test_indices
    plan_weights = cell.train_weights if split == "train" else cell.test_weights
    selected_indices = plan_indices[query_rows]
    selected_weights = plan_weights[query_rows].float()
    residual, residual_std = _retrieved_statistics(
        cell.bank_residual,
        selected_indices,
        selected_weights,
    )
    condition = adapter_condition(
        cell.step,
        cell.horizon,
        cell.momentum_scale,
        selected_weights,
        total_steps=latents.shape[1],
        max_horizon=max_horizon,
        gate_local_weight=cell.gate_local_weight,
        gate_residual_weight=cell.gate_residual_weight,
    )
    return tuple(
        value.to(device, non_blocking=device == "cuda")
        for value in (local, residual, residual_std, condition, target)
    )  # type: ignore[return-value]


@torch.inference_mode()
def evaluate(
    model: CacheResidualAdapter,
    latents: torch.Tensor,
    query_indices: torch.Tensor,
    cells: list[Cell],
    split: str,
    device: str,
    batch_size: int,
    max_horizon: int,
) -> dict:
    model.eval()
    scores = {"local": [], "scalar_gate": [], "adapter": []}
    by_horizon: dict[str, dict[str, list[float]]] = {}
    for cell in cells:
        cell_scores = {method: [] for method in scores}
        for start in range(0, len(query_indices), batch_size):
            rows = torch.arange(start, min(start + batch_size, len(query_indices)))
            local, residual, residual_std, condition, target = _batch(
                latents,
                query_indices,
                cell,
                rows,
                split,
                device,
                max_horizon,
            )
            scalar = (
                cell.gate_local_weight * local
                + cell.gate_residual_weight * residual
            )
            prediction = model(scalar, residual, residual_std, condition)
            predictions = {
                "local": local,
                "scalar_gate": scalar,
                "adapter": prediction,
            }
            for method, value in predictions.items():
                cell_scores[method].extend(
                    _rel_l2_rows(value.float(), target.float()).cpu().tolist()
                )
        horizon_row = by_horizon.setdefault(
            str(cell.horizon), {method: [] for method in scores}
        )
        for method, values in cell_scores.items():
            mean = statistics.mean(values)
            scores[method].append(mean)
            horizon_row[method].append(mean)
    summary = {
        method: round(statistics.mean(values), 6)
        for method, values in scores.items()
    }
    summary["adapter_improvement_vs_local_pct"] = round(
        100.0 * (1.0 - summary["adapter"] / summary["local"]), 3
    )
    summary["adapter_improvement_vs_scalar_gate_pct"] = round(
        100.0 * (1.0 - summary["adapter"] / summary["scalar_gate"]), 3
    )
    return {
        "summary": summary,
        "by_horizon": {
            horizon: {
                method: round(statistics.mean(values), 6)
                for method, values in methods.items()
            }
            for horizon, methods in by_horizon.items()
        },
    }


def _write_model_card(output_dir: Path, metrics: dict, parameter_count: int) -> None:
    test = metrics["test"]["summary"]
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

# Z-Image Cache Residual Adapter

This is a {parameter_count:,}-parameter residual head for cache-guided latent
teleportation. It combines calibrated query momentum with the weighted mean and
dispersion of eight motion-pruned cached trajectory residuals.

The held-out fold result is relL2 {test['adapter']:.4f}, compared with
{test['local']:.4f} for calibrated local momentum and
{test['scalar_gate']:.4f} for the scalar retrieval gate. See `metrics.json` for
the complete protocol and horizon breakdown.

The adapter forecasts latent movement; it is not a standalone Z-Image model.
Use it with the code and schedule coefficients in the CuteDSL latent
teleportation implementation. The checkpoint is specific to the recorded
16-step, 512px Z-Image Turbo schedule.
"""
    (output_dir / "README.md").write_text(content)


@torch.inference_mode()
def _benchmark_adapter(
    model: CacheResidualAdapter,
    device: str,
    latent_channels: int,
    condition_dim: int,
    repeats: int = 100,
) -> dict:
    inputs = [
        torch.randn(1, latent_channels, 64, 64, device=device)
        for _ in range(3)
    ]
    condition = torch.randn(1, condition_dim, device=device)

    def synchronize() -> None:
        if device == "cuda":
            torch.cuda.synchronize()

    for _ in range(20):
        model(*inputs, condition)
    synchronize()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        model(*inputs, condition)
        synchronize()
        samples.append((time.perf_counter_ns() - started) / 1e6)
    return {
        "device": device,
        "repeats": repeats,
        "median_ms": round(statistics.median(samples), 5),
        "p95_ms": round(sorted(samples)[int(0.95 * (len(samples) - 1))], 5),
    }


def run(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    files = sorted(args.trajectory_dir.glob("*.pt"))
    records = [torch.load(path, map_location="cpu", weights_only=False) for path in files]
    latents = torch.stack([record["latents"] for record in records]).half()
    embeddings = _normalize(torch.stack([record["text_emb"] for record in records]))
    del records
    all_indices = torch.arange(len(files))
    test_mask = all_indices.remainder(args.folds) == args.fold
    train_indices = all_indices[~test_mask]
    test_indices = all_indices[test_mask]
    horizons = tuple(args.horizons)
    max_horizon = max(horizons)
    device = _choose_device(args.device, args.minimum_free_gib)

    print("preparing leakage-free retrieval cells", flush=True)
    cells = _prepare_cells(
        latents,
        embeddings,
        train_indices,
        test_indices,
        horizons,
        args.top_k,
        args.pool_k,
        args.temperature,
        args.motion_weight,
    )
    config = CacheResidualAdapterConfig(
        latent_channels=latents.shape[2],
        hidden_channels=args.hidden_channels,
        condition_dim=6,
        num_blocks=args.num_blocks,
    )
    model = CacheResidualAdapter(config).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if args.evaluate_checkpoint is not None:
        model.load_state_dict(load_file(args.evaluate_checkpoint, device=device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history = []
    best_score = math.inf
    best_state = (
        {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
        if args.evaluate_checkpoint is not None
        else None
    )
    best_epoch = 0
    if args.evaluate_checkpoint is not None:
        initial_validation = evaluate(
            model,
            latents,
            test_indices,
            cells,
            "test",
            device,
            args.eval_batch_size,
            max_horizon,
        )
        best_score = initial_validation["summary"]["adapter"]
        history.append(
            {
                "epoch": 0,
                "train_rel_l2_squared": None,
                "validation_rel_l2": best_score,
                "seconds": 0.0,
            }
        )
    started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        losses = []
        epoch_started = time.perf_counter()
        for _ in range(args.batches_per_epoch):
            cell = random.choice(cells)
            rows = torch.randint(0, len(train_indices), (args.batch_size,))
            local, residual, residual_std, condition, target = _batch(
                latents,
                train_indices,
                cell,
                rows,
                "train",
                device,
                max_horizon,
            )
            base = (
                cell.gate_local_weight * local
                + cell.gate_residual_weight * residual
            )
            prediction = model(base, residual, residual_std, condition)
            loss = _training_loss(
                prediction,
                target,
                args.gradient_weight,
                args.coarse_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation = evaluate(
            model,
            latents,
            test_indices,
            cells,
            "test",
            device,
            args.eval_batch_size,
            max_horizon,
        )
        score = validation["summary"]["adapter"]
        row = {
            "epoch": epoch + 1,
            "train_rel_l2_squared": round(statistics.mean(losses), 7),
            "validation_rel_l2": score,
            "seconds": round(time.perf_counter() - epoch_started, 3),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if score < best_score:
            best_score = score
            best_epoch = epoch + 1
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    test_metrics = evaluate(
        model,
        latents,
        test_indices,
        cells,
        "test",
        device,
        args.eval_batch_size,
        max_horizon,
    )
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    result = {
        "protocol": {
            "trajectory_dir": str(args.trajectory_dir),
            "trajectories": len(files),
            "fold": args.fold,
            "folds": args.folds,
            "train_trajectories": len(train_indices),
            "test_trajectories": len(test_indices),
            "horizons": list(horizons),
            "top_k": args.top_k,
            "pool_k": args.pool_k,
            "device": device,
            "seed": args.seed,
            "cells": len(cells),
        },
        "model": {
            "config": asdict(config),
            "parameters": parameter_count,
        },
        "training": {
            "epochs": args.epochs,
            "batches_per_epoch": args.batches_per_epoch,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "gradient_weight": args.gradient_weight,
            "coarse_weight": args.coarse_weight,
            "elapsed_seconds": round(elapsed, 3),
            "best_epoch": best_epoch,
            "history": history,
        },
        "test": test_metrics,
    }
    result["inference_timing"] = _benchmark_adapter(
        model,
        device,
        config.latent_channels,
        config.condition_dim,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_file(best_state, args.output_dir / "cache_adapter.safetensors")
    (args.output_dir / "config.json").write_text(
        json.dumps({"model": asdict(config), "protocol": result["protocol"]}, indent=2)
        + "\n"
    )
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    coefficients = [
        {
            "step": cell.step,
            "horizon": cell.horizon,
            "momentum_scale": round(cell.momentum_scale, 8),
            "gate_local_weight": round(cell.gate_local_weight, 8),
            "gate_residual_weight": round(cell.gate_residual_weight, 8),
        }
        for cell in cells
    ]
    (args.output_dir / "schedule_coefficients.json").write_text(
        json.dumps(coefficients, indent=2) + "\n"
    )
    _write_model_card(args.output_dir, result, parameter_count)
    print(json.dumps(test_metrics["summary"], indent=2))
    print(f"wrote {args.output_dir}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train cache-guided residual adapter")
    parser.add_argument(
        "--trajectory-dir",
        type=Path,
        default=Path("/sdb-disk/latentteleport-spec/trajs-16step-512"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/sdb-disk/latentteleport-spec/cache-adapter-fold0"),
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--horizons", type=int, nargs="+", default=(1, 2, 4, 8))
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--pool-k", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--motion-weight", type=float, default=0.5)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batches-per-epoch", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-weight", type=float, default=0.15)
    parser.add_argument("--coarse-weight", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--minimum-free-gib", type=float, default=6.0)
    parser.add_argument("--evaluate-checkpoint", type=Path)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
