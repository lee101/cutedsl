"""Trajectory priors for low-step latent teleportation.

Uses nearest cached text embeddings to estimate a latent-space delta field
between neighboring denoising steps, then nudges the combined latent before
running the remaining sampler steps.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from latentteleport.cache import LatentCache
from latentteleport.tokenizer import VisualUnit


@dataclass
class TrajectoryStats:
    neighbors_used: int = 0
    repel_neighbors_used: int = 0
    virtual_steps_applied: int = 0
    mean_similarity: float = 0.0
    candidates_scanned: int = 0
    mean_motion_similarity: float = 0.0


def _motion_descriptor(delta: torch.Tensor, size: int = 8) -> torch.Tensor:
    """Compact a latent movement for cheap cosine-similarity pruning."""
    value = delta.detach().float()
    if value.ndim < 2:
        value = value.reshape(1, 1, -1, 1)
    else:
        value = value.reshape(-1, 1, value.shape[-2], value.shape[-1])
    pooled = torch.nn.functional.adaptive_avg_pool2d(value, (size, size)).flatten()
    return torch.nn.functional.normalize(pooled, dim=0, eps=1e-8)


def forecast_with_pruned_knn_residual(
    cache: LatentCache,
    current_latent: torch.Tensor,
    previous_latent: torch.Tensor,
    embedding: torch.Tensor,
    start_step: int,
    horizon: int,
    *,
    top_k: int = 8,
    candidate_pool: int = 16,
    momentum_scale: float = 1.0,
    local_weight: float = 1.0,
    residual_weight: float = 1.0,
    motion_weight: float = 0.5,
    temperature: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """Forecast with local momentum plus motion-pruned neighbour residuals.

    Prompt similarity selects a small candidate pool. Agreement between the
    query and candidate's observed motion prunes it to ``top_k`` records. Only
    the candidate correction away from calibrated momentum is transported.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if top_k <= 0 or candidate_pool <= 0:
        raise ValueError("top_k and candidate_pool must be positive")
    if not 0.0 <= motion_weight <= 1.0:
        raise ValueError("motion_weight must be in [0, 1]")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    query_delta = current_latent.float() - previous_latent.float()
    local_move = momentum_scale * horizon * query_delta
    candidates = cache.find_nearest(embedding, top_k=max(top_k, candidate_pool))
    query_motion = _motion_descriptor(query_delta)
    rows = []
    for unit_id, text, prompt_similarity in candidates:
        unit = VisualUnit(text=text, unit_id=unit_id)
        previous = cache.load_latent(unit, start_step - 1)
        current = cache.load_latent(unit, start_step)
        target = cache.load_latent(unit, start_step + horizon)
        if previous is None or current is None or target is None:
            continue
        if current.shape != current_latent.shape:
            continue
        neighbor_delta = current.float() - previous.float()
        motion_similarity = float(
            torch.dot(query_motion, _motion_descriptor(neighbor_delta))
        )
        score = (
            (1.0 - motion_weight) * float(prompt_similarity)
            + motion_weight * motion_similarity
        )
        residual = (
            target.float()
            - current.float()
            - momentum_scale * horizon * neighbor_delta
        )
        rows.append((score, float(prompt_similarity), motion_similarity, residual))

    rows.sort(key=lambda row: row[0], reverse=True)
    selected = rows[:top_k]
    stats = TrajectoryStats(candidates_scanned=len(rows))
    if not selected:
        prediction = current_latent.float() + local_weight * local_move
        return prediction.to(current_latent.dtype), stats.__dict__

    scores = torch.tensor([row[0] for row in selected], dtype=torch.float32)
    weights = torch.softmax(scores / temperature, dim=0)
    residuals = torch.stack([row[3] for row in selected])
    weight_shape = (len(selected),) + (1,) * (residuals.ndim - 1)
    retrieved_residual = (residuals * weights.view(weight_shape)).sum(dim=0)
    prediction = (
        current_latent.float()
        + local_weight * local_move
        + residual_weight * retrieved_residual
    )
    stats.neighbors_used = len(selected)
    stats.virtual_steps_applied = horizon
    stats.mean_similarity = float(
        sum(row[1] for row in selected) / len(selected)
    )
    stats.mean_motion_similarity = float(
        sum(row[2] for row in selected) / len(selected)
    )
    return prediction.to(current_latent.dtype), stats.__dict__


def _weighted_mean_delta(
    cache: LatentCache,
    embedding: torch.Tensor,
    start_step: int,
    next_step: int,
    top_k: int,
) -> tuple[torch.Tensor | None, TrajectoryStats]:
    neighbors = cache.find_nearest(embedding, top_k=top_k)
    deltas: list[torch.Tensor] = []
    weights: list[float] = []
    for unit_id, text, sim in neighbors:
        unit = VisualUnit(text=text, unit_id=unit_id)
        start_latent = cache.load_latent(unit, start_step)
        next_latent = cache.load_latent(unit, next_step)
        if start_latent is None or next_latent is None:
            continue
        deltas.append((next_latent - start_latent).float())
        weights.append(max(sim, 0.0))

    if not deltas:
        return None, TrajectoryStats()

    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    if float(weight_tensor.sum()) <= 0.0:
        weight_tensor = torch.ones_like(weight_tensor)
    weight_tensor = weight_tensor / weight_tensor.sum()
    stacked = torch.stack(deltas, dim=0)
    weight_shape = (stacked.shape[0],) + (1,) * (stacked.ndim - 1)
    delta = (stacked * weight_tensor.view(weight_shape)).sum(dim=0)
    return delta, TrajectoryStats(
        neighbors_used=len(deltas),
        mean_similarity=float(sum(weights) / max(len(weights), 1)),
    )


def apply_knn_trajectory_prior(
    cache: LatentCache,
    combined_latent: torch.Tensor,
    embeddings: list[torch.Tensor],
    repel_embeddings: list[torch.Tensor] | None,
    start_step: int,
    top_k: int = 4,
    scale: float = 0.35,
    repel_scale: float = 0.0,
    virtual_steps: int = 1,
) -> tuple[torch.Tensor, dict]:
    """Apply one or more virtual denoising steps from nearest-neighbor trajectories."""
    if not embeddings or virtual_steps <= 0:
        return combined_latent, TrajectoryStats().__dict__

    updated = combined_latent.float()
    agg_stats = TrajectoryStats()
    current_step = start_step
    for _ in range(virtual_steps):
        next_step = current_step + 1
        deltas = []
        sims = []
        for emb in embeddings:
            delta, stats = _weighted_mean_delta(cache, emb, current_step, next_step, top_k)
            if delta is None:
                continue
            deltas.append(delta)
            sims.append(stats.mean_similarity)
            agg_stats.neighbors_used += stats.neighbors_used
        if not deltas:
            break
        mean_delta = torch.stack(deltas, dim=0).mean(dim=0)
        if repel_embeddings and repel_scale > 0.0:
            repel_deltas = []
            for emb in repel_embeddings:
                repel_delta, stats = _weighted_mean_delta(cache, emb, current_step, next_step, top_k)
                if repel_delta is None:
                    continue
                repel_deltas.append(repel_delta)
                agg_stats.repel_neighbors_used += stats.neighbors_used
            if repel_deltas:
                mean_delta = mean_delta - repel_scale * torch.stack(repel_deltas, dim=0).mean(dim=0)
        updated = updated + scale * mean_delta.to(updated.device)
        agg_stats.virtual_steps_applied += 1
        agg_stats.mean_similarity = float(sum(sims) / max(len(sims), 1))
        current_step = next_step

    return updated.to(combined_latent.dtype), agg_stats.__dict__
