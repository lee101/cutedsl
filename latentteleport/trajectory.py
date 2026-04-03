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
