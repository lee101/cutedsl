"""Tests for k-NN trajectory priors."""

import tempfile

import torch

from latentteleport.cache import LatentCache
from latentteleport.tokenizer import VisualUnit
from latentteleport.trajectory import apply_knn_trajectory_prior


def test_apply_knn_trajectory_prior_uses_neighbor_deltas():
    tmpdir = tempfile.mkdtemp()
    cache = LatentCache(tmpdir, resolution=(512, 512))
    unit = VisualUnit.from_text("grass")
    emb = torch.ones(8)
    cache.store_latents(
        unit,
        {
            6: torch.zeros(1, 2, 2),
            7: torch.ones(1, 2, 2),
        },
        text_embedding=emb,
    )

    combined = torch.zeros(1, 2, 2)
    updated, stats = apply_knn_trajectory_prior(
        cache,
        combined,
        [emb],
        None,
        start_step=6,
        top_k=1,
        scale=0.5,
        virtual_steps=1,
    )

    assert torch.allclose(updated, torch.full_like(combined, 0.5))
    assert stats["neighbors_used"] == 1
    assert stats["virtual_steps_applied"] == 1


def test_apply_knn_trajectory_prior_can_repel_negative_neighbors():
    tmpdir = tempfile.mkdtemp()
    cache = LatentCache(tmpdir, resolution=(512, 512))
    pos_unit = VisualUnit.from_text("grass")
    neg_unit = VisualUnit.from_text("artifact")
    pos_emb = torch.tensor([1.0, 0.0, 0.0, 0.0])
    neg_emb = torch.tensor([0.0, 1.0, 0.0, 0.0])
    cache.store_latents(
        pos_unit,
        {6: torch.zeros(1, 2, 2), 7: torch.ones(1, 2, 2)},
        text_embedding=pos_emb,
    )
    cache.store_latents(
        neg_unit,
        {6: torch.zeros(1, 2, 2), 7: torch.full((1, 2, 2), 2.0)},
        text_embedding=neg_emb,
    )

    updated, stats = apply_knn_trajectory_prior(
        cache,
        torch.zeros(1, 2, 2),
        [pos_emb],
        [neg_emb],
        start_step=6,
        top_k=1,
        scale=1.0,
        repel_scale=0.25,
        virtual_steps=1,
    )

    assert torch.allclose(updated, torch.full((1, 2, 2), 0.5))
    assert stats["neighbors_used"] == 1
    assert stats["repel_neighbors_used"] == 1
