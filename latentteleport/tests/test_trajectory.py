"""Tests for k-NN trajectory priors."""

import tempfile

import torch

from latentteleport.cache import LatentCache
from latentteleport.tokenizer import VisualUnit
from latentteleport.trajectory import (
    apply_knn_trajectory_prior,
    forecast_with_pruned_knn_residual,
)


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


def test_pruned_knn_residual_transports_only_forecast_correction():
    cache = LatentCache(tempfile.mkdtemp(), resolution=(512, 512))
    embedding = torch.tensor([1.0, 0.0, 0.0, 0.0])
    for index, residual in enumerate((0.2, 0.4)):
        unit = VisualUnit.from_text(f"neighbor {index}")
        cache.store_latents(
            unit,
            {
                0: torch.zeros(1, 8, 8),
                1: torch.ones(1, 8, 8),
                3: torch.full((1, 8, 8), 2.0 + residual),
            },
            text_embedding=embedding + index * 0.01,
        )

    prediction, stats = forecast_with_pruned_knn_residual(
        cache,
        current_latent=torch.ones(1, 8, 8),
        previous_latent=torch.zeros(1, 8, 8),
        embedding=embedding,
        start_step=1,
        horizon=2,
        top_k=2,
        candidate_pool=2,
        momentum_scale=0.5,
        local_weight=1.0,
        residual_weight=1.0,
        motion_weight=0.5,
    )

    assert torch.all(prediction > 2.19)
    assert torch.all(prediction < 2.41)
    assert stats["neighbors_used"] == 2
    assert stats["candidates_scanned"] == 2
    assert stats["mean_motion_similarity"] > 0.99


def test_pruned_knn_residual_falls_back_to_calibrated_local_move():
    cache = LatentCache(tempfile.mkdtemp(), resolution=(512, 512))
    prediction, stats = forecast_with_pruned_knn_residual(
        cache,
        current_latent=torch.ones(1, 4, 4),
        previous_latent=torch.zeros(1, 4, 4),
        embedding=torch.ones(4),
        start_step=1,
        horizon=3,
        momentum_scale=0.25,
    )

    torch.testing.assert_close(prediction, torch.full((1, 4, 4), 1.75))
    assert stats["neighbors_used"] == 0


def test_pruned_knn_residual_validates_configuration():
    cache = LatentCache(tempfile.mkdtemp(), resolution=(512, 512))
    try:
        forecast_with_pruned_knn_residual(
            cache,
            torch.ones(1, 2, 2),
            torch.zeros(1, 2, 2),
            torch.ones(4),
            start_step=1,
            horizon=0,
        )
    except ValueError as error:
        assert "horizon" in str(error)
    else:
        raise AssertionError("expected an invalid horizon to fail")
