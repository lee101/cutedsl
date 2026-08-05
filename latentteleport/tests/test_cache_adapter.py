from __future__ import annotations

import json

import torch
from safetensors.torch import save_file

from latentteleport.cache_adapter import (
    CacheResidualAdapter,
    CacheResidualAdapterConfig,
    adapter_condition,
    load_cache_adapter,
    weighted_residual_statistics,
)
from scripts.spec_train_cache_adapter import _training_loss


def test_zero_initialized_adapter_is_exact_local_forecast() -> None:
    model = CacheResidualAdapter()
    local = torch.randn(2, 16, 8, 8)
    residual = torch.randn_like(local)
    residual_std = torch.rand_like(local)
    condition = torch.randn(2, 6)

    prediction = model(local, residual, residual_std, condition)

    torch.testing.assert_close(prediction, local)


def test_adapter_supports_compact_configuration_and_gradients() -> None:
    model = CacheResidualAdapter(
        CacheResidualAdapterConfig(
            latent_channels=4,
            hidden_channels=8,
            condition_dim=4,
            num_blocks=1,
        )
    )
    inputs = [torch.randn(2, 4, 8, 8) for _ in range(3)]
    prediction = model(*inputs, torch.randn(2, 4))
    prediction.square().mean().backward()

    assert prediction.shape == inputs[0].shape
    assert model.output_projection.weight.grad is not None
    assert model.config_dict()["hidden_channels"] == 8


def test_weighted_residual_statistics_match_manual_values() -> None:
    residuals = torch.tensor([[[[[1.0]]], [[[3.0]]]]])
    weights = torch.tensor([[0.25, 0.75]])

    mean, std = weighted_residual_statistics(residuals, weights)

    torch.testing.assert_close(mean, torch.tensor([[[[2.5]]]]))
    torch.testing.assert_close(std, torch.tensor([[[[0.8660254]]]]))


def test_weighted_residual_statistics_validate_shapes() -> None:
    try:
        weighted_residual_statistics(torch.ones(2, 3, 4), torch.ones(2, 3))
    except ValueError as error:
        assert "residuals" in str(error)
    else:
        raise AssertionError("expected invalid residual shape to fail")


def test_adapter_condition_encodes_schedule_and_neighbor_confidence() -> None:
    uniform = adapter_condition(
        step=4,
        horizon=2,
        momentum_scale=0.75,
        weights=torch.tensor([[0.5, 0.5]]),
        total_steps=16,
        max_horizon=8,
    )
    peaked = adapter_condition(
        step=4,
        horizon=2,
        momentum_scale=0.75,
        weights=torch.tensor([[0.99, 0.01]]),
        total_steps=16,
        max_horizon=8,
    )

    torch.testing.assert_close(uniform[0, :3], torch.tensor([4 / 15, 0.25, 0.75]))
    assert abs(float(uniform[0, 5])) < 1e-6
    assert float(peaked[0, 5]) > 0.9


def test_structural_training_loss_is_zero_for_exact_target() -> None:
    target = torch.randn(2, 4, 8, 8)
    exact = _training_loss(target, target, gradient_weight=0.15, coarse_weight=0.15)
    shifted = _training_loss(
        target.roll(1, dims=-1),
        target,
        gradient_weight=0.15,
        coarse_weight=0.15,
    )

    assert float(exact) == 0.0
    assert float(shifted) > 0.0


def test_load_cache_adapter_restores_weights_and_schedule(tmp_path) -> None:
    model = CacheResidualAdapter()
    (tmp_path / "config.json").write_text(
        json.dumps({"model": model.config_dict(), "protocol": {}})
    )
    save_file(model.state_dict(), tmp_path / "cache_adapter.safetensors")
    (tmp_path / "schedule_coefficients.json").write_text(
        json.dumps(
            [
                {
                    "step": 2,
                    "horizon": 4,
                    "momentum_scale": 0.5,
                    "gate_local_weight": 1.0,
                    "gate_residual_weight": 0.25,
                }
            ]
        )
    )

    loaded, schedule = load_cache_adapter(tmp_path)
    local = torch.randn(1, 16, 4, 4)
    prediction = loaded(
        local,
        torch.randn_like(local),
        torch.rand_like(local),
        torch.randn(1, 6),
    )

    torch.testing.assert_close(prediction, local)
    assert schedule[(2, 4)] == {
        "momentum_scale": 0.5,
        "gate_local_weight": 1.0,
        "gate_residual_weight": 0.25,
    }
