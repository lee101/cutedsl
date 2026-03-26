from __future__ import annotations

import torch

from cutechronos.tests.conftest import build_cute_only


def test_chronos_tubroquant_hook_runs_and_reports():
    model = build_cute_only()
    model.enable_turboquant_kv(bits=4, key_mode="prod", value_mode="mse")

    torch.manual_seed(0)
    context = torch.randn(1, 128) * 0.1 + 100

    with torch.no_grad():
        out = model(context)

    summary = model.get_tubroquant_summary()
    assert out.shape == (1, 21, 16)
    assert summary["enabled"] is True
    assert summary["compression_ratio"] > 1.0
    assert len(summary["layers"]) == model.config.num_layers


def test_chronos_tubroquant_disable_clears_hook():
    model = build_cute_only()
    model.enable_turboquant_kv(bits=4)
    model.disable_turboquant_kv()
    summary = model.get_tubroquant_summary()

    assert summary["enabled"] is False
    assert summary["layers"] == []
