"""Tests for GPU offload/onload on CuteChronos2Model and CuteChronos2Pipeline."""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest
import torch

from cutechronos.model import CuteChronos2Config, CuteChronos2Model
from cutechronos.pipeline import CuteChronos2Pipeline

from .conftest import _BASE_CONFIG_KWARGS


def _make_small_model() -> CuteChronos2Model:
    """Build a small model for fast tests."""
    cfg = CuteChronos2Config(
        **{**_BASE_CONFIG_KWARGS, "num_layers": 1, "d_model": 64, "d_kv": 8, "d_ff": 128, "num_heads": 8}
    )
    model = CuteChronos2Model(cfg)
    model.eval()
    return model


class TestModelOffload:
    def test_initial_state_not_offloaded(self):
        model = _make_small_model()
        assert not model.is_offloaded

    def test_offload_moves_params_to_cpu(self):
        model = _make_small_model()
        model.offload_to_cpu()
        assert model.is_offloaded
        for p in model.parameters():
            assert p.device == torch.device("cpu")

    def test_onload_moves_params_to_device(self):
        model = _make_small_model()
        model.offload_to_cpu()
        assert model.is_offloaded
        model.onload_to_gpu("cpu")  # use cpu since no GPU in tests
        assert not model.is_offloaded

    def test_offload_onload_roundtrip_preserves_weights(self):
        model = _make_small_model()
        weight_before = model.shared.weight.clone()
        model.offload_to_cpu()
        model.onload_to_gpu("cpu")
        assert torch.equal(model.shared.weight, weight_before)

    def test_offload_warns_on_compiled_model(self, caplog):
        model = _make_small_model()
        # Simulate torch.compile by adding _orig_mod attribute
        model._orig_mod = True
        with caplog.at_level(logging.WARNING, logger="cutechronos.model"):
            model.offload_to_cpu()
        assert "torch.compile" in caplog.text
        # Cleanup
        del model._orig_mod

    def test_is_offloaded_property_defaults_false(self):
        model = _make_small_model()
        # Manually delete _offloaded to test getattr fallback
        del model._offloaded
        assert not model.is_offloaded


class TestPipelineOffload:
    def _make_pipeline(self) -> CuteChronos2Pipeline:
        model = _make_small_model()
        return CuteChronos2Pipeline(model, device="cpu", _is_cute=True)

    def test_pipeline_offload(self):
        pipe = self._make_pipeline()
        pipe.offload()
        assert pipe.model.is_offloaded

    def test_pipeline_onload(self):
        pipe = self._make_pipeline()
        pipe.offload()
        pipe.onload("cpu")
        assert not pipe.model.is_offloaded

    def test_predict_auto_onloads(self):
        pipe = self._make_pipeline()
        pipe.offload()
        assert pipe.model.is_offloaded
        ctx = torch.randn(1, 64)
        pipe.predict(ctx, prediction_length=16)
        assert not pipe.model.is_offloaded

    def test_pipeline_offload_non_cute(self):
        model = _make_small_model()
        pipe = CuteChronos2Pipeline(model, device="cpu", _is_cute=False)
        pipe.offload()
        for p in pipe.model.parameters():
            assert p.device == torch.device("cpu")
