"""Batch-size invariance: a series must forecast identically alone, in a batch
of 2, and in a large batch (trading-critical: the live worker mixes batch sizes).
"""
import json
import tempfile
from pathlib import Path

import pytest
import torch

from cutechronos.model import CuteChronos2Config, CuteChronos2Model
from cutechronos.pipeline import CuteChronos2Pipeline


def _small_pipeline() -> CuteChronos2Pipeline:
    torch.manual_seed(0)
    config = CuteChronos2Config(
        d_model=64, d_kv=16, d_ff=128, num_layers=2, num_heads=4,
        dropout_rate=0.0, layer_norm_epsilon=1e-6, dense_act_fn="relu",
        rope_theta=10000.0, vocab_size=2,
        context_length=64, input_patch_size=8, input_patch_stride=8,
        output_patch_size=8, use_reg_token=True, use_arcsinh=True,
    )
    model = CuteChronos2Model(config)
    model.eval()
    return CuteChronos2Pipeline(model.to(dtype=torch.float32), device="cpu", _is_cute=True)


def _series(seed: int, n: int = 64) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, generator=g).cumsum(0)


@pytest.mark.parametrize("batch", [2, 8, 32])
def test_solo_equals_batched(batch):
    pipe = _small_pipeline()
    target = _series(1)
    others = [_series(100 + i) for i in range(batch - 1)]

    solo = pipe.predict([target], prediction_length=8)[0]
    batched = pipe.predict([target] + others, prediction_length=8)[0]

    torch.testing.assert_close(solo, batched, rtol=1e-4, atol=1e-5)


def test_order_invariance():
    pipe = _small_pipeline()
    a, b = _series(1), _series(2)
    ab = pipe.predict([a, b], prediction_length=8)
    ba = pipe.predict([b, a], prediction_length=8)
    torch.testing.assert_close(ab[0], ba[1], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(ab[1], ba[0], rtol=1e-4, atol=1e-5)


def test_repeat_determinism():
    pipe = _small_pipeline()
    a = _series(3)
    p1 = pipe.predict([a], prediction_length=8)[0]
    p2 = pipe.predict([a], prediction_length=8)[0]
    torch.testing.assert_close(p1, p2, rtol=0.0, atol=0.0)


def test_cross_learning_changes_but_stays_finite():
    """cross_learning intentionally couples series; sanity that it runs and
    produces finite output for batch >= 2 (n=2 is the trading edge case)."""
    pipe = _small_pipeline()
    a, b = _series(4), _series(5)
    out = pipe.predict([a, b], prediction_length=8, cross_learning=True)
    assert len(out) == 2
    for t in out:
        assert torch.isfinite(t).all()
