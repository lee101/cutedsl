from __future__ import annotations

from argparse import Namespace

import pytest
import torch

from examples.multivariate_pair_eval import (
    build_partner_sets,
    predict_quantiles_with_options,
    resolve_context_lengths,
    resolve_cross_learning_options,
    resolve_strides,
    select_frontier,
)


def test_resolve_context_lengths_prefers_explicit_list() -> None:
    args = Namespace(context_length=128, context_lengths=[64, 128, -1])
    assert resolve_context_lengths(args) == [64, 128]


def test_resolve_strides_falls_back_to_single_stride() -> None:
    args = Namespace(stride=24, strides=None)
    assert resolve_strides(args) == [24]


def test_resolve_cross_learning_options_normalizes_and_dedupes() -> None:
    assert resolve_cross_learning_options(["off", "true", "1", "false"]) == [False, True]


def test_build_partner_sets_excludes_target_and_dedupes() -> None:
    partner_sets = build_partner_sets("BTCUSDT", ["ETHUSDT", "BTCUSDT", "ETHUSDT", "SOLUSDT"], [1, 2])
    assert partner_sets == [("ETHUSDT",), ("SOLUSDT",), ("ETHUSDT", "SOLUSDT")]


def test_select_frontier_keeps_latency_accuracy_tradeoff() -> None:
    rows = [
        {"avg_latency_ms": 5.0, "mean_mape_pct": 2.5, "name": "a"},
        {"avg_latency_ms": 6.0, "mean_mape_pct": 2.7, "name": "dominated"},
        {"avg_latency_ms": 8.0, "mean_mape_pct": 2.2, "name": "b"},
        {"avg_latency_ms": 9.0, "mean_mape_pct": 2.3, "name": "dominated2"},
    ]
    frontier = select_frontier(rows)
    assert [row["name"] for row in frontier] == ["a", "b"]


def test_predict_quantiles_with_options_raises_clear_error_when_cross_learning_unsupported() -> None:
    class LegacyPipeline:
        quantiles = [0.1, 0.5, 0.9]

        def predict_quantiles(self, context, prediction_length=None, quantile_levels=None):
            return [], []

        def predict(self, context, prediction_length=None, limit_prediction_length=False):
            return [torch.tensor([[[1.0, 2.0], [10.0, 20.0], [100.0, 200.0]]], dtype=torch.float32)]

    with pytest.raises(RuntimeError, match="does not support cross_learning"):
        predict_quantiles_with_options(
            LegacyPipeline(),
            torch.tensor([1.0, 2.0, 3.0]),
            prediction_length=2,
            quantile_levels=[0.5],
            cross_learning=True,
        )
