"""Tests for predict_df, cross_learning, and batch_size chunking."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from cutechronos.model import CuteChronos2Config, CuteChronos2Model
from cutechronos.pipeline import CuteChronos2Pipeline
from cutechronos.tests.conftest import _BASE_CONFIG_KWARGS


@pytest.fixture(scope="module")
def pipe():
    torch.manual_seed(42)
    config = CuteChronos2Config(**_BASE_CONFIG_KWARGS)
    model = CuteChronos2Model(config)
    model.eval()
    return CuteChronos2Pipeline(model, device="cpu", _is_cute=True)


def _make_df(n_series: int = 3, length: int = 64) -> pd.DataFrame:
    np.random.seed(42)
    rows = []
    for i in range(n_series):
        ts = pd.date_range("2024-01-01", periods=length, freq="h")
        vals = np.random.randn(length).cumsum() + 100
        for t, v in zip(ts, vals):
            rows.append({"item_id": f"series_{i}", "timestamp": t, "target": v})
    return pd.DataFrame(rows)


class TestCrossLearning:
    def test_cross_learning_changes_output(self, pipe):
        ctx = torch.randn(3, 64)
        pred_normal = pipe.predict(ctx, prediction_length=16, cross_learning=False)
        pred_cross = pipe.predict(ctx, prediction_length=16, cross_learning=True)
        assert len(pred_normal) == 3
        assert len(pred_cross) == 3
        combined_normal = torch.cat(pred_normal, dim=0)
        combined_cross = torch.cat(pred_cross, dim=0)
        assert combined_normal.shape == combined_cross.shape

    def test_cross_learning_single_series(self, pipe):
        ctx = torch.randn(1, 64)
        pred_normal = pipe.predict(ctx, prediction_length=16, cross_learning=False)
        pred_cross = pipe.predict(ctx, prediction_length=16, cross_learning=True)
        torch.testing.assert_close(pred_normal[0], pred_cross[0])


class TestBatchSizeChunking:
    def test_batch_size_chunking_matches_full(self, pipe):
        ctx = torch.randn(6, 64)
        pred_full = pipe.predict(ctx, prediction_length=16)
        pred_chunked = pipe.predict(ctx, prediction_length=16, batch_size=2)
        assert len(pred_full) == len(pred_chunked) == 6
        for a, b in zip(pred_full, pred_chunked):
            torch.testing.assert_close(a, b)

    def test_batch_size_larger_than_input(self, pipe):
        ctx = torch.randn(3, 64)
        pred = pipe.predict(ctx, prediction_length=16, batch_size=100)
        assert len(pred) == 3

    def test_batch_size_equals_input(self, pipe):
        ctx = torch.randn(4, 64)
        pred_full = pipe.predict(ctx, prediction_length=16)
        pred_eq = pipe.predict(ctx, prediction_length=16, batch_size=4)
        for a, b in zip(pred_full, pred_eq):
            torch.testing.assert_close(a, b)

    def test_batch_size_one(self, pipe):
        ctx = torch.randn(3, 64)
        pred_full = pipe.predict(ctx, prediction_length=16)
        pred_one = pipe.predict(ctx, prediction_length=16, batch_size=1)
        assert len(pred_one) == 3
        for a, b in zip(pred_full, pred_one):
            torch.testing.assert_close(a, b)


class TestPredictDf:
    def test_basic_output(self, pipe):
        df = _make_df(n_series=2, length=64)
        pred_len = 16
        result = pipe.predict_df(df, prediction_length=pred_len)
        assert isinstance(result, pd.DataFrame)
        assert "item_id" in result.columns
        assert "step" in result.columns
        assert len(result) == 2 * pred_len

    def test_quantile_columns(self, pipe):
        df = _make_df(n_series=1, length=64)
        qs = [0.1, 0.5, 0.9]
        result = pipe.predict_df(df, prediction_length=16, quantile_levels=qs)
        for q in qs:
            assert str(q) in result.columns

    def test_default_quantile_levels(self, pipe):
        df = _make_df(n_series=1, length=64)
        result = pipe.predict_df(df, prediction_length=16)
        for q in pipe.quantiles:
            assert str(q) in result.columns

    def test_item_ids_preserved(self, pipe):
        df = _make_df(n_series=3, length=64)
        result = pipe.predict_df(df, prediction_length=16)
        assert set(result["item_id"]) == {"series_0", "series_1", "series_2"}

    def test_cross_learning_flag(self, pipe):
        df = _make_df(n_series=2, length=64)
        result_normal = pipe.predict_df(df, prediction_length=16, cross_learning=False)
        result_cross = pipe.predict_df(df, prediction_length=16, cross_learning=True)
        assert result_normal.shape == result_cross.shape

    def test_batch_size_param(self, pipe):
        df = _make_df(n_series=4, length=64)
        result = pipe.predict_df(df, prediction_length=16, batch_size=2)
        assert len(result) == 4 * 16

    def test_no_nan_in_output(self, pipe):
        df = _make_df(n_series=2, length=128)
        result = pipe.predict_df(df, prediction_length=16)
        numeric_cols = [c for c in result.columns if c not in ("item_id", "step")]
        assert not result[numeric_cols].isna().any().any()

    def test_custom_column_names(self, pipe):
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "sid": ["a"] * 64 + ["b"] * 64,
                "ts": list(pd.date_range("2024-01-01", periods=64, freq="h")) * 2,
                "value": np.random.randn(128).cumsum(),
            }
        )
        result = pipe.predict_df(
            df,
            id_column="sid",
            timestamp_column="ts",
            target="value",
            prediction_length=16,
        )
        assert "sid" in result.columns
        assert set(result["sid"]) == {"a", "b"}


class TestModelGroupIds:
    def test_forward_accepts_group_ids(self):
        torch.manual_seed(42)
        config = CuteChronos2Config(**_BASE_CONFIG_KWARGS)
        model = CuteChronos2Model(config)
        model.eval()
        ctx = torch.randn(3, 64)
        gids = torch.tensor([0, 0, 1], dtype=torch.long)
        with torch.inference_mode():
            out = model(ctx, group_ids=gids)
        assert out.shape[0] == 3

    def test_forward_none_group_ids_matches_default(self):
        torch.manual_seed(42)
        config = CuteChronos2Config(**_BASE_CONFIG_KWARGS)
        model = CuteChronos2Model(config)
        model.eval()
        ctx = torch.randn(2, 64)
        with torch.inference_mode():
            out_none = model(ctx, group_ids=None)
            gids = torch.arange(2, dtype=torch.long)
            out_explicit = model(ctx, group_ids=gids)
        torch.testing.assert_close(out_none, out_explicit)
