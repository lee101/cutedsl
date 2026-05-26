from __future__ import annotations

import torch

from cutechronos.pipeline import CuteChronos2Pipeline, _left_pad_and_cat_2d


class _DummyPipeline:
    def __init__(self) -> None:
        self.quantiles = [0.1, 0.5, 0.9]
        self.calls: list[tuple[str, dict[str, object]]] = []

    def predict(self, context, prediction_length=None, limit_prediction_length=True, cross_learning=False, batch_size=None):
        self.calls.append(
            (
                "predict",
                {
                    "context": context,
                    "prediction_length": prediction_length,
                    "limit_prediction_length": limit_prediction_length,
                    "cross_learning": cross_learning,
                    "batch_size": batch_size,
                },
            )
        )
        prediction = torch.tensor(
            [[[1.0, 2.0], [10.0, 20.0], [100.0, 200.0]]],
            dtype=torch.float32,
        )
        return [prediction]


def test_predict_quantiles_forwards_cross_learning_and_batch_size() -> None:
    pipeline = object.__new__(CuteChronos2Pipeline)
    pipeline._get_config = lambda: type("Cfg", (), {"quantiles": [0.1, 0.5, 0.9]})()  # type: ignore[attr-defined]

    dummy = _DummyPipeline()
    pipeline.predict = dummy.predict  # type: ignore[method-assign]

    quantiles, mean = pipeline.predict_quantiles(
        torch.tensor([1.0, 2.0, 3.0]),
        prediction_length=2,
        quantile_levels=[0.5],
        cross_learning=True,
        batch_size=7,
    )

    assert len(dummy.calls) == 1
    name, kwargs = dummy.calls[0]
    assert name == "predict"
    assert torch.equal(kwargs["context"], torch.tensor([1.0, 2.0, 3.0]))
    assert kwargs["prediction_length"] == 2
    assert kwargs["limit_prediction_length"] is True
    assert kwargs["cross_learning"] is True
    assert kwargs["batch_size"] == 7
    assert quantiles[0].shape == (1, 2, 1)
    assert mean[0].shape == (1, 2)


def test_left_pad_and_stack_rows_matches_cat_reference() -> None:
    rows = [
        torch.tensor([1.0, 2.0]),
        torch.tensor([3.0, 4.0, 5.0, 6.0]),
        torch.tensor([7.0]),
    ]

    expected = torch.tensor(
        [
            [float("nan"), float("nan"), 1.0, 2.0],
            [3.0, 4.0, 5.0, 6.0],
            [float("nan"), float("nan"), float("nan"), 7.0],
        ]
    )

    actual = CuteChronos2Pipeline._left_pad_and_stack_rows(rows)
    torch.testing.assert_close(actual, expected, equal_nan=True)


def test_left_pad_and_cat_2d_matches_cat_reference() -> None:
    tensors = [
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        torch.tensor([[5.0, 6.0, 7.0]]),
        torch.tensor([[8.0]]),
    ]

    expected = torch.tensor(
        [
            [float("nan"), 1.0, 2.0],
            [float("nan"), 3.0, 4.0],
            [5.0, 6.0, 7.0],
            [float("nan"), float("nan"), 8.0],
        ]
    )

    actual = _left_pad_and_cat_2d(tensors)
    torch.testing.assert_close(actual, expected, equal_nan=True)
