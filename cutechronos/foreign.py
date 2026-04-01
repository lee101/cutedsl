"""Embedded-Python bridge for CuteChronos2 wrapper examples.

This module provides a tiny stable API for non-Python hosts:

- ``init_pipeline(...)`` loads either CuteChronos2 or upstream Chronos-2.
- ``predict_median(...)`` runs one forecast and returns the median series.
- ``destroy_pipeline(...)`` releases a handle from the in-process registry.

The C shared library in ``examples/chronos_wrappers`` embeds CPython and calls
these functions so C, Go, and Python/ctypes all hit the exact same inference
path.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from cutechronos.pipeline import CuteChronos2Pipeline

_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass
class _PipelineSession:
    backend: str
    device: str
    dtype_name: str
    pipeline: Any


_PIPELINES: dict[int, _PipelineSession] = {}
_NEXT_HANDLE = 1


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _maybe_add_local_chronos_checkout() -> None:
    env_path = os.environ.get("CHRONOS_FORECASTING_SRC")
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(_repo_root().parent / "stock" / "chronos-forecasting" / "src")

    for candidate in candidates:
        if candidate.exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    try:
        return _DTYPE_MAP[dtype_name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported dtype '{dtype_name}'. Expected one of {sorted(_DTYPE_MAP)}"
        ) from exc


def _resolve_device(device: str | None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _sync_if_needed(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def init_pipeline(
    model_id: str,
    backend: str = "cute",
    device: str | None = None,
    dtype_name: str = "bfloat16",
    compile_mode: str | None = None,
) -> int:
    """Load a pipeline and return a numeric handle."""
    global _NEXT_HANDLE

    device = _resolve_device(device)
    dtype = _resolve_dtype(dtype_name)

    if backend == "cute":
        pipeline = CuteChronos2Pipeline.from_pretrained(
            model_id,
            device=device,
            dtype=dtype,
            use_cute=True,
            compile_mode=compile_mode,
        )
    elif backend == "original":
        _maybe_add_local_chronos_checkout()
        from chronos.chronos2 import Chronos2Pipeline

        pipeline = Chronos2Pipeline.from_pretrained(model_id, dtype=dtype)
        pipeline.model = pipeline.model.to(device)
    else:
        raise ValueError(f"Unsupported backend '{backend}'. Expected 'cute' or 'original'.")

    handle = _NEXT_HANDLE
    _NEXT_HANDLE += 1
    _PIPELINES[handle] = _PipelineSession(
        backend=backend,
        device=device,
        dtype_name=dtype_name,
        pipeline=pipeline,
    )
    return handle


def destroy_pipeline(handle: int) -> bool:
    """Remove a pipeline handle from the registry."""
    session = _PIPELINES.pop(handle, None)
    if session is None:
        return False

    pipeline = session.pipeline
    if session.backend == "cute" and hasattr(pipeline, "offload"):
        try:
            pipeline.offload()
        except Exception:
            pass
    elif hasattr(pipeline, "model"):
        try:
            pipeline.model = pipeline.model.to("cpu")
        except Exception:
            pass
    return True


def predict_median(
    handle: int,
    context_values: list[float],
    prediction_length: int,
) -> tuple[list[float], float]:
    """Run one inference and return ``(forecast, latency_ms)``."""
    return predict_quantile(handle, context_values, prediction_length, 0.5)


def predict_quantile(
    handle: int,
    context_values: list[float],
    prediction_length: int,
    quantile_level: float,
) -> tuple[list[float], float]:
    """Run one inference and return ``(forecast, latency_ms)`` for one quantile."""
    session = _PIPELINES[handle]
    context = torch.tensor(context_values, dtype=torch.float32)

    _sync_if_needed(session.device)
    start = time.perf_counter()

    if session.backend == "cute":
        quantiles, _ = session.pipeline.predict_quantiles(
            context,
            prediction_length=prediction_length,
            quantile_levels=[quantile_level],
        )
    else:
        quantiles, _ = session.pipeline.predict_quantiles(
            [context],
            prediction_length=prediction_length,
            quantile_levels=[quantile_level],
        )

    forecast = quantiles[0][0, :, 0]

    _sync_if_needed(session.device)
    end = time.perf_counter()

    return forecast.tolist(), (end - start) * 1000.0


def available_backends() -> list[str]:
    return ["cute", "original"]


__all__ = [
    "available_backends",
    "destroy_pipeline",
    "init_pipeline",
    "predict_quantile",
    "predict_median",
]
