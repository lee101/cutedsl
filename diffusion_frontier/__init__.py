"""Reproducible latency/quality evaluation for text-to-image pipelines."""

from .catalog import MODEL_CATALOG, ModelSpec, get_model
from .metrics import compare_images
from .preflight import inspect_model, system_snapshot

__all__ = [
    "MODEL_CATALOG",
    "ModelSpec",
    "compare_images",
    "get_model",
    "inspect_model",
    "system_snapshot",
]
