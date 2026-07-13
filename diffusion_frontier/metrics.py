"""Paired image metrics used for same-model quality-retention checks."""

from __future__ import annotations

import math

import numpy as np
import torch
from PIL import Image

from cutezimage.image_metrics import compare_images as _compare_tensors


def _rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def compare_images(candidate: Image.Image, reference: Image.Image) -> dict:
    """Compare deterministic outputs at equal dimensions.

    These metrics measure retention relative to a same-model, same-seed
    reference. They must not be interpreted as absolute or cross-model quality.
    """
    if candidate.size != reference.size:
        candidate = candidate.resize(reference.size, Image.Resampling.LANCZOS)
    values = _compare_tensors(torch.from_numpy(_rgb_array(candidate)), torch.from_numpy(_rgb_array(reference)))
    # JSON has no portable representation for Infinity.
    if math.isinf(values["psnr_db"]):
        values["psnr_db"] = 99.0
    return values


def optional_lpips(candidate: Image.Image, reference: Image.Image) -> float | None:
    try:
        import lpips
    except ImportError:
        return None
    model = lpips.LPIPS(net="alex")
    a = torch.from_numpy(_rgb_array(candidate)).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1
    b = torch.from_numpy(_rgb_array(reference)).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1
    with torch.inference_mode():
        return float(model(a, b).item())
