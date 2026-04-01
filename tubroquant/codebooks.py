"""Lloyd-Max codebook generation for TurboQuant-style scalar quantizers."""

from __future__ import annotations

import functools
import math

import numpy as np
import torch


def _sphere_coordinate_pdf(x: np.ndarray, dim: int) -> np.ndarray:
    coef = math.exp(math.lgamma(dim / 2.0) - 0.5 * math.log(math.pi) - math.lgamma((dim - 1.0) / 2.0))
    power = max((dim - 3.0) / 2.0, 0.0)
    base = np.clip(1.0 - x * x, 0.0, None)
    return coef * np.power(base, power, dtype=np.float64)


@functools.lru_cache(maxsize=None)
def _solve_codebook(dim: int, bits: int, grid_size: int = 32769) -> tuple[np.ndarray, np.ndarray]:
    if bits <= 0:
        return np.array([0.0], dtype=np.float64), np.empty(0, dtype=np.float64)

    levels = 1 << bits
    x = np.linspace(-1.0, 1.0, grid_size, dtype=np.float64)
    pdf = _sphere_coordinate_pdf(x, dim)
    weights = pdf / max(pdf.sum(), 1e-12)
    cdf = np.cumsum(weights)
    cdf /= cdf[-1]

    targets = (np.arange(levels, dtype=np.float64) + 0.5) / levels
    centroids = np.interp(targets, cdf, x)
    centroids = np.sort(centroids)

    for _ in range(256):
        boundaries = np.empty(levels + 1, dtype=np.float64)
        boundaries[0] = -1.0
        boundaries[-1] = 1.0
        boundaries[1:-1] = 0.5 * (centroids[:-1] + centroids[1:])

        assignments = np.searchsorted(boundaries[1:-1], x, side="right")
        updated = centroids.copy()
        for idx in range(levels):
            mask = assignments == idx
            bucket_mass = weights[mask].sum()
            if bucket_mass > 0:
                updated[idx] = float((weights[mask] * x[mask]).sum() / bucket_mass)

        if np.max(np.abs(updated - centroids)) < 1e-8:
            centroids = updated
            break
        centroids = updated

    boundaries = np.empty(levels + 1, dtype=np.float64)
    boundaries[0] = -1.0
    boundaries[-1] = 1.0
    boundaries[1:-1] = 0.5 * (centroids[:-1] + centroids[1:])
    return centroids.astype(np.float32), boundaries[1:-1].astype(np.float32)


def get_codebook(dim: int, bits: int, *, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the centroid codebook and bucket boundaries for `(dim, bits)`."""
    centroids, boundaries = _solve_codebook(dim, bits)
    codebook = torch.from_numpy(centroids)
    cuts = torch.from_numpy(boundaries)
    if device is not None:
        codebook = codebook.to(device=device)
        cuts = cuts.to(device=device)
    return codebook, cuts
