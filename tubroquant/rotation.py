"""Random orthogonal rotations for TurboQuant-style quantization."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _fwht(x: torch.Tensor) -> torch.Tensor:
    dim = x.shape[-1]
    if dim & (dim - 1):
        raise ValueError(f"FWHT requires a power-of-two dim, got {dim}")

    y = x
    h = 1
    while h < dim:
        y = y.reshape(*y.shape[:-1], -1, 2 * h)
        a = y[..., :h]
        b = y[..., h:]
        y = torch.cat((a + b, a - b), dim=-1)
        y = y.reshape(*x.shape[:-1], dim)
        h *= 2
    return y


class HadamardRotation(nn.Module):
    """Structured orthogonal rotation using random signs, permutation, and FWHT."""

    def __init__(self, dim: int, seed: int = 0):
        super().__init__()
        if dim & (dim - 1):
            raise ValueError(f"Hadamard rotation requires a power-of-two dim, got {dim}")
        generator = torch.Generator().manual_seed(seed)
        signs = torch.where(torch.rand(dim, generator=generator) > 0.5, 1.0, -1.0)
        perm = torch.randperm(dim, generator=generator)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(dim)

        self.dim = dim
        self.scale = dim ** -0.5
        self.register_buffer("signs", signs, persistent=False)
        self.register_buffer("perm", perm, persistent=False)
        self.register_buffer("inv_perm", inv_perm, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x[..., self.perm] * self.signs.to(dtype=x.dtype)
        return _fwht(y) * self.scale

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        y = _fwht(x) * self.scale
        y = y * self.signs.to(dtype=x.dtype)
        return y[..., self.inv_perm]


class DenseRotation(nn.Module):
    """Dense QR-based random orthogonal rotation."""

    def __init__(self, dim: int, seed: int = 0):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        q, _ = torch.linalg.qr(torch.randn(dim, dim, generator=generator))
        self.register_buffer("matrix", q, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.matrix.t().to(dtype=x.dtype)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.matrix.to(dtype=x.dtype)


def build_rotation(dim: int, kind: str = "hadamard", seed: int = 0) -> nn.Module:
    """Construct a rotation module."""
    if kind == "hadamard":
        if dim & (dim - 1):
            raise ValueError(f"hadamard rotation requires power-of-two dim, got {dim}")
        return HadamardRotation(dim, seed=seed)
    if kind == "qr":
        return DenseRotation(dim, seed=seed)
    raise ValueError(f"Unknown rotation kind: {kind}")
