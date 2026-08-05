"""Small learned residual head for cache-guided latent teleportation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file


@dataclass
class CacheResidualAdapterConfig:
    latent_channels: int = 16
    hidden_channels: int = 32
    condition_dim: int = 6
    num_blocks: int = 2


class SpatialMixerBlock(nn.Module):
    """Depthwise spatial mixing followed by inexpensive channel mixing."""

    def __init__(self, channels: int):
        super().__init__()
        groups = min(8, channels)
        self.norm = nn.GroupNorm(groups, channels)
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)
        self.activation = nn.SiLU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.norm(value)
        residual = self.depthwise(residual)
        residual = self.activation(residual)
        residual = self.pointwise(residual)
        return value + residual


class CacheResidualAdapter(nn.Module):
    """Fuse local motion with the mean and dispersion of cached residuals.

    The zero-initialized output projection makes the untrained network exactly
    equal to its supplied base move.  In the retained architecture that base is
    the fitted scalar local-plus-cache gate, so training can only add a spatial
    correction on top of the best inexpensive retrieval rule.
    """

    def __init__(self, config: CacheResidualAdapterConfig | None = None):
        super().__init__()
        self.config = config or CacheResidualAdapterConfig()
        input_channels = 3 * self.config.latent_channels
        hidden = self.config.hidden_channels
        self.input_projection = nn.Conv2d(input_channels, hidden, kernel_size=1)
        self.condition = nn.Sequential(
            nn.Linear(self.config.condition_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * hidden),
        )
        self.blocks = nn.ModuleList(
            [SpatialMixerBlock(hidden) for _ in range(self.config.num_blocks)]
        )
        self.output_projection = nn.Conv2d(
            hidden,
            self.config.latent_channels,
            kernel_size=1,
        )
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        base_move: torch.Tensor,
        retrieved_residual: torch.Tensor,
        residual_std: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat(
            (base_move, retrieved_residual, residual_std), dim=1
        )
        hidden = self.input_projection(features)
        scale, shift = self.condition(condition.float()).chunk(2, dim=1)
        hidden = hidden * (1.0 + scale[:, :, None, None])
        hidden = hidden + shift[:, :, None, None]
        for block in self.blocks:
            hidden = block(hidden)
        correction = self.output_projection(torch.nn.functional.silu(hidden))
        return base_move + correction

    def config_dict(self) -> dict:
        return asdict(self.config)


def weighted_residual_statistics(
    residuals: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute weighted mean and standard deviation over k neighbours."""
    if residuals.ndim != 5:
        raise ValueError("residuals must have shape [batch, k, channels, height, width]")
    if weights.shape != residuals.shape[:2]:
        raise ValueError("weights must have shape [batch, k]")
    normalized = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    expanded = normalized[:, :, None, None, None]
    mean = (residuals * expanded).sum(dim=1)
    variance = ((residuals - mean[:, None]).square() * expanded).sum(dim=1)
    return mean, variance.clamp_min(0.0).sqrt()


def adapter_condition(
    step: int,
    horizon: int,
    momentum_scale: float,
    weights: torch.Tensor,
    *,
    total_steps: int,
    max_horizon: int,
    gate_local_weight: float = 1.0,
    gate_residual_weight: float = 0.0,
) -> torch.Tensor:
    """Build a compact schedule and neighbour-confidence condition."""
    normalized = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    entropy = -(normalized * normalized.clamp_min(1e-8).log()).sum(dim=1)
    max_entropy = torch.log(
        torch.tensor(float(weights.shape[1]), device=weights.device)
    ).clamp_min(1e-8)
    confidence = 1.0 - entropy / max_entropy
    constants = torch.tensor(
        [
            step / max(total_steps - 1, 1),
            horizon / max(max_horizon, 1),
            momentum_scale,
            gate_local_weight,
            gate_residual_weight,
        ],
        dtype=torch.float32,
        device=weights.device,
    )
    constants = constants[None].expand(weights.shape[0], -1)
    return torch.cat((constants, confidence[:, None]), dim=1)


def load_cache_adapter(
    directory: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[CacheResidualAdapter, dict[tuple[int, int], dict[str, float]]]:
    """Load a published adapter and its schedule-specific scalar table."""
    root = Path(directory)
    raw_config = json.loads((root / "config.json").read_text())
    model = CacheResidualAdapter(
        CacheResidualAdapterConfig(**raw_config["model"])
    ).to(device)
    model.load_state_dict(
        load_file(root / "cache_adapter.safetensors", device=str(device))
    )
    model.eval()
    rows = json.loads((root / "schedule_coefficients.json").read_text())
    coefficients = {
        (int(row["step"]), int(row["horizon"])): {
            "momentum_scale": float(row["momentum_scale"]),
            "gate_local_weight": float(row["gate_local_weight"]),
            "gate_residual_weight": float(row["gate_residual_weight"]),
        }
        for row in rows
    }
    return model, coefficients
