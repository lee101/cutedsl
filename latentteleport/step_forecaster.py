"""Neural diffusion-step forecaster for latent teleportation.

Predicts a latent delta or next latent from the current latent, timestep,
and optional pooled text embedding. This is the learned replacement for
heuristic Taylor-style finite differencing.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class StepForecasterConfig:
    latent_channels: int = 16
    hidden_channels: int = 64
    text_dim: int = 2560
    time_embed_dim: int = 128
    num_res_blocks: int = 4
    predict_mode: str = "delta"  # "delta" or "next"


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class LatentStepForecaster(nn.Module):
    def __init__(self, config: StepForecasterConfig | None = None):
        super().__init__()
        self.config = config or StepForecasterConfig()
        c = self.config.hidden_channels

        self.in_proj = nn.Conv2d(self.config.latent_channels, c, kernel_size=3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, self.config.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.config.time_embed_dim, c),
        )
        self.text_mlp = nn.Sequential(
            nn.Linear(self.config.text_dim, c),
            nn.SiLU(),
            nn.Linear(c, c),
        )
        self.blocks = nn.ModuleList([ResidualBlock(c) for _ in range(self.config.num_res_blocks)])
        self.out_proj = nn.Conv2d(c, self.config.latent_channels, kernel_size=3, padding=1)

    def forward(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        text_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.in_proj(latent)
        time_cond = self.time_mlp(timestep.view(-1, 1).float()).unsqueeze(-1).unsqueeze(-1)
        x = x + time_cond
        if text_embedding is not None:
            if text_embedding.dim() == 3:
                text_embedding = text_embedding.mean(dim=1)
            text_cond = self.text_mlp(text_embedding.float()).unsqueeze(-1).unsqueeze(-1)
            x = x + text_cond
        for block in self.blocks:
            x = block(x)
        pred = self.out_proj(F.silu(x))
        if self.config.predict_mode == "next":
            return pred
        return latent + pred

