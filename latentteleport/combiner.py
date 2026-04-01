"""Combiners: merge cached latents via SLERP, neural network, or tree."""

from __future__ import annotations

import math
from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from latentteleport.config import CombinerConfig


class LatentCombiner(Protocol):
    def combine(
        self,
        latent_a: torch.Tensor,
        latent_b: torch.Tensor,
        emb_a: torch.Tensor | None = None,
        emb_b: torch.Tensor | None = None,
        t: float = 0.5,
    ) -> torch.Tensor: ...


# --- SLERP ---

def slerp(z0: torch.Tensor, z1: torch.Tensor, t: float, eps: float = 1e-8) -> torch.Tensor:
    """Spherical linear interpolation between two latent tensors."""
    flat0 = z0.reshape(-1).float()
    flat1 = z1.reshape(-1).float()
    n0 = flat0 / (flat0.norm() + eps)
    n1 = flat1 / (flat1.norm() + eps)
    dot = torch.clamp((n0 * n1).sum(), -1.0, 1.0)
    omega = torch.acos(dot)
    if omega.abs() < eps:
        return ((1 - t) * z0 + t * z1).to(z0.dtype)
    so = torch.sin(omega)
    w0 = torch.sin((1 - t) * omega) / so
    w1 = torch.sin(t * omega) / so
    result = w0 * z0.float() + w1 * z1.float()
    return result.to(z0.dtype)


class SLERPCombiner:
    def combine(
        self,
        latent_a: torch.Tensor,
        latent_b: torch.Tensor,
        emb_a: torch.Tensor | None = None,
        emb_b: torch.Tensor | None = None,
        t: float = 0.5,
    ) -> torch.Tensor:
        return slerp(latent_a, latent_b, t)


# --- Neural Combiner ---

class NeuralCombinerNet(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        clip_dim: int = 2560,
        hidden_dim: int = 1024,
        num_layers: int = 4,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        input_dim = latent_dim * 2 + clip_dim * 2
        layers = []
        dim = input_dim
        for i in range(num_layers - 1):
            out = hidden_dim if i < num_layers - 2 else latent_dim
            layers.extend([nn.Linear(dim, out), nn.SiLU()])
            dim = out
        layers.append(nn.Linear(dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        latent_a: torch.Tensor,
        latent_b: torch.Tensor,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
    ) -> torch.Tensor:
        shape = latent_a.shape
        flat_a = latent_a.reshape(latent_a.shape[0], -1)
        flat_b = latent_b.reshape(latent_b.shape[0], -1)
        x = torch.cat([flat_a, flat_b, emb_a, emb_b], dim=-1)
        out = self.net(x)
        return out.reshape(shape)


class NeuralCombiner:
    def __init__(self, net: NeuralCombinerNet):
        self.net = net

    @torch.no_grad()
    def combine(
        self,
        latent_a: torch.Tensor,
        latent_b: torch.Tensor,
        emb_a: torch.Tensor | None = None,
        emb_b: torch.Tensor | None = None,
        t: float = 0.5,
    ) -> torch.Tensor:
        if emb_a is None or emb_b is None:
            return slerp(latent_a, latent_b, t)
        squeezed = False
        if latent_a.dim() < 5:
            latent_a = latent_a.unsqueeze(0)
            latent_b = latent_b.unsqueeze(0)
            squeezed = True
        if emb_a.dim() == 1:
            emb_a = emb_a.unsqueeze(0)
            emb_b = emb_b.unsqueeze(0)
        result = self.net(latent_a, latent_b, emb_a, emb_b)
        return result.squeeze(0) if squeezed else result


# --- Tree Combiner ---

class TreeCombiner:
    """Recursively combine N visual units pairwise, ordered by similarity."""

    def __init__(self, base_combiner: LatentCombiner):
        self.base = base_combiner

    def combine_tree(
        self,
        latents: list[torch.Tensor],
        embeddings: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if len(latents) == 1:
            return latents[0]
        if embeddings and len(embeddings) == len(latents):
            order = self._similarity_order(embeddings)
            latents = [latents[i] for i in order]
            embeddings = [embeddings[i] for i in order]

        while len(latents) > 1:
            next_latents = []
            next_embs = []
            for i in range(0, len(latents), 2):
                if i + 1 < len(latents):
                    emb_a = embeddings[i] if embeddings else None
                    emb_b = embeddings[i + 1] if embeddings else None
                    combined = self.base.combine(latents[i], latents[i + 1], emb_a, emb_b)
                    next_latents.append(combined)
                    if emb_a is not None and emb_b is not None:
                        next_embs.append((emb_a + emb_b) / 2)
                else:
                    next_latents.append(latents[i])
                    if embeddings:
                        next_embs.append(embeddings[i])
            latents = next_latents
            embeddings = next_embs if next_embs else None
        return latents[0]

    @staticmethod
    def _similarity_order(embeddings: list[torch.Tensor]) -> list[int]:
        """Order embeddings so most similar pairs are adjacent."""
        n = len(embeddings)
        if n <= 2:
            return list(range(n))
        embs = torch.stack([e.float() for e in embeddings])
        embs = F.normalize(embs, dim=-1)
        sims = embs @ embs.T
        used = [False] * n
        order = [0]
        used[0] = True
        for _ in range(n - 1):
            last = order[-1]
            best_idx, best_sim = -1, -2.0
            for j in range(n):
                if not used[j] and sims[last, j] > best_sim:
                    best_sim = sims[last, j].item()
                    best_idx = j
            order.append(best_idx)
            used[best_idx] = True
        return order


def create_combiner(config: CombinerConfig) -> LatentCombiner | TreeCombiner:
    if config.method == "slerp":
        return SLERPCombiner()
    elif config.method == "neural":
        latent_dim = config.latent_channels * config.latent_h * config.latent_w
        net = NeuralCombinerNet(
            latent_dim=latent_dim,
            clip_dim=config.clip_dim,
            hidden_dim=config.neural_hidden_dim,
            num_layers=config.neural_num_layers,
        )
        return NeuralCombiner(net)
    elif config.method == "tree":
        base = SLERPCombiner()
        return TreeCombiner(base)
    raise ValueError(f"Unknown method: {config.method}")
