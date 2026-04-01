"""Order-aware embedding sequence combiner.

Takes an ordered sequence of visual unit embeddings and produces one target
embedding. Order matters: "cat on mat" != "mat on cat".

Two architectures:
1. Small cross-attention transformer (more expressive, trainable)
2. Positional-weighted mean (fast, no training needed)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceCombinerTransformer(nn.Module):
    """Small transformer that combines ordered visual unit embeddings into one.

    Input: (B, N, D) sequence of N visual unit embeddings, each D-dim
    Output: (B, D) single combined embedding

    Uses learned [CLS]-style query token to aggregate via cross-attention.
    """

    def __init__(
        self,
        embed_dim: int = 2560,
        num_heads: int = 8,
        num_layers: int = 2,
        max_units: int = 32,
        ffn_ratio: float = 2.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Positional encoding for unit order
        self.pos_embed = nn.Embedding(max_units, embed_dim)

        # Learnable query token for aggregation
        self.query_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Cross-attention layers: query attends to visual unit sequence
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(CrossAttentionBlock(embed_dim, num_heads, ffn_ratio))

        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            embeddings: (B, N, D) visual unit embeddings in order
            mask: (B, N) bool mask, True = valid position

        Returns:
            (B, D) combined embedding
        """
        B, N, D = embeddings.shape
        positions = torch.arange(N, device=embeddings.device)
        pos_enc = self.pos_embed(positions).unsqueeze(0).expand(B, -1, -1)
        kv = embeddings + pos_enc

        query = self.query_token.expand(B, -1, -1)  # (B, 1, D)

        for layer in self.layers:
            query = layer(query, kv, mask)

        return self.final_norm(query.squeeze(1))  # (B, D)


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_ratio: float = 2.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_ff = nn.LayerNorm(dim)
        hidden = int(dim * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, dim),
        )

    def forward(
        self, query: torch.Tensor, kv: torch.Tensor, mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.norm_q(query)
        k = v = self.norm_kv(kv)
        key_padding_mask = ~mask if mask is not None else None
        attn_out, _ = self.attn(q, k, v, key_padding_mask=key_padding_mask)
        query = query + attn_out
        query = query + self.ffn(self.norm_ff(query))
        return query


class SequenceCombinerLatent(nn.Module):
    """Combines ordered visual unit LATENTS (not just embeddings) into one.

    Takes N latent tensors + their text embeddings, produces one combined latent.
    Uses the SequenceCombinerTransformer for text guidance, then applies learned
    spatial mixing in latent space.
    """

    def __init__(
        self,
        latent_channels: int = 16,
        latent_spatial: int = 64 * 64,
        clip_dim: int = 2560,
        hidden_dim: int = 512,
        num_heads: int = 8,
        max_units: int = 16,
    ):
        super().__init__()
        self.text_combiner = SequenceCombinerTransformer(
            embed_dim=clip_dim, num_heads=num_heads, num_layers=2, max_units=max_units,
        )
        # Spatial attention weights per unit conditioned on combined text embedding
        self.spatial_weight_net = nn.Sequential(
            nn.Linear(clip_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_channels),
        )
        # Per-channel mixing bias
        self.channel_bias = nn.Parameter(torch.zeros(latent_channels))

    def forward(
        self,
        latents: torch.Tensor,
        text_embeddings: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            latents: (B, N, C, H, W) stacked visual unit latents
            text_embeddings: (B, N, D) corresponding text embeddings
            mask: (B, N) validity mask

        Returns:
            (B, C, H, W) combined latent
        """
        B, N, C, H, W = latents.shape

        # 1. Combine text embeddings to get guidance signal
        combined_text = self.text_combiner(text_embeddings, mask)  # (B, D)

        # 2. Compute per-channel attention weights
        channel_weights = self.spatial_weight_net(combined_text)  # (B, C)
        channel_weights = channel_weights + self.channel_bias
        channel_weights = torch.sigmoid(channel_weights)  # (B, C)

        # 3. Weighted combination of latents
        # Simple approach: mean across units with channel-wise modulation
        if mask is not None:
            mask_expanded = mask.float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B, N, 1, 1, 1)
            latents_masked = latents * mask_expanded
            combined = latents_masked.sum(1) / mask_expanded.sum(1).clamp(min=1)
        else:
            combined = latents.mean(1)  # (B, C, H, W)

        # 4. Apply channel-wise modulation
        combined = combined * channel_weights.unsqueeze(-1).unsqueeze(-1)

        return combined


class PositionalWeightedMean(nn.Module):
    """Fast, lightweight combiner: positional-weighted mean of embeddings.

    No cross-attention, just learned position-dependent weights.
    Good baseline and very fast at inference.
    """

    def __init__(self, embed_dim: int = 2560, max_units: int = 32):
        super().__init__()
        self.position_weights = nn.Parameter(torch.ones(max_units) / max_units)
        self.embed_dim = embed_dim

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, N, D = embeddings.shape
        w = self.position_weights[:N].softmax(0)  # (N,)
        if mask is not None:
            w = w.unsqueeze(0) * mask.float()  # (B, N)
            w = w / w.sum(-1, keepdim=True).clamp(min=1e-8)
        else:
            w = w.unsqueeze(0).expand(B, -1)  # (B, N)
        return (embeddings * w.unsqueeze(-1)).sum(1)  # (B, D)
