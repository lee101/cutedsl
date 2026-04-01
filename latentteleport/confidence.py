"""Confidence-gated adaptive refinement inspired by CG-Taylor.

Instead of binary skip/compute, maps confidence score to number of refinement
steps needed. High confidence (cached latent is close to target) -> fewer steps.
Low confidence -> more steps.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class ConfidenceConfig:
    max_refinement_steps: int = 15
    min_refinement_steps: int = 1
    base_threshold: float = 0.15
    cache_window: int = 5
    # Taylor expansion order for predicting quality improvement
    taylor_order: int = 2
    # Adaptive threshold bounds
    threshold_low: float = 0.08
    threshold_high: float = 0.25
    # How quickly to adapt threshold
    adapt_rate: float = 0.1


class ConfidenceGate:
    """Predicts how many refinement steps a cached latent needs.

    Uses Taylor-style derivative tracking of prediction errors to estimate
    when the denoising trajectory has converged enough.
    """

    def __init__(self, config: ConfidenceConfig | None = None):
        self.config = config or ConfidenceConfig()
        self._error_history: deque[float] = deque(maxlen=self.config.cache_window)
        self._step_history: deque[int] = deque(maxlen=self.config.cache_window)
        self._threshold = self.config.base_threshold
        self._stats = {"total_calls": 0, "total_steps_saved": 0, "avg_confidence": 0.0}

    def estimate_steps(
        self,
        cached_latent: torch.Tensor,
        target_latent: torch.Tensor | None = None,
        text_similarity: float | None = None,
    ) -> int:
        """Estimate refinement steps needed for a cached/combined latent.

        Args:
            cached_latent: The teleported/combined latent
            target_latent: Optional reference latent (from full generation) for calibration
            text_similarity: Cosine similarity between prompt and cached unit (0-1)

        Returns:
            Number of refinement steps to run
        """
        self._stats["total_calls"] += 1
        confidence = self._compute_confidence(cached_latent, target_latent, text_similarity)

        # Map confidence [0,1] -> steps [min, max]
        # High confidence -> fewer steps
        steps = self.config.min_refinement_steps + int(
            (1.0 - confidence) * (self.config.max_refinement_steps - self.config.min_refinement_steps)
        )
        steps = max(self.config.min_refinement_steps, min(self.config.max_refinement_steps, steps))

        steps_saved = self.config.max_refinement_steps - steps
        self._stats["total_steps_saved"] += steps_saved
        self._stats["avg_confidence"] = (
            self._stats["avg_confidence"] * 0.95 + confidence * 0.05
        )

        self._step_history.append(steps)
        return steps

    def _compute_confidence(
        self,
        cached_latent: torch.Tensor,
        target_latent: torch.Tensor | None,
        text_similarity: float | None,
    ) -> float:
        """Compute confidence score in [0, 1]."""
        scores = []

        # 1. If we have a reference, direct comparison
        if target_latent is not None:
            error = torch.mean(torch.abs(cached_latent.float() - target_latent.float())).item()
            self._error_history.append(error)
            self._adapt_threshold(error)
            error_score = max(0.0, 1.0 - error / (self._threshold * 3))
            scores.append(error_score * 0.6)

        # 2. Text similarity as proxy for cache quality
        if text_similarity is not None:
            scores.append(text_similarity * 0.3)

        # 3. Historical error trend (Taylor-inspired)
        if len(self._error_history) >= 2:
            errors = list(self._error_history)
            # First derivative: is error decreasing?
            de = errors[-1] - errors[-2]
            trend_score = 1.0 if de < 0 else max(0.0, 1.0 - de / self._threshold)
            scores.append(trend_score * 0.1)

            # Second derivative: is improvement accelerating?
            if len(errors) >= 3:
                d2e = errors[-1] - 2 * errors[-2] + errors[-3]
                accel_score = 1.0 if d2e < 0 else 0.5
                scores.append(accel_score * 0.05)

        # 4. Latent statistics as quality proxy
        lat_std = cached_latent.float().std().item()
        lat_mean = cached_latent.float().mean().abs().item()
        # Well-formed latents have reasonable variance
        stat_score = min(1.0, lat_std / 1.0) * min(1.0, 1.0 / (lat_mean + 0.1))
        scores.append(min(0.3, stat_score * 0.3))

        if not scores:
            return 0.5

        total_weight = sum(scores)
        return min(1.0, max(0.0, total_weight))

    def _adapt_threshold(self, error: float):
        """Adapt threshold based on recent errors."""
        rate = self.config.adapt_rate
        if error < self._threshold * 0.5:
            # Errors consistently low -> tighten threshold
            self._threshold = max(
                self.config.threshold_low,
                self._threshold * (1 - rate),
            )
        elif error > self._threshold * 1.5:
            # Errors high -> relax threshold
            self._threshold = min(
                self.config.threshold_high,
                self._threshold * (1 + rate),
            )

    def calibrate(
        self,
        cached_latents: list[torch.Tensor],
        target_latents: list[torch.Tensor],
    ):
        """Calibrate thresholds from a set of cached vs target pairs."""
        errors = []
        for c, t in zip(cached_latents, target_latents):
            err = torch.mean(torch.abs(c.float() - t.float())).item()
            errors.append(err)
            self._error_history.append(err)

        if errors:
            mean_err = sum(errors) / len(errors)
            # Set threshold at 1.5x mean error (allow slightly worse than average)
            self._threshold = max(
                self.config.threshold_low,
                min(self.config.threshold_high, mean_err * 1.5),
            )

    def get_stats(self) -> dict:
        avg_steps = (
            sum(self._step_history) / len(self._step_history)
            if self._step_history else self.config.max_refinement_steps
        )
        return {
            **self._stats,
            "current_threshold": self._threshold,
            "avg_steps": avg_steps,
            "avg_steps_saved_per_call": (
                self._stats["total_steps_saved"] / max(1, self._stats["total_calls"])
            ),
        }


class LearnedConfidenceGate(nn.Module):
    """Small network that predicts refinement steps from latent features.

    Trainable alternative to the heuristic ConfidenceGate. Input features:
    - Latent statistics (mean, std, norm per channel)
    - Text embedding similarity
    - Cache match quality
    """

    def __init__(self, latent_channels: int = 16, clip_dim: int = 2560, hidden: int = 128):
        super().__init__()
        # Latent features: per-channel mean/std/norm = 3 * channels
        # + text similarity scalar + cache similarity scalar
        input_dim = latent_channels * 3 + clip_dim + 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),  # output in [0, 1] = confidence
        )

    def forward(
        self,
        latent: torch.Tensor,
        text_emb: torch.Tensor,
        text_sim: torch.Tensor,
        cache_sim: torch.Tensor,
    ) -> torch.Tensor:
        B = latent.shape[0]
        flat = latent.reshape(B, latent.shape[1], -1)  # (B, C, H*W)
        ch_mean = flat.mean(-1)  # (B, C)
        ch_std = flat.std(-1)    # (B, C)
        ch_norm = flat.norm(dim=-1)  # (B, C)
        latent_feats = torch.cat([ch_mean, ch_std, ch_norm], dim=-1)  # (B, 3C)
        if text_emb.dim() == 3:
            text_emb = text_emb.mean(1)  # pool over sequence
        x = torch.cat([latent_feats, text_emb, text_sim.unsqueeze(-1), cache_sim.unsqueeze(-1)], dim=-1)
        return self.net(x).squeeze(-1)
