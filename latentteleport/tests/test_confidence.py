"""Tests for confidence gating."""

import torch

from latentteleport.confidence import ConfidenceConfig, ConfidenceGate, LearnedConfidenceGate


class TestConfidenceGate:
    def test_estimate_steps_default(self):
        gate = ConfidenceGate()
        latent = torch.randn(16, 1, 64, 64)
        steps = gate.estimate_steps(latent)
        assert 1 <= steps <= 15

    def test_high_similarity_fewer_steps(self):
        gate = ConfidenceGate()
        latent = torch.randn(16, 1, 64, 64)
        steps_low = gate.estimate_steps(latent, text_similarity=0.2)
        gate2 = ConfidenceGate()
        steps_high = gate2.estimate_steps(latent, text_similarity=0.95)
        assert steps_high <= steps_low

    def test_with_target(self):
        gate = ConfidenceGate()
        cached = torch.randn(16, 1, 64, 64)
        target = cached + torch.randn_like(cached) * 0.01  # very close
        steps = gate.estimate_steps(cached, target_latent=target)
        assert steps < gate.config.max_refinement_steps

    def test_calibrate(self):
        gate = ConfidenceGate()
        cached = [torch.randn(16, 1, 64, 64) for _ in range(5)]
        targets = [c + torch.randn_like(c) * 0.1 for c in cached]
        gate.calibrate(cached, targets)
        stats = gate.get_stats()
        assert stats["current_threshold"] > 0

    def test_stats(self):
        gate = ConfidenceGate()
        for _ in range(3):
            gate.estimate_steps(torch.randn(16, 1, 64, 64))
        stats = gate.get_stats()
        assert stats["total_calls"] == 3


class TestLearnedConfidenceGate:
    def test_forward(self):
        net = LearnedConfidenceGate(latent_channels=16, clip_dim=2560)
        latent = torch.randn(2, 16, 64, 64)
        text_emb = torch.randn(2, 2560)
        text_sim = torch.tensor([0.8, 0.3])
        cache_sim = torch.tensor([0.9, 0.5])
        confidence = net(latent, text_emb, text_sim, cache_sim)
        assert confidence.shape == (2,)
        assert (confidence >= 0).all() and (confidence <= 1).all()
