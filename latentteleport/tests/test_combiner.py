"""Tests for latent combiners."""

import torch

from latentteleport.combiner import (
    SLERPCombiner, NeuralCombinerNet, NeuralCombiner,
    TreeCombiner, slerp, create_combiner,
)
from latentteleport.config import CombinerConfig


class TestSLERP:
    def test_endpoints(self):
        z0 = torch.randn(16, 1, 64, 64)
        z1 = torch.randn(16, 1, 64, 64)
        assert torch.allclose(slerp(z0, z1, 0.0), z0, atol=1e-5)
        assert torch.allclose(slerp(z0, z1, 1.0), z1, atol=1e-5)

    def test_midpoint_shape(self):
        z0 = torch.randn(16, 1, 64, 64)
        z1 = torch.randn(16, 1, 64, 64)
        mid = slerp(z0, z1, 0.5)
        assert mid.shape == z0.shape

    def test_combiner_interface(self):
        c = SLERPCombiner()
        z0 = torch.randn(16, 1, 64, 64)
        z1 = torch.randn(16, 1, 64, 64)
        result = c.combine(z0, z1, t=0.3)
        assert result.shape == z0.shape


class TestNeuralCombiner:
    def test_forward_shape(self):
        net = NeuralCombinerNet(latent_dim=16 * 64 * 64, clip_dim=2560)
        z0 = torch.randn(2, 16, 1, 64, 64)
        z1 = torch.randn(2, 16, 1, 64, 64)
        e0 = torch.randn(2, 2560)
        e1 = torch.randn(2, 2560)
        out = net(z0, z1, e0, e1)
        assert out.shape == z0.shape

    def test_combiner_wrapper(self):
        net = NeuralCombinerNet(latent_dim=16 * 64 * 64, clip_dim=2560)
        c = NeuralCombiner(net)
        z0 = torch.randn(16, 1, 64, 64)
        z1 = torch.randn(16, 1, 64, 64)
        e0 = torch.randn(2560)
        e1 = torch.randn(2560)
        result = c.combine(z0, z1, e0, e1)
        assert result.shape == z0.shape


class TestTreeCombiner:
    def test_single(self):
        tc = TreeCombiner(SLERPCombiner())
        z = torch.randn(16, 1, 64, 64)
        result = tc.combine_tree([z])
        assert torch.allclose(result, z)

    def test_pair(self):
        tc = TreeCombiner(SLERPCombiner())
        z0 = torch.randn(16, 1, 64, 64)
        z1 = torch.randn(16, 1, 64, 64)
        result = tc.combine_tree([z0, z1])
        assert result.shape == z0.shape

    def test_many(self):
        tc = TreeCombiner(SLERPCombiner())
        latents = [torch.randn(16, 1, 64, 64) for _ in range(5)]
        embeddings = [torch.randn(2560) for _ in range(5)]
        result = tc.combine_tree(latents, embeddings)
        assert result.shape == latents[0].shape


class TestCreateCombiner:
    def test_slerp(self):
        c = create_combiner(CombinerConfig(method="slerp"))
        assert isinstance(c, SLERPCombiner)

    def test_tree(self):
        c = create_combiner(CombinerConfig(method="tree"))
        assert isinstance(c, TreeCombiner)

    def test_neural(self):
        c = create_combiner(CombinerConfig(method="neural"))
        assert isinstance(c, NeuralCombiner)
