"""Tests for order-aware sequence combiner."""

import torch

from latentteleport.sequence_combiner import (
    SequenceCombinerTransformer,
    SequenceCombinerLatent,
    PositionalWeightedMean,
)


class TestSequenceCombinerTransformer:
    def test_forward_shape(self):
        model = SequenceCombinerTransformer(embed_dim=256, num_heads=4, num_layers=2)
        embs = torch.randn(2, 5, 256)
        out = model(embs)
        assert out.shape == (2, 256)

    def test_with_mask(self):
        model = SequenceCombinerTransformer(embed_dim=256, num_heads=4, num_layers=2)
        embs = torch.randn(2, 5, 256)
        mask = torch.tensor([[True, True, True, False, False],
                             [True, True, True, True, False]])
        out = model(embs, mask)
        assert out.shape == (2, 256)

    def test_order_matters(self):
        model = SequenceCombinerTransformer(embed_dim=128, num_heads=4, num_layers=2)
        a = torch.randn(1, 1, 128)
        b = torch.randn(1, 1, 128)
        out_ab = model(torch.cat([a, b], dim=1))
        out_ba = model(torch.cat([b, a], dim=1))
        # Outputs should differ because of positional encoding
        assert not torch.allclose(out_ab, out_ba, atol=1e-5)

    def test_single_unit(self):
        model = SequenceCombinerTransformer(embed_dim=128, num_heads=4, num_layers=2)
        embs = torch.randn(1, 1, 128)
        out = model(embs)
        assert out.shape == (1, 128)


class TestSequenceCombinerLatent:
    def test_forward_shape(self):
        model = SequenceCombinerLatent(
            latent_channels=16, latent_spatial=64*64, clip_dim=256,
            hidden_dim=128, num_heads=4,
        )
        latents = torch.randn(2, 3, 16, 64, 64)
        text_embs = torch.randn(2, 3, 256)
        out = model(latents, text_embs)
        assert out.shape == (2, 16, 64, 64)

    def test_with_mask(self):
        model = SequenceCombinerLatent(
            latent_channels=16, latent_spatial=64*64, clip_dim=256,
            hidden_dim=128, num_heads=4,
        )
        latents = torch.randn(2, 3, 16, 64, 64)
        text_embs = torch.randn(2, 3, 256)
        mask = torch.tensor([[True, True, False], [True, True, True]])
        out = model(latents, text_embs, mask)
        assert out.shape == (2, 16, 64, 64)


class TestPositionalWeightedMean:
    def test_forward_shape(self):
        model = PositionalWeightedMean(embed_dim=256, max_units=32)
        embs = torch.randn(2, 5, 256)
        out = model(embs)
        assert out.shape == (2, 256)

    def test_single_input(self):
        model = PositionalWeightedMean(embed_dim=128)
        embs = torch.randn(1, 1, 128)
        out = model(embs)
        assert out.shape == (1, 128)
