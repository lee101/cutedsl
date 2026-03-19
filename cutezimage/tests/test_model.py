"""Tests for CuteZImage model components."""

import pytest
import torch

from cutezimage.model import (
    CuteZImageConfig,
    CuteZImageTransformerBlock,
    SiLUGatedFFN,
    RMSNorm,
    TimestepEmbedder,
    FinalLayer,
)


class TestRMSNormModule:
    def test_output_shape(self):
        norm = RMSNorm(256)
        x = torch.randn(2, 16, 256)
        out = norm(x)
        assert out.shape == x.shape

    def test_deterministic(self):
        norm = RMSNorm(256)
        x = torch.randn(2, 16, 256)
        out1 = norm(x)
        out2 = norm(x)
        assert torch.equal(out1, out2)


class TestSiLUGatedFFN:
    def test_output_shape(self):
        ffn = SiLUGatedFFN(256, 512)
        x = torch.randn(2, 16, 256)
        out = ffn(x)
        assert out.shape == x.shape

    def test_parameter_count(self):
        ffn = SiLUGatedFFN(256, 512)
        # w1: 256*512, w2: 512*256, w3: 256*512 (no bias)
        expected = 256 * 512 * 3
        actual = sum(p.numel() for p in ffn.parameters())
        assert actual == expected


class TestTimestepEmbedder:
    def test_output_shape(self):
        embedder = TimestepEmbedder(3072, mid_size=1024)
        t = torch.tensor([0.5, 0.1])
        out = embedder(t)
        assert out.shape == (2, 3072)

    def test_different_timesteps(self):
        embedder = TimestepEmbedder(256, mid_size=128)
        t1 = torch.tensor([0.0])
        t2 = torch.tensor([1.0])
        out1 = embedder(t1)
        out2 = embedder(t2)
        assert not torch.equal(out1, out2)


class TestTransformerBlock:
    def test_output_shape(self):
        block = CuteZImageTransformerBlock(
            layer_id=0, dim=256, n_heads=4, n_kv_heads=4,
        )
        x = torch.randn(2, 16, 256)
        out = block(x)
        assert out.shape == x.shape

    def test_modulated_block(self):
        block = CuteZImageTransformerBlock(
            layer_id=0, dim=256, n_heads=4, n_kv_heads=4, modulation=True,
        )
        x = torch.randn(2, 16, 256)
        adaln = torch.randn(2, min(256, 3072))
        out = block(x, adaln_input=adaln)
        assert out.shape == x.shape

    def test_with_attention_mask(self):
        block = CuteZImageTransformerBlock(
            layer_id=0, dim=256, n_heads=4, n_kv_heads=4,
        )
        x = torch.randn(2, 16, 256)
        mask = torch.ones(2, 16)
        mask[:, 8:] = 0
        out = block(x, attn_mask=mask)
        assert out.shape == x.shape

    def test_gqa(self):
        """Test grouped-query attention (fewer KV heads)."""
        block = CuteZImageTransformerBlock(
            layer_id=0, dim=256, n_heads=8, n_kv_heads=4,
        )
        x = torch.randn(2, 16, 256)
        out = block(x)
        assert out.shape == x.shape

    def test_deterministic(self):
        block = CuteZImageTransformerBlock(
            layer_id=0, dim=256, n_heads=4, n_kv_heads=4,
        )
        block.eval()
        x = torch.randn(2, 16, 256)
        with torch.no_grad():
            out1 = block(x.clone())
            out2 = block(x.clone())
        assert torch.equal(out1, out2)


class TestFinalLayer:
    def test_output_shape(self):
        layer = FinalLayer(256, 64)
        x = torch.randn(2, 16, 256)
        c = torch.randn(2, min(256, 3072))
        out = layer(x, c)
        assert out.shape == (2, 16, 64)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestCUDA:
    def test_block_on_cuda(self):
        block = CuteZImageTransformerBlock(
            layer_id=0, dim=256, n_heads=4, n_kv_heads=4,
        ).cuda().to(torch.bfloat16).eval()
        x = torch.randn(2, 16, 256, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            out = block(x)
        assert out.shape == x.shape
        assert out.device.type == "cuda"

    def test_ffn_on_cuda(self):
        ffn = SiLUGatedFFN(256, 512).cuda().to(torch.bfloat16)
        x = torch.randn(2, 16, 256, device="cuda", dtype=torch.bfloat16)
        out = ffn(x)
        assert out.shape == x.shape
        assert out.device.type == "cuda"
