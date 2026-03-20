"""Tests for CuteZImage model components."""

import pytest
import torch

from cutezimage.model import (
    ADALN_EMBED_DIM,
    CuteZImageConfig,
    CuteZImageTransformer,
    CuteZImageTransformerBlock,
    RopeEmbedder,
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
        adaln = torch.randn(2, min(256, ADALN_EMBED_DIM))
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
        c = torch.randn(2, min(256, ADALN_EMBED_DIM))
        out = layer(x, c)
        assert out.shape == (2, 16, 64)


class TestRopeEmbedder:
    def test_output_shape(self):
        embedder = RopeEmbedder(theta=256.0, axes_dims=[16, 24, 24], axes_lens=[256, 128, 128])
        ids = torch.tensor([[10, 5, 3], [20, 10, 7]], dtype=torch.int32)
        out = embedder(ids)
        # Output should have sum(axes_dims)//2 complex values = sum(d//2 for d in dims)
        # Each axis produces d//2 complex values, total = 8 + 12 + 12 = 32 complex
        assert out.shape == (2, 32)
        assert out.dtype == torch.complex64

    def test_deterministic(self):
        embedder = RopeEmbedder(theta=256.0, axes_dims=[16, 24, 24], axes_lens=[256, 128, 128])
        ids = torch.tensor([[10, 5, 3]], dtype=torch.int32)
        out1 = embedder(ids)
        out2 = embedder(ids)
        assert torch.equal(out1, out2)


class TestCuteZImageTransformerForward:
    """Test full forward pass of CuteZImageTransformer."""

    @pytest.fixture
    def small_model(self):
        config = CuteZImageConfig(
            patch_size=2,
            f_patch_size=1,
            in_channels=4,
            dim=128,
            n_layers=2,
            n_refiner_layers=1,
            n_heads=4,
            n_kv_heads=4,
            cap_feat_dim=64,
            rope_theta=256.0,
            axes_dims=[8, 12, 12],
            axes_lens=[256, 128, 128],
        )
        return CuteZImageTransformer(config)

    def test_forward_output_shape(self, small_model):
        model = small_model.eval()
        B, C, F, H, W = 1, 4, 1, 16, 16
        x = [torch.randn(C, F, H, W)]
        t = torch.tensor([0.5])
        cap_feats = [torch.randn(10, 64)]
        with torch.no_grad():
            out = model(x, t, cap_feats, return_dict=False)
        # Output is a list of (C, F, H, W) tensors
        assert len(out[0]) == 1
        assert out[0][0].shape == (C, F, H, W)

    def test_forward_batch(self, small_model):
        model = small_model.eval()
        C = 4
        x = [torch.randn(C, 1, 16, 16), torch.randn(C, 1, 16, 16)]
        t = torch.tensor([0.5, 0.3])
        cap_feats = [torch.randn(10, 64), torch.randn(8, 64)]
        with torch.no_grad():
            out = model(x, t, cap_feats, return_dict=False)
        assert len(out[0]) == 2
        for img in out[0]:
            assert img.shape == (C, 1, 16, 16)

    def test_forward_deterministic(self, small_model):
        model = small_model.eval()
        x = [torch.randn(4, 1, 16, 16)]
        t = torch.tensor([0.5])
        cap_feats = [torch.randn(10, 64)]
        with torch.no_grad():
            out1 = model(x, t, cap_feats, return_dict=False)
            out2 = model(x, t, cap_feats, return_dict=False)
        assert torch.equal(out1[0][0], out2[0][0])

    def test_forward_return_dict(self, small_model):
        model = small_model.eval()
        x = [torch.randn(4, 1, 16, 16)]
        t = torch.tensor([0.5])
        cap_feats = [torch.randn(10, 64)]
        with torch.no_grad():
            out = model(x, t, cap_feats, return_dict=True)
        assert "sample" in out
        assert len(out["sample"]) == 1


class TestADALNDim:
    """Verify ADALN_EMBED_DIM matches diffusers (256, not 3072)."""

    def test_constant_value(self):
        assert ADALN_EMBED_DIM == 256

    def test_modulation_input_dim(self):
        """For dim=3840 (production size), modulation input should be 256."""
        block = CuteZImageTransformerBlock(
            layer_id=0, dim=512, n_heads=4, n_kv_heads=4, modulation=True,
        )
        # min(512, 256) = 256
        assert block.adaLN_modulation[0].in_features == 256

    def test_main_layers_have_modulation(self):
        """Main layers should have modulation=True matching diffusers default."""
        config = CuteZImageConfig(
            dim=128, n_layers=2, n_refiner_layers=1, n_heads=4, n_kv_heads=4,
            cap_feat_dim=64, axes_dims=[8, 12, 12], axes_lens=[256, 128, 128],
        )
        model = CuteZImageTransformer(config)
        for layer in model.layers:
            assert layer.modulation is True, "Main layers must have modulation=True"

    def test_context_refiner_no_modulation(self):
        config = CuteZImageConfig(
            dim=128, n_layers=2, n_refiner_layers=1, n_heads=4, n_kv_heads=4,
            cap_feat_dim=64, axes_dims=[8, 12, 12], axes_lens=[256, 128, 128],
        )
        model = CuteZImageTransformer(config)
        for layer in model.context_refiner:
            assert layer.modulation is False

    def test_noise_refiner_has_modulation(self):
        config = CuteZImageConfig(
            dim=128, n_layers=2, n_refiner_layers=1, n_heads=4, n_kv_heads=4,
            cap_feat_dim=64, axes_dims=[8, 12, 12], axes_lens=[256, 128, 128],
        )
        model = CuteZImageTransformer(config)
        for layer in model.noise_refiner:
            assert layer.modulation is True


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
