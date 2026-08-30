"""Tests for CuteZImage model components."""

import os

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
    _residual_rms_update,
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


class TestResidualRMSUpdate:
    def test_fallback_matches_existing_expression(self):
        torch.manual_seed(42)
        residual = torch.randn(2, 4, 8)
        branch = torch.randn(2, 4, 8)
        gate = torch.randn(2, 1, 8).tanh()
        norm = RMSNorm(8)

        expected = residual + gate * norm(branch)
        actual = _residual_rms_update(residual, branch, norm, gate)

        assert torch.equal(actual, expected)

    def test_fused_dispatch_is_inference_only(self, monkeypatch):
        residual = torch.randn(2, 4, 8)
        branch = torch.randn(2, 4, 8)
        gate = torch.randn(2, 1, 8)
        norm = RMSNorm(8)
        calls = []

        def fake_fused(res, br, weight, gate=None, eps=1e-5):
            calls.append((res, br, weight, gate, eps))
            return torch.full_like(res, 3.0)

        monkeypatch.setattr("cutezimage.model._get_fused_residual_rms", lambda: fake_fused)

        class _CudaLikeTensor(torch.Tensor):
            @property
            def is_cuda(self):
                return True

        residual_cuda_like = residual.as_subclass(_CudaLikeTensor)
        with torch.inference_mode():
            out = _residual_rms_update(residual_cuda_like, branch, norm, gate)

        assert len(calls) == 1
        assert torch.equal(out, torch.full_like(residual, 3.0))


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

    def test_uses_fused_ffn_helper_on_cuda_inputs(self, monkeypatch):
        ffn = SiLUGatedFFN(8, 16)
        x = torch.randn(2, 4, 8)
        calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []

        def fake_fused_ffn(inp, w1, w2, w3):
            calls.append((inp, w1, w2, w3))
            return torch.full_like(inp, 7.0)

        monkeypatch.setattr("cutezimage.model._get_fused_silu_gate_ffn", lambda: fake_fused_ffn)

        class _CudaLikeTensor(torch.Tensor):
            @property
            def is_cuda(self):
                return True

        x_cuda_like = x.as_subclass(_CudaLikeTensor)
        out = ffn(x_cuda_like)

        assert len(calls) == 1
        assert out.shape == x.shape
        assert torch.equal(out, torch.full_like(x, 7.0))

    def test_uses_hf_silu_gate_when_enabled_on_cuda_inputs(self, monkeypatch):
        ffn = SiLUGatedFFN(8, 16)
        x = torch.randn(2, 4, 8)
        calls: list[tuple[torch.Tensor, torch.Tensor]] = []

        def fake_hf_silu_gate(x1, x3):
            calls.append((x1, x3))
            return torch.zeros_like(x1)

        def fail_regular_gate():
            raise AssertionError("regular silu gate should not run when HF gate succeeds")

        monkeypatch.setenv("CUTEZIMAGE_USE_HF_ACTIVATION_KERNELS", "1")
        monkeypatch.setattr("cutezimage.model._get_fused_silu_gate_ffn", lambda: None)
        monkeypatch.setattr("cutezimage.model._hf_silu_gate", fake_hf_silu_gate)
        monkeypatch.setattr("cutezimage.model._get_silu_gate", fail_regular_gate)

        class _CudaLikeTensor(torch.Tensor):
            @property
            def is_cuda(self):
                return True

        x_cuda_like = x.as_subclass(_CudaLikeTensor)
        out = ffn(x_cuda_like)

        assert len(calls) == 1
        assert out.shape == x.shape
        assert torch.equal(out, torch.zeros_like(x))


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

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_fused_residual_block_matches_reference(self, monkeypatch):
        torch.manual_seed(42)
        block = CuteZImageTransformerBlock(
            layer_id=0, dim=256, n_heads=4, n_kv_heads=4, modulation=True,
        ).to(device="cuda", dtype=torch.bfloat16).eval()
        x = torch.randn(2, 16, 256, device="cuda", dtype=torch.bfloat16)
        adaln = torch.randn(
            2, min(256, ADALN_EMBED_DIM), device="cuda", dtype=torch.bfloat16,
        )

        with torch.inference_mode():
            monkeypatch.setenv("CUTEZIMAGE_FUSED_RESIDUAL", "0")
            reference = block(x, adaln_input=adaln)
            monkeypatch.setenv("CUTEZIMAGE_FUSED_RESIDUAL", "1")
            fused = block(x, adaln_input=adaln)

        error = (fused.float() - reference.float()).abs()
        assert error.max().item() < 0.05
        assert error.mean().item() < 0.003

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

    def test_fused_qkv_matches_unfused(self, monkeypatch):
        block = CuteZImageTransformerBlock(
            layer_id=0, dim=256, n_heads=4, n_kv_heads=4, modulation=True,
        ).eval()
        x = torch.randn(2, 16, 256)
        adaln = torch.randn(2, min(256, ADALN_EMBED_DIM))
        freqs_real = torch.randn(1, 16, 32)
        freqs_imag = torch.randn(1, 16, 32)
        freqs_cis = torch.complex(freqs_real, freqs_imag)

        monkeypatch.setenv("CUTEZIMAGE_FUSED_QKV", "0")
        with torch.no_grad():
            baseline = block(x, freqs_cis=freqs_cis, adaln_input=adaln)

        monkeypatch.setenv("CUTEZIMAGE_FUSED_QKV", "1")
        with torch.no_grad():
            fused = block(x, freqs_cis=freqs_cis, adaln_input=adaln)

        assert torch.equal(baseline, fused)


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

    def test_regional_compile_targets_only_repeated_blocks(self, small_model, monkeypatch):
        calls = []

        def fake_compile(function, *, mode, fullgraph):
            calls.append((function, mode, fullgraph))
            return function

        monkeypatch.setattr(torch, "compile", fake_compile)
        count = small_model.compile_repeated_blocks(mode="max-autotune")

        assert count == 4
        assert len(calls) == 4
        assert all(mode == "max-autotune" and fullgraph is False
                   for _, mode, fullgraph in calls)

    def test_regional_compile_mode_uses_default_inductor_mode(self, small_model, monkeypatch):
        calls = []

        def fake_compile(function, *, mode, fullgraph):
            calls.append((mode, fullgraph))
            return function

        monkeypatch.setattr(torch, "compile", fake_compile)
        result = CuteZImageTransformer._apply_compile(small_model, "regional")

        assert result is small_model
        assert calls == [("reduce-overhead", False)] * 4


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


@pytest.mark.skipif(
    not os.environ.get("CUTEZIMAGE_MODEL_ID"),
    reason="Set CUTEZIMAGE_MODEL_ID env var to run overlap tests (requires model download)",
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestDiffusersOverlap:
    """Validate CuteZImage output matches diffusers ZImageTransformer2DModel."""

    @pytest.fixture(scope="class")
    def models(self):
        """Load both diffusers and CuteZImage transformers."""
        model_id = os.environ["CUTEZIMAGE_MODEL_ID"]
        from diffusers import ZImagePipeline

        pipe = ZImagePipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        orig = pipe.transformer.to("cuda", torch.bfloat16).eval()
        cute = CuteZImageTransformer.from_diffusers(orig).to("cuda", torch.bfloat16).eval()
        return orig, cute, pipe

    def test_transformer_output_matches(self, models):
        """Transformer outputs should match within bfloat16 tolerance."""
        orig, cute, _ = models
        # Create synthetic inputs
        torch.manual_seed(42)
        cfg = cute.config
        x = [torch.randn(cfg.in_channels, 1, 128, 128, device="cuda", dtype=torch.bfloat16)]
        t = torch.tensor([0.5], device="cuda", dtype=torch.bfloat16)
        cap = [torch.randn(77, cfg.cap_feat_dim, device="cuda", dtype=torch.bfloat16)]

        with torch.no_grad():
            out_orig = orig(x, t, cap, return_dict=False)[0]
            out_cute = cute(x, t, cap, return_dict=False)[0]

        for o, c in zip(out_orig, out_cute):
            diff = (o.float() - c.float()).abs()
            max_err = diff.max().item()
            mean_err = diff.mean().item()
            assert max_err < 0.01, f"Max error {max_err} >= 0.01"
            assert mean_err < 0.001, f"Mean error {mean_err} >= 0.001"

    def test_pipeline_image_similarity(self, models):
        """Full pipeline images should have high SSIM and PSNR."""
        _, cute, pipe = models
        from cutezimage.image_metrics import compare_images, pil_to_tensor

        seed = 42
        gen = torch.Generator(device="cuda").manual_seed(seed)
        prompt = "a red apple on a wooden table"

        # Original pipeline
        gen.manual_seed(seed)
        orig_result = pipe(
            prompt=prompt,
            width=512,
            height=512,
            num_inference_steps=9,
            guidance_scale=0.0,
            generator=gen,
        )
        orig_img = orig_result.images[0]

        # Replace with CuteZImage
        pipe.transformer = cute
        gen.manual_seed(seed)
        cute_result = pipe(
            prompt=prompt,
            width=512,
            height=512,
            num_inference_steps=9,
            guidance_scale=0.0,
            generator=gen,
        )
        cute_img = cute_result.images[0]

        # Compare
        metrics = compare_images(pil_to_tensor(orig_img), pil_to_tensor(cute_img))
        assert metrics["ssim"] > 0.99, f"SSIM {metrics['ssim']} <= 0.99"
        assert metrics["psnr_db"] > 40.0, f"PSNR {metrics['psnr_db']} <= 40.0 dB"
        assert metrics["max_pixel_error"] < 5.0, f"Max pixel error {metrics['max_pixel_error']} >= 5"
