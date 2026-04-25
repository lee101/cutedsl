import torch

from cutechronos.kernels import _fallback_preprocess
from cutechronos.model import CuteChronos2Config, CuteChronos2Model


def _small_config() -> CuteChronos2Config:
    return CuteChronos2Config(
        d_model=64,
        d_kv=16,
        d_ff=128,
        num_layers=1,
        num_heads=4,
        dropout_rate=0.0,
        layer_norm_epsilon=1e-6,
        dense_act_fn="relu",
        rope_theta=10000.0,
        vocab_size=2,
        reg_token_id=1,
        context_length=32,
        input_patch_size=4,
        input_patch_stride=4,
        output_patch_size=4,
        quantiles=[0.1, 0.5, 0.9],
        use_reg_token=True,
        use_arcsinh=True,
    )


def test_prepare_patched_context_matches_fallback_preprocess() -> None:
    torch.manual_seed(0)
    model = CuteChronos2Model(_small_config()).eval()
    context = torch.randn(3, 18)
    context[0, :2] = float("nan")
    context[1, 5:9] = float("nan")

    patched, attn, (loc, scale) = model._prepare_patched_context(context)
    ref_patched, ref_attn, ref_loc, ref_scale = _fallback_preprocess(
        context,
        patch_size=model.config.input_patch_size,
        context_length=model.config.context_length,
        use_arcsinh=model.config.use_arcsinh,
    )

    assert torch.allclose(patched.float(), ref_patched.float(), atol=1e-5)
    assert torch.allclose(attn.float(), ref_attn.float(), atol=1e-5)
    assert torch.allclose(loc.float(), ref_loc.float(), atol=1e-5)
    assert torch.allclose(scale.float(), ref_scale.float(), atol=1e-5)


def test_prepare_patched_context_respects_explicit_mask() -> None:
    torch.manual_seed(0)
    model = CuteChronos2Model(_small_config()).eval()
    context = torch.randn(2, 16)
    context_mask = torch.ones_like(context)
    context_mask[:, :4] = 0.0

    patched, attn, (loc, scale) = model._prepare_patched_context(context, context_mask=context_mask)
    masked_context = torch.where(context_mask > 0, context, torch.full_like(context, float("nan")))
    ref_patched, ref_attn, ref_loc, ref_scale = _fallback_preprocess(
        masked_context,
        patch_size=model.config.input_patch_size,
        context_length=model.config.context_length,
        use_arcsinh=model.config.use_arcsinh,
    )

    assert torch.allclose(patched.float(), ref_patched.float(), atol=1e-5)
    assert torch.allclose(attn.float(), ref_attn.float(), atol=1e-5)
    assert torch.allclose(loc.float(), ref_loc.float(), atol=1e-5)
    assert torch.allclose(scale.float(), ref_scale.float(), atol=1e-5)
