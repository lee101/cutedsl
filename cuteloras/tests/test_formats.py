import torch
from safetensors.torch import save_file

from cuteloras.formats import detect_lora_format, load_lora_factors


def test_detect_zimage_native():
    assert detect_lora_format(["diffusion_model.layers.0.attention.to_q.lora_A.weight"]) == "zimage-native"


def test_detect_peft():
    assert detect_lora_format(["base_model.model.layers.0.q_proj.lora_A.weight"]) == "peft"


def test_detect_diffusers():
    assert detect_lora_format(["transformer.transformer_blocks.0.attn.to_q.lora_A.weight"]) == "diffusers"


def test_detect_kohya():
    assert detect_lora_format(["lora_unet_down_blocks_0.lora_down.weight"]) == "kohya"


def test_load_zimage_factors(lora_file):
    factors = load_lora_factors(str(lora_file))
    assert factors.format == "zimage-native"
    assert "layers.0.attention.to_q" in factors.factors
    assert "layers.1.attention.to_k" in factors.factors
    a, b, alpha = factors.factors["layers.0.attention.to_q"]
    assert a.shape == (4, 32)
    assert b.shape == (32, 4)
    assert alpha == 1.0


def test_load_peft_factors(tmp_path):
    path = tmp_path / "peft.safetensors"
    save_file(
        {
            "base_model.model.layers.0.q_proj.lora_A.weight": torch.randn(4, 32),
            "base_model.model.layers.0.q_proj.lora_B.weight": torch.randn(32, 4),
        },
        str(path),
    )
    factors = load_lora_factors(str(path))
    assert factors.format == "peft"
    assert "layers.0.q_proj" in factors.factors


def test_kohya_alpha_scaling(tmp_path):
    path = tmp_path / "kohya.safetensors"
    save_file(
        {
            "lora_unet_mid_block.lora_down.weight": torch.randn(8, 32),
            "lora_unet_mid_block.lora_up.weight": torch.randn(32, 8),
            "lora_unet_mid_block.alpha": torch.tensor(4.0),
        },
        str(path),
    )
    factors = load_lora_factors(str(path))
    _, _, alpha_scale = factors.factors["mid.block"]
    assert alpha_scale == 0.5


def test_kohya_zimage_joint_qkv_and_projection_mapping(tmp_path):
    path = tmp_path / "zimage-kohya.safetensors"
    a = torch.randn(12, 32)
    b = torch.randn(96, 12)
    save_file(
        {
            "lora_unet_layers_2_attention_qkv.lora_down.weight": a,
            "lora_unet_layers_2_attention_qkv.lora_up.weight": b,
            "lora_unet_layers_2_attention_out.lora_down.weight": torch.randn(4, 32),
            "lora_unet_layers_2_attention_out.lora_up.weight": torch.randn(32, 4),
            "lora_unet_layers_2_feed_forward_w1.lora_down.weight": torch.randn(4, 32),
            "lora_unet_layers_2_feed_forward_w1.lora_up.weight": torch.randn(64, 4),
        },
        str(path),
    )

    factors = load_lora_factors(str(path)).factors

    assert "layers.2.attention.to_out.0" in factors
    assert "layers.2.feed_forward.w1" in factors
    for name in ("to_q", "to_k", "to_v"):
        qkv_a, qkv_b, scale = factors[f"layers.2.attention.{name}"]
        assert qkv_a.shape == (12, 32)
        assert qkv_b.shape == (32, 12)
        assert scale == 1.0
