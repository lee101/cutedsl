from types import SimpleNamespace

import torch

from cutezimage.model import CuteZImageConfig, CuteZImageTransformer, CuteZImageTransformerBlock
from zimageaccelerated.model import AcceleratedZImageTransformer, AcceleratedZImageTransformerBlock


def _make_freqs(seq_len: int, head_dim: int) -> torch.Tensor:
    return torch.complex(
        torch.randn(1, seq_len, head_dim // 2, dtype=torch.float32),
        torch.randn(1, seq_len, head_dim // 2, dtype=torch.float32),
    )


def _fake_diffusers_from_cute(model: CuteZImageTransformer):
    config = SimpleNamespace(
        all_patch_size=[model.config.patch_size],
        all_f_patch_size=[model.config.f_patch_size],
        in_channels=model.config.in_channels,
        dim=model.config.dim,
        n_layers=model.config.n_layers,
        n_refiner_layers=model.config.n_refiner_layers,
        n_heads=model.config.n_heads,
        n_kv_heads=model.config.n_kv_heads,
        norm_eps=model.config.norm_eps,
        qk_norm=model.config.qk_norm,
        cap_feat_dim=model.config.cap_feat_dim,
        rope_theta=model.config.rope_theta,
        t_scale=model.config.t_scale,
        axes_dims=list(model.config.axes_dims),
        axes_lens=list(model.config.axes_lens),
    )
    ps = f"{model.config.patch_size}-{model.config.f_patch_size}"
    sd = {
        f"all_x_embedder.{ps}.weight": model.x_embedder.weight.detach().clone(),
        f"all_x_embedder.{ps}.bias": model.x_embedder.bias.detach().clone(),
        "cap_embedder.0.weight": model.cap_embedder[0].weight.detach().clone(),
        "cap_embedder.1.weight": model.cap_embedder[1].weight.detach().clone(),
        "cap_embedder.1.bias": model.cap_embedder[1].bias.detach().clone(),
        "t_embedder.mlp.0.weight": model.t_embedder.mlp[0].weight.detach().clone(),
        "t_embedder.mlp.0.bias": model.t_embedder.mlp[0].bias.detach().clone(),
        "t_embedder.mlp.2.weight": model.t_embedder.mlp[2].weight.detach().clone(),
        "t_embedder.mlp.2.bias": model.t_embedder.mlp[2].bias.detach().clone(),
        "x_pad_token": model.x_pad_token.detach().clone(),
        "cap_pad_token": model.cap_pad_token.detach().clone(),
        f"all_final_layer.{ps}.linear.weight": model.final_layer.linear.weight.detach().clone(),
        f"all_final_layer.{ps}.linear.bias": model.final_layer.linear.bias.detach().clone(),
        f"all_final_layer.{ps}.adaLN_modulation.1.weight": model.final_layer.adaLN_modulation[1].weight.detach().clone(),
        f"all_final_layer.{ps}.adaLN_modulation.1.bias": model.final_layer.adaLN_modulation[1].bias.detach().clone(),
    }

    def add_blocks(prefix: str, blocks) -> None:
        for i, block in enumerate(blocks):
            p = f"{prefix}.{i}"
            sd[f"{p}.attention.to_q.weight"] = block.q_proj.weight.detach().clone()
            sd[f"{p}.attention.to_k.weight"] = block.k_proj.weight.detach().clone()
            sd[f"{p}.attention.to_v.weight"] = block.v_proj.weight.detach().clone()
            sd[f"{p}.attention.to_out.0.weight"] = block.o_proj.weight.detach().clone()
            if block.qk_norm:
                sd[f"{p}.attention.norm_q.weight"] = block.q_norm.weight.detach().clone()
                sd[f"{p}.attention.norm_k.weight"] = block.k_norm.weight.detach().clone()
            sd[f"{p}.feed_forward.w1.weight"] = block.feed_forward.w1.weight.detach().clone()
            sd[f"{p}.feed_forward.w2.weight"] = block.feed_forward.w2.weight.detach().clone()
            sd[f"{p}.feed_forward.w3.weight"] = block.feed_forward.w3.weight.detach().clone()
            sd[f"{p}.attention_norm1.weight"] = block.attention_norm1.weight.detach().clone()
            sd[f"{p}.attention_norm2.weight"] = block.attention_norm2.weight.detach().clone()
            sd[f"{p}.ffn_norm1.weight"] = block.ffn_norm1.weight.detach().clone()
            sd[f"{p}.ffn_norm2.weight"] = block.ffn_norm2.weight.detach().clone()
            if block.modulation:
                sd[f"{p}.adaLN_modulation.0.weight"] = block.adaLN_modulation[0].weight.detach().clone()
                sd[f"{p}.adaLN_modulation.0.bias"] = block.adaLN_modulation[0].bias.detach().clone()

    add_blocks("noise_refiner", model.noise_refiner)
    add_blocks("context_refiner", model.context_refiner)
    add_blocks("layers", model.layers)
    return SimpleNamespace(
        config=config,
        state_dict=lambda: sd,
        parameters=lambda: model.parameters(),
    )


def test_accelerated_block_matches_baseline_non_modulated():
    torch.manual_seed(0)
    baseline = CuteZImageTransformerBlock(
        layer_id=0,
        dim=256,
        n_heads=4,
        n_kv_heads=4,
        modulation=False,
    ).eval()
    accelerated = AcceleratedZImageTransformerBlock.from_cutezimage_block(baseline).eval()

    x = torch.randn(2, 16, 256)
    attn_mask = torch.ones(2, 16)
    freqs = _make_freqs(16, 64)

    with torch.no_grad():
        baseline_out = baseline(x, attn_mask=attn_mask, freqs_cis=freqs)
        accelerated_out = accelerated(x, attn_mask=attn_mask, freqs_cis=freqs)

    assert baseline_out.shape == accelerated_out.shape
    assert torch.allclose(baseline_out, accelerated_out, atol=1e-5, rtol=1e-5)


def test_accelerated_block_matches_baseline_modulated():
    torch.manual_seed(0)
    baseline = CuteZImageTransformerBlock(
        layer_id=0,
        dim=256,
        n_heads=4,
        n_kv_heads=4,
        modulation=True,
    ).eval()
    accelerated = AcceleratedZImageTransformerBlock.from_cutezimage_block(baseline).eval()

    x = torch.randn(2, 16, 256)
    attn_mask = torch.ones(2, 16)
    freqs = _make_freqs(16, 64)
    adaln_input = torch.randn(2, 256)

    with torch.no_grad():
        baseline_out = baseline(x, attn_mask=attn_mask, freqs_cis=freqs, adaln_input=adaln_input)
        accelerated_out = accelerated(x, attn_mask=attn_mask, freqs_cis=freqs, adaln_input=adaln_input)

    assert baseline_out.shape == accelerated_out.shape
    assert torch.allclose(baseline_out, accelerated_out, atol=1e-5, rtol=1e-5)


def test_accelerated_block_falls_back_for_gqa():
    torch.manual_seed(0)
    baseline = CuteZImageTransformerBlock(
        layer_id=0,
        dim=256,
        n_heads=8,
        n_kv_heads=4,
        modulation=False,
    ).eval()
    accelerated = AcceleratedZImageTransformerBlock.from_cutezimage_block(baseline).eval()

    x = torch.randn(1, 8, 256)
    with torch.no_grad():
        baseline_out = baseline(x)
        accelerated_out = accelerated(x)

    assert accelerated.use_fused_qkv is False
    assert torch.allclose(baseline_out, accelerated_out, atol=1e-5, rtol=1e-5)


def test_accelerated_transformer_matches_baseline():
    torch.manual_seed(0)
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
        axes_dims=[8, 12, 12],
        axes_lens=[256, 128, 128],
    )
    baseline = CuteZImageTransformer(config).eval()
    accelerated = AcceleratedZImageTransformer.from_cutezimage(baseline).eval()

    x = [torch.randn(4, 1, 16, 16)]
    t = torch.tensor([0.5])
    cap_feats = [torch.randn(10, 64)]

    with torch.no_grad():
        baseline_out = baseline(x, t, cap_feats, return_dict=False)[0][0]
        accelerated_out = accelerated(x, t, cap_feats, return_dict=False)[0][0]

    assert baseline_out.shape == accelerated_out.shape
    assert torch.allclose(baseline_out, accelerated_out, atol=1e-5, rtol=1e-5)


def test_accelerated_transformer_direct_from_diffusers_matches_two_step_path():
    torch.manual_seed(0)
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
        axes_dims=[8, 12, 12],
        axes_lens=[256, 128, 128],
    )
    cute = CuteZImageTransformer(config).eval()
    fake_diffusers = _fake_diffusers_from_cute(cute)

    direct = AcceleratedZImageTransformer.from_diffusers(fake_diffusers).eval()
    via_cute = AcceleratedZImageTransformer.from_cutezimage(cute).eval()

    for direct_param, via_cute_param in zip(direct.parameters(), via_cute.parameters()):
        assert torch.allclose(direct_param, via_cute_param)
