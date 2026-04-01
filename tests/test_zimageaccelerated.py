import torch

from cutezimage.model import CuteZImageConfig, CuteZImageTransformer, CuteZImageTransformerBlock
from zimageaccelerated.model import AcceleratedZImageTransformer, AcceleratedZImageTransformerBlock


def _make_freqs(seq_len: int, head_dim: int) -> torch.Tensor:
    return torch.complex(
        torch.randn(1, seq_len, head_dim // 2, dtype=torch.float32),
        torch.randn(1, seq_len, head_dim // 2, dtype=torch.float32),
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
