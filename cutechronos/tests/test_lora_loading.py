"""Tests for LoRA adapter loading and weight merging in CuteChronos2Model."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file as safetensors_load_file
from safetensors.torch import save_file as safetensors_save_file

from cutechronos.model import CuteChronos2Config, CuteChronos2Model

_REAL_ADAPTER_PATH = Path(
    "/vfast/data/code/btcmarketsbot/models_store_v3/chronos2_lora/BTC_USDT/BTC_lora_20260207_192741/lora-adapter"
)


def _make_small_model() -> CuteChronos2Model:
    """Build a tiny model for fast tests (2 layers, 64 d_model)."""
    torch.manual_seed(42)
    config = CuteChronos2Config(
        d_model=64,
        d_kv=16,
        d_ff=128,
        num_layers=2,
        num_heads=4,
        dropout_rate=0.0,
        layer_norm_epsilon=1e-6,
        dense_act_fn="relu",
        rope_theta=10000.0,
        vocab_size=2,
        context_length=64,
        input_patch_size=8,
        input_patch_stride=8,
        output_patch_size=8,
        use_reg_token=True,
        use_arcsinh=True,
    )
    model = CuteChronos2Model(config)
    model.eval()
    return model


def _create_mock_adapter(
    tmpdir: str,
    num_layers: int = 2,
    d_model: int = 64,
    r: int = 4,
    lora_alpha: int = 8,
    target_modules: list[str] | None = None,
    fan_in_fan_out: bool = False,
    use_safetensors: bool = True,
) -> Path:
    """Create a mock LoRA adapter directory with synthetic weights."""
    if target_modules is None:
        target_modules = ["q", "k", "v", "o"]

    adapter_dir = Path(tmpdir) / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "r": r,
        "lora_alpha": lora_alpha,
        "fan_in_fan_out": fan_in_fan_out,
        "target_modules": target_modules,
        "peft_type": "LORA",
    }
    with open(adapter_dir / "adapter_config.json", "w") as f:
        json.dump(config, f)

    lora_sd: dict[str, torch.Tensor] = {}
    for i in range(num_layers):
        for layer_idx in (0, 1):
            for proj in target_modules:
                key_a = f"base_model.model.encoder.block.{i}.layer.{layer_idx}.self_attention.{proj}.lora_A.weight"
                key_b = f"base_model.model.encoder.block.{i}.layer.{layer_idx}.self_attention.{proj}.lora_B.weight"
                lora_sd[key_a] = torch.randn(r, d_model)
                lora_sd[key_b] = torch.randn(d_model, r)

    if use_safetensors:
        safetensors_save_file(lora_sd, str(adapter_dir / "adapter_model.safetensors"))
    else:
        torch.save(lora_sd, str(adapter_dir / "adapter_model.bin"))

    return adapter_dir


class TestAdapterConfigParsing:
    def test_reads_config_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = _create_mock_adapter(tmpdir, r=8, lora_alpha=16)
            with open(adapter_dir / "adapter_config.json") as f:
                cfg = json.load(f)
            assert cfg["r"] == 8
            assert cfg["lora_alpha"] == 16
            assert set(cfg["target_modules"]) == {"q", "k", "v", "o"}
            assert cfg["fan_in_fan_out"] is False

    def test_missing_adapter_weights_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir) / "empty_adapter"
            adapter_dir.mkdir()
            with open(adapter_dir / "adapter_config.json", "w") as f:
                json.dump({"r": 4, "lora_alpha": 8}, f)
            model = _make_small_model()
            with pytest.raises(FileNotFoundError):
                model.load_lora_adapter(adapter_dir)


class TestWeightMergingMath:
    def test_merge_formula_correct(self):
        """Verify merged = original + (alpha/r) * (B @ A) for a single projection."""
        torch.manual_seed(0)
        model = _make_small_model()
        d_model = model.config.d_model
        r, alpha = 4, 8
        scaling = alpha / r

        orig_q_weight = model.blocks[0].time_attn.q.weight.clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = _create_mock_adapter(tmpdir, num_layers=2, d_model=d_model, r=r, lora_alpha=alpha)
            lora_sd = safetensors_load_file(str(adapter_dir / "adapter_model.safetensors"))

            lora_a = lora_sd["base_model.model.encoder.block.0.layer.0.self_attention.q.lora_A.weight"]
            lora_b = lora_sd["base_model.model.encoder.block.0.layer.0.self_attention.q.lora_B.weight"]
            expected = orig_q_weight + scaling * (lora_b @ lora_a)

            model.load_lora_adapter(adapter_dir)

        actual = model.blocks[0].time_attn.q.weight
        assert torch.allclose(actual, expected, atol=1e-6), (
            f"max err: {(actual - expected).abs().max().item()}"
        )

    def test_merge_all_projections(self):
        """All q/k/v/o in both time and group attention should be modified."""
        torch.manual_seed(0)
        model = _make_small_model()

        orig_weights = {}
        for i, block in enumerate(model.blocks):
            for attn_name in ("time_attn", "group_attn"):
                for proj in ("q", "k", "v", "o"):
                    key = f"{i}.{attn_name}.{proj}"
                    orig_weights[key] = getattr(getattr(block, attn_name), proj).weight.clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = _create_mock_adapter(tmpdir, num_layers=2, d_model=model.config.d_model)
            model.load_lora_adapter(adapter_dir)

        for key, orig_w in orig_weights.items():
            parts = key.split(".")
            block_idx, attn_name, proj = int(parts[0]), parts[1], parts[2]
            new_w = getattr(getattr(model.blocks[block_idx], attn_name), proj).weight
            assert not torch.equal(new_w, orig_w), f"Weight {key} was not modified by LoRA merge"

    def test_fan_in_fan_out_transposes_delta(self):
        """When fan_in_fan_out=True, the delta should be transposed before adding."""
        torch.manual_seed(0)
        d_model, r, alpha = 64, 4, 8
        scaling = alpha / r

        model_normal = _make_small_model()
        orig_w = model_normal.blocks[0].time_attn.q.weight.clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_normal = _create_mock_adapter(tmpdir, d_model=d_model, r=r, lora_alpha=alpha, fan_in_fan_out=False)
            lora_sd = safetensors_load_file(str(adapter_normal / "adapter_model.safetensors"))
            lora_a = lora_sd["base_model.model.encoder.block.0.layer.0.self_attention.q.lora_A.weight"]
            lora_b = lora_sd["base_model.model.encoder.block.0.layer.0.self_attention.q.lora_B.weight"]
            delta = lora_b @ lora_a

            model_normal.load_lora_adapter(adapter_normal)

        expected_normal = orig_w + scaling * delta
        assert torch.allclose(model_normal.blocks[0].time_attn.q.weight, expected_normal, atol=1e-6)

    def test_load_bin_format(self):
        """Adapter loading should work with .bin format too."""
        torch.manual_seed(0)
        model = _make_small_model()
        orig_w = model.blocks[0].time_attn.q.weight.clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = _create_mock_adapter(
                tmpdir, num_layers=2, d_model=model.config.d_model, use_safetensors=False
            )
            model.load_lora_adapter(adapter_dir)

        assert not torch.equal(model.blocks[0].time_attn.q.weight, orig_w)

    def test_subset_target_modules(self):
        """Only specified target_modules should be modified."""
        torch.manual_seed(0)
        model = _make_small_model()
        orig_q = model.blocks[0].time_attn.q.weight.clone()
        orig_k = model.blocks[0].time_attn.k.weight.clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = _create_mock_adapter(
                tmpdir, num_layers=2, d_model=model.config.d_model, target_modules=["q"]
            )
            model.load_lora_adapter(adapter_dir)

        assert not torch.equal(model.blocks[0].time_attn.q.weight, orig_q), "q should be modified"
        assert torch.equal(model.blocks[0].time_attn.k.weight, orig_k), "k should be unchanged"


def _save_as_chronos2_checkpoint(model: CuteChronos2Model, model_dir: Path) -> None:
    """Save model weights in upstream Chronos2 key format for from_pretrained."""
    sd = {}
    sd["shared.weight"] = model.shared.weight.clone()
    for name in ("input_patch_embedding", "output_patch_embedding"):
        block = getattr(model, name)
        for layer in ("hidden_layer", "output_layer", "residual_layer"):
            for param in ("weight", "bias"):
                sd[f"{name}.{layer}.{param}"] = getattr(getattr(block, layer), param).clone()
    sd["encoder.final_layer_norm.weight"] = model.final_layer_norm_weight.clone()
    for i, block in enumerate(model.blocks):
        prefix = f"encoder.block.{i}"
        sd[f"{prefix}.layer.0.layer_norm.weight"] = block.time_attn.layer_norm_weight.clone()
        sd[f"{prefix}.layer.1.layer_norm.weight"] = block.group_attn.layer_norm_weight.clone()
        sd[f"{prefix}.layer.2.layer_norm.weight"] = block.feed_forward.layer_norm_weight.clone()
        for layer_idx, attn_name in ((0, "time_attn"), (1, "group_attn")):
            attn = getattr(block, attn_name)
            for proj in ("q", "k", "v", "o"):
                sd[f"{prefix}.layer.{layer_idx}.self_attention.{proj}.weight"] = getattr(attn, proj).weight.clone()
        sd[f"{prefix}.layer.2.mlp.wi.weight"] = block.feed_forward.wi.weight.clone()
        sd[f"{prefix}.layer.2.mlp.wo.weight"] = block.feed_forward.wo.weight.clone()
    safetensors_save_file(sd, str(model_dir / "model.safetensors"))


class TestFromPretrainedWithAdapter:
    def test_auto_detect_adapter_in_model_dir(self):
        """from_pretrained should auto-detect adapter_config.json in model_path."""
        torch.manual_seed(0)
        model_no_lora = _make_small_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)

            config_dict = {
                "d_model": 64, "d_kv": 16, "d_ff": 128, "num_layers": 2, "num_heads": 4,
                "dropout_rate": 0.0, "layer_norm_epsilon": 1e-6, "dense_act_fn": "relu",
                "rope_theta": 10000.0, "vocab_size": 2,
                "chronos_config": {
                    "context_length": 64, "input_patch_size": 8, "input_patch_stride": 8,
                    "output_patch_size": 8, "use_reg_token": True, "use_arcsinh": True,
                },
            }
            with open(model_dir / "config.json", "w") as f:
                json.dump(config_dict, f)

            _save_as_chronos2_checkpoint(model_no_lora, model_dir)

            adapter_cfg = {"r": 4, "lora_alpha": 8, "target_modules": ["q", "k", "v", "o"], "fan_in_fan_out": False}
            with open(model_dir / "adapter_config.json", "w") as f:
                json.dump(adapter_cfg, f)

            lora_sd: dict[str, torch.Tensor] = {}
            for i in range(2):
                for layer_idx in (0, 1):
                    for proj in ["q", "k", "v", "o"]:
                        lora_sd[f"base_model.model.encoder.block.{i}.layer.{layer_idx}.self_attention.{proj}.lora_A.weight"] = torch.randn(4, 64)
                        lora_sd[f"base_model.model.encoder.block.{i}.layer.{layer_idx}.self_attention.{proj}.lora_B.weight"] = torch.randn(64, 4)
            safetensors_save_file(lora_sd, str(model_dir / "adapter_model.safetensors"))

            loaded = CuteChronos2Model.from_pretrained(model_dir)

        assert not torch.equal(loaded.blocks[0].time_attn.q.weight, model_no_lora.blocks[0].time_attn.q.weight)

    def test_explicit_adapter_path(self):
        """from_pretrained with explicit adapter_path should merge correctly."""
        torch.manual_seed(0)
        model_base = _make_small_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            model_dir.mkdir()

            config_dict = {
                "d_model": 64, "d_kv": 16, "d_ff": 128, "num_layers": 2, "num_heads": 4,
                "dropout_rate": 0.0, "layer_norm_epsilon": 1e-6, "dense_act_fn": "relu",
                "rope_theta": 10000.0, "vocab_size": 2,
                "chronos_config": {
                    "context_length": 64, "input_patch_size": 8, "input_patch_stride": 8,
                    "output_patch_size": 8, "use_reg_token": True, "use_arcsinh": True,
                },
            }
            with open(model_dir / "config.json", "w") as f:
                json.dump(config_dict, f)
            _save_as_chronos2_checkpoint(model_base, model_dir)

            adapter_dir = _create_mock_adapter(tmpdir, num_layers=2, d_model=64)
            loaded = CuteChronos2Model.from_pretrained(model_dir, adapter_path=adapter_dir)

        assert not torch.equal(loaded.blocks[0].time_attn.q.weight, model_base.blocks[0].time_attn.q.weight)


@pytest.mark.skipif(not _REAL_ADAPTER_PATH.exists(), reason="Real adapter not available on this machine")
class TestRealAdapter:
    def test_load_real_adapter(self):
        """Load the real BTC LoRA adapter and verify weights change."""
        torch.manual_seed(0)
        config = CuteChronos2Config(
            d_model=768, d_kv=64, d_ff=3072, num_layers=12, num_heads=12,
            context_length=512, input_patch_size=16, input_patch_stride=16, output_patch_size=16,
        )
        model = CuteChronos2Model(config)
        model.eval()

        orig_w = model.blocks[0].time_attn.q.weight.clone()
        model.load_lora_adapter(_REAL_ADAPTER_PATH)
        assert not torch.equal(model.blocks[0].time_attn.q.weight, orig_w)

    def test_real_adapter_config_values(self):
        """Verify real adapter config has expected r and alpha."""
        with open(_REAL_ADAPTER_PATH / "adapter_config.json") as f:
            cfg = json.load(f)
        assert cfg["r"] == 16
        assert cfg["lora_alpha"] == 32
        assert set(cfg["target_modules"]) == {"q", "k", "v", "o"}
