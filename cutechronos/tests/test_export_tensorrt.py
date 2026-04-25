import json
from pathlib import Path

import torch
from safetensors.torch import save_file as safetensors_save_file

from cutechronos.export_tensorrt import analyze_cutechronos_tensorrt
from cutechronos.model import CuteChronos2Config, CuteChronos2Model


def _make_small_config() -> CuteChronos2Config:
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
        context_length=64,
        input_patch_size=4,
        input_patch_stride=4,
        output_patch_size=4,
        quantiles=[0.1, 0.5, 0.9],
        use_reg_token=True,
        use_arcsinh=True,
    )


def _save_model_to_dir(model: CuteChronos2Model, target_dir: Path) -> None:
    config = model.config
    config_dict = {
        "d_model": config.d_model,
        "d_kv": config.d_kv,
        "d_ff": config.d_ff,
        "num_layers": config.num_layers,
        "num_heads": config.num_heads,
        "dropout_rate": config.dropout_rate,
        "layer_norm_epsilon": config.layer_norm_epsilon,
        "dense_act_fn": config.dense_act_fn,
        "rope_theta": config.rope_theta,
        "vocab_size": config.vocab_size,
        "reg_token_id": config.reg_token_id,
        "chronos_config": {
            "context_length": config.context_length,
            "input_patch_size": config.input_patch_size,
            "input_patch_stride": config.input_patch_stride,
            "output_patch_size": config.output_patch_size,
            "quantiles": config.quantiles,
            "use_reg_token": config.use_reg_token,
            "use_arcsinh": config.use_arcsinh,
            "time_encoding_scale": config.time_encoding_scale,
        },
    }
    (target_dir / "config.json").write_text(json.dumps(config_dict))

    state_dict = {
        "shared.weight": model.shared.weight.data,
        "encoder.final_layer_norm.weight": model.final_layer_norm_weight.data,
    }
    for name in ("input_patch_embedding", "output_patch_embedding"):
        block = getattr(model, name)
        for layer in ("hidden_layer", "output_layer", "residual_layer"):
            for param in ("weight", "bias"):
                key = f"{name}.{layer}.{param}"
                state_dict[key] = getattr(getattr(block, layer), param).data

    for i, block in enumerate(model.blocks):
        prefix = f"encoder.block.{i}"
        state_dict[f"{prefix}.layer.0.layer_norm.weight"] = block.time_attn.layer_norm_weight.data
        for proj in ("q", "k", "v", "o"):
            state_dict[f"{prefix}.layer.0.self_attention.{proj}.weight"] = getattr(block.time_attn, proj).weight.data
        state_dict[f"{prefix}.layer.1.layer_norm.weight"] = block.group_attn.layer_norm_weight.data
        for proj in ("q", "k", "v", "o"):
            state_dict[f"{prefix}.layer.1.self_attention.{proj}.weight"] = getattr(block.group_attn, proj).weight.data
        state_dict[f"{prefix}.layer.2.layer_norm.weight"] = block.feed_forward.layer_norm_weight.data
        state_dict[f"{prefix}.layer.2.mlp.wi.weight"] = block.feed_forward.wi.weight.data
        state_dict[f"{prefix}.layer.2.mlp.wo.weight"] = block.feed_forward.wo.weight.data

    safetensors_save_file(state_dict, str(target_dir / "model.safetensors"))


def test_analyze_cutechronos_tensorrt_exports_onnx(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model = CuteChronos2Model(_make_small_config()).eval()
    _save_model_to_dir(model, model_dir)

    report = analyze_cutechronos_tensorrt(
        model_path=str(model_dir),
        output_dir=str(tmp_path / "artifacts"),
        device="cpu",
        dtype_name="float32",
        batch_size=2,
        context_length=16,
        num_output_patches=1,
        max_batch_size=4,
        max_context_length=32,
        build_engine=False,
    )

    assert report["dynamo_explain"]["available"] is True
    assert report["torch_export"]["available"] is True
    assert report["torch_export"]["success"] is True
    constraints = report["onnx_export"]["context_shape_constraints"]
    assert constraints["effective_min"] % report["input_patch_size"] == 0
    assert constraints["effective_max"] % report["input_patch_size"] == 0
    if report["onnx_export"]["success"]:
        assert Path(report["onnx_export"]["path"]).exists()
    else:
        error_text = f"{report['onnx_export'].get('error', '')} {report['onnx_export'].get('dynamo_export_error', '')}"
        assert "onnx" in error_text.lower()
    assert Path(report["report_path"]).exists()
    assert report["backend_overrides"]["CUTECHRONOS_ATTENTION_BACKEND"] == "torch"
