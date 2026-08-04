import copy

import torch

from cuteloras.registry import LoRARecord, LoRARegistry
from cuteloras.swapper import LoRASwapper

from .conftest import make_zimage_lora


def _registry_with(tmp_path, n=1):
    registry = LoRARegistry(cache_dir=str(tmp_path / "cache"))
    ids = []
    for i in range(n):
        path = make_zimage_lora(tmp_path / f"lora{i}.safetensors", seed=i + 1)
        lora_id = f"lora{i}"
        registry.add(LoRARecord(id=lora_id, path=str(path)))
        ids.append(lora_id)
    return registry, ids


def test_apply_changes_output(transformer, tmp_path):
    registry, ids = _registry_with(tmp_path)
    swapper = LoRASwapper(transformer, registry)
    x = torch.randn(2, 8, 32)
    base_out = transformer(x)
    info = swapper.activate(ids[0])
    assert info["swapped"] and info["params"] == 4
    assert not torch.allclose(transformer(x), base_out)


def test_exact_restore(transformer, tmp_path):
    registry, ids = _registry_with(tmp_path)
    original = copy.deepcopy(transformer.state_dict())
    swapper = LoRASwapper(transformer, registry, pin_snapshots=False)
    for _ in range(50):
        swapper.activate(ids[0])
        swapper.deactivate()
    for key, value in transformer.state_dict().items():
        assert torch.equal(value, original[key]), key


def test_swap_equals_fresh_apply(transformer, tmp_path):
    registry, ids = _registry_with(tmp_path, n=2)
    swapper = LoRASwapper(transformer, registry, pin_snapshots=False)
    swapper.activate(ids[1])
    direct = {k: v.clone() for k, v in transformer.state_dict().items()}
    swapper.deactivate()
    for _ in range(3):
        swapper.activate(ids[0])
        swapper.activate(ids[1])
    for key, value in transformer.state_dict().items():
        assert torch.equal(value, direct[key]), key


def test_activate_same_is_noop(transformer, tmp_path):
    registry, ids = _registry_with(tmp_path)
    swapper = LoRASwapper(transformer, registry)
    assert swapper.activate(ids[0])["swapped"]
    assert not swapper.activate(ids[0])["swapped"]


def test_bf16_fused_path_is_used_for_single_adapter(transformer, tmp_path, monkeypatch):
    registry, ids = _registry_with(tmp_path)
    monkeypatch.setenv("CUTELORAS_FUSED_BF16", "1")
    import cuteloras.fused_apply as fused_apply

    original = fused_apply.fused_lora_delta
    calls = []

    def counted(*args, **kwargs):
        calls.append(kwargs.get("compute_dtype"))
        return original(*args, **kwargs)

    monkeypatch.setattr(fused_apply, "fused_lora_delta", counted)
    swapper = LoRASwapper(transformer, registry, pin_snapshots=False, pin_factors=False)
    before = copy.deepcopy(transformer.state_dict())
    info = swapper.activate(ids[0])

    assert info["params"] == 4
    assert calls == [torch.bfloat16] * 4
    swapper.deactivate()
    for key, value in transformer.state_dict().items():
        assert torch.equal(value, before[key]), key


def test_scale_composition(transformer, tmp_path):
    registry, ids = _registry_with(tmp_path)
    registry.get(ids[0]).scale = 0.5
    swapper = LoRASwapper(transformer, registry, pin_snapshots=False)
    base = transformer.layers[0].q_proj.weight.clone()
    swapper.activate([(ids[0], 2.0)])
    delta_full = transformer.layers[0].q_proj.weight - base
    swapper.deactivate()
    swapper.activate([(ids[0], 1.0)])
    delta_half = transformer.layers[0].q_proj.weight - base
    assert torch.allclose(delta_full, delta_half * 2, atol=1e-5)


def test_multi_lora_stack(transformer, tmp_path):
    registry, ids = _registry_with(tmp_path, n=2)
    swapper = LoRASwapper(transformer, registry, pin_snapshots=False)
    base = transformer.layers[0].q_proj.weight.clone()
    swapper.activate(ids[0])
    d0 = transformer.layers[0].q_proj.weight - base
    swapper.activate(ids[1])
    d1 = transformer.layers[0].q_proj.weight - base
    swapper.activate([ids[0], ids[1]])
    stacked = transformer.layers[0].q_proj.weight - base
    assert torch.allclose(stacked, d0 + d1, atol=1e-5)


def test_fused_qkv_resolution(fused_transformer, tmp_path):
    registry = LoRARegistry(cache_dir=str(tmp_path))
    path = make_zimage_lora(tmp_path / "fused.safetensors", mods=("to_q", "to_k", "to_v", "to_out.0"), adaln=True)
    registry.add(LoRARecord(id="fused", path=str(path)))
    swapper = LoRASwapper(fused_transformer, registry, pin_snapshots=False)

    import copy

    original = copy.deepcopy(fused_transformer.state_dict())
    dim = 32
    qkv_before = original["layers.0.qkv_proj.weight"]
    info = swapper.activate("fused")
    assert info["params"] == 2 * (3 + 1 + 1)
    qkv_after = fused_transformer.layers[0].qkv_proj.weight
    assert not torch.equal(qkv_after[:dim], qkv_before[:dim])
    assert not torch.equal(qkv_after[dim : 2 * dim], qkv_before[dim : 2 * dim])
    assert not torch.equal(
        fused_transformer.layers[0].adaLN_modulation[0].weight, original["layers.0.adaLN_modulation.0.weight"]
    )
    swapper.deactivate()
    for key, value in fused_transformer.state_dict().items():
        assert torch.equal(value, original[key]), key


def test_diffusers_naming_resolution(diffusers_transformer, tmp_path):
    registry = LoRARegistry(cache_dir=str(tmp_path))
    path = make_zimage_lora(tmp_path / "d.safetensors", mods=("to_q", "to_out.0"))
    registry.add(LoRARecord(id="d", path=str(path)))
    swapper = LoRASwapper(diffusers_transformer, registry, pin_snapshots=False, pin_factors=False)

    import copy

    original = copy.deepcopy(diffusers_transformer.state_dict())
    info = swapper.activate("d")
    assert info["params"] == 4
    assert not torch.equal(
        diffusers_transformer.layers[0].attention.to_q.weight, original["layers.0.attention.to_q.weight"]
    )
    assert not torch.equal(
        diffusers_transformer.layers[0].attention.to_out[0].weight, original["layers.0.attention.to_out.0.weight"]
    )
    swapper.deactivate()
    for key, value in diffusers_transformer.state_dict().items():
        assert torch.equal(value, original[key]), key


def test_lru_eviction_still_restores(transformer, tmp_path):
    registry, ids = _registry_with(tmp_path, n=3)
    swapper = LoRASwapper(transformer, registry, max_cached_loras=1, pin_snapshots=False)
    original = copy.deepcopy(transformer.state_dict())
    for lora_id in ids:
        swapper.activate(lora_id)
    swapper.deactivate()
    for key, value in transformer.state_dict().items():
        assert torch.equal(value, original[key]), key


def test_unresolvable_adapter_fails_instead_of_claiming_activation(transformer, tmp_path):
    from safetensors.torch import save_file

    path = tmp_path / "bad.safetensors"
    save_file(
        {
            "base_model.model.missing.module.lora_A.weight": torch.randn(4, 32),
            "base_model.model.missing.module.lora_B.weight": torch.randn(32, 4),
        },
        str(path),
    )
    registry = LoRARegistry([LoRARecord(id="bad", path=str(path))])
    swapper = LoRASwapper(transformer, registry, pin_snapshots=False, pin_factors=False)

    import pytest

    with pytest.raises(RuntimeError, match="zero parameters"):
        swapper.activate("bad")
    assert swapper.active == ()


def test_switch_to_unresolvable_adapter_restores_nothing_and_keeps_valid_active(transformer, tmp_path):
    """Removing the old adapter must not count as applying the new adapter."""
    from safetensors.torch import save_file
    import pytest

    registry, ids = _registry_with(tmp_path)
    bad_path = tmp_path / "bad-switch.safetensors"
    save_file(
        {
            "base_model.model.missing.module.lora_A.weight": torch.randn(4, 32),
            "base_model.model.missing.module.lora_B.weight": torch.randn(32, 4),
        },
        str(bad_path),
    )
    registry.add(LoRARecord(id="bad", path=str(bad_path)))
    swapper = LoRASwapper(transformer, registry, pin_snapshots=False, pin_factors=False)

    swapper.activate(ids[0])
    active_state = copy.deepcopy(transformer.state_dict())

    with pytest.raises(RuntimeError, match="zero parameters"):
        swapper.activate("bad")

    assert swapper.active == ((ids[0], 1.0),)
    for key, value in transformer.state_dict().items():
        assert torch.equal(value, active_state[key]), key


def test_kohya_joint_qkv_applies_to_vanilla_diffusers_transformer(diffusers_transformer, tmp_path):
    from safetensors.torch import save_file

    rank = 4
    dim = diffusers_transformer.layers[0].attention.to_q.in_features
    path = tmp_path / "joint-qkv.safetensors"
    save_file(
        {
            "lora_unet_layers_0_attention_qkv.lora_down.weight": torch.randn(rank, dim),
            "lora_unet_layers_0_attention_qkv.lora_up.weight": torch.randn(dim * 3, rank),
            "lora_unet_layers_0_attention_out.lora_down.weight": torch.randn(rank, dim),
            "lora_unet_layers_0_attention_out.lora_up.weight": torch.randn(dim, rank),
        },
        str(path),
    )
    registry = LoRARegistry([LoRARecord(id="kohya", path=str(path))])
    swapper = LoRASwapper(diffusers_transformer, registry, pin_snapshots=False, pin_factors=False)
    original = copy.deepcopy(diffusers_transformer.state_dict())

    info = swapper.activate("kohya")

    assert info["params"] == 4
    for name in ("to_q", "to_k", "to_v"):
        assert not torch.equal(
            getattr(diffusers_transformer.layers[0].attention, name).weight,
            original[f"layers.0.attention.{name}.weight"],
        )
    assert not torch.equal(
        diffusers_transformer.layers[0].attention.to_out[0].weight,
        original["layers.0.attention.to_out.0.weight"],
    )
