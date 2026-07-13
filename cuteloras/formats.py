"""LoRA weight-format detection and key mapping to CuteDSL transformer parameter paths.

Supported formats:
- zimage-native: ``diffusion_model.layers.N.attention.to_q.lora_A.weight`` (Z-Image community LoRAs)
- peft: ``base_model.model.<path>.lora_A.weight`` (PEFT adapters)
- diffusers: ``transformer.transformer_blocks.N.attn.to_q.lora_A.weight`` (Flux-style)
- kohya: ``lora_unet_<path>.lora_down.weight`` with per-module alpha
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import torch

logger = logging.getLogger("cuteloras")

ZIMAGE_MODULE_MAP = {
    "attention.to_q": "q_proj",
    "attention.to_k": "k_proj",
    "attention.to_v": "v_proj",
    "attention.to_out.0": "o_proj",
    "feed_forward.w1": "feed_forward.w1",
    "feed_forward.w2": "feed_forward.w2",
    "feed_forward.w3": "feed_forward.w3",
    "adaLN_modulation.0": "adaLN_modulation.1",
}

ZIMAGE_BLOCK_PREFIXES = ("layers", "noise_refiner", "context_refiner")


@dataclass
class LoRAFactors:
    """Low-rank factors keyed by target parameter path (relative to the transformer root).

    Each entry maps ``<module_path>`` -> ``(A, B, alpha_scale)`` where the weight delta is
    ``B @ A * alpha_scale``. Tensors are kept on CPU until applied.
    """

    factors: dict[str, tuple[torch.Tensor, torch.Tensor, float]]
    format: str
    source: str = ""

    def param_paths(self) -> set[str]:
        return {f"{path}.weight" for path in self.factors}


def detect_lora_format(keys: list[str]) -> str:
    for k in keys:
        if k.startswith("diffusion_model."):
            return "zimage-native"
        if k.startswith("base_model.model."):
            return "peft"
        if k.startswith("transformer."):
            return "diffusers"
        if k.startswith(("lora_unet_", "lora_te_")):
            return "kohya"
    return "unknown"


def _pair_keys(keys: list[str], down: str, up: str) -> dict[str, tuple[str, str]]:
    pairs: dict[str, tuple[str, str]] = {}
    for k in keys:
        if k.endswith(f".{down}.weight"):
            module = k[: -len(f".{down}.weight")]
            up_key = f"{module}.{up}.weight"
            if up_key in keys:
                pairs[module] = (k, up_key)
    return pairs


def _known_zimage_module(module: str) -> bool:
    parts = module.split(".")
    if parts[0] not in ZIMAGE_BLOCK_PREFIXES or len(parts) < 3 or not parts[1].isdigit():
        return False
    return ".".join(parts[2:]) in ZIMAGE_MODULE_MAP


def _kohya_module_factors(
    module: str,
    a: torch.Tensor,
    b: torch.Tensor,
    scale: float,
) -> dict[str, tuple[torch.Tensor, torch.Tensor, float]]:
    """Decode Kohya's underscore Z-Image paths, including joint QKV LoRAs."""
    raw = module.removeprefix("lora_unet_")
    match = re.fullmatch(
        r"(layers|noise_refiner|context_refiner)_(\d+)_(attention_out|attention_qkv|feed_forward_w[123])",
        raw,
    )
    if match is None:
        return {raw.replace("_", "."): (a, b, scale)}

    block, index, target = match.groups()
    prefix = f"{block}.{index}"
    if target == "attention_out":
        return {f"{prefix}.attention.to_out.0": (a, b, scale)}
    if target == "attention_qkv":
        if b.shape[0] % 3:
            raise ValueError(f"joint QKV LoRA has indivisible output rows: {tuple(b.shape)}")
        q, k, v = b.chunk(3, dim=0)
        return {
            f"{prefix}.attention.to_q": (a, q.contiguous(), scale),
            f"{prefix}.attention.to_k": (a, k.contiguous(), scale),
            f"{prefix}.attention.to_v": (a, v.contiguous(), scale),
        }
    projection = target.removeprefix("feed_forward_")
    return {f"{prefix}.feed_forward.{projection}": (a, b, scale)}


def load_lora_factors(path: str, fmt: str | None = None) -> LoRAFactors:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as sf:
        keys = list(sf.keys())
        fmt = fmt or detect_lora_format(keys)
        key_set = set(keys)

        factors: dict[str, tuple[torch.Tensor, torch.Tensor, float]] = {}

        if fmt == "zimage-native":
            pairs = _pair_keys(keys, "lora_A", "lora_B")
            for module, (a_key, b_key) in pairs.items():
                raw = module.removeprefix("diffusion_model.")
                if not _known_zimage_module(raw):
                    continue
                factors[raw] = (sf.get_tensor(a_key), sf.get_tensor(b_key), 1.0)
        elif fmt == "peft":
            pairs = _pair_keys(keys, "lora_A", "lora_B")
            for module, (a_key, b_key) in pairs.items():
                factors[module.removeprefix("base_model.model.")] = (
                    sf.get_tensor(a_key),
                    sf.get_tensor(b_key),
                    1.0,
                )
        elif fmt == "diffusers":
            pairs = _pair_keys(keys, "lora_A", "lora_B")
            for module, (a_key, b_key) in pairs.items():
                factors[module.removeprefix("transformer.")] = (
                    sf.get_tensor(a_key),
                    sf.get_tensor(b_key),
                    1.0,
                )
        elif fmt == "kohya":
            pairs = _pair_keys(keys, "lora_down", "lora_up")
            for module, (a_key, b_key) in pairs.items():
                if not module.startswith("lora_unet_"):
                    continue
                a = sf.get_tensor(a_key)
                alpha_key = f"{module}.alpha"
                alpha = float(sf.get_tensor(alpha_key)) if alpha_key in key_set else float(a.shape[0])
                factors.update(
                    _kohya_module_factors(
                        module,
                        a,
                        sf.get_tensor(b_key),
                        alpha / a.shape[0],
                    )
                )
        else:
            raise ValueError(f"unrecognized LoRA format in {path} (first keys: {keys[:5]})")

    if not factors:
        raise ValueError(f"no mappable LoRA modules found in {path} (format={fmt})")
    logger.info("loaded %d LoRA modules from %s (format=%s)", len(factors), path, fmt)
    return LoRAFactors(factors=factors, format=fmt, source=path)
