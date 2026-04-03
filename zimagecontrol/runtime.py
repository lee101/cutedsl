"""Runtime helpers shared by Z-Image ControlNet scripts."""

from __future__ import annotations

from typing import Iterable

import torch


def parse_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[key]


def parse_index_spec(spec: str, limit: int) -> list[int]:
    if limit <= 0:
        return []
    spec = spec.strip().lower()
    if spec in {"", "all"}:
        return list(range(limit))

    result: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_s, end_s = chunk.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError(f"Invalid range: {chunk}")
            result.extend(range(start, end + 1))
        else:
            result.append(int(chunk))

    deduped = sorted(set(result))
    if deduped and (deduped[0] < 0 or deduped[-1] >= limit):
        raise ValueError(f"Index spec {spec!r} is out of range 0..{limit - 1}")
    return deduped


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def retrieve_latents(encoder_output, *, sample_mode: str = "sample", generator=None) -> torch.Tensor:
    if hasattr(encoder_output, "latent_dist"):
        if sample_mode == "sample":
            return encoder_output.latent_dist.sample(generator=generator)
        if sample_mode in {"argmax", "mode"}:
            return encoder_output.latent_dist.mode()
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not retrieve latents from encoder output")


def encode_vae_image(vae, image: torch.Tensor, *, generator=None, sample_mode: str = "sample") -> torch.Tensor:
    latents = retrieve_latents(vae.encode(image), generator=generator, sample_mode=sample_mode)
    return (latents - vae.config.shift_factor) * vae.config.scaling_factor


def build_zimage_controlnet(
    transformer,
    *,
    control_layers_spec: str = "all",
    control_refiner_layers_spec: str = "all",
    add_control_noise_refiner: str | None = None,
):
    from diffusers import ZImageControlNetModel

    cfg = transformer.config
    control_layers = parse_index_spec(control_layers_spec, cfg.n_layers)
    control_refiner_layers = parse_index_spec(control_refiner_layers_spec, cfg.n_refiner_layers)
    controlnet = ZImageControlNetModel(
        control_layers_places=control_layers,
        control_refiner_layers_places=control_refiner_layers,
        control_in_dim=transformer.in_channels,
        add_control_noise_refiner=add_control_noise_refiner,
        all_patch_size=tuple(cfg.all_patch_size),
        all_f_patch_size=tuple(cfg.all_f_patch_size),
        dim=cfg.dim,
        n_refiner_layers=cfg.n_refiner_layers,
        n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads,
        norm_eps=cfg.norm_eps,
        qk_norm=cfg.qk_norm,
    )
    return ZImageControlNetModel.from_transformer(controlnet, transformer)


def freeze_module(module) -> None:
    module.requires_grad_(False)
    module.eval()


def iter_trainable_controlnet_parameters(controlnet) -> Iterable[torch.nn.Parameter]:
    prefixes = ("control_layers.", "control_all_x_embedder.", "control_noise_refiner.")
    for name, parameter in controlnet.named_parameters():
        trainable = name.startswith(prefixes)
        parameter.requires_grad_(trainable)
        if trainable:
            yield parameter

