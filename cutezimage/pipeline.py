"""Accelerated Z-Image Turbo pipeline helpers.

This mirrors the lazy-loading utility style used in the sibling inference
server, but swaps the diffusers ``ZImageTransformer2DModel`` for
``CuteZImageTransformer`` so the same fused kernels are used in:

- text-to-image via ``ZImagePipeline``
- img2img via ``ZImageImg2ImgPipeline``
- ControlNet text-to-image via ``ZImageControlNetPipeline``

The ControlNet path accelerates the shared Z-Image transformer while keeping
the diffusers ControlNet model unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from typing import Any

import torch

from cutezimage.model import CuteZImageTransformer

try:  # pragma: no cover - exercised via monkeypatch in tests when absent
    from diffusers import (
        ZImageControlNetModel,
        ZImageControlNetPipeline,
        ZImageImg2ImgPipeline,
        ZImagePipeline,
    )
except ImportError as exc:  # pragma: no cover - depends on optional deps
    ZImagePipeline = None
    ZImageImg2ImgPipeline = None
    ZImageControlNetPipeline = None
    ZImageControlNetModel = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

try:  # pragma: no cover - optional dependency pulled in by diffusers
    from huggingface_hub import hf_hub_download
except ImportError:  # pragma: no cover - depends on optional deps
    hf_hub_download = None


ZIMAGE_MODEL_PATH = os.getenv("ZIMAGE_MODEL_PATH", "Tongyi-MAI/Z-Image-Turbo")
ZIMAGE_TURBO_STEPS = int(os.getenv("ZIMAGE_TURBO_STEPS", "9"))
ZIMAGE_TURBO_GUIDANCE_SCALE = float(os.getenv("ZIMAGE_TURBO_GUIDANCE_SCALE", "0.0"))
ZIMAGE_COMPILE_MODE = os.getenv("ZIMAGE_COMPILE_MODE", "reduce-overhead") or None
ZIMAGE_CONTROLNET_MODEL_PATH = os.getenv("ZIMAGE_CONTROLNET_MODEL_PATH")
ZIMAGE_CONTROLNET_FILENAME = os.getenv("ZIMAGE_CONTROLNET_FILENAME")

_PIPELINE_LOCK = Lock()
_ZIMAGE_PIPELINE_CACHE: dict[tuple[Any, ...], tuple[Any, Any]] = {}
_ZIMAGE_CONTROLNET_CACHE: dict[tuple[Any, ...], Any] = {}


def _dtype_key(dtype: torch.dtype | None) -> str:
    return str(dtype) if dtype is not None else "auto"


def _default_device(device: str | torch.device | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _default_dtype(
    dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
) -> torch.dtype:
    if dtype is not None:
        return dtype
    return torch.bfloat16 if _default_device(device).type == "cuda" else torch.float32


def _build_generator(
    seed: int = 0,
    device: str | torch.device | None = None,
) -> torch.Generator:
    target_device = _default_device(device)
    return torch.Generator(device=target_device.type).manual_seed(seed)


def _configure_pipeline(
    pipe: Any,
    *,
    device: str | torch.device | None = None,
    enable_cpu_offload: bool | None = None,
) -> None:
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()

    if enable_cpu_offload is None:
        enable_cpu_offload = torch.cuda.is_available()

    if enable_cpu_offload and torch.cuda.is_available() and hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
        return

    if hasattr(pipe, "to"):
        pipe.to(_default_device(device))


def _require_zimage_support(*, controlnet: bool = False) -> None:
    if _IMPORT_ERROR is not None:  # pragma: no cover - depends on optional deps
        if controlnet:
            raise RuntimeError(
                "Z-Image ControlNet support requires diffusers with "
                "ZImageControlNetPipeline and ZImageControlNetModel "
                "(diffusers >= 0.36.0)."
            ) from _IMPORT_ERROR
        raise RuntimeError(
            "Z-Image support requires diffusers with ZImagePipeline and "
            "ZImageImg2ImgPipeline (diffusers >= 0.36.0)."
        ) from _IMPORT_ERROR


def _infer_module_device_dtype(module: torch.nn.Module) -> tuple[torch.device, torch.dtype]:
    try:
        parameter = next(module.parameters())
    except StopIteration:
        return torch.device("cpu"), torch.float32
    return parameter.device, parameter.dtype


def build_cute_transformer(
    transformer: torch.nn.Module,
    *,
    compile_mode: str | None = ZIMAGE_COMPILE_MODE,
    device: str | torch.device | None = None,
    torch_dtype: torch.dtype | None = None,
) -> CuteZImageTransformer:
    """Convert a diffusers transformer into ``CuteZImageTransformer``.

    If the transformer is already accelerated, it is returned unchanged.
    """
    if isinstance(transformer, CuteZImageTransformer):
        return transformer

    source_device, source_dtype = _infer_module_device_dtype(transformer)
    target_device = torch.device(device) if device is not None else source_device
    target_dtype = torch_dtype if torch_dtype is not None else source_dtype

    if compile_mode:
        cute = CuteZImageTransformer.from_diffusers_compiled(transformer, compile_mode=compile_mode)
    else:
        cute = CuteZImageTransformer.from_diffusers(transformer)

    cute = cute.to(device=target_device, dtype=target_dtype)
    cute.eval()
    return cute


def accelerate_zimage_pipeline(
    pipe: Any,
    *,
    compile_mode: str | None = ZIMAGE_COMPILE_MODE,
    device: str | torch.device | None = None,
    torch_dtype: torch.dtype | None = None,
    configure: bool = True,
    enable_cpu_offload: bool | None = None,
) -> Any:
    """Replace a diffusers Z-Image pipeline transformer with CuteZImage."""
    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        raise ValueError("Expected pipeline to expose a `transformer` attribute.")

    pipe.transformer = build_cute_transformer(
        transformer,
        compile_mode=compile_mode,
        device=device,
        torch_dtype=torch_dtype,
    )
    setattr(pipe, "_uses_cutezimage", True)

    if configure:
        _configure_pipeline(pipe, device=device, enable_cpu_offload=enable_cpu_offload)
    return pipe


def _controlnet_cache_key(
    model_path: str,
    controlnet_model: str | None,
    controlnet_filename: str | None,
    torch_dtype: torch.dtype | None,
    use_cute: bool,
    compile_mode: str | None,
    device: str | torch.device | None,
    enable_cpu_offload: bool | None,
) -> tuple[Any, ...]:
    return (
        model_path,
        controlnet_model,
        controlnet_filename,
        _dtype_key(torch_dtype),
        use_cute,
        compile_mode,
        str(device) if device is not None else "auto",
        enable_cpu_offload,
    )


def _pipeline_cache_key(
    model_path: str,
    torch_dtype: torch.dtype | None,
    use_cute: bool,
    compile_mode: str | None,
    device: str | torch.device | None,
    enable_cpu_offload: bool | None,
) -> tuple[Any, ...]:
    return (
        model_path,
        _dtype_key(torch_dtype),
        use_cute,
        compile_mode,
        str(device) if device is not None else "auto",
        enable_cpu_offload,
    )


def clear_pipeline_caches() -> None:
    """Clear all cached Z-Image pipeline instances."""
    with _PIPELINE_LOCK:
        _ZIMAGE_PIPELINE_CACHE.clear()
        _ZIMAGE_CONTROLNET_CACHE.clear()


def get_zimage_pipelines(
    model_path: str = ZIMAGE_MODEL_PATH,
    *,
    torch_dtype: torch.dtype | None = None,
    use_cute: bool = True,
    compile_mode: str | None = ZIMAGE_COMPILE_MODE,
    device: str | torch.device | None = None,
    enable_cpu_offload: bool | None = None,
):
    """Load and cache accelerated Z-Image text2img and img2img pipelines."""
    _require_zimage_support(controlnet=False)

    resolved_dtype = _default_dtype(torch_dtype, device)
    key = _pipeline_cache_key(
        model_path,
        resolved_dtype,
        use_cute,
        compile_mode,
        device,
        enable_cpu_offload,
    )

    cached = _ZIMAGE_PIPELINE_CACHE.get(key)
    if cached is not None:
        return cached

    with _PIPELINE_LOCK:
        cached = _ZIMAGE_PIPELINE_CACHE.get(key)
        if cached is not None:
            return cached

        zimage_pipe = ZImagePipeline.from_pretrained(
            model_path,
            torch_dtype=resolved_dtype,
            low_cpu_mem_usage=False,
        )
        if use_cute:
            accelerate_zimage_pipeline(
                zimage_pipe,
                compile_mode=compile_mode,
                device=device,
                torch_dtype=resolved_dtype,
                configure=False,
            )

        zimage_img2img_pipe = ZImageImg2ImgPipeline(**zimage_pipe.components)
        if use_cute:
            zimage_img2img_pipe.transformer = zimage_pipe.transformer

        _configure_pipeline(zimage_pipe, device=device, enable_cpu_offload=enable_cpu_offload)
        _configure_pipeline(zimage_img2img_pipe, device=device, enable_cpu_offload=enable_cpu_offload)

        cached = (zimage_pipe, zimage_img2img_pipe)
        _ZIMAGE_PIPELINE_CACHE[key] = cached
        return cached


def _load_controlnet_model(
    controlnet_model: Any = None,
    *,
    controlnet_filename: str | None = None,
    torch_dtype: torch.dtype,
):
    _require_zimage_support(controlnet=True)

    model_ref = controlnet_model or ZIMAGE_CONTROLNET_MODEL_PATH
    filename = controlnet_filename or ZIMAGE_CONTROLNET_FILENAME

    if model_ref is None:
        raise ValueError(
            "Provide `controlnet_model` or set `ZIMAGE_CONTROLNET_MODEL_PATH` to load a Z-Image ControlNet model."
        )

    if not isinstance(model_ref, str):
        return model_ref

    path = Path(model_ref)
    if path.is_file() or path.suffix in {".safetensors", ".bin", ".ckpt", ".pt"}:
        return ZImageControlNetModel.from_single_file(str(path), torch_dtype=torch_dtype)

    if filename:
        if hf_hub_download is None:
            raise RuntimeError(
                "Loading a single-file Z-Image ControlNet repo requires `huggingface_hub`."
            )
        resolved_path = hf_hub_download(model_ref, filename=filename)
        return ZImageControlNetModel.from_single_file(resolved_path, torch_dtype=torch_dtype)

    try:
        return ZImageControlNetModel.from_pretrained(model_ref, torch_dtype=torch_dtype)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load Z-Image ControlNet model. If you are pointing at a single-file Hub repo, "
            "also pass `controlnet_filename`."
        ) from exc


def get_zimage_controlnet_pipeline(
    *,
    model_path: str = ZIMAGE_MODEL_PATH,
    controlnet_model: Any = None,
    controlnet_filename: str | None = None,
    torch_dtype: torch.dtype | None = None,
    use_cute: bool = True,
    compile_mode: str | None = ZIMAGE_COMPILE_MODE,
    device: str | torch.device | None = None,
    enable_cpu_offload: bool | None = None,
):
    """Load and cache an accelerated Z-Image ControlNet pipeline."""
    _require_zimage_support(controlnet=True)

    if isinstance(controlnet_model, str) or controlnet_model is None:
        key = _controlnet_cache_key(
            model_path,
            controlnet_model or ZIMAGE_CONTROLNET_MODEL_PATH,
            controlnet_filename or ZIMAGE_CONTROLNET_FILENAME,
            torch_dtype,
            use_cute,
            compile_mode,
            device,
            enable_cpu_offload,
        )
        cached = _ZIMAGE_CONTROLNET_CACHE.get(key)
        if cached is not None:
            return cached
    else:
        key = None

    resolved_dtype = _default_dtype(torch_dtype, device)
    base_pipe, _ = get_zimage_pipelines(
        model_path=model_path,
        torch_dtype=resolved_dtype,
        use_cute=use_cute,
        compile_mode=compile_mode,
        device=device,
        enable_cpu_offload=enable_cpu_offload,
    )
    controlnet = _load_controlnet_model(
        controlnet_model,
        controlnet_filename=controlnet_filename,
        torch_dtype=resolved_dtype,
    )

    pipeline_components = dict(base_pipe.components)
    pipeline_components["controlnet"] = controlnet

    pipe = ZImageControlNetPipeline(**pipeline_components)
    if use_cute:
        pipe.transformer = base_pipe.transformer

    _configure_pipeline(pipe, device=device, enable_cpu_offload=enable_cpu_offload)

    if key is not None:
        with _PIPELINE_LOCK:
            cached = _ZIMAGE_CONTROLNET_CACHE.get(key)
            if cached is None:
                _ZIMAGE_CONTROLNET_CACHE[key] = pipe
            else:
                pipe = cached
    return pipe


def create_image_with_zimage(
    prompt: str,
    width: int,
    height: int,
    *,
    seed: int = 0,
    num_inference_steps: int = ZIMAGE_TURBO_STEPS,
    guidance_scale: float = ZIMAGE_TURBO_GUIDANCE_SCALE,
    model_path: str = ZIMAGE_MODEL_PATH,
    use_cute: bool = True,
    compile_mode: str | None = ZIMAGE_COMPILE_MODE,
    device: str | torch.device | None = None,
    torch_dtype: torch.dtype | None = None,
    enable_cpu_offload: bool | None = None,
    **kwargs: Any,
):
    """Generate an image with an accelerated Z-Image Turbo pipeline."""
    zimage_pipe, _ = get_zimage_pipelines(
        model_path=model_path,
        torch_dtype=torch_dtype,
        use_cute=use_cute,
        compile_mode=compile_mode,
        device=device,
        enable_cpu_offload=enable_cpu_offload,
    )
    result = zimage_pipe(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=_build_generator(seed, device=device),
        **kwargs,
    )
    return result.images[0]


def style_transfer_with_zimage(
    prompt: str,
    image: Any,
    strength: float,
    *,
    seed: int = 0,
    num_inference_steps: int = ZIMAGE_TURBO_STEPS,
    guidance_scale: float = ZIMAGE_TURBO_GUIDANCE_SCALE,
    model_path: str = ZIMAGE_MODEL_PATH,
    use_cute: bool = True,
    compile_mode: str | None = ZIMAGE_COMPILE_MODE,
    device: str | torch.device | None = None,
    torch_dtype: torch.dtype | None = None,
    enable_cpu_offload: bool | None = None,
    **kwargs: Any,
):
    """Run accelerated Z-Image img2img editing."""
    _, zimage_img2img_pipe = get_zimage_pipelines(
        model_path=model_path,
        torch_dtype=torch_dtype,
        use_cute=use_cute,
        compile_mode=compile_mode,
        device=device,
        enable_cpu_offload=enable_cpu_offload,
    )
    result = zimage_img2img_pipe(
        prompt=prompt,
        image=image,
        strength=strength,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=_build_generator(seed, device=device),
        **kwargs,
    )
    return result.images[0]


def create_image_with_zimage_controlnet(
    prompt: str,
    control_image: Any,
    *,
    width: int,
    height: int,
    controlnet_model: Any = None,
    controlnet_filename: str | None = None,
    controlnet_conditioning_scale: float = 0.75,
    seed: int = 0,
    num_inference_steps: int = ZIMAGE_TURBO_STEPS,
    guidance_scale: float = ZIMAGE_TURBO_GUIDANCE_SCALE,
    model_path: str = ZIMAGE_MODEL_PATH,
    use_cute: bool = True,
    compile_mode: str | None = ZIMAGE_COMPILE_MODE,
    device: str | torch.device | None = None,
    torch_dtype: torch.dtype | None = None,
    enable_cpu_offload: bool | None = None,
    **kwargs: Any,
):
    """Generate an image with Z-Image ControlNet and an accelerated transformer."""
    pipe = get_zimage_controlnet_pipeline(
        model_path=model_path,
        controlnet_model=controlnet_model,
        controlnet_filename=controlnet_filename,
        torch_dtype=torch_dtype,
        use_cute=use_cute,
        compile_mode=compile_mode,
        device=device,
        enable_cpu_offload=enable_cpu_offload,
    )
    result = pipe(
        prompt=prompt,
        control_image=control_image,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=_build_generator(seed, device=device),
        **kwargs,
    )
    return result.images[0]
