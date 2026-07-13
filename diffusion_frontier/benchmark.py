"""Benchmark public diffusion models on one latency/quality frontier.

Examples:

    cutedsl-diffusion-frontier --list-models
    cutedsl-diffusion-frontier --preflight --models krea2-turbo flux-schnell
    cutedsl-diffusion-frontier --models zimage-turbo --api-base-url http://localhost:8100
    cutedsl-diffusion-frontier --models realvisxl-v4 --allow-download --steps 4 8 20 40

Downloads are never implicit: uncached local models require --allow-download.
"""

from __future__ import annotations

import argparse
import base64
import gc
import io
import json
import random
import shutil
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw, ImageFont

from .catalog import MODEL_CATALOG, ModelSpec, get_model
from .flow_resume import capture_flow_latent, resume_flow_latent
from .metrics import compare_images, optional_lpips
from .preflight import inspect_model, system_snapshot

DEFAULT_PROMPTS = [
    "a red fox standing in fresh snow, photorealistic, detailed fur, golden hour",
    "a clean pixel-art grass block inventory icon, centered, plain background",
    "editorial portrait of an elderly astronaut, natural skin texture, soft window light",
    "a glass teapot on a wooden table, readable label saying TEA, product photography",
]


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sync_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _peak_vram_mib() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1024**2, 1)
    except Exception:
        pass
    return None


def _reset_peak_vram() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _generator(seed: int):
    import torch

    # CPU generators work with Diffusers CPU offload and keep seeds comparable.
    return torch.Generator(device="cpu").manual_seed(seed)


def _source_for(spec: ModelSpec, local_files_only: bool = False) -> str:
    if spec.loader != "single_file":
        return spec.repo_id
    from huggingface_hub import hf_hub_download, hf_hub_url

    if not spec.source_file:
        raise ValueError(f"{spec.key} is a single-file model without source_file")
    if local_files_only:
        return hf_hub_download(spec.repo_id, spec.source_file, local_files_only=True)
    return hf_hub_url(spec.repo_id, spec.source_file)


def _single_file_config(spec: ModelSpec, local_files_only: bool = False) -> str:
    """Resolve the small local Diffusers config tree for a single-file model."""
    from huggingface_hub import snapshot_download

    return snapshot_download(
        spec.repo_id,
        allow_patterns=[
            "model_index.json",
            "scheduler/*",
            "text_encoder/config.json",
            "text_encoder_2/config.json",
            "tokenizer/*",
            "tokenizer_2/*",
            "unet/config.json",
            "vae/config.json",
        ],
        local_files_only=local_files_only,
    )


def _load_pipeline(spec: ModelSpec, args, check: dict, *, skip_text_encoder: bool = False):
    import diffusers
    import torch

    cls = getattr(diffusers, spec.pipeline_class)
    kwargs = {
        "torch_dtype": torch.bfloat16,
        "local_files_only": not args.allow_download,
    }
    if args.quantization:
        from diffusers import PipelineQuantizationConfig, TorchAoConfig
        from torchao.quantization import (
            Float8DynamicActivationFloat8WeightConfig,
            Float8WeightOnlyConfig,
            Int4WeightOnlyConfig,
            Int8WeightOnlyConfig,
        )

        ao_configs = {
            "torchao-fp8dq": Float8DynamicActivationFloat8WeightConfig,
            "torchao-fp8wo": Float8WeightOnlyConfig,
            "torchao-int8wo": Int8WeightOnlyConfig,
            "torchao-int4wo": Int4WeightOnlyConfig,
        }
        kwargs["quantization_config"] = PipelineQuantizationConfig(
            quant_mapping={"transformer": TorchAoConfig(ao_configs[args.quantization]())}
        )
        if args.offload == "cuda":
            kwargs["device_map"] = "cuda"
    if skip_text_encoder:
        # Krea's 4B Qwen encoder and 13B denoiser only barely coexist on a
        # 32 GB card. Conditioning can be staged first and supplied directly,
        # so do not load components that would immediately be released.
        kwargs.update({"text_encoder": None, "tokenizer": None})
    if spec.family == "sdxl":
        kwargs["use_safetensors"] = True
    if spec.variant:
        kwargs["variant"] = spec.variant
    started = time.perf_counter()
    if spec.loader == "single_file":
        pipe = cls.from_single_file(
            _source_for(spec, local_files_only=not args.allow_download),
            config=_single_file_config(spec, local_files_only=not args.allow_download),
            **kwargs,
        )
    else:
        pipe = cls.from_pretrained(_source_for(spec), **kwargs)

    mode = args.offload
    if mode == "auto":
        mode = check.get("recommended_offload") or spec.recommended_offload
    if mode == "cuda":
        if "device_map" not in kwargs:
            pipe.to("cuda")
    elif mode == "model":
        pipe.enable_model_cpu_offload()
    elif mode == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        raise ValueError(f"unsupported offload mode: {mode}")

    if args.lora:
        if not hasattr(pipe, "load_lora_weights"):
            raise RuntimeError(f"{spec.key} pipeline has no Diffusers LoRA loader")
        pipe.load_lora_weights(args.lora, adapter_name="frontier")
        if hasattr(pipe, "set_adapters"):
            pipe.set_adapters(["frontier"], adapter_weights=[args.lora_scale])

    compiled_component = None
    if args.compile:
        name = "transformer" if hasattr(pipe, "transformer") else "unet"
        component = getattr(pipe, name, None)
        if component is None:
            raise RuntimeError(f"{spec.key} has no transformer/unet to compile")
        setattr(pipe, name, torch.compile(component, mode=args.compile, fullgraph=args.compile_fullgraph))
        compiled_component = name

    return pipe, {
        "load_s": time.perf_counter() - started,
        "offload": mode,
        "compiled_component": compiled_component,
        "compile_fullgraph": args.compile_fullgraph if compiled_component else None,
        "quantization": args.quantization,
        "prompt_embeddings_cached": args.cache_prompt_embeds,
        "lora": args.lora,
        "lora_scale": args.lora_scale if args.lora else None,
        "text_encoder_load_skipped": skip_text_encoder,
    }


def _should_stage_krea_conditioning(spec: ModelSpec, args) -> bool:
    return bool(
        spec.family == "krea2"
        and args.offload == "cuda"
        and args.quantization
        and args.cache_prompt_embeds
        and args.release_text_encoders
    )


def _stage_krea_conditioning(spec: ModelSpec, prompts: list[str], args) -> tuple[dict[int, dict], dict]:
    """Encode Krea prompts before loading the resident quantized denoiser.

    Krea-2's BF16 Qwen3-VL encoder plus its FP8 transformer can exhaust a
    32 GB GPU during prompt encoding. Loading the public encoder alone,
    retaining its outputs on CPU, and then omitting it from the final pipeline
    keeps prompt encoding outside the hot denoiser footprint.
    """
    import diffusers
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, Qwen3VLModel

    started = time.perf_counter()
    source = _source_for(spec)
    common = {"local_files_only": not args.allow_download}
    snapshot = Path(
        snapshot_download(
            source,
            allow_patterns=["tokenizer/*", "text_encoder/*"],
            **common,
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(snapshot / "tokenizer", **common)
    text_encoder = Qwen3VLModel.from_pretrained(
        snapshot / "text_encoder",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        **common,
    )
    cls = getattr(diffusers, spec.pipeline_class)
    encoder_pipe = cls(
        scheduler=None,
        vae=None,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=None,
        is_distilled=True,
    )
    conditioning = {
        index: {
            key: value.detach().cpu()
            for key, value in _encode_conditioning(encoder_pipe, spec, prompt).items()
        }
        for index, prompt in enumerate(prompts)
    }
    del encoder_pipe, text_encoder, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return conditioning, {
        "conditioning_stage_s": time.perf_counter() - started,
        "conditioning_stage_device": "cuda",
        "conditioning_staged_components": ["text_encoder", "tokenizer"],
    }


def _release_text_encoders(pipe) -> list[str]:
    released = []
    for name in ("text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2"):
        if getattr(pipe, name, None) is not None:
            setattr(pipe, name, None)
            released.append(name)
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return released


def _encode_conditioning(pipe, spec: ModelSpec, prompt: str) -> dict:
    if spec.family == "flux":
        encoded = pipe.encode_prompt(
            prompt=prompt,
            device="cuda",
            num_images_per_prompt=1,
            max_sequence_length=spec.max_sequence_length or 512,
        )
        return {"prompt_embeds": encoded[0], "pooled_prompt_embeds": encoded[1]}
    if spec.family == "krea2":
        prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(
            prompt=prompt,
            device="cuda",
            num_images_per_prompt=1,
            max_sequence_length=spec.max_sequence_length or 512,
        )
        return {"prompt_embeds": prompt_embeds, "prompt_embeds_mask": prompt_embeds_mask}
    raise ValueError(f"prompt embedding cache is not implemented for {spec.family}")


def _local_pipeline_kwargs(spec: ModelSpec, prompt: str, seed: int, steps: int, args, conditioning=None) -> dict:
    kwargs = {
        "height": args.height,
        "width": args.width,
        "num_inference_steps": steps,
        "guidance_scale": args.guidance if args.guidance is not None else spec.guidance_scale,
        "generator": _generator(seed),
    }
    if conditioning is None:
        kwargs["prompt"] = prompt
    else:
        kwargs.update(conditioning)
    if spec.max_sequence_length is not None:
        kwargs["max_sequence_length"] = spec.max_sequence_length
    return kwargs


def _local_generate(
    pipe,
    spec: ModelSpec,
    prompt: str,
    seed: int,
    steps: int,
    args,
    conditioning: dict | None = None,
) -> tuple[Image.Image, dict]:
    kwargs = _local_pipeline_kwargs(spec, prompt, seed, steps, args, conditioning)
    _reset_peak_vram()
    _sync_cuda()
    started = time.perf_counter()
    result = pipe(**kwargs)
    _sync_cuda()
    elapsed = time.perf_counter() - started
    return result.images[0].convert("RGB"), {
        "wall_s": elapsed,
        "server_s": None,
        "peak_vram_mib": _peak_vram_mib(),
    }


def _local_flow_resume_pair(pipe, spec, prompt, seed, steps, args, conditioning=None):
    kwargs = _local_pipeline_kwargs(spec, prompt, seed, steps, args, conditioning)
    _reset_peak_vram()
    full = capture_flow_latent(pipe, kwargs, steps, args.flow_resume_after)
    full_peak = _peak_vram_mib()
    _reset_peak_vram()
    resumed = resume_flow_latent(pipe, full.latent, kwargs, steps, args.flow_resume_after)
    resume_peak = _peak_vram_mib()
    return [
        (
            full.image.convert("RGB"),
            {
                "wall_s": full.wall_s,
                "server_s": None,
                "peak_vram_mib": full_peak,
                "physical_steps": full.physical_steps,
                "profile": f"full-{steps}",
            },
        ),
        (
            resumed.image.convert("RGB"),
            {
                "wall_s": resumed.wall_s,
                "server_s": None,
                "peak_vram_mib": resume_peak,
                "physical_steps": resumed.physical_steps,
                "profile": f"resume-after-{args.flow_resume_after + 1}",
            },
        ),
    ]


def _api_generate(base_url: str, prompt: str, seed: int, steps: int, args) -> tuple[Image.Image, dict]:
    payload = json.dumps(
        {
            "prompt": prompt,
            "width": args.width,
            "height": args.height,
            "seed": seed,
            "num_inference_steps": steps,
            "guidance_scale": args.guidance if args.guidance is not None else 0.0,
            "lora_id": args.lora,
            "lora_scale": args.lora_scale,
            "auto_lora": False,
            "low_priority": args.low_priority,
            "quality_retry": False,
            "enhance_prompt": False,
            "best_of": 1,
            "score_aesthetic": False,
        }
    ).encode()
    request = Request(
        f"{base_url.rstrip('/')}/generate_image",
        data=payload,
        headers={"content-type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urlopen(request, timeout=args.timeout) as response:
        data = json.load(response)
    elapsed = time.perf_counter() - started
    image = Image.open(io.BytesIO(base64.b64decode(data["image_base64"]))).convert("RGB")
    return image, {
        "wall_s": elapsed,
        "server_s": data.get("inference_time_ms", 0) / 1000,
        "peak_vram_mib": None,
        "response": {
            key: data.get(key)
            for key in ("lora", "quality_retry", "quality_warning", "prompt_used", "prompt_embedding_cache")
        },
    }


def _release_pipeline(pipe):
    del pipe
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    return None


def _summarize(values: list[float]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "min": min(ordered),
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "max": max(ordered),
    }


def _annotate_quality(rows: list[dict], include_lpips: bool) -> None:
    groups: dict[tuple[str, int, int, int], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["model"], row["prompt_index"], row["seed"], row["run"]), []).append(row)
    for group in groups.values():
        reference = max(group, key=lambda row: row["steps"])
        reference_image = Image.open(reference["image_path"]).convert("RGB")
        for row in group:
            image = Image.open(row["image_path"]).convert("RGB")
            row["reference_steps"] = reference["steps"]
            row["reference_profile"] = reference.get("profile")
            row["quality_retention"] = compare_images(image, reference_image)
            if include_lpips:
                row["quality_retention"]["lpips"] = optional_lpips(image, reference_image)


def _pareto_flags(rows: list[dict]) -> None:
    for row in rows:
        quality = row.get("quality_retention", {}).get("ssim", float("-inf"))
        group = (row["model"], row.get("prompt_index"), row.get("run"))
        row["pareto"] = not any(
            other is not row
            and (other["model"], other.get("prompt_index"), other.get("run")) == group
            and other["wall_s"] <= row["wall_s"]
            and other.get("quality_retention", {}).get("ssim", float("-inf")) >= quality
            and (
                other["wall_s"] < row["wall_s"]
                or other.get("quality_retention", {}).get("ssim", float("-inf")) > quality
            )
            for other in rows
        )


def _review_manifest(rows: list[dict], seed: int, output_dir: Path | None = None) -> list[dict]:
    review = [
        {
            "candidate_id": f"candidate-{index + 1:04d}",
            "prompt_index": row["prompt_index"],
            "image_path": row["image_path"],
            "rating_1_to_5": None,
            "prompt_match_1_to_5": None,
            "artifact_notes": "",
        }
        for index, row in enumerate(rows)
    ]
    random.Random(seed).shuffle(review)
    if output_dir is not None:
        review_dir = output_dir / "review"
        review_dir.mkdir(exist_ok=True)
        for index, item in enumerate(review):
            source_image_path = item["image_path"]
            suffix = Path(source_image_path).suffix
            opaque = review_dir / f"candidate-{index + 1:04d}{suffix}"
            shutil.copy2(source_image_path, opaque)
            item["candidate_id"] = opaque.stem
            item["image_path"] = str(opaque)
            item["_source_image_path"] = source_image_path
    return review


def _contact_sheet(rows: list[dict], output: Path) -> None:
    if not rows:
        return
    thumb = 256
    label = 52
    cols = min(4, len(rows))
    grid_rows = (len(rows) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, grid_rows * (thumb + label)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, row in enumerate(rows):
        image = Image.open(row["image_path"]).convert("RGB")
        image.thumbnail((thumb, thumb), Image.Resampling.LANCZOS)
        x = (index % cols) * thumb
        y = (index // cols) * (thumb + label)
        sheet.paste(image, (x, y))
        draw.text(
            (x + 4, y + thumb + 4),
            f"{row['model']}{'/' + row['profile'] if row.get('profile') else ''} {row['steps']} steps\n"
            f"{row['wall_s']:.2f}s p{row['prompt_index']}",
            fill="black",
            font=font,
        )
    sheet.save(output, quality=92)


def _markdown(report: dict) -> str:
    lines = [
        "# Diffusion Frontier",
        "",
        f"Generated: `{report['timestamp']}`",
        f"Target: `<={report['target_latency_s']:.3f}s` warm batch-1 latency",
        "",
        "Pixel/SSIM/LPIPS values below are same-model, same-seed retention metrics. They are not cross-model quality scores; use `review_manifest.json` for blind model preference.",
        "",
        "| Model | Profile | Steps | Prompt | Run | Wall s | Server s | Peak VRAM MiB | SSIM to max-step | PSNR dB | <= target | Pareto |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in report["rows"]:
        quality = row.get("quality_retention", {})
        server = "" if row.get("server_s") is None else f"{row['server_s']:.3f}"
        peak = "" if row.get("peak_vram_mib") is None else f"{row['peak_vram_mib']:.1f}"
        lines.append(
            f"| {row['model']} | {row.get('profile', '')} | {row['steps']} | {row['prompt_index']} | {row['run']} | {row['wall_s']:.3f} | {server} | {peak} | "
            f"{quality.get('ssim', 0):.4f} | {quality.get('psnr_db', 0):.2f} | "
            f"{'yes' if row['meets_latency_target'] else 'no'} | {'yes' if row.get('pareto') else 'no'} |"
        )
    lines.extend(["", "## Runtime", "", "```json", json.dumps(report["system"], indent=2), "```", ""])
    return "\n".join(lines)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_CATALOG))
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--api-base-url", help="Use the existing Z-Image HTTP server instead of loading it locally")
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--steps", nargs="+", type=int)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--target-latency-s", type=float, default=1.0)
    parser.add_argument("--guidance", type=float)
    parser.add_argument("--offload", choices=("auto", "cuda", "model", "sequential"), default="auto")
    parser.add_argument("--compile", choices=("default", "reduce-overhead", "max-autotune"))
    parser.add_argument("--compile-fullgraph", action="store_true")
    parser.add_argument(
        "--quantization",
        choices=("torchao-fp8dq", "torchao-fp8wo", "torchao-int8wo", "torchao-int4wo"),
        help="Quantize the Diffusers transformer during loading (local pipelines only)",
    )
    parser.add_argument(
        "--cache-prompt-embeds",
        action="store_true",
        help="Pre-encode FLUX/Krea prompts outside timed generation, matching a warm prompt service",
    )
    parser.add_argument(
        "--release-text-encoders",
        action="store_true",
        help="Release text encoders after --cache-prompt-embeds to maximize resident denoiser headroom",
    )
    parser.add_argument(
        "--flow-resume-after",
        type=int,
        help="Also time exact FLUX/Krea resume after this many completed zero-based steps",
    )
    parser.add_argument("--lora", help="LoRA path/repository for local pipelines, or LoRA id for --api-base-url")
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--low-priority", action="store_true")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output-dir", default="experiments/results")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.list_models:
        print(json.dumps({key: spec.to_dict() for key, spec in MODEL_CATALOG.items()}, indent=2))
        return 0
    if not args.models:
        raise SystemExit("--models is required unless --list-models is used")
    if args.api_base_url and args.models != ["zimage-turbo"]:
        raise SystemExit("--api-base-url currently supports exactly --models zimage-turbo")
    if args.compile_fullgraph and not args.compile:
        raise SystemExit("--compile-fullgraph requires --compile")
    if args.release_text_encoders and not args.cache_prompt_embeds:
        raise SystemExit("--release-text-encoders requires --cache-prompt-embeds")
    if args.flow_resume_after is not None:
        if args.api_base_url or any(get_model(key).family not in {"flux", "krea2"} for key in args.models):
            raise SystemExit("--flow-resume-after supports local FLUX/Krea models only")

    system = system_snapshot(args.output_dir if Path(args.output_dir).exists() else ".")
    checks = {key: inspect_model(get_model(key), system, allow_download=args.allow_download) for key in args.models}
    if args.api_base_url:
        checks["zimage-turbo"]["ready"] = True
        checks["zimage-turbo"]["blockers"] = []
        checks["zimage-turbo"]["warnings"].append("using the HTTP server; local weight/pipeline checks do not apply")
    if args.preflight:
        print(json.dumps({"system": system, "models": checks}, indent=2))
        return 0 if all(item["ready"] for item in checks.values()) else 2
    blocked = {key: check["blockers"] for key, check in checks.items() if not check["ready"]}
    if blocked:
        print(json.dumps({"error": "preflight failed", "models": blocked}, indent=2))
        return 2

    run_dir = Path(args.output_dir) / f"diffusion_frontier_{_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    prompts = args.prompt or DEFAULT_PROMPTS
    rows = []
    runtimes = {}
    for model_key in args.models:
        spec = get_model(model_key)
        pipe = None
        conditioning_cache = {}
        stage_metadata = {}
        stage_conditioning = _should_stage_krea_conditioning(spec, args)
        if stage_conditioning:
            conditioning_cache, stage_metadata = _stage_krea_conditioning(spec, prompts, args)
        if not args.api_base_url:
            pipe, runtimes[model_key] = _load_pipeline(
                spec,
                args,
                checks[model_key],
                skip_text_encoder=stage_conditioning,
            )
            runtimes[model_key].update(stage_metadata)
        else:
            runtimes[model_key] = {"load_s": 0.0, "offload": "remote-api", "lora": args.lora}

        if stage_conditioning:
            conditioning_cache = {
                index: {key: value.to("cuda") for key, value in values.items()}
                for index, values in conditioning_cache.items()
            }
            runtimes[model_key]["released_components"] = ["text_encoder", "tokenizer"]
        elif pipe is not None and args.cache_prompt_embeds:
            conditioning_cache = {
                index: _encode_conditioning(pipe, spec, prompt) for index, prompt in enumerate(prompts)
            }
            if args.release_text_encoders:
                runtimes[model_key]["released_components"] = _release_text_encoders(pipe)

        steps_list = sorted(set(args.steps or [spec.default_steps, spec.reference_steps]))
        if pipe is not None and args.warmups:
            for _ in range(args.warmups):
                _local_generate(
                    pipe,
                    spec,
                    prompts[0],
                    args.seed,
                    steps_list[0],
                    args,
                    conditioning_cache.get(0),
                )

        for prompt_index, prompt in enumerate(prompts):
            seed = args.seed + prompt_index
            for steps in steps_list:
                for run in range(args.runs):
                    if args.api_base_url:
                        generated = [(*_api_generate(args.api_base_url, prompt, seed, steps, args),)]
                    elif args.flow_resume_after is not None:
                        generated = _local_flow_resume_pair(
                            pipe,
                            spec,
                            prompt,
                            seed,
                            steps,
                            args,
                            conditioning_cache.get(prompt_index),
                        )
                    else:
                        generated = [
                            _local_generate(
                                pipe,
                                spec,
                                prompt,
                                seed,
                                steps,
                                args,
                                conditioning_cache.get(prompt_index),
                            )
                        ]
                    for image, timing in generated:
                        suffix = f"_{timing['profile']}" if timing.get("profile") else ""
                        path = run_dir / f"{model_key}_p{prompt_index:02d}_s{steps:02d}_r{run:02d}{suffix}.webp"
                        image.save(path, format="WEBP", quality=95)
                        rows.append(
                            {
                                "model": model_key,
                                "family": spec.family,
                                "backend": args.quantization or "bf16",
                                "prompt_index": prompt_index,
                                "prompt": prompt,
                                "seed": seed,
                                "width": args.width,
                                "height": args.height,
                                "steps": steps,
                                "run": run,
                                "image_path": str(path),
                                **timing,
                                "meets_latency_target": timing["wall_s"] <= args.target_latency_s,
                            }
                        )
                        print(
                            json.dumps(
                                {
                                    key: rows[-1].get(key)
                                    for key in (
                                        "model",
                                        "profile",
                                        "prompt_index",
                                        "steps",
                                        "run",
                                        "wall_s",
                                        "image_path",
                                    )
                                }
                            )
                        )
        if pipe is not None:
            pipe = _release_pipeline(pipe)

    _annotate_quality(rows, args.lpips)
    _pareto_flags(rows)
    summary = {}
    for model_key in args.models:
        summary[model_key] = {}
        model_rows = [row for row in rows if row["model"] == model_key]
        for profile in sorted({row.get("profile") or f"{row['steps']}-step" for row in model_rows}):
            values = [
                row["wall_s"]
                for row in model_rows
                if (row.get("profile") or f"{row['steps']}-step") == profile
            ]
            summary[model_key][profile] = _summarize(values)
    report = {
        "schema_version": 1,
        "timestamp": _stamp(),
        "target_latency_s": args.target_latency_s,
        "system": system,
        "preflight": checks,
        "runtime": runtimes,
        "prompts": prompts,
        "summary": summary,
        "rows": rows,
    }
    (run_dir / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    review = _review_manifest(rows, args.seed, run_dir)
    row_by_path = {row["image_path"]: row for row in rows}
    answer_key = {}
    for item in review:
        source_path = item.pop("_source_image_path")
        row = row_by_path[source_path]
        answer_key[item["candidate_id"]] = {
            key: row.get(key)
            for key in ("model", "backend", "profile", "prompt_index", "seed", "steps", "physical_steps", "run")
        }
    (run_dir / "review_manifest.json").write_text(json.dumps(review, indent=2))
    (run_dir / "review_answer_key.json").write_text(json.dumps(answer_key, indent=2))
    (run_dir / "report.md").write_text(_markdown(report))
    _contact_sheet(rows, run_dir / "contact_sheet.jpg")
    print(run_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
