"""Scheduler-aware native versus distilled SDXL frontier benchmark.

This evaluator keeps the base checkpoint fixed and compares a native 40-step
reference with public LCM-LoRA and Hyper-SD acceleration profiles. Downloads
remain opt-in through ``--allow-download``.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import shutil
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from .benchmark import (
    DEFAULT_PROMPTS,
    _contact_sheet,
    _generator,
    _peak_vram_mib,
    _release_pipeline,
    _reset_peak_vram,
    _single_file_config,
    _source_for,
    _stamp,
    _sync_cuda,
)
from .catalog import MODEL_CATALOG, ModelSpec, get_model
from .metrics import compare_images, optional_lpips
from .preflight import inspect_model, system_snapshot


@dataclass(frozen=True)
class SDXLProfile:
    key: str
    steps: int
    guidance_scale: float
    scheduler: str
    adapter: str | None = None
    adapter_repo: str | None = None
    adapter_weight: str | None = None
    source: str = ""


SDXL_PROFILES: dict[str, SDXLProfile] = {
    "native20": SDXLProfile(
        key="native20",
        steps=20,
        guidance_scale=7.0,
        scheduler="dpmpp-2m-karras",
        source="https://huggingface.co/docs/diffusers/en/using-diffusers/schedulers",
    ),
    "native40": SDXLProfile(
        key="native40",
        steps=40,
        guidance_scale=7.0,
        scheduler="dpmpp-2m-karras",
        source="https://huggingface.co/docs/diffusers/en/using-diffusers/schedulers",
    ),
    "lcm4": SDXLProfile(
        key="lcm4",
        steps=4,
        guidance_scale=1.0,
        scheduler="lcm",
        adapter="lcm-sdxl",
        adapter_repo="latent-consistency/lcm-lora-sdxl",
        adapter_weight="pytorch_lora_weights.safetensors",
        source="https://huggingface.co/docs/diffusers/main/using-diffusers/inference_with_lcm_lora",
    ),
    "lcm8": SDXLProfile(
        key="lcm8",
        steps=8,
        guidance_scale=1.0,
        scheduler="lcm",
        adapter="lcm-sdxl",
        adapter_repo="latent-consistency/lcm-lora-sdxl",
        adapter_weight="pytorch_lora_weights.safetensors",
        source="https://huggingface.co/docs/diffusers/main/using-diffusers/inference_with_lcm_lora",
    ),
    "hyper4": SDXLProfile(
        key="hyper4",
        steps=4,
        guidance_scale=0.0,
        scheduler="ddim-trailing",
        adapter="hyper-sdxl-4",
        adapter_repo="ByteDance/Hyper-SD",
        adapter_weight="Hyper-SDXL-4steps-lora.safetensors",
        source="https://huggingface.co/ByteDance/Hyper-SD",
    ),
    "hyper8": SDXLProfile(
        key="hyper8",
        steps=8,
        guidance_scale=0.0,
        scheduler="ddim-trailing",
        adapter="hyper-sdxl-8",
        adapter_repo="ByteDance/Hyper-SD",
        adapter_weight="Hyper-SDXL-8steps-lora.safetensors",
        source="https://huggingface.co/ByteDance/Hyper-SD",
    ),
}

DEFAULT_PROFILES = ("native20", "native40", "lcm4", "lcm8", "hyper4")


def _adapter_specs(profiles: list[SDXLProfile]) -> dict[str, SDXLProfile]:
    return {profile.adapter: profile for profile in profiles if profile.adapter is not None}


def _load_pipeline(spec: ModelSpec, args, profiles: list[SDXLProfile]):
    import diffusers
    import torch

    cls = getattr(diffusers, spec.pipeline_class)
    kwargs = {
        "torch_dtype": torch.float16,
        "local_files_only": not args.allow_download,
        "use_safetensors": True,
    }
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
        pipe = cls.from_pretrained(spec.repo_id, **kwargs)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    adapters = {}
    for adapter_name, profile in _adapter_specs(profiles).items():
        pipe.load_lora_weights(
            profile.adapter_repo,
            weight_name=profile.adapter_weight,
            adapter_name=adapter_name,
            local_files_only=not args.allow_download,
        )
        adapters[adapter_name] = {
            "repo_id": profile.adapter_repo,
            "weight_name": profile.adapter_weight,
        }
    if args.style_lora:
        style_kwargs = {
            "adapter_name": "style",
            "local_files_only": not args.allow_download,
        }
        if args.style_lora_weight_name:
            style_kwargs["weight_name"] = args.style_lora_weight_name
        pipe.load_lora_weights(args.style_lora, **style_kwargs)
        adapters["style"] = {
            "repo_id": args.style_lora,
            "weight_name": args.style_lora_weight_name,
            "scale": args.style_lora_scale,
        }

    return pipe, {
        "load_s": time.perf_counter() - started,
        "dtype": "float16",
        "offload": "cuda",
        "prompt_embeddings_cached": not args.include_text_encode,
        "adapters": adapters,
    }


def _configure_profile(pipe, profile: SDXLProfile, base_scheduler_config: dict, args) -> None:
    from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, LCMScheduler

    if profile.scheduler == "dpmpp-2m-karras":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            base_scheduler_config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True,
        )
    elif profile.scheduler == "lcm":
        pipe.scheduler = LCMScheduler.from_config(base_scheduler_config)
    elif profile.scheduler == "ddim-trailing":
        pipe.scheduler = DDIMScheduler.from_config(base_scheduler_config, timestep_spacing="trailing")
    else:
        raise ValueError(f"unsupported SDXL scheduler profile: {profile.scheduler}")

    adapter_names = [name for name in (profile.adapter, "style" if args.style_lora else None) if name]
    adapter_weights = [1.0 if name != "style" else args.style_lora_scale for name in adapter_names]
    fused_adapters = getattr(pipe, "_frontier_fused_adapters", ())
    target_adapters = tuple(adapter_names)
    if fused_adapters and fused_adapters != target_adapters:
        pipe.unfuse_lora(components=["unet"])
        pipe._frontier_fused_adapters = ()

    if not adapter_names:
        pipe.disable_lora()
    else:
        pipe.enable_lora()
        pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
        if fused_adapters != target_adapters:
            pipe.fuse_lora(components=["unet"], adapter_names=adapter_names)
            pipe._frontier_fused_adapters = target_adapters


def _encode_prompt(pipe, prompt: str) -> dict:
    prompt_embeds, negative_prompt_embeds, pooled, negative_pooled = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=None,
        device="cuda",
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt="",
        negative_prompt_2=None,
    )
    return {
        "prompt_embeds": prompt_embeds,
        "negative_prompt_embeds": negative_prompt_embeds,
        "pooled_prompt_embeds": pooled,
        "negative_pooled_prompt_embeds": negative_pooled,
    }


def _generate(pipe, profile: SDXLProfile, prompt: str, seed: int, args, conditioning: dict | None):
    kwargs = {
        "height": args.height,
        "width": args.width,
        "num_inference_steps": profile.steps,
        "guidance_scale": profile.guidance_scale,
        "generator": _generator(seed),
    }
    if conditioning is None:
        kwargs["prompt"] = prompt
    else:
        kwargs.update(conditioning)
    _reset_peak_vram()
    _sync_cuda()
    started = time.perf_counter()
    image = pipe(**kwargs).images[0].convert("RGB")
    _sync_cuda()
    return image, time.perf_counter() - started, _peak_vram_mib()


def _annotate_native_retention(rows: list[dict], include_lpips: bool = False) -> None:
    references = {
        (row["model"], row["prompt_index"], row["seed"], row["run"]): row
        for row in rows
        if row["profile"] == "native40"
    }
    for row in rows:
        key = (row["model"], row["prompt_index"], row["seed"], row["run"])
        reference = references.get(key)
        if reference is None:
            raise ValueError(f"missing native40 reference for {key}")
        image = Image.open(row["image_path"]).convert("RGB")
        reference_image = Image.open(reference["image_path"]).convert("RGB")
        row["quality_reference"] = "native40"
        row["quality_retention"] = compare_images(image, reference_image)
        if include_lpips:
            row["quality_retention"]["lpips"] = optional_lpips(image, reference_image)


def _pareto_flags(rows: list[dict]) -> None:
    for row in rows:
        quality = row["quality_retention"]["ssim"]
        peers = [
            other
            for other in rows
            if other["model"] == row["model"]
            and other["prompt_index"] == row["prompt_index"]
            and other["run"] == row["run"]
        ]
        row["pareto"] = not any(
            other is not row
            and other["wall_s"] <= row["wall_s"]
            and other["quality_retention"]["ssim"] >= quality
            and (other["wall_s"] < row["wall_s"] or other["quality_retention"]["ssim"] > quality)
            for other in peers
        )


def _review_artifacts(rows: list[dict], output_dir: Path, seed: int) -> tuple[list[dict], dict]:
    review_dir = output_dir / "review"
    review_dir.mkdir(exist_ok=True)
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    manifest = []
    answer_key = {}
    for candidate_number, row_index in enumerate(order, 1):
        row = rows[row_index]
        candidate_id = f"candidate-{candidate_number:04d}"
        target = review_dir / f"{candidate_id}{Path(row['image_path']).suffix}"
        shutil.copy2(row["image_path"], target)
        manifest.append(
            {
                "candidate_id": candidate_id,
                "prompt_index": row["prompt_index"],
                "image_path": str(target),
                "quality_1_to_5": None,
                "prompt_match_1_to_5": None,
                "artifact_notes": "",
            }
        )
        answer_key[candidate_id] = {
            "model": row["model"],
            "profile": row["profile"],
            "prompt_index": row["prompt_index"],
            "seed": row["seed"],
            "run": row["run"],
            "source_image_path": row["image_path"],
        }
    return manifest, answer_key


def _summary(rows: list[dict]) -> dict:
    result = {}
    for model in sorted({row["model"] for row in rows}):
        result[model] = {}
        for profile in sorted({row["profile"] for row in rows if row["model"] == model}):
            group = [row for row in rows if row["model"] == model and row["profile"] == profile]
            walls = [row["wall_s"] for row in group]
            ssims = [row["quality_retention"]["ssim"] for row in group]
            result[model][profile] = {
                "n": len(group),
                "wall_s": {
                    "min": min(walls),
                    "median": statistics.median(walls),
                    "mean": statistics.fmean(walls),
                    "max": max(walls),
                },
                "ssim_to_native40": {
                    "min": min(ssims),
                    "median": statistics.median(ssims),
                    "mean": statistics.fmean(ssims),
                    "max": max(ssims),
                },
                "latency_passes": sum(row["meets_latency_target"] for row in group),
            }
    return result


def _markdown(report: dict) -> str:
    lines = [
        "# SDXL Native/Distilled Frontier",
        "",
        f"Generated: `{report['timestamp']}`",
        f"Target: `<={report['target_latency_s']:.3f}s` warm batch-1 latency",
        "",
        "All SSIM/PSNR values are same-base-model, same-seed retention against `native40`. They are not perceptual model rankings; complete the blind review manifest.",
        "",
        "| Model | Profile | Steps | Prompt | Run | Wall s | Peak VRAM MiB | SSIM to native40 | PSNR dB | <= target | Pareto |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in report["rows"]:
        quality = row["quality_retention"]
        lines.append(
            f"| {row['model']} | {row['profile']} | {row['physical_steps']} | {row['prompt_index']} | "
            f"{row['run']} | {row['wall_s']:.3f} | {row['peak_vram_mib']:.1f} | {quality['ssim']:.4f} | "
            f"{quality['psnr_db']:.2f} | {'yes' if row['meets_latency_target'] else 'no'} | "
            f"{'yes' if row['pareto'] else 'no'} |"
        )
    lines.extend(["", "## Summary", "", "```json", json.dumps(report["summary"], indent=2), "```", ""])
    return "\n".join(lines)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(key for key, spec in MODEL_CATALOG.items() if spec.family == "sdxl"),
        required=True,
    )
    parser.add_argument("--profiles", nargs="+", choices=sorted(SDXL_PROFILES), default=list(DEFAULT_PROFILES))
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--target-latency-s", type=float, default=1.0)
    parser.add_argument("--include-text-encode", action="store_true")
    parser.add_argument("--style-lora", help="Optional SDXL style LoRA path or Hub repository")
    parser.add_argument("--style-lora-weight-name", help="Specific safetensors filename inside --style-lora")
    parser.add_argument("--style-lora-scale", type=float, default=1.0)
    parser.add_argument("--compile", choices=("default", "reduce-overhead", "max-autotune"))
    parser.add_argument("--compile-fullgraph", action="store_true")
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--output-dir", default="experiments/results")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    if "native40" not in args.profiles:
        raise SystemExit("--profiles must include native40 as the quality reference")
    if args.compile_fullgraph and not args.compile:
        raise SystemExit("--compile-fullgraph requires --compile")
    profiles = [SDXL_PROFILES[key] for key in args.profiles]
    accelerated_adapters = {profile.adapter for profile in profiles if profile.adapter}
    if args.compile and len(accelerated_adapters) > 1:
        raise SystemExit("--compile supports one accelerated adapter per run plus native references")
    if args.compile:
        first_accelerated = next((index for index, profile in enumerate(profiles) if profile.adapter), len(profiles))
        if any(profile.adapter is None for profile in profiles[first_accelerated:]):
            raise SystemExit("with --compile, put native reference profiles before the accelerated profile")
    system = system_snapshot(args.output_dir if Path(args.output_dir).exists() else ".")
    checks = {key: inspect_model(get_model(key), system, allow_download=args.allow_download) for key in args.models}
    blocked = {key: check["blockers"] for key, check in checks.items() if not check["ready"]}
    if blocked:
        print(json.dumps({"error": "preflight failed", "models": blocked}, indent=2))
        return 2

    run_dir = Path(args.output_dir) / f"sdxl_frontier_{_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    prompts = args.prompt or DEFAULT_PROMPTS
    rows = []
    runtimes = {}
    for model_key in args.models:
        spec = get_model(model_key)
        pipe, runtimes[model_key] = _load_pipeline(spec, args, profiles)
        base_scheduler_config = dict(pipe.scheduler.config)
        conditioning_cache = {}
        if not args.include_text_encode:
            if args.style_lora:
                pipe.enable_lora()
                pipe.set_adapters(["style"], adapter_weights=[args.style_lora_scale])
            else:
                pipe.disable_lora()
            conditioning_cache = {index: _encode_prompt(pipe, prompt) for index, prompt in enumerate(prompts)}

        compiled = False
        for profile in profiles:
            _configure_profile(pipe, profile, base_scheduler_config, args)
            if args.compile and profile.adapter is not None and not compiled:
                import torch

                pipe.unet = torch.compile(pipe.unet, mode=args.compile, fullgraph=args.compile_fullgraph)
                runtimes[model_key]["compiled_component"] = "unet"
                runtimes[model_key]["compile_mode"] = args.compile
                runtimes[model_key]["compile_fullgraph"] = args.compile_fullgraph
                compiled = True
            if args.warmups:
                conditioning = conditioning_cache.get(0)
                for _ in range(args.warmups):
                    _generate(pipe, profile, prompts[0], args.seed, args, conditioning)
            for prompt_index, prompt in enumerate(prompts):
                conditioning = conditioning_cache.get(prompt_index)
                for run in range(args.runs):
                    image, wall_s, peak_vram_mib = _generate(
                        pipe, profile, prompt, args.seed + prompt_index, args, conditioning
                    )
                    path = run_dir / f"{model_key}_{profile.key}_p{prompt_index:02d}_r{run:02d}.webp"
                    image.save(path, format="WEBP", quality=95)
                    row = {
                        "model": model_key,
                        "family": "sdxl",
                        "profile": profile.key,
                        "scheduler": profile.scheduler,
                        "adapter": profile.adapter,
                        "physical_steps": profile.steps,
                        "guidance_scale": profile.guidance_scale,
                        "prompt_index": prompt_index,
                        "prompt": prompt,
                        "seed": args.seed + prompt_index,
                        "width": args.width,
                        "height": args.height,
                        "run": run,
                        "image_path": str(path),
                        "wall_s": wall_s,
                        "peak_vram_mib": peak_vram_mib,
                        "meets_latency_target": wall_s <= args.target_latency_s,
                    }
                    rows.append(row)
                    print(json.dumps({key: row[key] for key in ("model", "profile", "prompt_index", "run", "wall_s")}))
        conditioning_cache.clear()
        gc.collect()
        pipe = _release_pipeline(pipe)

    _annotate_native_retention(rows, include_lpips=args.lpips)
    _pareto_flags(rows)
    report = {
        "schema_version": 1,
        "timestamp": _stamp(),
        "target_latency_s": args.target_latency_s,
        "system": system,
        "preflight": checks,
        "runtime": runtimes,
        "profiles": {profile.key: asdict(profile) for profile in profiles},
        "prompts": prompts,
        "summary": _summary(rows),
        "rows": rows,
    }
    manifest, answer_key = _review_artifacts(rows, run_dir, args.seed)
    (run_dir / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    (run_dir / "review_manifest.json").write_text(json.dumps(manifest, indent=2))
    (run_dir / "review_answer_key.json").write_text(json.dumps(answer_key, indent=2))
    (run_dir / "report.md").write_text(_markdown(report))
    _contact_sheet(
        [
            {**row, "model": f"{row['model']}/{row['profile']}", "steps": row["physical_steps"]}
            for row in rows
        ],
        run_dir / "contact_sheet.jpg",
    )
    print(run_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
