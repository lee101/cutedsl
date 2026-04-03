"""Run Z-Image baseline, LeMiCa, and MeanCache generations into review folders."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from diffusers import ZImagePipeline
from diffusers.models.transformers import ZImageTransformer2DModel
from PIL import Image


DEFAULT_PROMPT = (
    "A cinematic, melancholic photograph of a solitary hooded figure walking "
    "through a sprawling, rain-slicked metropolis at night. The city lights are "
    "a chaotic blur of neon orange and cool blue, reflecting on the wet asphalt. "
    "Superimposed over the scene in a sleek modern font is the quote: "
    "'THE CITY IS A CIRCUIT BOARD, AND I AM A BROKEN TRANSISTOR.'"
)

LEMICA_MODES = {
    "slow": [0, 1, 2, 3, 5, 7, 8, 9],
    "medium": [0, 1, 2, 4, 6, 8, 9],
    "fast": [0, 1, 2, 5, 8, 9],
}

MEANCACHE_CALC_DICT = {
    25: [0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 15, 16, 17, 18, 20, 22, 29, 37, 42, 45, 47, 48, 49],
    20: [0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 30, 38, 43, 46, 48, 49],
    17: [0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 19, 27, 35, 43, 46, 48, 49],
    15: [0, 1, 2, 3, 5, 7, 10, 12, 17, 25, 33, 41, 45, 48, 49],
    13: [0, 1, 2, 3, 5, 9, 15, 22, 29, 36, 43, 47, 49],
}
MEANCACHE_MAPPING_RULES = {
    "v_diff_mean": 1,
    "v_diff_mean_jvp1_s2": 2,
    "v_diff_mean_jvp1_s3": 3,
    "v_diff_mean_jvp1_s4": 4,
    "v_diff_mean_jvp1_s5": 5,
    "v_diff_mean_jvp1_s6": 6,
    "chain": 0,
}
MEANCACHE_EDGE_SOURCE = {
    25: ["chain", "chain", "chain", "chain", "chain", "chain", "v_diff_mean_jvp1_s2", "v_diff_mean_jvp1_s2", "chain", "chain", "chain", "v_diff_mean_jvp1_s2", "chain", "chain", "chain", "v_diff_mean_jvp1_s5", "v_diff_mean_jvp1_s5", "v_diff_mean", "v_diff_mean", "v_diff_mean", "v_diff_mean_jvp1_s5", "v_diff_mean_jvp1_s5", "chain", "chain"],
    20: ["chain", "chain", "chain", "chain", "v_diff_mean_jvp1_s2", "v_diff_mean_jvp1_s2", "v_diff_mean_jvp1_s2", "v_diff_mean_jvp1_s2", "v_diff_mean_jvp1_s2", "v_diff_mean_jvp1_s5", "v_diff_mean_jvp1_s4", "v_diff_mean_jvp1_s5", "v_diff_mean_jvp1_s5", "v_diff_mean", "v_diff_mean", "v_diff_mean", "v_diff_mean_jvp1_s5", "v_diff_mean_jvp1_s2", "chain"],
    17: ["chain", "chain", "chain", "chain", "v_diff_mean_jvp1_s2", "v_diff_mean_jvp1_s2", "v_diff_mean_jvp1_s2", "v_diff_mean_jvp1_s2", "v_diff_mean_jvp1_s2", "v_diff_mean", "v_diff_mean", "v_diff_mean", "v_diff_mean", "v_diff_mean_jvp1_s5", "v_diff_mean_jvp1_s2", "chain"],
    15: ["chain", "chain", "chain", "v_diff_mean_jvp1_s2", "v_diff_mean_jvp1_s2", "v_diff_mean_jvp1_s2", "v_diff_mean_jvp1_s2", "v_diff_mean", "v_diff_mean", "v_diff_mean", "v_diff_mean", "v_diff_mean_jvp1_s5", "v_diff_mean_jvp1_s4", "chain"],
    13: ["chain", "chain", "chain", "v_diff_mean_jvp1_s2", "v_diff_mean", "v_diff_mean", "v_diff_mean", "v_diff_mean", "v_diff_mean", "v_diff_mean", "v_diff_mean_jvp1_s5", "v_diff_mean_jvp1_s2"],
}


def load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    module.Any = Any
    module.Callable = Callable
    module.Dict = Dict
    module.List = List
    module.Optional = Optional
    module.Union = Union
    spec.loader.exec_module(module)
    return module


def image_stats(image: Image.Image) -> dict[str, float]:
    arr = np.asarray(image).astype(np.float32) / 255.0
    gray = arr.mean(axis=2)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "near_black_fraction": float((gray < 0.02).mean()),
        "near_white_fraction": float((gray > 0.98).mean()),
    }


def save_result(out_path: Path, image: Image.Image, meta: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    meta = dict(meta)
    meta["image_stats"] = image_stats(image)
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


def build_meancache_lists(cache_jvp: int, total_steps: int = 50) -> tuple[list[bool], list[int]]:
    calc_steps = MEANCACHE_CALC_DICT[cache_jvp]
    bool_list = [False] * total_steps
    for idx in calc_steps:
        if idx < total_steps:
            bool_list[idx] = True
    edge_rule_names = MEANCACHE_EDGE_SOURCE[cache_jvp]
    edge_order = [MEANCACHE_MAPPING_RULES[name] for name in edge_rule_names]
    result_edge_order = [0] * total_steps
    for i in range(len(calc_steps) - 1):
        start = calc_steps[i]
        end = calc_steps[i + 1]
        for pos in range(start, end):
            result_edge_order[pos] = edge_order[i]
    return bool_list, result_edge_order


def run_pipeline(pipe: ZImagePipeline, prompt: str, height: int, width: int, steps: int, guidance: float, seed: int) -> tuple[Image.Image, float]:
    generator = torch.Generator("cuda").manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    start = time.perf_counter()
    image = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    ).images[0]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return image, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Z-Image baselines into artifacts folders")
    parser.add_argument("--output-root", default="artifacts/zimage_review")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--turbo-model-id", default="Tongyi-MAI/Z-Image-Turbo")
    parser.add_argument("--base-model-id", default="Tongyi-MAI/Z-Image")
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--turbo-steps", type=int, default=9)
    parser.add_argument("--base-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--meancache-guidance-scale", type=float, default=4.0)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-lemica", action="store_true")
    parser.add_argument("--skip-meancache", action="store_true")
    parser.add_argument("--cpu-offload", action="store_true")
    args = parser.parse_args()

    root = Path(args.output_root)
    dtype = torch.bfloat16
    repo_root = Path(__file__).resolve().parent.parent
    lemica_module = load_module_from_path(
        "lemica_zimage",
        repo_root / "external" / "LeMiCa" / "LeMiCa4Z-Image" / "inference_zimage.py",
    )
    meancache_module = load_module_from_path(
        "meancache_zimage",
        repo_root / "external" / "MeanCache" / "MeanCache4Z-Image" / "MC_zimage.py",
    )
    lemica_call = lemica_module.Lemica_call
    meancache_inference_fn = meancache_module.meancache_inference

    def lemica_forward_wrapper(self, x, t, cap_feats, **kwargs):
        kwargs.pop("return_dict", None)
        return lemica_call(self, x, t, cap_feats, **kwargs)

    if not args.skip_baseline or not args.skip_lemica:
        turbo_pipe = ZImagePipeline.from_pretrained(
            args.turbo_model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        )
        if args.cpu_offload:
            turbo_pipe.enable_model_cpu_offload()
        else:
            turbo_pipe.to("cuda")

        if not args.skip_baseline:
            image, elapsed = run_pipeline(
                turbo_pipe,
                args.prompt,
                args.height,
                args.width,
                args.turbo_steps,
                args.guidance_scale,
                args.seed,
            )
            save_result(
                root / "01_baseline" / "baseline_turbo.png",
                image,
                {
                    "variant": "baseline_turbo",
                    "elapsed_s": elapsed,
                    "steps": args.turbo_steps,
                    "guidance_scale": args.guidance_scale,
                    "prompt": args.prompt,
                    "model_id": args.turbo_model_id,
                },
            )

        if not args.skip_lemica:
            original_forward = ZImageTransformer2DModel.forward
            try:
                ZImageTransformer2DModel.forward = lemica_forward_wrapper
                turbo_pipe.transformer.__class__.enable_lemica = True
                turbo_pipe.transformer.__class__.cnt = 0
                turbo_pipe.transformer.__class__.num_steps = args.turbo_steps
                turbo_pipe.transformer.__class__.previous_residual = None
                turbo_pipe.transformer.__class__.store = []
                for mode, calc_steps in LEMICA_MODES.items():
                    turbo_pipe.transformer.__class__.bool_list = [i in calc_steps for i in range(args.turbo_steps + 1)]
                    turbo_pipe.transformer.__class__.cnt = 0
                    turbo_pipe.transformer.__class__.previous_residual = None
                    turbo_pipe.transformer.__class__.store = []
                    image, elapsed = run_pipeline(
                        turbo_pipe,
                        args.prompt,
                        args.height,
                        args.width,
                        args.turbo_steps,
                        args.guidance_scale,
                        args.seed,
                    )
                    save_result(
                        root / "02_lemica" / f"lemica_{mode}.png",
                        image,
                        {
                            "variant": f"lemica_{mode}",
                            "elapsed_s": elapsed,
                            "steps": args.turbo_steps,
                            "guidance_scale": args.guidance_scale,
                            "prompt": args.prompt,
                            "model_id": args.turbo_model_id,
                            "calc_steps": calc_steps,
                        },
                    )
            finally:
                ZImageTransformer2DModel.forward = original_forward
                del turbo_pipe
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    if not args.skip_meancache:
        base_pipe = ZImagePipeline.from_pretrained(
            args.base_model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        )
        if args.cpu_offload:
            base_pipe.enable_model_cpu_offload()
        else:
            base_pipe.to("cuda")
        original_call = ZImagePipeline.__call__
        try:
            baseline_image, baseline_elapsed = run_pipeline(
                base_pipe,
                args.prompt,
                args.height,
                args.width,
                args.base_steps,
                args.meancache_guidance_scale,
                args.seed,
            )
            save_result(
                root / "03_meancache" / "base_50step.png",
                baseline_image,
                {
                    "variant": "base_50step",
                    "elapsed_s": baseline_elapsed,
                    "steps": args.base_steps,
                    "guidance_scale": args.meancache_guidance_scale,
                    "prompt": args.prompt,
                    "model_id": args.base_model_id,
                },
            )

            ZImagePipeline.__call__ = meancache_inference_fn
            for cache_jvp in [25, 20, 15, 13]:
                bool_list, edge_order = build_meancache_lists(cache_jvp, args.base_steps)
                ZImagePipeline.should_calc_list = bool_list
                ZImagePipeline.edge_order = edge_order
                ZImagePipeline.cache_jvp = True
                image, elapsed = run_pipeline(
                    base_pipe,
                    args.prompt,
                    args.height,
                    args.width,
                    args.base_steps,
                    args.meancache_guidance_scale,
                    args.seed,
                )
                save_result(
                    root / "03_meancache" / f"meancache_{cache_jvp}.png",
                    image,
                    {
                        "variant": f"meancache_{cache_jvp}",
                        "elapsed_s": elapsed,
                        "steps": args.base_steps,
                        "guidance_scale": args.meancache_guidance_scale,
                        "prompt": args.prompt,
                        "model_id": args.base_model_id,
                        "cache_jvp": cache_jvp,
                    },
                )
        finally:
            ZImagePipeline.__call__ = original_call


if __name__ == "__main__":
    main()
