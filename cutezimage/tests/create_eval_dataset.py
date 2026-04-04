"""Create a regression eval dataset for CuteZImage vs original Z-Image.

Runs a fixed set of prompts through:
  1. Original ZImagePipeline  – saves images + per-step latents
  2. CuteZImage pipeline       – computes pixel error vs originals

Output layout::

    eval_dataset/
        manifest.json
        prompts/
            <slug>/
                original.png
                cute.png
                latents_original/   step_00.pt ... step_N.pt
                latents_cute/       step_00.pt ... step_N.pt

Usage::

    # Generate full dataset (needs model on disk / HF cache)
    python -m cutezimage.tests.create_eval_dataset

    # Quick smoke-test (fewer prompts, smaller image, fewer steps)
    python -m cutezimage.tests.create_eval_dataset --quick

    # Reload an existing dataset and recompute pixel errors only
    python -m cutezimage.tests.create_eval_dataset --errors-only --dataset-dir eval_dataset
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Eval prompt suite
# ---------------------------------------------------------------------------

EVAL_PROMPTS = [
    "anime fairy cute winged fairy glowing magical forest",
    "photo realistic portrait woman golden hour soft lighting",
    "sci-fi cityscape neon lights rain reflections cyberpunk",
    "oil painting impressionist sunset ocean waves",
    "minimalist vector art cat sitting window abstract",
    "fantasy dragon mountain castle epic landscape",
    "product photo luxury watch black background studio",
    "watercolor flowers pastel spring garden",
]

QUICK_PROMPTS = EVAL_PROMPTS[:2]


def _slug(prompt: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", prompt.lower())[:48].strip("_")


# ---------------------------------------------------------------------------
# Latent capture via callback
# ---------------------------------------------------------------------------

class LatentCapture:
    """Collect latents at each denoising step via pipeline callback."""

    def __init__(self):
        self.latents: list[torch.Tensor] = []

    def __call__(self, pipe: Any, step: int, timestep: int, callback_kwargs: dict) -> dict:
        latent = callback_kwargs.get("latents")
        if latent is not None:
            self.latents.append(latent.cpu().clone())
        return callback_kwargs

    def reset(self):
        self.latents = []


# ---------------------------------------------------------------------------
# Image pixel error
# ---------------------------------------------------------------------------

def pixel_error(img_a, img_b) -> dict:
    a = np.array(img_a, dtype=np.float32)
    b = np.array(img_b, dtype=np.float32)
    diff = np.abs(a - b)
    return {
        "max_pixel_error": float(diff.max()),
        "mean_pixel_error": float(diff.mean()),
        "psnr_db": float(10 * np.log10(255.0**2 / (np.mean(diff**2) + 1e-8))),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    dataset_dir = Path(args.dataset_dir)
    prompts_dir = dataset_dir / "prompts"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(exist_ok=True)

    prompts = QUICK_PROMPTS if args.quick else EVAL_PROMPTS
    width = height = args.size
    steps = args.steps
    seed = args.seed
    model_id = args.model_id

    print(f"Dataset dir : {dataset_dir}")
    print(f"Model       : {model_id}")
    print(f"Prompts     : {len(prompts)}")
    print(f"Size        : {width}x{height}  steps={steps}  seed={seed}")
    print()

    from diffusers import ZImagePipeline
    from cutezimage.model import CuteZImageTransformer

    # ------------------------------------------------------------------
    # Load pipelines
    # ------------------------------------------------------------------
    print("Loading original ZImagePipeline...")
    t0 = time.perf_counter()
    orig_pipe = ZImagePipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    orig_pipe.enable_model_cpu_offload()
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")

    if not args.errors_only:
        print("Building CuteZImageTransformer...")
        cute_transformer = CuteZImageTransformer.from_diffusers(orig_pipe.transformer)
        cute_transformer = cute_transformer.to("cuda", torch.bfloat16).eval()

        # Patch a second pipeline instance using the same components
        from diffusers import ZImagePipeline as _P
        cute_pipe = _P(**orig_pipe.components)
        cute_pipe.transformer = cute_transformer
        cute_pipe.enable_model_cpu_offload()
        print("  done")

    capture_orig = LatentCapture()
    capture_cute = LatentCapture() if not args.errors_only else None

    manifest: list[dict] = []

    for i, prompt in enumerate(prompts):
        slug = _slug(prompt)
        prompt_dir = prompts_dir / slug
        prompt_dir.mkdir(exist_ok=True)
        lat_orig_dir = prompt_dir / "latents_original"
        lat_cute_dir = prompt_dir / "latents_cute"

        print(f"[{i+1}/{len(prompts)}] {prompt[:60]}")

        gen = lambda: torch.Generator("cuda").manual_seed(seed)

        # --- original pipeline ---
        if not (prompt_dir / "original.png").exists() or args.regenerate:
            capture_orig.reset()
            t0 = time.perf_counter()
            res_orig = orig_pipe(
                prompt=prompt,
                width=width, height=height,
                num_inference_steps=steps,
                guidance_scale=0.0,
                generator=gen(),
                callback_on_step_end=capture_orig,
                callback_on_step_end_tensor_inputs=["latents"],
            )
            orig_ms = (time.perf_counter() - t0) * 1000
            orig_img = res_orig.images[0]
            orig_img.save(prompt_dir / "original.png")

            # save latents
            lat_orig_dir.mkdir(exist_ok=True)
            for s, lat in enumerate(capture_orig.latents):
                torch.save(lat, lat_orig_dir / f"step_{s:02d}.pt")
            print(f"  orig: {orig_ms:.0f}ms  latents={len(capture_orig.latents)}")
        else:
            from PIL import Image
            orig_img = Image.open(prompt_dir / "original.png")
            orig_ms = None
            print(f"  orig: (cached)")

        # --- cute pipeline ---
        if not args.errors_only and (not (prompt_dir / "cute.png").exists() or args.regenerate):
            capture_cute.reset()
            t0 = time.perf_counter()
            res_cute = cute_pipe(
                prompt=prompt,
                width=width, height=height,
                num_inference_steps=steps,
                guidance_scale=0.0,
                generator=gen(),
                callback_on_step_end=capture_cute,
                callback_on_step_end_tensor_inputs=["latents"],
            )
            cute_ms = (time.perf_counter() - t0) * 1000
            cute_img = res_cute.images[0]
            cute_img.save(prompt_dir / "cute.png")

            lat_cute_dir.mkdir(exist_ok=True)
            for s, lat in enumerate(capture_cute.latents):
                torch.save(lat, lat_cute_dir / f"step_{s:02d}.pt")
            print(f"  cute: {cute_ms:.0f}ms  latents={len(capture_cute.latents)}")
        else:
            from PIL import Image
            cute_img = Image.open(prompt_dir / "cute.png") if (prompt_dir / "cute.png").exists() else None
            cute_ms = None

        # --- pixel error ---
        errors = {}
        if cute_img is not None:
            errors = pixel_error(orig_img, cute_img)
            print(f"  pixel error: max={errors['max_pixel_error']:.2f}/255  "
                  f"mean={errors['mean_pixel_error']:.4f}/255  "
                  f"PSNR={errors['psnr_db']:.1f}dB")

        # --- per-step latent error ---
        step_errors = []
        if lat_orig_dir.exists() and lat_cute_dir.exists():
            orig_steps = sorted(lat_orig_dir.glob("step_*.pt"))
            cute_steps = sorted(lat_cute_dir.glob("step_*.pt"))
            for ost, cst in zip(orig_steps, cute_steps):
                lo = torch.load(ost, weights_only=True).float()
                lc = torch.load(cst, weights_only=True).float()
                diff = (lo - lc).abs()
                step_errors.append({
                    "step": int(ost.stem.split("_")[1]),
                    "max_abs_error": float(diff.max()),
                    "mean_abs_error": float(diff.mean()),
                })

        entry = {
            "slug": slug,
            "prompt": prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "seed": seed,
            "orig_ms": orig_ms,
            "cute_ms": cute_ms,
            "pixel_errors": errors,
            "latent_step_errors": step_errors,
        }
        manifest.append(entry)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 64)
    print("EVAL DATASET SUMMARY")
    print("=" * 64)
    print(f"{'Prompt':<40} {'MaxPx':>7} {'MeanPx':>7} {'PSNR':>7}")
    print("-" * 64)
    for e in manifest:
        px = e.get("pixel_errors", {})
        print(f"{e['slug'][:40]:<40} "
              f"{px.get('max_pixel_error', float('nan')):>7.2f} "
              f"{px.get('mean_pixel_error', float('nan')):>7.4f} "
              f"{px.get('psnr_db', float('nan')):>7.1f}")

    manifest_path = dataset_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Create CuteZImage eval dataset")
    parser.add_argument("--model-id", default="Tongyi-MAI/Z-Image-Turbo")
    parser.add_argument("--dataset-dir", default="eval_dataset")
    parser.add_argument("--size", type=int, default=512, help="Image size (width=height)")
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="Use only 2 prompts, 256px, 4 steps")
    parser.add_argument("--errors-only", action="store_true",
                        help="Skip generation, recompute errors from saved files")
    parser.add_argument("--regenerate", action="store_true", help="Overwrite existing outputs")
    args = parser.parse_args()

    if args.quick:
        args.size = min(args.size, 256)
        args.steps = min(args.steps, 4)

    run(args)


if __name__ == "__main__":
    main()
