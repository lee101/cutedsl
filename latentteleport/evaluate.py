"""Evaluation harness: FID, LPIPS, SSIM, PSNR, latency."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from cutezimage.image_metrics import compare_images, pil_to_tensor
from latentteleport.config import EvalConfig

log = logging.getLogger(__name__)


def load_image_tensor(path: str | Path) -> torch.Tensor:
    return pil_to_tensor(Image.open(path).convert("RGB"))


def compute_pairwise_metrics(
    generated_dir: str,
    reference_dir: str,
    max_samples: int = 0,
) -> dict:
    """Compare generated vs reference images pairwise."""
    gen_path = Path(generated_dir)
    ref_path = Path(reference_dir)
    gen_files = sorted(gen_path.glob("*.png"))
    ref_files = sorted(ref_path.glob("*.png"))
    pairs = list(zip(gen_files, ref_files))
    if max_samples > 0:
        pairs = pairs[:max_samples]

    all_metrics = []
    for gf, rf in pairs:
        gen_img = load_image_tensor(gf)
        ref_img = load_image_tensor(rf)
        m = compare_images(gen_img, ref_img)
        all_metrics.append(m)

    if not all_metrics:
        return {}

    agg = {}
    for key in all_metrics[0]:
        if key == "images_identical":
            agg[key] = sum(1 for m in all_metrics if m[key]) / len(all_metrics)
        else:
            vals = [m[key] for m in all_metrics if not (key == "psnr_db" and m[key] == float("inf"))]
            agg[f"mean_{key}"] = float(np.mean(vals)) if vals else 0.0
            agg[f"std_{key}"] = float(np.std(vals)) if vals else 0.0
    agg["num_pairs"] = len(all_metrics)
    return agg


def compute_fid(generated_dir: str, reference_dir: str) -> float | None:
    """FID using clean-fid."""
    try:
        from cleanfid import fid
        score = fid.compute_fid(generated_dir, reference_dir)
        return float(score)
    except ImportError:
        log.warning("clean-fid not installed, skipping FID")
        return None


def compute_lpips_batch(
    generated_dir: str,
    reference_dir: str,
    max_samples: int = 0,
) -> float | None:
    """Mean LPIPS across paired images."""
    try:
        import lpips
    except ImportError:
        log.warning("lpips not installed, skipping")
        return None

    loss_fn = lpips.LPIPS(net="alex")
    gen_path = Path(generated_dir)
    ref_path = Path(reference_dir)
    gen_files = sorted(gen_path.glob("*.png"))
    ref_files = sorted(ref_path.glob("*.png"))
    pairs = list(zip(gen_files, ref_files))
    if max_samples > 0:
        pairs = pairs[:max_samples]

    scores = []
    for gf, rf in pairs:
        gen_t = lpips.im2tensor(lpips.load_image(str(gf)))
        ref_t = lpips.im2tensor(lpips.load_image(str(rf)))
        with torch.no_grad():
            d = loss_fn(gen_t, ref_t)
        scores.append(d.item())
    return float(np.mean(scores)) if scores else None


def evaluate_teleportation(
    pipe,
    cache,
    tokenizer,
    combiner,
    config: EvalConfig,
    teleport_config,
    prompts: list[str],
    output_dir: str,
) -> dict:
    """Full evaluation: generate images via teleportation and compare to references."""
    from latentteleport.dataset import capture_intermediates

    out = Path(output_dir)
    gen_dir = out / "generated"
    ref_dir = out / "references"
    gen_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)

    latencies_teleport = []
    latencies_baseline = []

    for idx, prompt in enumerate(prompts[:config.num_samples]):
        # Baseline: full generation
        t0 = time.time()
        result_ref, _ = capture_intermediates(
            pipe, prompt, teleport_config.height, teleport_config.width,
            teleport_config.num_steps, teleport_config.seed, teleport_config.device,
        )
        latencies_baseline.append(time.time() - t0)
        if result_ref.images:
            result_ref.images[0].save(str(ref_dir / f"{idx:04d}.png"))

        # Teleported: look up cached units, combine, refine
        t0 = time.time()
        units = tokenizer.tokenize(prompt)
        latents = []
        embeddings = []
        step_idx = int(teleport_config.num_steps * 0.3)  # teleport timestep

        for u in units:
            lat = cache.load_latent(u, step_idx)
            emb = cache.load_text_embedding(u)
            if lat is not None:
                latents.append(lat)
                if emb is not None:
                    embeddings.append(emb)

        if not latents:
            # Cache miss: fall back to full generation
            result_tp, _ = capture_intermediates(
                pipe, prompt, teleport_config.height, teleport_config.width,
                teleport_config.num_steps, teleport_config.seed, teleport_config.device,
            )
        else:
            if hasattr(combiner, "combine_tree"):
                combined = combiner.combine_tree(latents, embeddings or None)
            elif len(latents) >= 2:
                combined = latents[0]
                for i in range(1, len(latents)):
                    ea = embeddings[i - 1] if embeddings else None
                    eb = embeddings[i] if embeddings else None
                    combined = combiner.combine(combined, latents[i], ea, eb)
            else:
                combined = latents[0]

            # TODO: inject combined latent at timestep and run remaining steps
            # For now, save the combined latent decoded through VAE
            result_tp = None  # placeholder until refinement pipeline is wired

        latencies_teleport.append(time.time() - t0)
        if result_tp and result_tp.images:
            result_tp.images[0].save(str(gen_dir / f"{idx:04d}.png"))

        log.info(f"[{idx+1}/{min(len(prompts), config.num_samples)}] done")

    metrics = compute_pairwise_metrics(str(gen_dir), str(ref_dir))
    fid_score = compute_fid(str(gen_dir), str(ref_dir))
    lpips_score = compute_lpips_batch(str(gen_dir), str(ref_dir))

    results = {
        **metrics,
        "fid": fid_score,
        "mean_lpips": lpips_score,
        "mean_latency_baseline_s": float(np.mean(latencies_baseline)) if latencies_baseline else 0,
        "mean_latency_teleport_s": float(np.mean(latencies_teleport)) if latencies_teleport else 0,
    }
    if results["mean_latency_baseline_s"] > 0:
        results["speedup"] = results["mean_latency_baseline_s"] / max(results["mean_latency_teleport_s"], 1e-6)

    results_path = out / "results.json"
    results_path.write_text(json.dumps(results, indent=2))
    log.info(f"Results saved to {results_path}")
    return results
