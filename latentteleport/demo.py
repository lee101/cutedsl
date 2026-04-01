"""End-to-end demo: latent teleportation vs baseline turbo generation.

Compares:
1. Full 20-step generation (quality reference)
2. 4-step turbo generation (speed baseline)
3. Cached single-token teleportation + refinement
4. Cached bigram teleportation + refinement
5. SLERP combination + refinement
6. Neural combination + refinement (if trained)

Measures wall-clock time and SSIM/PSNR vs 20-step reference.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch

from cutezimage.image_metrics import compare_images, pil_to_tensor
from latentteleport.cache import LatentCache
from latentteleport.combiner import SLERPCombiner, TreeCombiner, create_combiner
from latentteleport.config import CombinerConfig, TeleportConfig, TokenizerConfig
from latentteleport.dataset import capture_intermediates, DEFAULT_PROMPTS
from latentteleport.refine import TeleportPipeline
from latentteleport.tokenizer import create_tokenizer

log = logging.getLogger(__name__)


def run_baseline(pipe, prompt, height, width, num_steps, seed, device):
    """Full N-step generation."""
    t0 = time.time()
    gen = torch.Generator(device=device if device == "cuda" else "cpu").manual_seed(seed)
    result = pipe(
        prompt=prompt, height=height, width=width,
        num_inference_steps=num_steps, guidance_scale=0.0, generator=gen,
    )
    elapsed = time.time() - t0
    return result.images[0], elapsed


def run_demo(
    model_id: str = "Tongyi-MAI/Z-Image-Turbo",
    cache_dir: str = "/vfast/latentteleport/cache",
    output_dir: str = "/vfast/latentteleport/demo_output",
    prompts: list[str] | None = None,
    height: int = 512,
    width: int = 512,
    seed: int = 42,
    device: str = "cuda",
    cpu_offload: bool = False,
    populate_if_empty: bool = True,
):
    dtype = torch.bfloat16
    from diffusers import ZImagePipeline
    pipe = ZImagePipeline.from_pretrained(model_id, torch_dtype=dtype)
    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)

    cache = LatentCache(cache_dir, resolution=(height, width))
    prompts = prompts or DEFAULT_PROMPTS[:5]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tok_config = TokenizerConfig(strategy="curated")
    tokenizer = create_tokenizer(tok_config)

    # Populate cache if empty
    if populate_if_empty and cache.stats()["num_units"] == 0:
        log.info("Cache empty, populating with visual units from prompts...")
        from latentteleport.dataset import populate_cache
        teleport_cfg = TeleportConfig(
            model_id=model_id, device=device, dtype="bfloat16",
            height=height, width=width, num_steps=20, seed=seed, cache_dir=cache_dir,
        )
        populate_cache(pipe, cache, prompts, teleport_cfg, tok_config)
        log.info(f"Cache populated: {cache.stats()}")

    # Set up teleportation pipeline
    comb_config = CombinerConfig(
        method="slerp", refinement_steps=5, teleport_timestep=0.3,
        latent_h=height // 8, latent_w=width // 8,
    )
    combiner = create_combiner(comb_config)
    teleport_cfg = TeleportConfig(
        model_id=model_id, device=device, dtype="bfloat16",
        height=height, width=width, num_steps=20, seed=seed, cache_dir=cache_dir,
    )
    tp = TeleportPipeline(pipe, cache, tokenizer, combiner, teleport_cfg, comb_config)

    all_results = []

    for idx, prompt in enumerate(prompts):
        log.info(f"\n{'='*60}")
        log.info(f"Prompt {idx+1}/{len(prompts)}: {prompt}")
        log.info(f"{'='*60}")

        result_row = {"prompt": prompt, "index": idx}

        # 1. 20-step reference
        ref_img, ref_time = run_baseline(pipe, prompt, height, width, 20, seed, device)
        ref_img.save(str(out / f"{idx:03d}_ref_20step.png"))
        result_row["ref_20step_time_s"] = ref_time
        log.info(f"  20-step: {ref_time:.2f}s")

        # 2. 4-step turbo baseline
        turbo_img, turbo_time = run_baseline(pipe, prompt, height, width, 4, seed, device)
        turbo_img.save(str(out / f"{idx:03d}_turbo_4step.png"))
        turbo_metrics = compare_images(pil_to_tensor(turbo_img), pil_to_tensor(ref_img))
        result_row["turbo_4step_time_s"] = turbo_time
        result_row["turbo_4step_ssim"] = turbo_metrics["ssim"]
        result_row["turbo_4step_psnr"] = turbo_metrics["psnr_db"]
        log.info(f"  4-step turbo: {turbo_time:.2f}s, SSIM={turbo_metrics['ssim']:.4f}, PSNR={turbo_metrics['psnr_db']:.1f}dB")

        # 3. 9-step turbo baseline
        turbo9_img, turbo9_time = run_baseline(pipe, prompt, height, width, 9, seed, device)
        turbo9_img.save(str(out / f"{idx:03d}_turbo_9step.png"))
        turbo9_metrics = compare_images(pil_to_tensor(turbo9_img), pil_to_tensor(ref_img))
        result_row["turbo_9step_time_s"] = turbo9_time
        result_row["turbo_9step_ssim"] = turbo9_metrics["ssim"]
        result_row["turbo_9step_psnr"] = turbo9_metrics["psnr_db"]
        log.info(f"  9-step turbo: {turbo9_time:.2f}s, SSIM={turbo9_metrics['ssim']:.4f}, PSNR={turbo9_metrics['psnr_db']:.1f}dB")

        # 4. Teleportation (SLERP combiner)
        tp_result = tp.generate(prompt, seed=seed)
        if tp_result["image"] is not None:
            tp_result["image"].save(str(out / f"{idx:03d}_teleport_slerp.png"))
            tp_metrics = compare_images(pil_to_tensor(tp_result["image"]), pil_to_tensor(ref_img))
            result_row["teleport_time_s"] = tp_result["elapsed_s"]
            result_row["teleport_ssim"] = tp_metrics["ssim"]
            result_row["teleport_psnr"] = tp_metrics["psnr_db"]
            result_row["teleport_method"] = tp_result["method"]
            result_row["teleport_cache_hits"] = tp_result["cache_hits"]
            result_row["teleport_units"] = tp_result.get("units", [])
            log.info(
                f"  Teleport: {tp_result['elapsed_s']:.2f}s, "
                f"SSIM={tp_metrics['ssim']:.4f}, PSNR={tp_metrics['psnr_db']:.1f}dB, "
                f"method={tp_result['method']}, hits={tp_result['cache_hits']}/{tp_result['total_units']}"
            )
        else:
            result_row["teleport_method"] = "failed"
            log.info("  Teleport: FAILED")

        all_results.append(result_row)

    # Summary
    results_path = out / "demo_results.json"
    results_path.write_text(json.dumps(all_results, indent=2, default=str))

    log.info(f"\n{'='*60}")
    log.info("SUMMARY")
    log.info(f"{'='*60}")

    for r in all_results:
        prompt_short = r["prompt"][:50]
        log.info(f"\n  {prompt_short}...")
        log.info(f"    20-step ref:    {r.get('ref_20step_time_s', 0):.2f}s")
        log.info(f"    4-step turbo:   {r.get('turbo_4step_time_s', 0):.2f}s  SSIM={r.get('turbo_4step_ssim', 0):.4f}")
        log.info(f"    9-step turbo:   {r.get('turbo_9step_time_s', 0):.2f}s  SSIM={r.get('turbo_9step_ssim', 0):.4f}")
        tp_time = r.get("teleport_time_s", 0)
        tp_ssim = r.get("teleport_ssim", 0)
        log.info(f"    Teleport:       {tp_time:.2f}s  SSIM={tp_ssim:.4f}  [{r.get('teleport_method', 'n/a')}]")

    log.info(f"\nResults saved to {results_path}")
    return all_results


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description="Latent teleportation demo")
    parser.add_argument("--model-id", default="Tongyi-MAI/Z-Image-Turbo")
    parser.add_argument("--cache-dir", default="/vfast/latentteleport/cache")
    parser.add_argument("--output-dir", default="/vfast/latentteleport/demo_output")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--num-prompts", type=int, default=5)
    parser.add_argument("--no-populate", action="store_true")
    args = parser.parse_args()

    prompts = args.prompt if args.prompt else DEFAULT_PROMPTS[:args.num_prompts]
    run_demo(
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        prompts=prompts,
        height=args.height,
        width=args.width,
        seed=args.seed,
        device=args.device,
        cpu_offload=args.cpu_offload,
        populate_if_empty=not args.no_populate,
    )


if __name__ == "__main__":
    main()
