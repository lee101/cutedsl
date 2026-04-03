"""Ablation sweep runner for latent teleportation experiments."""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import time
from pathlib import Path

import torch

from latentteleport.cache import LatentCache
from latentteleport.combiner import SLERPCombiner, TreeCombiner, slerp, create_combiner
from latentteleport.config import (
    AblationConfig, CombinerConfig, EvalConfig, TeleportConfig, TokenizerConfig,
)
from latentteleport.evaluate import compute_pairwise_metrics
from latentteleport.judge import create_judge
from latentteleport.refine import TeleportPipeline
from latentteleport.tokenizer import VisualUnit, create_tokenizer

log = logging.getLogger(__name__)


def run_single_config(
    pipe,
    cache: LatentCache,
    prompts: list[str],
    config: dict,
    output_base: str,
    judge_enabled: bool = False,
    negative_prompt: str = "",
    guidance_scale: float = 0.0,
    negative_trajectory_scale: float = 0.0,
) -> dict:
    """Run teleportation for one ablation config and measure quality."""
    from latentteleport.dataset import capture_intermediates

    tok_cfg = TokenizerConfig(strategy=config["tokenizer"])
    comb_cfg = CombinerConfig(
        method=config["combiner"],
        refinement_steps=config["refinement_steps"],
        teleport_timestep=config["teleport_timestep"],
    )

    tokenizer = create_tokenizer(tok_cfg)
    combiner = create_combiner(comb_cfg)
    tp_cfg = TeleportConfig(
        num_steps=20,
        height=512,
        width=512,
        device="cuda",
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        negative_trajectory_scale=negative_trajectory_scale,
        trajectory_virtual_steps=config.get("trajectory_virtual_steps", 0),
    )
    teleport = TeleportPipeline(pipe, cache, tokenizer, combiner, tp_cfg, comb_cfg)
    judge = create_judge(judge_enabled)

    run_id = (
        f"tok_{config['tokenizer']}_comb_{config['combiner']}"
        f"_ref{config['refinement_steps']}_tt{config['teleport_timestep']}"
    )
    run_dir = Path(output_base) / run_id
    gen_dir = run_dir / "generated"
    ref_dir = run_dir / "references"
    gen_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)

    latencies = []
    cache_hits = 0
    cache_misses = 0
    judge_scores = []

    for idx, prompt in enumerate(prompts):
        t0 = time.time()
        outcome = teleport.generate(prompt, negative_prompt=negative_prompt)
        cache_hits += int(outcome.get("cache_hits", 0))
        cache_misses += max(outcome.get("total_units", 0) - outcome.get("cache_hits", 0), 0)
        image = outcome.get("image")
        if image is not None:
            image.save(str(gen_dir / f"{idx:04d}.png"))

        latencies.append(time.time() - t0)

        # Also generate baseline reference
        result_ref, _ = capture_intermediates(pipe, prompt, 512, 512, 20, 42, "cuda")
        if result_ref.images:
            ref_image = result_ref.images[0]
            ref_image.save(str(ref_dir / f"{idx:04d}.png"))
            if judge is not None and image is not None:
                judge_result = judge.score_pair(prompt, image, ref_image)
                if judge_result.combined_score is not None:
                    judge_scores.append(
                        {
                            "prompt_score": judge_result.prompt_score,
                            "reference_score": judge_result.reference_score,
                            "combined_score": judge_result.combined_score,
                        }
                    )

    metrics = compute_pairwise_metrics(str(gen_dir), str(ref_dir))
    import numpy as np
    result = {
        **config,
        **metrics,
        "cache_hit_rate": cache_hits / max(cache_hits + cache_misses, 1),
        "mean_latency_s": float(np.mean(latencies)) if latencies else 0,
        "run_id": run_id,
        "negative_prompt": negative_prompt,
        "guidance_scale": guidance_scale,
        "negative_trajectory_scale": negative_trajectory_scale,
    }
    if judge_scores:
        result["judge_mean_prompt_score"] = float(np.mean([s["prompt_score"] for s in judge_scores]))
        result["judge_mean_reference_score"] = float(np.mean([s["reference_score"] for s in judge_scores]))
        result["judge_mean_combined_score"] = float(np.mean([s["combined_score"] for s in judge_scores]))
    (run_dir / "config.json").write_text(json.dumps(result, indent=2))
    return result


def run_ablation(
    pipe,
    cache: LatentCache,
    prompts: list[str],
    ablation: AblationConfig,
    output_dir: str,
):
    """Run full ablation sweep."""
    results = []
    configs = []
    for tok, comb, ref_steps, tt in itertools.product(
        ablation.tokenizer_strategies,
        ablation.combiner_methods,
        ablation.refinement_steps_list,
        ablation.teleport_timesteps,
    ):
        for tvs in ablation.trajectory_virtual_steps:
            configs.append({
                "tokenizer": tok,
                "combiner": comb,
                "refinement_steps": ref_steps,
                "teleport_timestep": tt,
                "trajectory_virtual_steps": tvs,
            })

    log.info(f"Ablation: {len(configs)} configs x {len(prompts)} prompts")

    for i, cfg in enumerate(configs):
        log.info(f"[{i+1}/{len(configs)}] {cfg}")
        try:
            result = run_single_config(pipe, cache, prompts, cfg, output_dir)
            results.append(result)
        except Exception as e:
            log.error(f"Failed: {e}")
            results.append({**cfg, "error": str(e)})

    out_path = Path(output_dir) / "ablation_results.jsonl"
    with out_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    log.info(f"Ablation results: {out_path}")
    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description="Run latent teleportation ablation sweep")
    parser.add_argument("--model-id", default="Tongyi-MAI/Z-Image-Turbo")
    parser.add_argument("--cache-dir", default="/vfast/latentteleport/cache")
    parser.add_argument("--output-dir", default="/vfast/latentteleport/eval/ablation")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--judge", action="store_true", help="Enable CLIP-based local prompt/reference judging")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--negative-trajectory-scale", type=float, default=0.0)
    args = parser.parse_args()

    from latentteleport.dataset import DEFAULT_PROMPTS
    prompts = DEFAULT_PROMPTS[:args.num_prompts]

    dtype = torch.bfloat16
    from diffusers import ZImagePipeline
    pipe = ZImagePipeline.from_pretrained(args.model_id, torch_dtype=dtype)
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(args.device)

    cache = LatentCache(args.cache_dir, resolution=(args.height, args.width))
    ablation = AblationConfig()
    results = []
    configs = []
    for tok, comb, ref_steps, tt in itertools.product(
        ablation.tokenizer_strategies,
        ablation.combiner_methods,
        ablation.refinement_steps_list,
        ablation.teleport_timesteps,
    ):
        for tvs in ablation.trajectory_virtual_steps:
            configs.append({
                "tokenizer": tok,
                "combiner": comb,
                "refinement_steps": ref_steps,
                "teleport_timestep": tt,
                "trajectory_virtual_steps": tvs,
            })
    log.info(f"Ablation: {len(configs)} configs x {len(prompts)} prompts")
    for i, cfg in enumerate(configs):
        log.info(f"[{i+1}/{len(configs)}] {cfg}")
        try:
            results.append(
                run_single_config(
                    pipe,
                    cache,
                    prompts,
                    cfg,
                    args.output_dir,
                    judge_enabled=args.judge,
                    negative_prompt=args.negative_prompt,
                    guidance_scale=args.guidance_scale,
                    negative_trajectory_scale=args.negative_trajectory_scale,
                )
            )
        except Exception as e:
            log.error(f"Failed: {e}")
            results.append({**cfg, "error": str(e)})
    out_path = Path(args.output_dir) / "ablation_results.jsonl"
    with out_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    log.info(f"Ablation results: {out_path}")


if __name__ == "__main__":
    main()
