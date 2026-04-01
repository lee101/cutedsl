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
from latentteleport.tokenizer import VisualUnit, create_tokenizer

log = logging.getLogger(__name__)


def run_single_config(
    pipe,
    cache: LatentCache,
    prompts: list[str],
    config: dict,
    output_base: str,
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
    target_step = int(20 * config["teleport_timestep"])

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

    for idx, prompt in enumerate(prompts):
        t0 = time.time()
        units = tokenizer.tokenize(prompt)

        latents = []
        embeddings = []
        for u in units:
            lat = cache.load_latent(u, target_step)
            emb = cache.load_text_embedding(u)
            if lat is not None:
                latents.append(lat)
                cache_hits += 1
                if emb is not None:
                    embeddings.append(emb)
            else:
                cache_misses += 1

        if not latents:
            # Full generation fallback
            result, _ = capture_intermediates(
                pipe, prompt, 512, 512, 20, 42, "cuda",
            )
            if result.images:
                result.images[0].save(str(gen_dir / f"{idx:04d}.png"))
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
            # TODO: wire refinement steps through pipeline

        latencies.append(time.time() - t0)

        # Also generate baseline reference
        result_ref, _ = capture_intermediates(pipe, prompt, 512, 512, 20, 42, "cuda")
        if result_ref.images:
            result_ref.images[0].save(str(ref_dir / f"{idx:04d}.png"))

    metrics = compute_pairwise_metrics(str(gen_dir), str(ref_dir))
    import numpy as np
    result = {
        **config,
        **metrics,
        "cache_hit_rate": cache_hits / max(cache_hits + cache_misses, 1),
        "mean_latency_s": float(np.mean(latencies)) if latencies else 0,
        "run_id": run_id,
    }
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
        configs.append({
            "tokenizer": tok,
            "combiner": comb,
            "refinement_steps": ref_steps,
            "teleport_timestep": tt,
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
    run_ablation(pipe, cache, prompts, ablation, args.output_dir)


if __name__ == "__main__":
    main()
