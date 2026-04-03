"""Generate latent cache and reference images for teleportation experiments."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch

from latentteleport.cache import LatentCache
from latentteleport.config import TeleportConfig, TokenizerConfig
from latentteleport.tokenizer import VisualUnit, create_tokenizer

log = logging.getLogger(__name__)

DEFAULT_PROMPTS = [
    "a red car on a sunny beach",
    "a black cat sitting on a windowsill",
    "a lighthouse in a storm at night",
    "a fairy with glowing wings in a dark forest",
    "a robot painting a portrait in a studio",
    "a dragon flying over a mountain village",
    "an astronaut floating above earth with stars",
    "a cozy cabin in snow with warm light inside",
    "a samurai standing in a field of cherry blossoms",
    "a pirate ship sailing through a thunderstorm",
    "a wizard casting spells in a crystal cave",
    "a knight on horseback in front of a castle",
    "a mermaid swimming near a coral reef",
    "an owl perched on an ancient book in candlelight",
    "a steampunk city with airships and brass towers",
    "a girl with an umbrella walking in rain on a cobblestone street",
    "a fox sitting in an autumn forest clearing",
    "a phoenix rising from flames against a dark sky",
    "a japanese garden with a red bridge and koi pond",
    "a wolf howling at the moon on a snowy cliff",
]

COMPOUND_PAIRS = [
    ("a red car", "a sunny beach"),
    ("a black cat", "a windowsill"),
    ("a lighthouse", "a storm at night"),
    ("a fairy with glowing wings", "a dark forest"),
    ("a robot", "a painting studio"),
    ("a dragon", "a mountain village"),
    ("an astronaut", "earth with stars"),
    ("a cozy cabin", "snow with warm light"),
    ("a samurai", "cherry blossoms"),
    ("a pirate ship", "a thunderstorm"),
]


def capture_intermediates(
    pipe,
    prompt: str,
    height: int,
    width: int,
    num_steps: int,
    seed: int,
    device: str,
    guidance_scale: float = 0.0,
) -> tuple[object, dict[int, torch.Tensor]]:
    """Run pipeline capturing latent at every step."""
    intermediates: dict[int, torch.Tensor] = {}

    def callback(pipe_obj, step_index, timestep, callback_kwargs):
        intermediates[step_index] = callback_kwargs["latents"].detach().clone().cpu()
        return callback_kwargs

    gen = torch.Generator(device=device if device == "cuda" else "cpu").manual_seed(seed)
    result = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=gen,
        callback_on_step_end=callback,
        callback_on_step_end_tensor_inputs=["latents"],
    )
    return result, intermediates


def extract_text_embedding(pipe, prompt: str) -> torch.Tensor:
    """Extract CLIP text embedding from the pipeline's text encoder."""
    with torch.no_grad():
        encoded = pipe.encode_prompt(prompt, do_classifier_free_guidance=False)
        if isinstance(encoded, tuple):
            return encoded[0].cpu()
        return encoded.cpu()


def populate_cache(
    pipe,
    cache: LatentCache,
    prompts: list[str],
    config: TeleportConfig,
    tokenizer_config: TokenizerConfig | None = None,
    shard_index: int = 0,
    num_shards: int = 1,
):
    """Generate and cache latents for a list of prompts."""
    tokenizer = create_tokenizer(tokenizer_config or TokenizerConfig())
    all_units: dict[str, VisualUnit] = {}

    for prompt in prompts:
        units = tokenizer.tokenize(prompt)
        for u in units:
            all_units[u.unit_id] = u
        all_units[VisualUnit.from_text(prompt).unit_id] = VisualUnit.from_text(prompt)

    unit_list = sorted(all_units.values(), key=lambda u: u.unit_id)
    if num_shards > 1:
        unit_list = [u for i, u in enumerate(unit_list) if i % num_shards == shard_index]

    total = len(unit_list)
    log.info(f"Populating cache: {total} units (shard {shard_index}/{num_shards})")

    for idx, unit in enumerate(unit_list):
        if cache.has_unit(unit):
            log.info(f"[{idx+1}/{total}] skip cached: {unit.text}")
            continue

        t0 = time.time()
        try:
            text_emb = extract_text_embedding(pipe, unit.text)
        except Exception as e:
            log.warning(f"[{idx+1}/{total}] embed failed for '{unit.text}': {e}")
            continue

        try:
            result, intermediates = capture_intermediates(
                pipe, unit.text, config.height, config.width,
                config.num_steps, config.seed, config.device, config.guidance_scale,
            )
        except Exception as e:
            log.warning(f"[{idx+1}/{total}] generation failed for '{unit.text}': {e}")
            continue

        cache.store_latents(
            unit, intermediates, text_embedding=text_emb,
            metadata={"prompt": unit.text, "seed": config.seed, "steps": config.num_steps},
        )

        if result.images:
            img_path = cache.unit_dir(unit) / "reference.png"
            result.images[0].save(str(img_path))

        elapsed = time.time() - t0
        log.info(f"[{idx+1}/{total}] cached '{unit.text}' ({len(intermediates)} steps, {elapsed:.1f}s)")


def cache_prompt_trajectory(
    pipe,
    cache: LatentCache,
    prompt: str,
    config: TeleportConfig,
    tokenizer_config: TokenizerConfig | None = None,
    include_bigrams: bool = True,
) -> tuple[object, dict[int, torch.Tensor]]:
    """Generate a full prompt once and use it to update online caches."""
    tokenizer = create_tokenizer(tokenizer_config or TokenizerConfig())
    result, intermediates = capture_intermediates(
        pipe,
        prompt,
        config.height,
        config.width,
        config.num_steps,
        config.seed,
        config.device,
        config.guidance_scale,
    )
    units = tokenizer.tokenize(prompt)
    for unit in units:
        text_emb = extract_text_embedding(pipe, unit.text)
        cache.store_latents(
            unit,
            intermediates,
            text_embedding=text_emb,
            metadata={"prompt": prompt, "source": "online_update", "seed": config.seed},
        )
    if include_bigrams:
        for i in range(len(units) - 1):
            cache.store_bigram(
                units[i],
                units[i + 1],
                intermediates,
                metadata={"prompt": prompt, "source": "online_update", "seed": config.seed},
            )
    return result, intermediates


def generate_pair_references(
    pipe,
    cache: LatentCache,
    pairs: list[tuple[str, str]],
    config: TeleportConfig,
    output_dir: str,
):
    """Generate 20-step reference images for compound prompts (unit pairs)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = []

    for idx, (a, b) in enumerate(pairs):
        compound = f"{a}, {b}"
        result, intermediates = capture_intermediates(
            pipe, compound, config.height, config.width,
            config.num_steps, config.seed, config.device, config.guidance_scale,
        )
        img_path = out / f"pair_{idx:04d}.png"
        if result.images:
            result.images[0].save(str(img_path))

        unit_compound = VisualUnit.from_text(compound)
        cache.store_latents(
            unit_compound, intermediates,
            metadata={"compound": True, "unit_a": a, "unit_b": b, "seed": config.seed},
        )

        records.append({
            "index": idx, "unit_a": a, "unit_b": b, "compound": compound,
            "image_path": str(img_path), "steps": config.num_steps,
        })
        log.info(f"[{idx+1}/{len(pairs)}] pair ref: {compound}")

    meta_path = out / "pair_metadata.jsonl"
    with meta_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def generate_bigram_cache(
    pipe,
    cache: LatentCache,
    pairs: list[tuple[str, str]],
    config: TeleportConfig,
):
    """Pre-cache latents for visual unit bigrams (ordered pairs)."""
    for idx, (a_text, b_text) in enumerate(pairs):
        unit_a = VisualUnit.from_text(a_text)
        unit_b = VisualUnit.from_text(b_text)
        if cache.has_bigram(unit_a, unit_b):
            log.info(f"[{idx+1}/{len(pairs)}] skip cached bigram: {a_text} + {b_text}")
            continue

        compound = f"{a_text}, {b_text}"
        try:
            _, intermediates = capture_intermediates(
                pipe, compound, config.height, config.width,
                config.num_steps, config.seed, config.device, config.guidance_scale,
            )
            cache.store_bigram(
                unit_a, unit_b, intermediates,
                metadata={"unit_a": a_text, "unit_b": b_text, "seed": config.seed},
            )
            log.info(f"[{idx+1}/{len(pairs)}] bigram cached: {a_text} + {b_text}")
        except Exception as e:
            log.warning(f"[{idx+1}/{len(pairs)}] bigram failed: {e}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description="Populate latent teleportation cache")
    parser.add_argument("--model-id", default="Tongyi-MAI/Z-Image-Turbo")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--cache-dir", default="/vfast/latentteleport/cache")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--prompts-file", default="")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--generate-pairs", action="store_true")
    parser.add_argument("--pairs-output-dir", default="/vfast/latentteleport/datasets/pairs_v1")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--tokenizer-strategy", default="nlp", choices=["nlp", "curated", "clip"])
    args = parser.parse_args()

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map.get(args.dtype, torch.bfloat16)

    config = TeleportConfig(
        model_id=args.model_id, device=args.device, dtype=args.dtype,
        height=args.height, width=args.width, num_steps=args.steps,
        guidance_scale=args.guidance_scale, cache_dir=args.cache_dir, seed=args.seed,
    )
    tok_config = TokenizerConfig(strategy=args.tokenizer_strategy)

    prompts = list(args.prompt)
    if args.prompts_file and Path(args.prompts_file).exists():
        prompts.extend(
            line.strip() for line in Path(args.prompts_file).read_text().splitlines() if line.strip()
        )
    if not prompts:
        prompts = DEFAULT_PROMPTS

    from diffusers import ZImagePipeline
    pipe = ZImagePipeline.from_pretrained(args.model_id, torch_dtype=dtype)
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(args.device)

    cache = LatentCache(args.cache_dir, resolution=(args.height, args.width))
    populate_cache(pipe, cache, prompts, config, tok_config, args.shard_index, args.num_shards)

    if args.generate_pairs:
        generate_pair_references(pipe, cache, COMPOUND_PAIRS, config, args.pairs_output_dir)

    # Also generate bigram cache entries
    generate_bigram_cache(pipe, cache, COMPOUND_PAIRS, config)

    stats = cache.stats()
    log.info(f"Cache stats: {json.dumps(stats)}")


if __name__ == "__main__":
    main()
