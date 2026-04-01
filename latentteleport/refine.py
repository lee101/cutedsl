"""Refinement pipeline: inject cached/combined latent at a timestep, run remaining steps."""

from __future__ import annotations

import logging
import time
from typing import Any

import torch

from latentteleport.cache import LatentCache
from latentteleport.combiner import LatentCombiner, TreeCombiner, slerp
from latentteleport.confidence import ConfidenceConfig, ConfidenceGate
from latentteleport.config import CombinerConfig, TeleportConfig
from latentteleport.tokenizer import TokenizerStrategy, VisualUnit

log = logging.getLogger(__name__)


def refine_from_latent(
    pipe,
    latent: torch.Tensor,
    prompt: str,
    start_step: int,
    num_total_steps: int = 20,
    height: int = 512,
    width: int = 512,
    guidance_scale: float = 0.0,
    seed: int = 42,
    device: str = "cuda",
) -> Any:
    """Inject a latent at start_step and denoise for the remaining steps.

    This is the core of latent teleportation: skip the first start_step
    denoising steps by starting from a pre-cached/combined latent.
    """
    scheduler = pipe.scheduler

    # Set up full timestep schedule
    # ZImage uses flow matching with mu-based schedule
    latent_image_ids = pipe._prepare_latent_image_ids(
        1, height // (pipe.vae_scale_factor * 2),
        width // (pipe.vae_scale_factor * 2),
        device, pipe.transformer.dtype,
    )

    # Handle mu calculation for flow matching
    mu = pipe._calculate_shift(
        latent_image_ids,
        scheduler.config.get("base_shift", 0.5),
        scheduler.config.get("max_shift", 1.15),
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 4096),
    ) if hasattr(pipe, "_calculate_shift") else None

    if mu is not None:
        scheduler.set_timesteps(num_total_steps, device=device, mu=mu)
    else:
        scheduler.set_timesteps(num_total_steps, device=device)

    # Slice timesteps: skip the first start_step steps
    all_timesteps = scheduler.timesteps
    remaining_timesteps = all_timesteps[start_step:]
    num_remaining = len(remaining_timesteps)

    if hasattr(scheduler, "set_begin_index"):
        scheduler.set_begin_index(start_step)

    # Encode prompt
    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=None,
        device=device,
    )

    # Ensure latent is on correct device/dtype
    latent = latent.to(device=device, dtype=pipe.transformer.dtype)
    if latent.dim() == 3:
        latent = latent.unsqueeze(0)  # add batch dim

    # Pack latent for transformer if needed
    latents = pipe._pack_latents(latent, 1, latent.shape[1], latent.shape[2], latent.shape[3])

    log.info(f"Refining: {num_remaining} steps from step {start_step}/{num_total_steps}")

    # Run remaining denoising steps
    for i, t in enumerate(remaining_timesteps):
        timestep = t.expand(latents.shape[0]).to(latents.dtype)

        noise_pred = pipe.transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]

        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    # Unpack and decode
    latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]
    image = pipe.image_processor.postprocess(image, output_type="pil")[0]

    return image


def refine_from_latent_simple(
    pipe,
    latent: torch.Tensor,
    prompt: str,
    strength: float = 0.7,
    num_total_steps: int = 20,
    height: int = 512,
    width: int = 512,
    guidance_scale: float = 0.0,
    seed: int = 42,
    device: str = "cuda",
) -> Any:
    """Simpler approach using img2img-style strength parameter.

    strength=0.7 means run 70% of steps (skip first 30%).
    Falls back to this if the pipeline supports img2img mode.
    """
    scheduler = pipe.scheduler
    num_steps = num_total_steps
    start_step = int(num_steps * (1.0 - strength))

    return refine_from_latent(
        pipe, latent, prompt, start_step, num_steps,
        height, width, guidance_scale, seed, device,
    )


class TeleportPipeline:
    """Full teleportation pipeline: tokenize -> cache lookup -> combine -> refine."""

    def __init__(
        self,
        pipe,
        cache: LatentCache,
        tokenizer: TokenizerStrategy,
        combiner: LatentCombiner | TreeCombiner,
        config: TeleportConfig,
        combiner_config: CombinerConfig,
        confidence_gate: ConfidenceGate | None = None,
    ):
        self.pipe = pipe
        self.cache = cache
        self.tokenizer = tokenizer
        self.combiner = combiner
        self.config = config
        self.comb_config = combiner_config
        self._start_step = int(config.num_steps * combiner_config.teleport_timestep)
        self.confidence_gate = confidence_gate or ConfidenceGate()

    def generate(self, prompt: str, seed: int | None = None) -> dict:
        """Generate an image via latent teleportation."""
        seed = seed or self.config.seed
        t0 = time.time()

        # 1. Tokenize into visual units
        units = self.tokenizer.tokenize(prompt)

        # 2. Look up cached latents (bigram first, then individual units)
        latents = []
        embeddings = []
        hits = 0
        bigram_hits = 0
        i = 0
        while i < len(units):
            # Try bigram with next unit
            if i + 1 < len(units):
                lat, method = self.cache.lookup_best([units[i], units[i + 1]], self._start_step)
                if method == "bigram":
                    latents.append(lat)
                    bigram_hits += 1
                    hits += 2
                    # Get embedding for first unit as proxy
                    emb = self.cache.load_text_embedding(units[i])
                    if emb is not None:
                        embeddings.append(emb)
                    i += 2
                    continue
            # Fall back to individual unit
            lat = self.cache.load_latent(units[i], self._start_step)
            emb = self.cache.load_text_embedding(units[i])
            if lat is not None:
                latents.append(lat)
                hits += 1
                if emb is not None:
                    embeddings.append(emb)
            i += 1

        cache_time = time.time() - t0

        # 3. Combine cached latents
        t1 = time.time()
        if not latents:
            log.info(f"Full cache miss for '{prompt}', falling back to full gen")
            from latentteleport.dataset import capture_intermediates
            result, _ = capture_intermediates(
                self.pipe, prompt, self.config.height, self.config.width,
                self.config.num_steps, seed, self.config.device, self.config.guidance_scale,
            )
            return {
                "image": result.images[0] if result.images else None,
                "method": "full_generation",
                "cache_hits": 0,
                "total_units": len(units),
                "elapsed_s": time.time() - t0,
            }

        if hasattr(self.combiner, "combine_tree"):
            combined = self.combiner.combine_tree(latents, embeddings or None)
        elif len(latents) >= 2:
            combined = latents[0]
            for i in range(1, len(latents)):
                ea = embeddings[i - 1] if i - 1 < len(embeddings) else None
                eb = embeddings[i] if i < len(embeddings) else None
                combined = self.combiner.combine(combined, latents[i], ea, eb)
        else:
            combined = latents[0]

        combine_time = time.time() - t1

        # 4. Confidence-gated adaptive refinement
        t2 = time.time()
        # Estimate how many refinement steps this combined latent needs
        text_sim = None
        if embeddings:
            # Use cache hit ratio as proxy for text match quality
            text_sim = hits / max(len(units), 1)
        adaptive_steps = self.confidence_gate.estimate_steps(
            combined, text_similarity=text_sim,
        )
        # Convert adaptive steps to start_step: more steps = earlier start
        adaptive_start = max(0, self.config.num_steps - adaptive_steps)
        log.info(f"Confidence gate: {adaptive_steps} refinement steps (start at {adaptive_start})")

        try:
            image = refine_from_latent(
                self.pipe, combined, prompt, adaptive_start,
                self.config.num_steps, self.config.height, self.config.width,
                self.config.guidance_scale, seed, self.config.device,
            )
        except Exception as e:
            log.warning(f"Refinement failed: {e}, falling back to full gen")
            from latentteleport.dataset import capture_intermediates
            result, _ = capture_intermediates(
                self.pipe, prompt, self.config.height, self.config.width,
                self.config.num_steps, seed, self.config.device, self.config.guidance_scale,
            )
            image = result.images[0] if result.images else None

        refine_time = time.time() - t2
        total_time = time.time() - t0

        return {
            "image": image,
            "method": "teleport",
            "cache_hits": hits,
            "bigram_hits": bigram_hits,
            "total_units": len(units),
            "start_step": adaptive_start,
            "refinement_steps": adaptive_steps,
            "elapsed_s": total_time,
            "cache_time_s": cache_time,
            "combine_time_s": combine_time,
            "refine_time_s": refine_time,
            "units": [u.text for u in units],
            "confidence_stats": self.confidence_gate.get_stats(),
        }
