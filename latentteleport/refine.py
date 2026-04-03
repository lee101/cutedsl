"""Refinement pipeline: inject cached/combined latent at a timestep, run remaining steps."""

from __future__ import annotations

import logging
import time
from typing import Any

import torch
from diffusers.pipelines.z_image.pipeline_z_image import calculate_shift, retrieve_timesteps

from latentteleport.cache import LatentCache
from latentteleport.combiner import LatentCombiner, TreeCombiner, slerp
from latentteleport.confidence import ConfidenceConfig, ConfidenceGate
from latentteleport.config import CombinerConfig, TeleportConfig
from latentteleport.tokenizer import TokenizerStrategy, VisualUnit
from latentteleport.trajectory import apply_knn_trajectory_prior
from latentteleport.step_forecaster import LatentStepForecaster

log = logging.getLogger(__name__)


def refine_from_latent(
    pipe,
    latent: torch.Tensor,
    prompt: str,
    negative_prompt: str | None,
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
    latent = latent.to(device=device, dtype=torch.float32)
    if latent.dim() == 3:
        latent = latent.unsqueeze(0)
    latents = latent

    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=guidance_scale > 0,
        device=device,
    )

    batch_size = latents.shape[0]
    image_seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)
    mu = calculate_shift(
        image_seq_len,
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_shift", 0.5),
        scheduler.config.get("max_shift", 1.15),
    )
    scheduler.sigma_min = 0.0
    timesteps, _ = retrieve_timesteps(
        scheduler,
        num_total_steps,
        device,
        sigmas=None,
        mu=mu,
    )
    remaining_timesteps = timesteps[start_step:]
    num_remaining = len(remaining_timesteps)

    log.info(f"Refining: {num_remaining} steps from step {start_step}/{num_total_steps}")

    # Run remaining denoising steps
    for i, t in enumerate(remaining_timesteps):
        timestep = t.expand(batch_size)
        timestep = (1000 - timestep) / 1000
        apply_cfg = guidance_scale > 0 and bool(negative_prompt_embeds)
        if apply_cfg:
            latent_model_input = latents.to(pipe.transformer.dtype).repeat(2, 1, 1, 1)
            prompt_embeds_model_input = prompt_embeds + negative_prompt_embeds
            timestep_model_input = timestep.repeat(2)
        else:
            latent_model_input = latents.to(pipe.transformer.dtype)
            prompt_embeds_model_input = prompt_embeds
            timestep_model_input = timestep

        latent_model_input = latent_model_input.unsqueeze(2)
        latent_model_input_list = list(latent_model_input.unbind(dim=0))
        model_out_list = pipe.transformer(
            latent_model_input_list,
            timestep_model_input,
            prompt_embeds_model_input,
            return_dict=False,
        )[0]

        if apply_cfg:
            pos_out = model_out_list[:batch_size]
            neg_out = model_out_list[batch_size:]
            noise_pred = []
            for j in range(batch_size):
                pos = pos_out[j].float()
                neg = neg_out[j].float()
                noise_pred.append(pos + guidance_scale * (pos - neg))
            noise_pred = torch.stack(noise_pred, dim=0)
        else:
            noise_pred = torch.stack([item.float() for item in model_out_list], dim=0)

        noise_pred = noise_pred.squeeze(2)
        noise_pred = -noise_pred
        latents = scheduler.step(noise_pred.to(torch.float32), t, latents, return_dict=False)[0]

    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]
    image = pipe.image_processor.postprocess(image, output_type="pil")[0]

    return image


def refine_from_latent_simple(
    pipe,
    latent: torch.Tensor,
    prompt: str,
    negative_prompt: str | None = None,
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
        negative_prompt, height, width, guidance_scale, seed, device,
    )


def _pooled_condition_embedding(
    pipe,
    cache: LatentCache,
    tokenizer: TokenizerStrategy,
    text: str | None,
    device: str,
) -> torch.Tensor | None:
    if not text:
        return None
    units = tokenizer.tokenize(text)
    cached = []
    for unit in units:
        emb = cache.load_text_embedding(unit)
        if emb is not None:
            cached.append(emb.float())
    if cached:
        return torch.stack(cached, dim=0).mean(dim=0)
    prompt_embeds, _negative = pipe.encode_prompt(
        prompt=text,
        negative_prompt=None,
        do_classifier_free_guidance=False,
        device=device,
    )
    if not prompt_embeds:
        return None
    pooled = [emb.float().mean(dim=0) for emb in prompt_embeds]
    return torch.stack(pooled, dim=0).mean(dim=0)


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
        step_forecaster: LatentStepForecaster | None = None,
    ):
        self.pipe = pipe
        self.cache = cache
        self.tokenizer = tokenizer
        self.combiner = combiner
        self.config = config
        self.comb_config = combiner_config
        self._start_step = int(config.num_steps * combiner_config.teleport_timestep)
        self.confidence_gate = confidence_gate or ConfidenceGate()
        self.step_forecaster = step_forecaster

    def generate(self, prompt: str, seed: int | None = None, negative_prompt: str | None = None) -> dict:
        """Generate an image via latent teleportation."""
        seed = seed or self.config.seed
        negative_prompt = self.config.negative_prompt if negative_prompt is None else negative_prompt
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
            from latentteleport.dataset import cache_prompt_trajectory, capture_intermediates
            if self.config.online_cache_updates:
                result, _ = cache_prompt_trajectory(
                    self.pipe,
                    self.cache,
                    prompt,
                    self.config,
                    include_bigrams=self.config.online_store_bigrams,
                )
            else:
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

        trajectory_stats = {}
        positive_anchor = _pooled_condition_embedding(
            self.pipe, self.cache, self.tokenizer, prompt, self.config.device,
        )
        negative_anchor = _pooled_condition_embedding(
            self.pipe, self.cache, self.tokenizer, negative_prompt, self.config.device,
        )
        repel_embeddings = [negative_anchor] if negative_anchor is not None else []
        if positive_anchor is not None:
            embeddings_for_guidance = [positive_anchor]
        else:
            embeddings_for_guidance = embeddings
        if embeddings and self.config.trajectory_virtual_steps > 0:
            if self.step_forecaster is not None:
                pooled_emb = (positive_anchor if positive_anchor is not None else torch.stack(embeddings).mean(dim=0)).clone()
                if negative_anchor is not None and self.config.negative_trajectory_scale > 0:
                    pooled_emb = pooled_emb - self.config.negative_trajectory_scale * negative_anchor
                pooled_emb = pooled_emb.unsqueeze(0).to(combined.device, combined.dtype)
                latent_in = combined.unsqueeze(0) if combined.dim() == 4 else combined
                timestep = torch.tensor([float(self._start_step)], device=combined.device, dtype=combined.dtype)
                with torch.no_grad():
                    combined = self.step_forecaster(latent_in.float(), timestep.float(), pooled_emb.float()).squeeze(0).to(combined.dtype)
                trajectory_stats = {
                    "mode": "learned_forecaster",
                    "virtual_steps_applied": 1,
                    "negative_prompt_used": bool(negative_prompt),
                }
            else:
                combined, trajectory_stats = apply_knn_trajectory_prior(
                    self.cache,
                    combined,
                    embeddings_for_guidance,
                    repel_embeddings,
                    self._start_step,
                    top_k=self.config.trajectory_top_k,
                    scale=self.config.trajectory_scale,
                    repel_scale=self.config.negative_trajectory_scale,
                    virtual_steps=self.config.trajectory_virtual_steps,
                )

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
                self.pipe, combined, prompt, negative_prompt, adaptive_start,
                self.config.num_steps, self.config.height, self.config.width,
                self.config.guidance_scale, seed, self.config.device,
            )
        except Exception as e:
            log.warning(f"Refinement failed: {e}, falling back to full gen")
            from latentteleport.dataset import cache_prompt_trajectory, capture_intermediates
            if self.config.online_cache_updates:
                result, _ = cache_prompt_trajectory(
                    self.pipe,
                    self.cache,
                    prompt,
                    self.config,
                    include_bigrams=self.config.online_store_bigrams,
                )
            else:
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
            "trajectory_stats": trajectory_stats,
            "negative_prompt_used": bool(negative_prompt),
        }
