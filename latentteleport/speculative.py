"""Speculative latent walking: don't walk the latent space alone.

Big model takes one real denoising step, a small walker drafts k cheap steps,
a gap interpolator teleports the draft endpoint onto the big model's likely
trajectory, and the big model verifies/corrects. Trained on captured
trajectories (scripts/spec_collect.py).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn


def timestep_embed(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / max(half - 1, 1))
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class FiLMResBlock(nn.Module):
    def __init__(self, c: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, c)
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)
        self.film = nn.Linear(cond_dim, 2 * c)

    def forward(self, x, cond):
        scale, shift = self.film(cond)[:, :, None, None].chunk(2, dim=1)
        h = self.conv1(torch.nn.functional.silu(self.norm1(x)))
        h = h * (1 + scale) + shift
        h = self.conv2(torch.nn.functional.silu(self.norm2(h)))
        return x + h


@dataclass
class SpecConfig:
    latent_channels: int = 16
    hidden: int = 128
    blocks: int = 6
    cond_dim: int = 256
    text_dim: int = 0  # 0 = unconditional on text


class LatentWalker(nn.Module):
    """Drafts x_{t+1} from (x_t, delta) where delta is the last step's movement
    — the trajectory's momentum. Rolled out k times for speculation, updating
    delta with its own predicted movement."""

    def __init__(self, cfg: SpecConfig | None = None):
        super().__init__()
        self.cfg = cfg or SpecConfig()
        c = self.cfg.hidden
        in_ch = self.cfg.latent_channels * 2
        self.inp = nn.Conv2d(in_ch, c, 3, padding=1)
        self.cond_mlp = nn.Sequential(
            nn.Linear(128 + 128 + self.cfg.text_dim, self.cfg.cond_dim),
            nn.SiLU(),
            nn.Linear(self.cfg.cond_dim, self.cfg.cond_dim),
        )
        self.blocks = nn.ModuleList(FiLMResBlock(c, self.cfg.cond_dim) for _ in range(self.cfg.blocks))
        self.out = nn.Conv2d(c, self.cfg.latent_channels, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, delta, t_frac, total_steps, text_emb=None):
        cond = [timestep_embed(t_frac, 128), timestep_embed(total_steps.float() / 32.0, 128)]
        if self.cfg.text_dim:
            cond.append(text_emb if text_emb is not None else x.new_zeros(x.shape[0], self.cfg.text_dim))
        cond = self.cond_mlp(torch.cat(cond, dim=-1))
        h = self.inp(torch.cat([x, delta], dim=1))
        for b in self.blocks:
            h = b(h, cond)
        # momentum prior: continue the last movement, learn the correction
        return x + delta + self.out(h)

    @torch.no_grad()
    def rollout(self, x, delta, t_idx: int, k: int, total_steps: int, text_emb=None):
        outs = []
        cur, d = x, delta
        n = torch.full((x.shape[0],), float(total_steps), device=x.device)
        for j in range(k):
            tf = torch.full((x.shape[0],), (t_idx + j) / max(total_steps - 1, 1), device=x.device)
            nxt = self.forward(cur, d, tf, n, text_emb)
            d = nxt - cur
            cur = nxt
            outs.append(cur)
        return outs


class GapInterpolator(nn.Module):
    """Teleport correction: predict big-model x_{t+k} from (anchor x_t, the
    anchor's real movement delta, walker draft endpoint). Residual on the
    draft, so identity = trust walker."""

    def __init__(self, cfg: SpecConfig | None = None):
        super().__init__()
        self.cfg = cfg or SpecConfig()
        c = self.cfg.hidden
        in_ch = self.cfg.latent_channels * 3
        self.inp = nn.Conv2d(in_ch, c, 3, padding=1)
        self.cond_mlp = nn.Sequential(
            nn.Linear(128 + 128 + self.cfg.text_dim, self.cfg.cond_dim),
            nn.SiLU(),
            nn.Linear(self.cfg.cond_dim, self.cfg.cond_dim),
        )
        self.blocks = nn.ModuleList(FiLMResBlock(c, self.cfg.cond_dim) for _ in range(self.cfg.blocks))
        self.out = nn.Conv2d(c, self.cfg.latent_channels, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, anchor, delta, draft, t_frac, k_frac, text_emb=None):
        cond = [timestep_embed(t_frac, 128), timestep_embed(k_frac, 128)]
        if self.cfg.text_dim:
            cond.append(text_emb if text_emb is not None else anchor.new_zeros(anchor.shape[0], self.cfg.text_dim))
        cond = self.cond_mlp(torch.cat(cond, dim=-1))
        h = self.inp(torch.cat([anchor, delta, draft], dim=1))
        for b in self.blocks:
            h = b(h, cond)
        return draft + self.out(h)


def taylor1(x_t, x_prev, k: int):
    return x_t + k * (x_t - x_prev)


def scaled_momentum(x_t, x_prev, k: int, scale: float):
    """Schedule-calibrated first-order trajectory forecast."""
    return x_t + scale * k * (x_t - x_prev)


def build_active_lane_plan(
    schedules: list[list[int]] | tuple[tuple[int, ...], ...],
    start_step: int = 0,
) -> list[tuple[int, tuple[int, ...]]]:
    """Group lane-specific anchor endpoints into compact denoiser events.

    Every horizon must be positive and every lane must cover the same total
    span, so the lanes reconverge at a shared final anchor.  The returned lane
    ids can be passed to ``index_select`` before a compact transformer call and
    scattered back into the full latent batch afterwards.
    """
    if not schedules:
        return []
    totals = {sum(schedule) for schedule in schedules}
    if len(totals) != 1:
        raise ValueError("lane schedules must cover the same total span")
    if any(not schedule or any(horizon <= 0 for horizon in schedule) for schedule in schedules):
        raise ValueError("lane schedules must contain positive horizons")

    events: dict[int, list[int]] = {}
    for lane, schedule in enumerate(schedules):
        step = start_step
        for horizon in schedule:
            step += horizon
            events.setdefault(step, []).append(lane)
    return [(step, tuple(lanes)) for step, lanes in sorted(events.items())]


@torch.no_grad()
def zimage_prepare(
    pipe,
    prompt: str,
    total_steps: int,
    height: int,
    width: int,
    seed: int,
    device: str = "cuda",
    prompt_embeds=None,
):
    """Encode prompt + prepare latents + full-schedule timesteps, mirroring
    ZImagePipeline.__call__ steps 1-5 (guidance 0 path)."""
    from diffusers.pipelines.z_image.pipeline_z_image import calculate_shift, retrieve_timesteps

    if prompt_embeds is None:
        prompt_embeds, _ = pipe.encode_prompt(prompt=prompt, device=device)
    gen = torch.Generator(device=device).manual_seed(seed)
    latents = pipe.prepare_latents(1, pipe.transformer.in_channels, height, width, torch.float32, device, gen, None)
    image_seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)
    mu = calculate_shift(
        image_seq_len,
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 4096),
        pipe.scheduler.config.get("base_shift", 0.5),
        pipe.scheduler.config.get("max_shift", 1.15),
    )
    pipe.scheduler.sigma_min = 0.0
    timesteps, _ = retrieve_timesteps(pipe.scheduler, total_steps, device, mu=mu)
    return prompt_embeds, latents, timesteps


@torch.no_grad()
def zimage_big_step(pipe, latents, t, prompt_embeds):
    timestep = t.expand(latents.shape[0])
    timestep = (1000 - timestep) / 1000
    inp = list(latents.to(pipe.transformer.dtype).unsqueeze(2).unbind(dim=0))
    out = pipe.transformer(inp, timestep, prompt_embeds, return_dict=False)[0]
    noise_pred = -torch.stack([o.float() for o in out], dim=0).squeeze(2)
    return pipe.scheduler.step(noise_pred.to(torch.float32), t, latents, return_dict=False)[0]


@torch.no_grad()
def zimage_decode(pipe, latents):
    lat = latents.to(pipe.vae.dtype)
    lat = (lat / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    image = pipe.vae.decode(lat, return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type="pil")


@torch.no_grad()
def speculative_denoise(
    pipe,
    prompt: str,
    walker: LatentWalker | None,
    interp: GapInterpolator | None,
    total_steps: int = 16,
    draft_k: int = 3,
    height: int = 512,
    width: int = 512,
    seed: int = 0,
    device: str = "cuda",
    mode: str = "spec",  # baseline | spec | taylor | scaled | skip
    momentum_scales: Mapping[tuple[int, int], float] | None = None,
    prompt_embeds=None,
):
    """Manual denoise on the FULL schedule, but every real step is followed by
    draft_k teleported steps — those transformer calls are skipped entirely.
    mode=spec: walker rollout + interpolator; taylor: x + k*(x - x_prev_real);
    scaled: schedule-calibrated Taylor using momentum_scales[(anchor_step, k)];
    skip: no correction. Last step is always real."""
    if mode not in {"baseline", "spec", "taylor", "scaled", "skip"}:
        raise ValueError(f"unknown speculative mode: {mode}")
    if mode == "scaled" and momentum_scales is None:
        raise ValueError("scaled mode requires fitted momentum_scales")
    cuda_timing = torch.device(device).type == "cuda" and torch.cuda.is_available()

    def synchronize():
        if cuda_timing:
            torch.cuda.synchronize(device)

    if cuda_timing:
        torch.cuda.reset_peak_memory_stats(device)
    synchronize()
    started = time.perf_counter()
    prompt_embeds, latents, timesteps = zimage_prepare(
        pipe, prompt, total_steps, height, width, seed, device,
        prompt_embeds=prompt_embeds,
    )
    synchronize()
    stats = {
        "big_steps": 0,
        "drafted_steps": 0,
        "scale_fallbacks": 0,
        "prepare_ms": (time.perf_counter() - started) * 1000.0,
    }
    denoise_started = time.perf_counter()
    i = 0
    n = len(timesteps)
    while i < n:
        before = latents
        latents = zimage_big_step(pipe, latents, timesteps[i], prompt_embeds)
        stats["big_steps"] += 1
        i += 1
        k = 0 if mode == "baseline" else min(draft_k, n - 1 - i)  # keep the final step real
        if k > 0:
            if mode == "spec" and walker is not None and interp is not None:
                delta = (latents - before).float()
                drafts = walker.rollout(latents.float(), delta, i, k, total_steps)
                tf = torch.full((latents.shape[0],), i / max(total_steps - 1, 1), device=latents.device)
                kf = torch.full((latents.shape[0],), k / max(total_steps - 1, 1), device=latents.device)
                latents = interp(latents.float(), delta, drafts[-1], tf, kf).to(torch.float32)
            elif mode == "taylor":
                latents = latents + k * (latents - before)
            elif mode == "scaled":
                # Stored trajectory step 0 is the latent after denoiser call 0,
                # while i is the number of consumed timesteps here.
                anchor_step = i - 1
                scale_key = (anchor_step, k)
                if scale_key not in momentum_scales:
                    stats["scale_fallbacks"] += 1
                scale = momentum_scales.get(scale_key, 1.0)
                latents = scaled_momentum(latents, before, k, scale)
            if hasattr(pipe.scheduler, "_step_index") and pipe.scheduler._step_index is not None:
                pipe.scheduler._step_index += k
            stats["drafted_steps"] += k
            i += k
    synchronize()
    stats["denoise_ms"] = (time.perf_counter() - denoise_started) * 1000.0
    decode_started = time.perf_counter()
    image = zimage_decode(pipe, latents)
    synchronize()
    stats["decode_ms"] = (time.perf_counter() - decode_started) * 1000.0
    stats["total_ms"] = stats["prepare_ms"] + stats["denoise_ms"] + stats["decode_ms"]
    if cuda_timing:
        stats["peak_vram_mb"] = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        stats["reserved_vram_mb"] = torch.cuda.memory_reserved(device) / (1024 * 1024)
    return image, stats
