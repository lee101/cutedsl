"""Exact same-prompt resume helpers for flow-matching Diffusers pipelines.

FLUX and Krea2 accept custom ``sigmas`` and packed intermediate ``latents``.
Replaying the unshifted tail of the original linear sigma schedule resumes the
same denoising trajectory without treating a shorter, newly-spaced schedule as
equivalent. This is an exact-cache primitive, not cross-prompt teleportation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch


@dataclass
class FlowResumeResult:
    image: object
    wall_s: float
    physical_steps: int
    latent: torch.Tensor | None = None


def remaining_linear_sigmas(total_steps: int, capture_after_step: int) -> list[float]:
    """Return the original unshifted sigma tail after a zero-based step."""
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if capture_after_step < 0 or capture_after_step >= total_steps - 1:
        raise ValueError("capture_after_step must leave at least one denoising step")
    if total_steps == 1:
        return []
    decrement = (1.0 - 1.0 / total_steps) / (total_steps - 1)
    full = [1.0 - decrement * index for index in range(total_steps)]
    return full[capture_after_step + 1 :]


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.inference_mode()
def capture_flow_latent(
    pipe,
    pipeline_kwargs: dict,
    total_steps: int,
    capture_after_step: int,
) -> FlowResumeResult:
    """Run a full trajectory and retain the packed latent after one step."""
    if not remaining_linear_sigmas(total_steps, capture_after_step):
        raise ValueError("capture point has no resumable sigma tail")
    captured = None

    def callback(_pipe, step_index, _timestep, callback_kwargs):
        nonlocal captured
        if step_index == capture_after_step:
            captured = callback_kwargs["latents"].detach().clone()
        return callback_kwargs

    kwargs = dict(pipeline_kwargs)
    kwargs.update(
        {
            "num_inference_steps": total_steps,
            "callback_on_step_end": callback,
            "callback_on_step_end_tensor_inputs": ["latents"],
        }
    )
    _sync_cuda()
    started = time.perf_counter()
    image = pipe(**kwargs).images[0]
    _sync_cuda()
    if captured is None:
        raise RuntimeError(f"pipeline did not expose latent at step {capture_after_step}")
    return FlowResumeResult(
        image=image,
        wall_s=time.perf_counter() - started,
        physical_steps=total_steps,
        latent=captured,
    )


@torch.inference_mode()
def resume_flow_latent(
    pipe,
    latent: torch.Tensor,
    pipeline_kwargs: dict,
    total_steps: int,
    capture_after_step: int,
) -> FlowResumeResult:
    """Resume from a captured packed latent using the original sigma tail."""
    sigmas = remaining_linear_sigmas(total_steps, capture_after_step)
    kwargs = dict(pipeline_kwargs)
    kwargs.update(
        {
            "latents": latent.detach().clone(),
            "sigmas": sigmas,
            "num_inference_steps": len(sigmas),
        }
    )
    _sync_cuda()
    started = time.perf_counter()
    image = pipe(**kwargs).images[0]
    _sync_cuda()
    return FlowResumeResult(
        image=image,
        wall_s=time.perf_counter() - started,
        physical_steps=len(sigmas),
    )
