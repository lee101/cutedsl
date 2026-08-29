"""Swap Cosmos/Anima transformer blocks onto the fused Triton kernels.

`apply_fused_blocks` is numerics-preserving: layer-norm statistics stay in fp32 and
the modulation algebra is unchanged, so only floating-point association differs.
Blocks that use ControlNet projections or image context fall back to diffusers.
"""

from __future__ import annotations

import types

import torch

from .triton_kernels.fused_adaln import adaln_modulate, gated_residual


def _modulation(norm, embedded_timestep: torch.Tensor, temb: torch.Tensor | None, chunks: int):
    values = norm.linear_2(norm.linear_1(norm.activation(embedded_timestep)))
    if temb is not None:
        values = values + (temb if chunks == 3 else temb[..., : values.shape[-1]])
    return values.chunk(chunks, dim=-1)


def _fused_block_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states=None,
    embedded_timestep: torch.Tensor = None,
    temb: torch.Tensor = None,
    image_rotary_emb=None,
    extra_pos_emb=None,
    attention_mask=None,
    controlnet_residual=None,
    latents=None,
    block_idx=None,
):
    if extra_pos_emb is not None:
        hidden_states = hidden_states + extra_pos_emb

    shift, scale, gate = _modulation(self.norm1, embedded_timestep, temb, 3)
    normed = adaln_modulate(hidden_states, shift, scale, self.norm1.norm.eps)
    hidden_states = gated_residual(hidden_states, self.attn1(normed, image_rotary_emb=image_rotary_emb), gate)

    shift, scale, gate = _modulation(self.norm2, embedded_timestep, temb, 3)
    normed = adaln_modulate(hidden_states, shift, scale, self.norm2.norm.eps)
    attn_output = self.attn2(normed, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask)
    hidden_states = gated_residual(hidden_states, attn_output, gate)

    shift, scale, gate = _modulation(self.norm3, embedded_timestep, temb, 3)
    normed = adaln_modulate(hidden_states, shift, scale, self.norm3.norm.eps)
    hidden_states = gated_residual(hidden_states, self.ff(normed), gate)

    if controlnet_residual is not None:
        hidden_states = hidden_states + controlnet_residual
    return hidden_states


def _supported(block) -> bool:
    return (
        getattr(block, "before_proj", None) is None
        and getattr(block, "after_proj", None) is None
        and not getattr(block, "img_context", False)
    )


def apply_fused_blocks(transformer) -> int:
    """Bind the fused forward to every eligible block; returns how many were patched."""
    patched = 0
    for block in transformer.transformer_blocks:
        if not _supported(block):
            continue
        block.forward = types.MethodType(_fused_block_forward, block)
        patched += 1
    return patched


def remove_fused_blocks(transformer) -> None:
    for block in transformer.transformer_blocks:
        if "forward" in block.__dict__:
            del block.__dict__["forward"]
