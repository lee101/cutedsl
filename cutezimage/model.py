"""CuteZImage: Accelerated Z-Image Turbo transformer with fused Triton kernels.

Replaces key operations in the ZImageTransformer2DModel with fused kernels:
- RMS normalization via Triton
- SiLU-gated FFN with fused activation
- AdaLN modulation fused with normalization
- Complex-valued RoPE via Triton

Weight loading from HuggingFace Z-Image checkpoints is supported via
the from_diffusers() factory method.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

# Module-level constants matching diffusers transformer_z_image.py
ADALN_EMBED_DIM = 256
SEQ_MULTI_OF = 32


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CuteZImageConfig:
    patch_size: int = 2
    f_patch_size: int = 1
    in_channels: int = 16
    dim: int = 3840
    n_layers: int = 30
    n_refiner_layers: int = 2
    n_heads: int = 30
    n_kv_heads: int = 30
    norm_eps: float = 1e-5
    qk_norm: bool = True
    cap_feat_dim: int = 2560
    rope_theta: float = 256.0
    t_scale: float = 1000.0
    axes_dims: list[int] = field(default_factory=lambda: [32, 48, 48])
    axes_lens: list[int] = field(default_factory=lambda: [1536, 512, 512])
    hidden_dim: int = 0  # auto-computed

    def __post_init__(self):
        if self.hidden_dim == 0:
            self.hidden_dim = int(self.dim / 3 * 8)
        self.head_dim = self.dim // self.n_heads


# ---------------------------------------------------------------------------
# RopeEmbedder
# ---------------------------------------------------------------------------

class RopeEmbedder:
    """Multi-axis RoPE frequency precomputation matching Z-Image's convention."""

    def __init__(
        self,
        theta: float = 256.0,
        axes_dims: list[int] | tuple[int, ...] = (16, 56, 56),
        axes_lens: list[int] | tuple[int, ...] = (64, 128, 128),
    ):
        self.theta = theta
        self.axes_dims = list(axes_dims)
        self.axes_lens = list(axes_lens)
        self.freqs_cis: list[torch.Tensor] | None = None

    @staticmethod
    def precompute_freqs_cis(
        dim: list[int], end: list[int], theta: float = 256.0,
    ) -> list[torch.Tensor]:
        with torch.device("cpu"):
            freqs_cis = []
            for d, e in zip(dim, end):
                freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
                timestep = torch.arange(e, device=freqs.device, dtype=torch.float64)
                freqs = torch.outer(timestep, freqs).float()
                freqs_cis_i = torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64)
                freqs_cis.append(freqs_cis_i)
            return freqs_cis

    def __call__(self, ids: torch.Tensor) -> torch.Tensor:
        device = ids.device
        if self.freqs_cis is None:
            self.freqs_cis = self.precompute_freqs_cis(self.axes_dims, self.axes_lens, theta=self.theta)
            self.freqs_cis = [f.to(device) for f in self.freqs_cis]
        elif self.freqs_cis[0].device != device:
            self.freqs_cis = [f.to(device) for f in self.freqs_cis]

        result = []
        for i in range(len(self.axes_dims)):
            index = ids[:, i]
            result.append(self.freqs_cis[i][index])
        return torch.cat(result, dim=-1)


# ---------------------------------------------------------------------------
# Fallback implementations (pure PyTorch)
# ---------------------------------------------------------------------------

def _rms_norm_fallback(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    orig_dtype = x.dtype
    variance = x.float().pow(2).mean(-1, keepdim=True)
    normed = x.float() * torch.rsqrt(variance + eps)
    normed = normed.to(orig_dtype)
    if weight is not None:
        normed = weight * normed
    return normed


def _silu_gate_fallback(x1: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
    return F.silu(x1) * x3


def _apply_rope_complex_fallback(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply complex-valued RoPE matching Z-Image's convention."""
    with torch.amp.autocast("cuda", enabled=False):
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        if freqs_cis.ndim < x_complex.ndim:
            freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x_complex * freqs_cis).flatten(x_complex.ndim - 1)
        return x_out.type_as(x)


# ---------------------------------------------------------------------------
# Dispatch: use Triton if available, else fallback
# ---------------------------------------------------------------------------

def _use_triton() -> bool:
    """Check if Triton kernels should be used (requires CUDA tensors)."""
    return torch.cuda.is_available()


def _get_rms_norm():
    if _use_triton():
        try:
            from cutezimage.triton_kernels.rms_norm import rms_norm
            return rms_norm
        except ImportError:
            pass
    return _rms_norm_fallback


def _get_silu_gate():
    if _use_triton():
        try:
            from cutezimage.triton_kernels.fused_silu_gate_ffn import fused_silu_gate
            return fused_silu_gate
        except ImportError:
            pass
    return _silu_gate_fallback


def _get_rope_complex():
    if _use_triton():
        try:
            from cutezimage.triton_kernels.rope_complex import apply_rope_complex
            return apply_rope_complex
        except ImportError:
            pass
    return _apply_rope_complex_fallback


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            fn = _get_rms_norm()
        else:
            fn = _rms_norm_fallback
        return fn(x, self.weight, self.eps)


class SiLUGatedFFN(nn.Module):
    """Z-Image gated FFN: w2(silu(w1(x)) * w3(x))."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.w1(x)
        x3 = self.w3(x)
        if x.is_cuda:
            silu_gate_fn = _get_silu_gate()
        else:
            silu_gate_fn = _silu_gate_fallback
        gated = silu_gate_fn(x1, x3)
        return self.w2(gated)


class TimestepEmbedder(nn.Module):
    def __init__(self, out_size: int, mid_size: int = 1024, freq_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, mid_size, bias=True),
            nn.SiLU(),
            nn.Linear(mid_size, out_size, bias=True),
        )
        self.freq_dim = freq_dim

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
        with torch.amp.autocast("cuda", enabled=False):
            half = dim // 2
            freqs = torch.exp(
                -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
            )
            args = t[:, None].float() * freqs[None, :]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.freq_dim)
        return self.mlp(t_freq)


class CuteZImageTransformerBlock(nn.Module):
    """Single Z-Image transformer block with fused operations.

    Architecture:
        x = x + gate_msa * attn(norm1(x) * scale_msa)
        x = x + gate_mlp * ffn(norm2(ffn_norm(x)) * scale_mlp)

    With optional AdaLN modulation for refiner blocks.
    """

    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float = 1e-5,
        qk_norm: bool = True,
        modulation: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.layer_id = layer_id
        self.modulation = modulation

        # Attention projections
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        # QK norm
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=1e-5)
            self.k_norm = RMSNorm(self.head_dim, eps=1e-5)

        # FFN
        hidden_dim = int(dim / 3 * 8)
        self.feed_forward = SiLUGatedFFN(dim, hidden_dim)

        # Norms
        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)
        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        # AdaLN modulation
        if modulation:
            self.adaLN_modulation = nn.Sequential(
                nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True)
            )

    def _apply_attention(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
        freqs_cis: torch.Tensor | None,
    ) -> torch.Tensor:
        B, S, D = x.shape

        q = self.q_proj(x).unflatten(-1, (self.n_heads, self.head_dim))
        k = self.k_proj(x).unflatten(-1, (self.n_kv_heads, self.head_dim))
        v = self.v_proj(x).unflatten(-1, (self.n_kv_heads, self.head_dim))

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Apply RoPE
        if freqs_cis is not None:
            rope_fn = _get_rope_complex() if q.is_cuda else _apply_rope_complex_fallback
            q = rope_fn(q, freqs_cis)
            k = rope_fn(k, freqs_cis)

        # Reshape for SDPA: (B, H, S, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Handle GQA by repeating KV heads
        if self.n_kv_heads < self.n_heads:
            repeat_factor = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        # Attention mask
        if attn_mask is not None and attn_mask.ndim == 2:
            attn_mask = attn_mask[:, None, None, :]

        # Use PyTorch SDPA (FlashAttention or efficient attention)
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False,
        )

        attn_out = attn_out.transpose(1, 2).reshape(B, S, D)
        return self.o_proj(attn_out)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        freqs_cis: torch.Tensor | None = None,
        adaln_input: torch.Tensor | None = None,
        noise_mask: torch.Tensor | None = None,
        adaln_noisy: torch.Tensor | None = None,
        adaln_clean: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.modulation:
            if noise_mask is not None:
                # Per-token modulation (omni mode)
                mod_noisy = self.adaLN_modulation(adaln_noisy)
                mod_clean = self.adaLN_modulation(adaln_clean)

                scale_msa_noisy, gate_msa_noisy, scale_mlp_noisy, gate_mlp_noisy = mod_noisy.chunk(4, dim=1)
                scale_msa_clean, gate_msa_clean, scale_mlp_clean, gate_mlp_clean = mod_clean.chunk(4, dim=1)

                gate_msa_noisy, gate_mlp_noisy = gate_msa_noisy.tanh(), gate_mlp_noisy.tanh()
                gate_msa_clean, gate_mlp_clean = gate_msa_clean.tanh(), gate_mlp_clean.tanh()
                scale_msa_noisy, scale_mlp_noisy = 1.0 + scale_msa_noisy, 1.0 + scale_mlp_noisy
                scale_msa_clean, scale_mlp_clean = 1.0 + scale_msa_clean, 1.0 + scale_mlp_clean

                seq_len = x.shape[1]
                noise_mask_expanded = noise_mask.unsqueeze(-1)
                scale_msa = torch.where(noise_mask_expanded == 1,
                                        scale_msa_noisy.unsqueeze(1).expand(-1, seq_len, -1),
                                        scale_msa_clean.unsqueeze(1).expand(-1, seq_len, -1))
                gate_msa = torch.where(noise_mask_expanded == 1,
                                       gate_msa_noisy.unsqueeze(1).expand(-1, seq_len, -1),
                                       gate_msa_clean.unsqueeze(1).expand(-1, seq_len, -1))
                scale_mlp = torch.where(noise_mask_expanded == 1,
                                        scale_mlp_noisy.unsqueeze(1).expand(-1, seq_len, -1),
                                        scale_mlp_clean.unsqueeze(1).expand(-1, seq_len, -1))
                gate_mlp = torch.where(noise_mask_expanded == 1,
                                       gate_mlp_noisy.unsqueeze(1).expand(-1, seq_len, -1),
                                       gate_mlp_clean.unsqueeze(1).expand(-1, seq_len, -1))
            else:
                # Global modulation (basic text-to-image mode)
                mod = self.adaLN_modulation(adaln_input)
                scale_msa, gate_msa, scale_mlp, gate_mlp = mod.unsqueeze(1).chunk(4, dim=2)
                gate_msa = gate_msa.tanh()
                gate_mlp = gate_mlp.tanh()
                scale_msa = 1.0 + scale_msa
                scale_mlp = 1.0 + scale_mlp

            # Attention with modulation
            normed = self.attention_norm1(x) * scale_msa
            attn_out = self._apply_attention(normed, attn_mask, freqs_cis)
            x = x + gate_msa * self.attention_norm2(attn_out)

            # FFN with modulation
            x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))
        else:
            # Standard (non-modulated) path
            normed = self.attention_norm1(x)
            attn_out = self._apply_attention(normed, attn_mask, freqs_cis)
            x = x + self.attention_norm2(attn_out)

            x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))

        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, ADALN_EMBED_DIM), hidden_size, bias=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor | None = None,
        noise_mask: torch.Tensor | None = None,
        c_noisy: torch.Tensor | None = None,
        c_clean: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noise_mask is not None:
            seq_len = x.shape[1]
            scale_noisy = 1.0 + self.adaLN_modulation(c_noisy)
            scale_clean = 1.0 + self.adaLN_modulation(c_clean)
            noise_mask_expanded = noise_mask.unsqueeze(-1)
            scale = torch.where(
                noise_mask_expanded == 1,
                scale_noisy.unsqueeze(1).expand(-1, seq_len, -1),
                scale_clean.unsqueeze(1).expand(-1, seq_len, -1),
            )
        else:
            scale = (1.0 + self.adaLN_modulation(c)).unsqueeze(1)

        x = self.norm_final(x) * scale
        return self.linear(x)


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class CuteZImageTransformer(nn.Module):
    """Accelerated Z-Image Transformer with fused Triton kernels.

    Architecture: 30 main transformer layers + 2 refiner layers per stream.
    - dim=3840, 30 heads, SiLU-gated FFN (hidden=10240)
    - Complex-valued RoPE, RMS LayerNorm, QK normalization
    - AdaLN timestep conditioning on refiner and main blocks

    Can load weights from a HuggingFace Z-Image checkpoint.
    """

    def __init__(self, config: CuteZImageConfig):
        super().__init__()
        self.config = config

        # Patch embedding
        patch_dim = config.f_patch_size * config.patch_size * config.patch_size * config.in_channels
        self.x_embedder = nn.Linear(patch_dim, config.dim, bias=True)

        # Caption embedding
        self.cap_embedder = nn.Sequential(
            RMSNorm(config.cap_feat_dim, eps=config.norm_eps),
            nn.Linear(config.cap_feat_dim, config.dim, bias=True),
        )

        # Timestep embedding
        adaln_dim = min(config.dim, ADALN_EMBED_DIM)
        self.t_embedder = TimestepEmbedder(adaln_dim, mid_size=1024)

        # Refiner blocks
        self.noise_refiner = nn.ModuleList([
            CuteZImageTransformerBlock(
                1000 + i, config.dim, config.n_heads, config.n_kv_heads,
                config.norm_eps, config.qk_norm, modulation=True,
            )
            for i in range(config.n_refiner_layers)
        ])
        self.context_refiner = nn.ModuleList([
            CuteZImageTransformerBlock(
                i, config.dim, config.n_heads, config.n_kv_heads,
                config.norm_eps, config.qk_norm, modulation=False,
            )
            for i in range(config.n_refiner_layers)
        ])

        # Main transformer layers (modulated, matching diffusers default)
        self.layers = nn.ModuleList([
            CuteZImageTransformerBlock(
                i, config.dim, config.n_heads, config.n_kv_heads,
                config.norm_eps, config.qk_norm, modulation=True,
            )
            for i in range(config.n_layers)
        ])

        # Output
        out_dim = config.patch_size * config.patch_size * config.f_patch_size * config.in_channels
        self.final_layer = FinalLayer(config.dim, out_dim)

        # Pad tokens
        self.x_pad_token = nn.Parameter(torch.empty(1, config.dim))
        self.cap_pad_token = nn.Parameter(torch.empty(1, config.dim))

        # RoPE embedder
        self.rope_embedder = RopeEmbedder(
            theta=config.rope_theta,
            axes_dims=config.axes_dims,
            axes_lens=config.axes_lens,
        )

    # -------------------------------------------------------------------
    # Forward pass helpers (matching diffusers ZImageTransformer2DModel)
    # -------------------------------------------------------------------

    @staticmethod
    def create_coordinate_grid(
        size: tuple, start: tuple | None = None, device: torch.device | None = None,
    ) -> torch.Tensor:
        if start is None:
            start = tuple(0 for _ in size)
        axes = [torch.arange(x0, x0 + span, dtype=torch.int32, device=device) for x0, span in zip(start, size)]
        grids = torch.meshgrid(axes, indexing="ij")
        return torch.stack(grids, dim=-1)

    def _patchify_image(self, image: torch.Tensor, patch_size: int, f_patch_size: int):
        """Patchify a single image tensor: (C, F, H, W) -> (num_patches, patch_dim)."""
        pH, pW, pF = patch_size, patch_size, f_patch_size
        C, F, H, W = image.size()
        F_tokens, H_tokens, W_tokens = F // pF, H // pH, W // pW
        image = image.view(C, F_tokens, pF, H_tokens, pH, W_tokens, pW)
        image = image.permute(1, 3, 5, 2, 4, 6, 0).reshape(F_tokens * H_tokens * W_tokens, pF * pH * pW * C)
        return image, (F, H, W), (F_tokens, H_tokens, W_tokens)

    def _pad_with_ids(
        self,
        feat: torch.Tensor,
        pos_grid_size: tuple,
        pos_start: tuple,
        device: torch.device,
    ):
        """Pad feature to SEQ_MULTI_OF, create position IDs and pad mask."""
        ori_len = len(feat)
        pad_len = (-ori_len) % SEQ_MULTI_OF

        # Pos IDs from coordinate grid
        ori_pos_ids = self.create_coordinate_grid(
            size=pos_grid_size, start=pos_start, device=device,
        ).flatten(0, -2)  # flatten spatial dims, keep last dim (3)

        if pad_len > 0:
            pad_pos_ids = (
                self.create_coordinate_grid(size=(1, 1, 1), start=(0, 0, 0), device=device)
                .flatten(0, -2)
                .repeat(pad_len, 1)
            )
            pos_ids = torch.cat([ori_pos_ids, pad_pos_ids], dim=0)
            padded_feat = torch.cat([feat, feat[-1:].repeat(pad_len, 1)], dim=0)
            pad_mask = torch.cat([
                torch.zeros(ori_len, dtype=torch.bool, device=device),
                torch.ones(pad_len, dtype=torch.bool, device=device),
            ])
        else:
            pos_ids = ori_pos_ids
            padded_feat = feat
            pad_mask = torch.zeros(ori_len, dtype=torch.bool, device=device)

        total_len = ori_len + pad_len
        return padded_feat, pos_ids, pad_mask, total_len

    def patchify_and_embed(
        self,
        all_image: list[torch.Tensor],
        all_cap_feats: list[torch.Tensor],
        patch_size: int,
        f_patch_size: int,
    ):
        """Patchify for basic mode: single image per batch item."""
        device = all_image[0].device
        all_img_out, all_img_size, all_img_pos_ids, all_img_pad_mask = [], [], [], []
        all_cap_out, all_cap_pos_ids, all_cap_pad_mask = [], [], []

        for image, cap_feat in zip(all_image, all_cap_feats):
            # Caption
            cap_padded_len = len(cap_feat) + (-len(cap_feat)) % SEQ_MULTI_OF
            cap_out, cap_pos_ids, cap_pad_mask, cap_len = self._pad_with_ids(
                cap_feat,
                (cap_padded_len, 1, 1),
                (1, 0, 0),
                device,
            )
            all_cap_out.append(cap_out)
            all_cap_pos_ids.append(cap_pos_ids)
            all_cap_pad_mask.append(cap_pad_mask)

            # Image
            img_patches, size, (F_t, H_t, W_t) = self._patchify_image(image, patch_size, f_patch_size)
            img_out, img_pos_ids, img_pad_mask, _ = self._pad_with_ids(
                img_patches,
                (F_t, H_t, W_t),
                (cap_len + 1, 0, 0),
                device,
            )
            all_img_out.append(img_out)
            all_img_size.append(size)
            all_img_pos_ids.append(img_pos_ids)
            all_img_pad_mask.append(img_pad_mask)

        return (
            all_img_out,
            all_cap_out,
            all_img_size,
            all_img_pos_ids,
            all_cap_pos_ids,
            all_img_pad_mask,
            all_cap_pad_mask,
        )

    def _prepare_sequence(
        self,
        feats: list[torch.Tensor],
        pos_ids: list[torch.Tensor],
        inner_pad_mask: list[torch.Tensor],
        pad_token: nn.Parameter,
        device: torch.device,
    ):
        """Prepare sequence: apply pad token, RoPE embed, pad to batch, create attention mask."""
        item_seqlens = [len(f) for f in feats]
        max_seqlen = max(item_seqlens)
        bsz = len(feats)

        # Replace padded positions with pad token
        feats_cat = torch.cat(feats, dim=0)
        feats_cat[torch.cat(inner_pad_mask)] = pad_token
        feats_list = list(feats_cat.split(item_seqlens, dim=0))

        # RoPE
        freqs_list = list(
            self.rope_embedder(torch.cat(pos_ids, dim=0))
            .split([len(p) for p in pos_ids], dim=0)
        )

        # Pad to batch
        feats_batched = pad_sequence(feats_list, batch_first=True, padding_value=0.0)
        freqs_batched = pad_sequence(freqs_list, batch_first=True, padding_value=0.0)[:, : feats_batched.shape[1]]

        # Attention mask
        attn_mask = torch.zeros((bsz, max_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(item_seqlens):
            attn_mask[i, :seq_len] = 1

        return feats_batched, freqs_batched, attn_mask, item_seqlens

    def _build_unified_sequence(
        self,
        x: torch.Tensor,
        x_freqs: torch.Tensor,
        x_seqlens: list[int],
        cap: torch.Tensor,
        cap_freqs: torch.Tensor,
        cap_seqlens: list[int],
        device: torch.device,
    ):
        """Build unified sequence [x, cap] for basic mode."""
        bsz = len(x_seqlens)
        unified = []
        unified_freqs = []

        for i in range(bsz):
            x_len, cap_len = x_seqlens[i], cap_seqlens[i]
            unified.append(torch.cat([x[i][:x_len], cap[i][:cap_len]]))
            unified_freqs.append(torch.cat([x_freqs[i][:x_len], cap_freqs[i][:cap_len]]))

        unified_seqlens = [a + b for a, b in zip(x_seqlens, cap_seqlens)]
        max_seqlen = max(unified_seqlens)

        # Pad to batch
        unified = pad_sequence(unified, batch_first=True, padding_value=0.0)
        unified_freqs = pad_sequence(unified_freqs, batch_first=True, padding_value=0.0)

        # Attention mask
        attn_mask = torch.zeros((bsz, max_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(unified_seqlens):
            attn_mask[i, :seq_len] = 1

        return unified, unified_freqs, attn_mask

    def unpatchify(
        self,
        x: list[torch.Tensor],
        size: list[tuple],
        patch_size: int,
        f_patch_size: int,
    ) -> list[torch.Tensor]:
        pH = pW = patch_size
        pF = f_patch_size
        for i in range(len(x)):
            F, H, W = size[i]
            ori_len = (F // pF) * (H // pH) * (W // pW)
            x[i] = (
                x[i][:ori_len]
                .view(F // pF, H // pH, W // pW, pF, pH, pW, self.config.in_channels)
                .permute(6, 0, 3, 1, 4, 2, 5)
                .reshape(self.config.in_channels, F, H, W)
            )
        return x

    # -------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------

    def forward(
        self,
        x: list[torch.Tensor],
        t: torch.Tensor,
        cap_feats: list[torch.Tensor],
        return_dict: bool = True,
        controlnet_block_samples: dict[int, torch.Tensor] | None = None,
        patch_size: int | None = None,
        f_patch_size: int | None = None,
    ):
        """Basic text-to-image forward pass.

        Parameters
        ----------
        x : list of (C, F, H, W) tensors
            Noisy latent images.
        t : (B,) tensor
            Timesteps (0..1 range, will be scaled by t_scale).
        cap_feats : list of (seq_len, cap_feat_dim) tensors
            Caption features from text encoder.
        """
        if patch_size is None:
            patch_size = self.config.patch_size
        if f_patch_size is None:
            f_patch_size = self.config.f_patch_size
        device = x[0].device

        # Timestep embedding
        adaln_input = self.t_embedder(t * self.config.t_scale).type_as(x[0])

        # Patchify
        (
            x_patches, cap_patches, x_size,
            x_pos_ids, cap_pos_ids,
            x_pad_mask, cap_pad_mask,
        ) = self.patchify_and_embed(x, cap_feats, patch_size, f_patch_size)

        # X embed & refine
        x_seqlens = [len(xi) for xi in x_patches]
        x_embedded = self.x_embedder(torch.cat(x_patches, dim=0))
        x_tensor, x_freqs, x_mask, x_item_seqlens = self._prepare_sequence(
            list(x_embedded.split(x_seqlens, dim=0)),
            x_pos_ids, x_pad_mask, self.x_pad_token, device,
        )

        for layer in self.noise_refiner:
            x_tensor = layer(x_tensor, x_mask, x_freqs, adaln_input=adaln_input)

        # Cap embed & refine
        cap_seqlens = [len(ci) for ci in cap_patches]
        cap_embedded = self.cap_embedder(torch.cat(cap_patches, dim=0))
        cap_tensor, cap_freqs, cap_mask, cap_item_seqlens = self._prepare_sequence(
            list(cap_embedded.split(cap_seqlens, dim=0)),
            cap_pos_ids, cap_pad_mask, self.cap_pad_token, device,
        )

        for layer in self.context_refiner:
            cap_tensor = layer(cap_tensor, cap_mask, cap_freqs)

        # Build unified sequence [x, cap]
        unified, unified_freqs, unified_mask = self._build_unified_sequence(
            x_tensor, x_freqs, x_item_seqlens,
            cap_tensor, cap_freqs, cap_item_seqlens,
            device,
        )

        # Main transformer layers
        for layer_idx, layer in enumerate(self.layers):
            unified = layer(unified, unified_mask, unified_freqs, adaln_input=adaln_input)
            if controlnet_block_samples is not None and layer_idx in controlnet_block_samples:
                unified = unified + controlnet_block_samples[layer_idx]

        # Final layer
        unified = self.final_layer(unified, c=adaln_input)

        # Unpatchify
        x_out = self.unpatchify(list(unified.unbind(dim=0)), x_size, patch_size, f_patch_size)

        if return_dict:
            return {"sample": x_out}
        return (x_out,)

    # -------------------------------------------------------------------
    # Weight loading from diffusers
    # -------------------------------------------------------------------

    @classmethod
    def from_diffusers(cls, model: "ZImageTransformer2DModel") -> "CuteZImageTransformer":
        """Create a CuteZImageTransformer from a diffusers ZImageTransformer2DModel.

        Copies all weights exactly.
        """
        cfg = model.config
        config = CuteZImageConfig(
            patch_size=cfg.all_patch_size[0],
            f_patch_size=cfg.all_f_patch_size[0],
            in_channels=cfg.in_channels,
            dim=cfg.dim,
            n_layers=cfg.n_layers,
            n_refiner_layers=cfg.n_refiner_layers,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads,
            norm_eps=cfg.norm_eps,
            qk_norm=cfg.qk_norm,
            cap_feat_dim=cfg.cap_feat_dim,
            rope_theta=cfg.rope_theta,
            t_scale=cfg.t_scale,
            axes_dims=list(cfg.axes_dims),
            axes_lens=list(cfg.axes_lens),
        )

        cute = cls(config)
        cute._copy_weights_from_diffusers(model)
        cute.eval()
        return cute

    def _copy_weights_from_diffusers(self, orig) -> None:
        """Copy weights from the original diffusers model."""
        sd = orig.state_dict()

        with torch.no_grad():
            # Embeddings
            ps = f"{self.config.patch_size}-{self.config.f_patch_size}"
            self.x_embedder.weight.copy_(sd[f"all_x_embedder.{ps}.weight"])
            self.x_embedder.bias.copy_(sd[f"all_x_embedder.{ps}.bias"])

            self.cap_embedder[0].weight.copy_(sd["cap_embedder.0.weight"])
            self.cap_embedder[1].weight.copy_(sd["cap_embedder.1.weight"])
            self.cap_embedder[1].bias.copy_(sd["cap_embedder.1.bias"])

            # Timestep embedder
            self.t_embedder.mlp[0].weight.copy_(sd["t_embedder.mlp.0.weight"])
            self.t_embedder.mlp[0].bias.copy_(sd["t_embedder.mlp.0.bias"])
            self.t_embedder.mlp[2].weight.copy_(sd["t_embedder.mlp.2.weight"])
            self.t_embedder.mlp[2].bias.copy_(sd["t_embedder.mlp.2.bias"])

            # Pad tokens
            self.x_pad_token.copy_(sd["x_pad_token"])
            self.cap_pad_token.copy_(sd["cap_pad_token"])

            # Final layer
            self.final_layer.norm_final.weight.copy_(sd[f"all_final_layer.{ps}.norm_final.weight"])
            self.final_layer.norm_final.bias.copy_(sd[f"all_final_layer.{ps}.norm_final.bias"])
            self.final_layer.linear.weight.copy_(sd[f"all_final_layer.{ps}.linear.weight"])
            self.final_layer.linear.bias.copy_(sd[f"all_final_layer.{ps}.linear.bias"])
            self.final_layer.adaLN_modulation[1].weight.copy_(sd[f"all_final_layer.{ps}.adaLN_modulation.1.weight"])
            self.final_layer.adaLN_modulation[1].bias.copy_(sd[f"all_final_layer.{ps}.adaLN_modulation.1.bias"])

            # Copy transformer block weights
            self._copy_block_weights(self.noise_refiner, sd, "noise_refiner")
            self._copy_block_weights(self.context_refiner, sd, "context_refiner")
            self._copy_block_weights(self.layers, sd, "layers")

    def _copy_block_weights(self, blocks: nn.ModuleList, sd: dict, prefix: str) -> None:
        """Copy weights for a list of transformer blocks."""
        for i, block in enumerate(blocks):
            p = f"{prefix}.{i}"

            # Attention projections
            block.q_proj.weight.copy_(sd[f"{p}.attention.to_q.weight"])
            block.k_proj.weight.copy_(sd[f"{p}.attention.to_k.weight"])
            block.v_proj.weight.copy_(sd[f"{p}.attention.to_v.weight"])
            block.o_proj.weight.copy_(sd[f"{p}.attention.to_out.0.weight"])

            # QK norm
            if block.qk_norm:
                block.q_norm.weight.copy_(sd[f"{p}.attention.norm_q.weight"])
                block.k_norm.weight.copy_(sd[f"{p}.attention.norm_k.weight"])

            # FFN
            block.feed_forward.w1.weight.copy_(sd[f"{p}.feed_forward.w1.weight"])
            block.feed_forward.w2.weight.copy_(sd[f"{p}.feed_forward.w2.weight"])
            block.feed_forward.w3.weight.copy_(sd[f"{p}.feed_forward.w3.weight"])

            # Norms
            block.attention_norm1.weight.copy_(sd[f"{p}.attention_norm1.weight"])
            block.attention_norm2.weight.copy_(sd[f"{p}.attention_norm2.weight"])
            block.ffn_norm1.weight.copy_(sd[f"{p}.ffn_norm1.weight"])
            block.ffn_norm2.weight.copy_(sd[f"{p}.ffn_norm2.weight"])

            # AdaLN modulation
            if block.modulation:
                block.adaLN_modulation[0].weight.copy_(sd[f"{p}.adaLN_modulation.0.weight"])
                block.adaLN_modulation[0].bias.copy_(sd[f"{p}.adaLN_modulation.0.bias"])

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
