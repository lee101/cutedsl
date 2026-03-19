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

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    axes_lens: list[int] = field(default_factory=lambda: [1024, 512, 512])
    hidden_dim: int = 0  # auto-computed

    def __post_init__(self):
        if self.hidden_dim == 0:
            self.hidden_dim = int(self.dim / 3 * 8)
        self.head_dim = self.dim // self.n_heads


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

    ADALN_EMBED_DIM = 3072

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
                nn.Linear(min(dim, self.ADALN_EMBED_DIM), 4 * dim, bias=True)
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
            rope_fn = _get_rope_complex()
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
        if self.modulation and adaln_input is not None:
            mod = self.adaLN_modulation(adaln_input)
            scale_msa, gate_msa, scale_mlp, gate_mlp = mod.chunk(4, dim=1)
            gate_msa = gate_msa.tanh()
            gate_mlp = gate_mlp.tanh()
            scale_msa = 1.0 + scale_msa
            scale_mlp = 1.0 + scale_mlp

            # Attention with modulation
            normed = self.attention_norm1(x) * scale_msa.unsqueeze(1)
            attn_out = self._apply_attention(normed, attn_mask, freqs_cis)
            x = x + gate_msa.unsqueeze(1) * self.attention_norm2(attn_out)

            # FFN with modulation
            normed_ff = self.ffn_norm1(x) * scale_mlp.unsqueeze(1)
            x = x + gate_mlp.unsqueeze(1) * self.ffn_norm2(self.feed_forward(normed_ff))
        else:
            # Standard (non-modulated) path
            normed = self.attention_norm1(x)
            attn_out = self._apply_attention(normed, attn_mask, freqs_cis)
            x = x + self.attention_norm2(attn_out)

            x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))

        return x


class FinalLayer(nn.Module):
    ADALN_EMBED_DIM = 3072

    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, self.ADALN_EMBED_DIM), hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        scale = 1.0 + self.adaLN_modulation(c)
        x = self.norm_final(x) * scale.unsqueeze(1)
        return self.linear(x)


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class CuteZImageTransformer(nn.Module):
    """Accelerated Z-Image Transformer with fused Triton kernels.

    Architecture: 30 main transformer layers + 2 refiner layers per stream.
    - dim=3840, 30 heads, SiLU-gated FFN (hidden=10240)
    - Complex-valued RoPE, RMS LayerNorm, QK normalization
    - AdaLN timestep conditioning on refiner blocks

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
        adaln_dim = min(config.dim, CuteZImageTransformerBlock.ADALN_EMBED_DIM)
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

        # Main transformer layers
        self.layers = nn.ModuleList([
            CuteZImageTransformerBlock(
                i, config.dim, config.n_heads, config.n_kv_heads,
                config.norm_eps, config.qk_norm, modulation=False,
            )
            for i in range(config.n_layers)
        ])

        # Output
        out_dim = config.patch_size * config.patch_size * config.f_patch_size * config.in_channels
        self.final_layer = FinalLayer(config.dim, out_dim)

        # Pad tokens
        self.x_pad_token = nn.Parameter(torch.empty(1, config.dim))
        self.cap_pad_token = nn.Parameter(torch.empty(1, config.dim))

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
