"""Experimental Z-Image acceleration blocks.

The first experiment here is conservative: fuse the common Q/K/V projection
path into a single linear for the `n_heads == n_kv_heads` case while keeping
all downstream math identical to `cutezimage.model.CuteZImageTransformerBlock`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from cutezimage.model import (
    ADALN_EMBED_DIM,
    CuteZImageConfig,
    CuteZImageTransformer,
    RMSNorm,
    SiLUGatedFFN,
    _apply_rope_complex_fallback,
    _get_fused_adaln_rms_norm,
    _get_fused_qk_norm,
    _get_rope_complex,
)


class AcceleratedZImageTransformerBlock(nn.Module):
    """Drop-in experimental block with fused QKV projection for equal Q/KV heads."""

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
        self.qk_norm = qk_norm

        self.use_fused_qkv = n_heads == n_kv_heads
        kv_dim = n_kv_heads * self.head_dim
        if self.use_fused_qkv:
            self.qkv_proj = nn.Linear(dim, dim + kv_dim + kv_dim, bias=False)
            self.q_proj = None
            self.k_proj = None
            self.v_proj = None
        else:
            self.qkv_proj = None
            self.q_proj = nn.Linear(dim, dim, bias=False)
            self.k_proj = nn.Linear(dim, kv_dim, bias=False)
            self.v_proj = nn.Linear(dim, kv_dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=1e-5)
            self.k_norm = RMSNorm(self.head_dim, eps=1e-5)

        hidden_dim = int(dim / 3 * 8)
        self.feed_forward = SiLUGatedFFN(dim, hidden_dim)

        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)
        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        if modulation:
            self.adaLN_modulation = nn.Sequential(
                nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True)
            )

    @classmethod
    def from_cutezimage_block(cls, block) -> "AcceleratedZImageTransformerBlock":
        """Clone weights from a CuteZImageTransformerBlock."""
        accelerated = cls(
            layer_id=block.layer_id,
            dim=block.dim,
            n_heads=block.n_heads,
            n_kv_heads=block.n_kv_heads,
            norm_eps=block.attention_norm1.eps,
            qk_norm=block.qk_norm,
            modulation=block.modulation,
        ).to(device=next(block.parameters()).device, dtype=next(block.parameters()).dtype)
        accelerated.copy_from_cutezimage_block(block)
        return accelerated

    def copy_from_cutezimage_block(self, block) -> None:
        with torch.no_grad():
            if self.use_fused_qkv:
                self.qkv_proj.weight.copy_(
                    torch.cat(
                        [
                            block.q_proj.weight,
                            block.k_proj.weight,
                            block.v_proj.weight,
                        ],
                        dim=0,
                    )
                )
            else:
                self.q_proj.weight.copy_(block.q_proj.weight)
                self.k_proj.weight.copy_(block.k_proj.weight)
                self.v_proj.weight.copy_(block.v_proj.weight)
            self.o_proj.weight.copy_(block.o_proj.weight)

            if self.qk_norm:
                self.q_norm.weight.copy_(block.q_norm.weight)
                self.k_norm.weight.copy_(block.k_norm.weight)

            self.feed_forward.load_state_dict(block.feed_forward.state_dict())
            self.attention_norm1.load_state_dict(block.attention_norm1.state_dict())
            self.ffn_norm1.load_state_dict(block.ffn_norm1.state_dict())
            self.attention_norm2.load_state_dict(block.attention_norm2.state_dict())
            self.ffn_norm2.load_state_dict(block.ffn_norm2.state_dict())

            if self.modulation:
                self.adaLN_modulation.load_state_dict(block.adaLN_modulation.state_dict())

    def _project_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.use_fused_qkv:
            qkv = self.qkv_proj(x)
            q, k, v = torch.split(
                qkv,
                [self.dim, self.n_kv_heads * self.head_dim, self.n_kv_heads * self.head_dim],
                dim=-1,
            )
        else:
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
        return q, k, v

    def _apply_attention(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
        freqs_cis: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape

        q, k, v = self._project_qkv(x)
        q = q.unflatten(-1, (self.n_heads, self.head_dim))
        k = k.unflatten(-1, (self.n_kv_heads, self.head_dim))
        v = v.unflatten(-1, (self.n_kv_heads, self.head_dim))

        if self.qk_norm:
            fused_qk_norm = _get_fused_qk_norm() if q.is_cuda and self.n_kv_heads == self.n_heads else None
            if fused_qk_norm is not None:
                q, k = fused_qk_norm(q, k, self.q_norm.weight, self.k_norm.weight, eps=1e-5)
            else:
                q = self.q_norm(q)
                k = self.k_norm(k)

        if freqs_cis is not None:
            rope_fn = _get_rope_complex() if q.is_cuda else _apply_rope_complex_fallback
            if self.n_heads == self.n_kv_heads:
                qk = torch.cat([q, k], dim=2)
                qk = rope_fn(qk, freqs_cis)
                q, k = torch.split(qk, [self.n_heads, self.n_kv_heads], dim=2)
            else:
                q = rope_fn(q, freqs_cis)
                k = rope_fn(k, freqs_cis)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.n_kv_heads < self.n_heads:
            repeat_factor = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        if attn_mask is not None and attn_mask.ndim == 2:
            attn_mask = attn_mask[:, None, None, :]

        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, seq_len, dim)
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
                scale_msa = torch.where(
                    noise_mask_expanded == 1,
                    scale_msa_noisy.unsqueeze(1).expand(-1, seq_len, -1),
                    scale_msa_clean.unsqueeze(1).expand(-1, seq_len, -1),
                )
                gate_msa = torch.where(
                    noise_mask_expanded == 1,
                    gate_msa_noisy.unsqueeze(1).expand(-1, seq_len, -1),
                    gate_msa_clean.unsqueeze(1).expand(-1, seq_len, -1),
                )
                scale_mlp = torch.where(
                    noise_mask_expanded == 1,
                    scale_mlp_noisy.unsqueeze(1).expand(-1, seq_len, -1),
                    scale_mlp_clean.unsqueeze(1).expand(-1, seq_len, -1),
                )
                gate_mlp = torch.where(
                    noise_mask_expanded == 1,
                    gate_mlp_noisy.unsqueeze(1).expand(-1, seq_len, -1),
                    gate_mlp_clean.unsqueeze(1).expand(-1, seq_len, -1),
                )

                normed = self.attention_norm1(x) * scale_msa
                attn_out = self._apply_attention(normed, attn_mask, freqs_cis)
                x = x + gate_msa * self.attention_norm2(attn_out)
                x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))
            else:
                mod = self.adaLN_modulation(adaln_input)
                scale_msa, gate_msa, scale_mlp, gate_mlp = mod.unsqueeze(1).chunk(4, dim=2)
                gate_msa = gate_msa.tanh()
                gate_mlp = gate_mlp.tanh()
                scale_msa = 1.0 + scale_msa
                scale_mlp = 1.0 + scale_mlp

                fused_adaln = _get_fused_adaln_rms_norm() if x.is_cuda else None
                if fused_adaln is not None:
                    normed = fused_adaln(x, scale_msa.squeeze(1), self.attention_norm1.weight, eps=self.attention_norm1.eps)
                else:
                    normed = self.attention_norm1(x) * scale_msa
                attn_out = self._apply_attention(normed, attn_mask, freqs_cis)
                x = x + gate_msa * self.attention_norm2(attn_out)

                if fused_adaln is not None:
                    ffn_input = fused_adaln(x, scale_mlp.squeeze(1), self.ffn_norm1.weight, eps=self.ffn_norm1.eps)
                else:
                    ffn_input = self.ffn_norm1(x) * scale_mlp
                x = x + gate_mlp * self.ffn_norm2(self.feed_forward(ffn_input))
        else:
            normed = self.attention_norm1(x)
            attn_out = self._apply_attention(normed, attn_mask, freqs_cis)
            x = x + self.attention_norm2(attn_out)
            x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))

        return x


class AcceleratedZImageTransformer(CuteZImageTransformer):
    """Full experimental transformer using accelerated blocks."""

    def __init__(self, config):
        super().__init__(config)
        self.noise_refiner = nn.ModuleList(
            [
                AcceleratedZImageTransformerBlock(
                    1000 + i,
                    config.dim,
                    config.n_heads,
                    config.n_kv_heads,
                    config.norm_eps,
                    config.qk_norm,
                    modulation=True,
                )
                for i in range(config.n_refiner_layers)
            ]
        )
        self.context_refiner = nn.ModuleList(
            [
                AcceleratedZImageTransformerBlock(
                    i,
                    config.dim,
                    config.n_heads,
                    config.n_kv_heads,
                    config.norm_eps,
                    config.qk_norm,
                    modulation=False,
                )
                for i in range(config.n_refiner_layers)
            ]
        )
        self.layers = nn.ModuleList(
            [
                AcceleratedZImageTransformerBlock(
                    i,
                    config.dim,
                    config.n_heads,
                    config.n_kv_heads,
                    config.norm_eps,
                    config.qk_norm,
                    modulation=True,
                )
                for i in range(config.n_layers)
            ]
        )

    @classmethod
    def from_cutezimage(cls, model: CuteZImageTransformer) -> "AcceleratedZImageTransformer":
        accelerated = cls(model.config)
        accelerated.load_state_dict(model.state_dict(), strict=False)

        accelerated.noise_refiner = nn.ModuleList(
            [AcceleratedZImageTransformerBlock.from_cutezimage_block(block) for block in model.noise_refiner]
        )
        accelerated.context_refiner = nn.ModuleList(
            [AcceleratedZImageTransformerBlock.from_cutezimage_block(block) for block in model.context_refiner]
        )
        accelerated.layers = nn.ModuleList(
            [AcceleratedZImageTransformerBlock.from_cutezimage_block(block) for block in model.layers]
        )
        accelerated.to(device=next(model.parameters()).device, dtype=next(model.parameters()).dtype)
        accelerated.eval()
        return accelerated

    @classmethod
    def from_diffusers(cls, model: "ZImageTransformer2DModel") -> "AcceleratedZImageTransformer":
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
        accelerated = cls(config)
        ref_param = next(model.parameters())
        accelerated.to(device=ref_param.device, dtype=ref_param.dtype)
        accelerated._copy_weights_from_diffusers(model)
        accelerated.eval()
        return accelerated

    def _copy_weights_from_diffusers(self, orig) -> None:
        """Copy weights directly from the diffusers transformer in one pass."""
        sd = orig.state_dict()

        with torch.no_grad():
            ps = f"{self.config.patch_size}-{self.config.f_patch_size}"
            self.x_embedder.weight.copy_(sd[f"all_x_embedder.{ps}.weight"])
            self.x_embedder.bias.copy_(sd[f"all_x_embedder.{ps}.bias"])

            self.cap_embedder[0].weight.copy_(sd["cap_embedder.0.weight"])
            self.cap_embedder[1].weight.copy_(sd["cap_embedder.1.weight"])
            self.cap_embedder[1].bias.copy_(sd["cap_embedder.1.bias"])

            self.t_embedder.mlp[0].weight.copy_(sd["t_embedder.mlp.0.weight"])
            self.t_embedder.mlp[0].bias.copy_(sd["t_embedder.mlp.0.bias"])
            self.t_embedder.mlp[2].weight.copy_(sd["t_embedder.mlp.2.weight"])
            self.t_embedder.mlp[2].bias.copy_(sd["t_embedder.mlp.2.bias"])

            self.x_pad_token.copy_(sd["x_pad_token"])
            self.cap_pad_token.copy_(sd["cap_pad_token"])

            self.final_layer.linear.weight.copy_(sd[f"all_final_layer.{ps}.linear.weight"])
            self.final_layer.linear.bias.copy_(sd[f"all_final_layer.{ps}.linear.bias"])
            self.final_layer.adaLN_modulation[1].weight.copy_(sd[f"all_final_layer.{ps}.adaLN_modulation.1.weight"])
            self.final_layer.adaLN_modulation[1].bias.copy_(sd[f"all_final_layer.{ps}.adaLN_modulation.1.bias"])

            self._copy_block_weights_from_diffusers(self.noise_refiner, sd, "noise_refiner")
            self._copy_block_weights_from_diffusers(self.context_refiner, sd, "context_refiner")
            self._copy_block_weights_from_diffusers(self.layers, sd, "layers")

    @staticmethod
    def _copy_block_weights_from_diffusers(blocks: nn.ModuleList, sd: dict, prefix: str) -> None:
        for i, block in enumerate(blocks):
            p = f"{prefix}.{i}"

            if block.use_fused_qkv:
                block.qkv_proj.weight.copy_(
                    torch.cat(
                        [
                            sd[f"{p}.attention.to_q.weight"],
                            sd[f"{p}.attention.to_k.weight"],
                            sd[f"{p}.attention.to_v.weight"],
                        ],
                        dim=0,
                    )
                )
            else:
                block.q_proj.weight.copy_(sd[f"{p}.attention.to_q.weight"])
                block.k_proj.weight.copy_(sd[f"{p}.attention.to_k.weight"])
                block.v_proj.weight.copy_(sd[f"{p}.attention.to_v.weight"])

            block.o_proj.weight.copy_(sd[f"{p}.attention.to_out.0.weight"])

            if block.qk_norm:
                block.q_norm.weight.copy_(sd[f"{p}.attention.norm_q.weight"])
                block.k_norm.weight.copy_(sd[f"{p}.attention.norm_k.weight"])

            block.feed_forward.w1.weight.copy_(sd[f"{p}.feed_forward.w1.weight"])
            block.feed_forward.w2.weight.copy_(sd[f"{p}.feed_forward.w2.weight"])
            block.feed_forward.w3.weight.copy_(sd[f"{p}.feed_forward.w3.weight"])

            block.attention_norm1.weight.copy_(sd[f"{p}.attention_norm1.weight"])
            block.attention_norm2.weight.copy_(sd[f"{p}.attention_norm2.weight"])
            block.ffn_norm1.weight.copy_(sd[f"{p}.ffn_norm1.weight"])
            block.ffn_norm2.weight.copy_(sd[f"{p}.ffn_norm2.weight"])

            if block.modulation:
                block.adaLN_modulation[0].weight.copy_(sd[f"{p}.adaLN_modulation.0.weight"])
                block.adaLN_modulation[0].bias.copy_(sd[f"{p}.adaLN_modulation.0.bias"])

    @staticmethod
    def _apply_compile(
        model: "AcceleratedZImageTransformer",
        compile_mode: str,
    ) -> "AcceleratedZImageTransformer":
        if hasattr(torch, "compile"):
            try:
                model.forward = torch.compile(  # type: ignore[assignment]
                    model.forward,
                    mode=compile_mode,
                    fullgraph=False,
                )
                print(f"[zimageaccelerated] torch.compile enabled (mode={compile_mode}).")
            except Exception as exc:
                print(f"[zimageaccelerated] torch.compile failed ({exc}); using eager mode.")
        return model

    @classmethod
    def from_cutezimage_compiled(
        cls,
        model: CuteZImageTransformer,
        compile_mode: str = "reduce-overhead",
    ) -> "AcceleratedZImageTransformer":
        accelerated = cls.from_cutezimage(model)
        return cls._apply_compile(accelerated, compile_mode)

    @classmethod
    def from_diffusers_compiled(
        cls,
        model: "ZImageTransformer2DModel",
        compile_mode: str = "reduce-overhead",
    ) -> "AcceleratedZImageTransformer":
        accelerated = cls.from_diffusers(model)
        return cls._apply_compile(accelerated, compile_mode)
