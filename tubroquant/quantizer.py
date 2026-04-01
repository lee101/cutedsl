"""TurboQuant-style MSE and product quantizers."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tubroquant.codebooks import get_codebook
from tubroquant.packing import pack_lowbit, pack_signs
from tubroquant.rotation import build_rotation


@dataclass
class EncodedVectors:
    """Packed representation and metadata for compressed vectors."""

    indices: torch.Tensor
    packed_indices: torch.Tensor
    norms: torch.Tensor
    qjl_signs: torch.Tensor | None
    packed_qjl: torch.Tensor | None
    residual_norms: torch.Tensor | None
    original_shape: tuple[int, ...]
    mode: str
    bits: int

    def quantized_bytes(self) -> int:
        total = self.packed_indices.numel() * self.packed_indices.element_size()
        total += self.norms.numel() * self.norms.element_size()
        if self.packed_qjl is not None:
            total += self.packed_qjl.numel() * self.packed_qjl.element_size()
        if self.residual_norms is not None:
            total += self.residual_norms.numel() * self.residual_norms.element_size()
        return int(total)


class TurboQuantizer(nn.Module):
    """Prototype TurboQuant quantizer for vectors with the last dimension `dim`."""

    def __init__(
        self,
        dim: int,
        bits: int = 4,
        *,
        mode: str = "mse",
        rotation: str = "hadamard",
        seed: int = 0,
        norm_dtype: str = "float16",
    ):
        super().__init__()
        if bits < 0:
            raise ValueError(f"bits must be >= 0, got {bits}")
        if mode not in {"mse", "prod"}:
            raise ValueError(f"mode must be 'mse' or 'prod', got {mode}")

        self.dim = dim
        self.bits = bits
        self.mode = mode
        self.mse_bits = bits if mode == "mse" else max(bits - 1, 0)
        self.rotation = build_rotation(dim, kind=rotation, seed=seed)
        self.norm_dtype = getattr(torch, norm_dtype)
        self.last_stats: dict[str, float] | None = None

        codebook, boundaries = get_codebook(dim, self.mse_bits)
        self.register_buffer("codebook", codebook, persistent=False)
        self.register_buffer("boundaries", boundaries, persistent=False)

        if mode == "prod":
            generator = torch.Generator().manual_seed(seed + 1729)
            proj = torch.randn(dim, dim, generator=generator)
            self.register_buffer("proj", proj, persistent=False)
        else:
            self.register_buffer("proj", torch.empty(0), persistent=False)

    def _norms_to_storage(self, norms: torch.Tensor) -> torch.Tensor:
        return norms.squeeze(-1).to(dtype=self.norm_dtype)

    def rotate_query(self, query: torch.Tensor) -> torch.Tensor:
        """Rotate query vectors into the same basis used by packed MSE codes."""
        if query.shape[-1] != self.dim:
            raise ValueError(f"Expected last dim {self.dim}, got {query.shape[-1]}")
        return self.rotation(query.float()).to(dtype=query.dtype)

    def _bucketize(self, rotated: torch.Tensor) -> torch.Tensor:
        if self.mse_bits == 0:
            return torch.zeros_like(rotated, dtype=torch.uint8)
        boundaries = self.boundaries.to(device=rotated.device)
        return torch.bucketize(rotated, boundaries).to(torch.uint8)

    def _decode_mse_unit(self, indices: torch.Tensor) -> torch.Tensor:
        if self.mse_bits == 0:
            return torch.zeros(*indices.shape[:-1], self.dim, device=indices.device, dtype=self.codebook.dtype)
        codebook = self.codebook.to(device=indices.device)
        rotated = codebook[indices.long()]
        return self.rotation.inverse(rotated.float())

    def _encode_qjl(self, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual_norms = residual.norm(dim=-1, keepdim=True)
        if self.bits == 0:
            signs = torch.ones(*residual.shape[:-1], self.dim, device=residual.device, dtype=torch.int8)
            return signs, residual_norms
        sr = torch.einsum("...d,ed->...e", residual.float(), self.proj.float().to(device=residual.device))
        signs = torch.where(sr >= 0, 1, -1).to(torch.int8)
        return signs, residual_norms

    def _decode_qjl(self, qjl_signs: torch.Tensor, residual_norms: torch.Tensor) -> torch.Tensor:
        if self.mode != "prod":
            return torch.zeros_like(qjl_signs, dtype=torch.float32)
        coeff = math.sqrt(math.pi / 2.0) / self.dim
        signs = qjl_signs.float()
        proj = self.proj.float().to(device=signs.device)
        direction = torch.einsum("...e,ed->...d", signs, proj)
        return coeff * residual_norms.float() * direction

    def encode(self, x: torch.Tensor) -> EncodedVectors:
        if x.shape[-1] != self.dim:
            raise ValueError(f"Expected last dim {self.dim}, got {x.shape[-1]}")

        x_float = x.float()
        norms = x_float.norm(dim=-1, keepdim=True)
        safe_norms = norms.clamp_min(1e-12)
        unit = torch.where(norms > 0, x_float / safe_norms, torch.zeros_like(x_float))

        rotated = self.rotation(unit)
        indices = self._bucketize(rotated)

        mse_unit = self._decode_mse_unit(indices)
        qjl_signs = None
        packed_qjl = None
        residual_norms = None

        if self.mode == "prod":
            residual = unit - mse_unit
            qjl_signs, residual_norms = self._encode_qjl(residual)
            packed_qjl = pack_signs(qjl_signs)

        packed_indices = pack_lowbit(indices, self.mse_bits)
        return EncodedVectors(
            indices=indices,
            packed_indices=packed_indices,
            norms=self._norms_to_storage(norms),
            qjl_signs=qjl_signs,
            packed_qjl=packed_qjl,
            residual_norms=None if residual_norms is None else self._norms_to_storage(residual_norms),
            original_shape=tuple(x.shape),
            mode=self.mode,
            bits=self.bits,
        )

    def decode(self, encoded: EncodedVectors) -> torch.Tensor:
        norms = encoded.norms.unsqueeze(-1).float()
        reconstructed_unit = self._decode_mse_unit(encoded.indices)

        if encoded.mode == "prod":
            assert encoded.qjl_signs is not None
            assert encoded.residual_norms is not None
            qjl = self._decode_qjl(encoded.qjl_signs, encoded.residual_norms.unsqueeze(-1))
            reconstructed_unit = reconstructed_unit + qjl

        return reconstructed_unit * norms

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        raw_bytes = x.numel() * x.element_size()
        quantized_bytes = encoded.quantized_bytes()
        self.last_stats = {
            "raw_bytes": float(raw_bytes),
            "quantized_bytes": float(quantized_bytes),
            "compression_ratio": float(raw_bytes / max(quantized_bytes, 1)),
            "mse": float(F.mse_loss(decoded.float(), x.float()).item()),
            "max_abs_error": float((decoded.float() - x.float()).abs().max().item()),
        }
        return decoded.to(dtype=x.dtype)
