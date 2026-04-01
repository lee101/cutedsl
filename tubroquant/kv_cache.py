"""KV-cache simulator backed by TurboQuant-style compressed vectors."""

from __future__ import annotations

import torch

from tubroquant.ops import qk_scores_mse
from tubroquant.quantizer import EncodedVectors, TurboQuantizer


class TurboQuantKVCache:
    """Append-only compressed KV cache with approximate attention replay."""

    def __init__(self, key_quantizer: TurboQuantizer, value_quantizer: TurboQuantizer):
        self.key_quantizer = key_quantizer
        self.value_quantizer = value_quantizer
        self._keys: list[EncodedVectors] = []
        self._values: list[EncodedVectors] = []

    def append(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        self._keys.append(self.key_quantizer.encode(keys))
        self._values.append(self.value_quantizer.encode(values))

    def materialize(self) -> tuple[torch.Tensor, torch.Tensor]:
        keys = [self.key_quantizer.decode(item) for item in self._keys]
        values = [self.value_quantizer.decode(item) for item in self._values]
        if not keys:
            raise ValueError("Cache is empty")
        return torch.cat(keys, dim=-2), torch.cat(values, dim=-2)

    def _stack_encoded(self) -> tuple[EncodedVectors, EncodedVectors]:
        if not self._keys or not self._values:
            raise ValueError("Cache is empty")

        key0 = self._keys[0]
        value0 = self._values[0]
        stacked_keys = EncodedVectors(
            indices=torch.cat([item.indices for item in self._keys], dim=-2),
            packed_indices=torch.cat([item.packed_indices for item in self._keys], dim=-2),
            norms=torch.cat([item.norms for item in self._keys], dim=-1),
            qjl_signs=None if key0.qjl_signs is None else torch.cat([item.qjl_signs for item in self._keys], dim=-2),
            packed_qjl=None if key0.packed_qjl is None else torch.cat([item.packed_qjl for item in self._keys], dim=-2),
            residual_norms=(
                None if key0.residual_norms is None else torch.cat([item.residual_norms for item in self._keys], dim=-1)
            ),
            original_shape=(
                *key0.original_shape[:-2],
                sum(item.original_shape[-2] for item in self._keys),
                key0.original_shape[-1],
            ),
            mode=key0.mode,
            bits=key0.bits,
        )
        stacked_values = EncodedVectors(
            indices=torch.cat([item.indices for item in self._values], dim=-2),
            packed_indices=torch.cat([item.packed_indices for item in self._values], dim=-2),
            norms=torch.cat([item.norms for item in self._values], dim=-1),
            qjl_signs=None if value0.qjl_signs is None else torch.cat([item.qjl_signs for item in self._values], dim=-2),
            packed_qjl=(
                None if value0.packed_qjl is None else torch.cat([item.packed_qjl for item in self._values], dim=-2)
            ),
            residual_norms=(
                None
                if value0.residual_norms is None
                else torch.cat([item.residual_norms for item in self._values], dim=-1)
            ),
            original_shape=(
                *value0.original_shape[:-2],
                sum(item.original_shape[-2] for item in self._values),
                value0.original_shape[-1],
            ),
            mode=value0.mode,
            bits=value0.bits,
        )
        return stacked_keys, stacked_values

    def attention(self, query: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.key_quantizer.mode == "mse":
            encoded_keys, _ = self._stack_encoded()
            flat_query = query.reshape(-1, query.shape[-1])
            rotated_query = self.key_quantizer.rotate_query(flat_query).contiguous()
            packed = encoded_keys.packed_indices.reshape(-1, encoded_keys.packed_indices.shape[-1]).contiguous()
            norms = encoded_keys.norms.reshape(-1).contiguous()
            codebook = self.key_quantizer.codebook.to(device=query.device, dtype=torch.float32).contiguous()
            scores = qk_scores_mse(
                rotated_query,
                packed,
                norms,
                codebook,
                dim=self.key_quantizer.dim,
                bits=self.key_quantizer.mse_bits,
            )
            scores = scores.reshape(*query.shape[:-1], encoded_keys.original_shape[-2])
            values = self.value_quantizer.decode(self._stack_encoded()[1])
        else:
            keys, values = self.materialize()
            scores = torch.matmul(query.float(), keys.transpose(-2, -1).float())

        if mask is not None:
            scores = scores + mask.float()
        weights = torch.softmax(scores, dim=-1).to(dtype=values.dtype)
        return torch.matmul(weights, values)

    def raw_bytes(self, *, dtype_bytes: int = 2) -> int:
        total = 0
        for encoded in self._keys + self._values:
            elems = 1
            for size in encoded.original_shape:
                elems *= size
            total += elems * dtype_bytes
        return total

    def quantized_bytes(self) -> int:
        return sum(item.quantized_bytes() for item in self._keys + self._values)

    def compression_ratio(self, *, dtype_bytes: int = 2) -> float:
        raw = self.raw_bytes(dtype_bytes=dtype_bytes)
        quantized = self.quantized_bytes()
        return raw / max(quantized, 1)
