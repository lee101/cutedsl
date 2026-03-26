"""Bit-packing helpers for TurboQuant-style compressed payloads."""

from __future__ import annotations

import torch


def pack_lowbit(values: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack low-bit integers along the last dimension into bytes."""
    if bits < 0 or bits > 8:
        raise ValueError(f"bits must be in [0, 8], got {bits}")
    if bits == 0:
        return torch.empty(*values.shape[:-1], 0, dtype=torch.uint8, device=values.device)

    values = values.to(torch.int32)
    dim = values.shape[-1]
    packed_len = (dim * bits + 7) // 8
    out = torch.zeros(*values.shape[:-1], packed_len, dtype=torch.uint8, device=values.device)
    mask = (1 << bits) - 1

    for idx in range(dim):
        shifted = (values[..., idx] & mask).to(torch.int32)
        bit_pos = idx * bits
        byte_idx = bit_pos // 8
        bit_offset = bit_pos % 8

        out[..., byte_idx] = out[..., byte_idx] | ((shifted << bit_offset) & 0xFF).to(torch.uint8)
        spill = bit_offset + bits - 8
        if spill > 0:
            out[..., byte_idx + 1] = out[..., byte_idx + 1] | (shifted >> (8 - bit_offset)).to(torch.uint8)

    return out


def unpack_lowbit(packed: torch.Tensor, bits: int, dim: int) -> torch.Tensor:
    """Unpack low-bit integers previously packed with `pack_lowbit`."""
    if bits < 0 or bits > 8:
        raise ValueError(f"bits must be in [0, 8], got {bits}")
    if bits == 0:
        return torch.zeros(*packed.shape[:-1], dim, dtype=torch.uint8, device=packed.device)

    packed_i32 = packed.to(torch.int32)
    out = torch.zeros(*packed.shape[:-1], dim, dtype=torch.uint8, device=packed.device)
    mask = (1 << bits) - 1

    for idx in range(dim):
        bit_pos = idx * bits
        byte_idx = bit_pos // 8
        bit_offset = bit_pos % 8

        value = packed_i32[..., byte_idx] >> bit_offset
        spill = bit_offset + bits - 8
        if spill > 0:
            value = value | (packed_i32[..., byte_idx + 1] << (8 - bit_offset))
        out[..., idx] = (value & mask).to(torch.uint8)

    return out


def pack_signs(signs: torch.Tensor) -> torch.Tensor:
    """Pack {-1, +1} signs into bits along the last dimension."""
    bits = (signs > 0).to(torch.uint8)
    return pack_lowbit(bits, bits=1)
