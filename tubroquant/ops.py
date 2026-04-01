"""Reference and CUDA-dispatched score ops for TurboQuant packed keys."""

from __future__ import annotations

import torch

from tubroquant.kernels import load_tubroquant_extension
from tubroquant.packing import unpack_lowbit


def qk_scores_mse_reference(
    rotated_query: torch.Tensor,
    packed_indices: torch.Tensor,
    norms: torch.Tensor,
    codebook: torch.Tensor,
    *,
    dim: int,
    bits: int,
) -> torch.Tensor:
    """Reference packed-key score path for MSE TurboQuant keys."""
    flat_q = rotated_query.reshape(-1, rotated_query.shape[-1]).float()
    flat_packed = packed_indices.reshape(-1, packed_indices.shape[-1])
    flat_norms = norms.reshape(-1).float()
    unpacked = unpack_lowbit(flat_packed, bits=bits, dim=dim).long()
    decoded_rotated = codebook.to(device=flat_q.device, dtype=torch.float32)[unpacked]
    decoded_rotated = decoded_rotated * flat_norms.unsqueeze(-1)
    scores = flat_q @ decoded_rotated.t()
    return scores.reshape(*rotated_query.shape[:-1], *packed_indices.shape[:-1])


def qk_scores_mse(
    rotated_query: torch.Tensor,
    packed_indices: torch.Tensor,
    norms: torch.Tensor,
    codebook: torch.Tensor,
    *,
    dim: int,
    bits: int,
    build_extension: bool = True,
) -> torch.Tensor:
    """Compute `QK^T` where K is stored as packed MSE TurboQuant codes."""
    ext = load_tubroquant_extension(build_extension=build_extension)
    if (
        ext is not None
        and rotated_query.is_cuda
        and packed_indices.is_cuda
        and norms.is_cuda
        and codebook.is_cuda
    ):
        return ext.qk_scores_mse(
            rotated_query.contiguous(),
            packed_indices.contiguous(),
            norms.contiguous(),
            codebook.contiguous(),
            dim,
            bits,
        )

    return qk_scores_mse_reference(
        rotated_query,
        packed_indices,
        norms,
        codebook,
        dim=dim,
        bits=bits,
    )
