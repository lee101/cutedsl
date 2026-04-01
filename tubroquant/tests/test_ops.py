from __future__ import annotations

import torch

from tubroquant.ops import qk_scores_mse_reference
from tubroquant.quantizer import TurboQuantizer


def test_qk_scores_reference_matches_decode_matmul():
    torch.manual_seed(0)
    quantizer = TurboQuantizer(dim=64, bits=4, mode="mse")

    keys = torch.randn(7, 64)
    query = torch.randn(3, 64)
    encoded = quantizer.encode(keys)

    rotated_query = quantizer.rotate_query(query)
    packed_scores = qk_scores_mse_reference(
        rotated_query,
        encoded.packed_indices,
        encoded.norms,
        quantizer.codebook.float(),
        dim=64,
        bits=quantizer.mse_bits,
    )

    decoded = quantizer.decode(encoded)
    ref_scores = query.float() @ decoded.float().t()
    assert torch.allclose(packed_scores, ref_scores, atol=1e-5, rtol=1e-5)
