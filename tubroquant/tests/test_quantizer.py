from __future__ import annotations

import torch

from tubroquant.codebooks import get_codebook
from tubroquant.kv_cache import TurboQuantKVCache
from tubroquant.packing import pack_lowbit, unpack_lowbit
from tubroquant.quantizer import TurboQuantizer


def test_codebook_is_sorted_and_symmetric():
    codebook, boundaries = get_codebook(64, 3)
    assert torch.all(codebook[1:] >= codebook[:-1])
    assert torch.allclose(codebook, -torch.flip(codebook, dims=(0,)), atol=3e-2)
    assert boundaries.numel() == (1 << 3) - 1


def test_pack_lowbit_roundtrip():
    values = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]], dtype=torch.uint8)
    packed = pack_lowbit(values, bits=2)
    unpacked = unpack_lowbit(packed, bits=2, dim=values.shape[-1])
    assert torch.equal(values, unpacked)


def test_more_bits_reduce_mse():
    torch.manual_seed(0)
    x = torch.randn(32, 64)
    q2 = TurboQuantizer(dim=64, bits=2, mode="mse")
    q4 = TurboQuantizer(dim=64, bits=4, mode="mse")

    x2 = q2(x)
    x4 = q4(x)
    mse2 = torch.mean((x2 - x) ** 2).item()
    mse4 = torch.mean((x4 - x) ** 2).item()
    assert mse4 < mse2


def test_prod_mode_reports_compression_stats():
    torch.manual_seed(0)
    x = torch.randn(8, 64)
    quantizer = TurboQuantizer(dim=64, bits=4, mode="prod")
    y = quantizer(x)

    assert y.shape == x.shape
    assert quantizer.last_stats is not None
    assert quantizer.last_stats["compression_ratio"] > 1.0
    assert quantizer.last_stats["quantized_bytes"] < quantizer.last_stats["raw_bytes"]


def test_kv_cache_attention_and_compression():
    torch.manual_seed(0)
    key_quant = TurboQuantizer(dim=64, bits=4, mode="prod")
    value_quant = TurboQuantizer(dim=64, bits=4, mode="mse")
    cache = TurboQuantKVCache(key_quant, value_quant)

    cache.append(torch.randn(2, 12, 5, 64), torch.randn(2, 12, 5, 64))
    cache.append(torch.randn(2, 12, 3, 64), torch.randn(2, 12, 3, 64))

    query = torch.randn(2, 12, 1, 64)
    out = cache.attention(query)

    assert out.shape == (2, 12, 1, 64)
    assert cache.compression_ratio() > 1.0
