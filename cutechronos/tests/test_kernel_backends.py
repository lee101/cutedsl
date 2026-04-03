from __future__ import annotations

import pytest
import torch

import cutechronos.kernel_backends as kernel_backends
from cutechronos.kernel_backends import rms_layernorm, unscaled_attention
from cutechronos.modules._fallbacks import rms_layernorm as torch_rms_layernorm
from cutechronos.modules._fallbacks import unscaled_attention as torch_unscaled_attention


def test_rms_auto_prefers_triton_on_pre_hopper(monkeypatch):
    x = torch.randn(2, 4, 8)
    weight = torch.randn(8)

    monkeypatch.setenv("CUTECHRONOS_RMS_BACKEND", "auto")
    monkeypatch.setattr(kernel_backends, "_HAS_TRITON_RMS", True)
    monkeypatch.setattr(kernel_backends, "_HAS_CUTLASS_RMS", True)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(kernel_backends, "_cuda_major", lambda device: 8)
    monkeypatch.setattr(kernel_backends, "_triton_rms_layernorm", lambda *args, **kwargs: torch.full_like(x, 1.0))
    monkeypatch.setattr(
        kernel_backends,
        "_cutlass_rms_layernorm",
        lambda *args, **kwargs: torch.full_like(x, 2.0),
        raising=False,
    )
    monkeypatch.setattr(kernel_backends, "_torch_rms_layernorm", lambda *args, **kwargs: torch.full_like(x, 3.0))

    out = rms_layernorm(x, weight)
    assert torch.equal(out, torch.full_like(x, 1.0))


def test_rms_auto_prefers_cutlass_on_hopper(monkeypatch):
    x = torch.randn(2, 4, 8)
    weight = torch.randn(8)

    monkeypatch.setenv("CUTECHRONOS_RMS_BACKEND", "auto")
    monkeypatch.setattr(kernel_backends, "_HAS_TRITON_RMS", True)
    monkeypatch.setattr(kernel_backends, "_HAS_CUTLASS_RMS", True)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(kernel_backends, "_cuda_major", lambda device: 9)
    monkeypatch.setattr(kernel_backends, "_triton_rms_layernorm", lambda *args, **kwargs: torch.full_like(x, 1.0))
    monkeypatch.setattr(
        kernel_backends,
        "_cutlass_rms_layernorm",
        lambda *args, **kwargs: torch.full_like(x, 2.0),
        raising=False,
    )
    monkeypatch.setattr(kernel_backends, "_torch_rms_layernorm", lambda *args, **kwargs: torch.full_like(x, 3.0))

    out = rms_layernorm(x, weight)
    assert torch.equal(out, torch.full_like(x, 2.0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cutlass_rms_backend_matches_reference(monkeypatch):
    monkeypatch.setenv("CUTECHRONOS_RMS_BACKEND", "cutlass")
    x = torch.randn(4, 128, 768, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(768, device="cuda", dtype=torch.bfloat16)
    ref = torch_rms_layernorm(x, weight)
    out = rms_layernorm(x, weight)
    assert torch.allclose(out, ref, atol=5e-3, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_sdpa_attention_backend_matches_reference(monkeypatch):
    monkeypatch.setenv("CUTECHRONOS_ATTENTION_BACKEND", "sdpa")
    batch, heads, seq_len, head_dim = 2, 12, 128, 64
    q = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16)
    mask = torch.zeros(batch, heads, seq_len, seq_len, device="cuda", dtype=torch.bfloat16)
    causal = torch.triu(torch.ones(seq_len, seq_len, device="cuda", dtype=torch.bool), diagonal=1)
    mask[:, :, causal] = torch.finfo(torch.bfloat16).min
    ref = torch_unscaled_attention(q, k, v, mask)
    out = unscaled_attention(q, k, v, mask)
    assert torch.allclose(out, ref, atol=1.5e-1, rtol=5e-2)
