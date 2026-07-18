import pytest

torch = pytest.importorskip("torch")

from cuteloras.fused_apply import fused_lora_delta, stock_lora_delta


def _terms(d, r, n, device="cpu"):
    g = torch.Generator().manual_seed(1)
    return [(torch.randn(r, d, generator=g) * 0.02,
             torch.randn(d, r, generator=g) * 0.02,
             0.7 + 0.1 * i) for i in range(n)]


@pytest.mark.parametrize("n", [1, 2, 4])
def test_fused_matches_stock_fp32(n):
    d, r = 512, 16
    terms = _terms(d, r, n)
    stock = stock_lora_delta(terms, torch.float32, "cpu").float()
    fused = fused_lora_delta(terms, torch.float32, "cpu", compute_dtype=torch.float32).float()
    # fp32 fused must match the fp32 loop tightly (same math, reordered)
    assert torch.allclose(stock, fused, atol=1e-5, rtol=1e-4)


def test_bf16_within_merge_tolerance():
    d, r = 1024, 32
    terms = _terms(d, r, 4)
    stock = stock_lora_delta(terms, torch.float32, "cpu").float()
    fused = fused_lora_delta(terms, torch.float32, "cpu", compute_dtype=torch.bfloat16).float()
    rel = (stock - fused).abs().max() / (stock.abs().max() + 1e-9)
    assert rel < 2e-2


def test_scale_folding_and_zero_skip():
    d, r = 256, 8
    terms = _terms(d, r, 3)
    terms[1] = (terms[1][0], terms[1][1], 0.0)  # zero-scale term must be skipped
    stock = stock_lora_delta(terms, torch.float32, "cpu").float()
    fused = fused_lora_delta(terms, torch.float32, "cpu", compute_dtype=torch.float32).float()
    assert torch.allclose(stock, fused, atol=1e-5, rtol=1e-4)


def test_all_zero_returns_none():
    d, r = 128, 8
    terms = [(t[0], t[1], 0.0) for t in _terms(d, r, 2)]
    assert fused_lora_delta(terms, torch.float32, "cpu") is None
