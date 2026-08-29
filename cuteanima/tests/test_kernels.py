import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels need CUDA")

from cuteanima.triton_kernels import (  # noqa: E402
    adaln_modulate,
    gated_residual,
    reference_adaln_modulate,
    reference_gated_residual,
)


@pytest.mark.parametrize("batch,seq_len,dim", [(1, 128, 2048), (2, 3952, 2048), (2, 77, 1024)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_adaln_modulate_matches_reference(batch, seq_len, dim, dtype):
    torch.manual_seed(0)
    hidden = torch.randn(batch, seq_len, dim, device="cuda", dtype=dtype)
    shift = torch.randn(batch, dim, device="cuda", dtype=dtype)
    scale = torch.randn(batch, dim, device="cuda", dtype=dtype) * 0.1

    fused = adaln_modulate(hidden, shift, scale).float()
    exact = reference_adaln_modulate(hidden.float(), shift.float(), scale.float())
    eager = reference_adaln_modulate(hidden, shift, scale).float()

    if dtype is torch.float32:
        assert torch.allclose(fused, exact, atol=1e-5, rtol=1e-5)
        return
    # bfloat16 keeps 8 mantissa bits, so a value of magnitude 4 already has a
    # 0.03 ulp. The kernel rounds once instead of three times, so it must land
    # within one ulp of exact and no further from exact than the eager chain.
    ulp = 2**-8 * exact.abs().clamp(min=1.0)
    assert (fused - exact).abs().max() <= (2 * ulp).max()
    assert (fused - exact).abs().mean() <= (eager - exact).abs().mean()


@pytest.mark.parametrize("batch,seq_len,dim", [(1, 512, 2048), (2, 3952, 2048)])
def test_gated_residual_matches_reference(batch, seq_len, dim):
    torch.manual_seed(0)
    hidden = torch.randn(batch, seq_len, dim, device="cuda", dtype=torch.bfloat16)
    delta = torch.randn(batch, seq_len, dim, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(batch, dim, device="cuda", dtype=torch.bfloat16)

    fused = gated_residual(hidden, delta, gate)
    expected = reference_gated_residual(hidden, delta, gate)
    assert torch.allclose(fused.float(), expected.float(), atol=2e-2, rtol=2e-2)


def test_adaln_modulate_rejects_wide_dim():
    hidden = torch.randn(1, 4, 70000, device="cuda", dtype=torch.bfloat16)
    shift = torch.zeros(1, 70000, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        adaln_modulate(hidden, shift, shift)
