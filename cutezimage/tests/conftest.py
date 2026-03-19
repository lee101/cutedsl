"""Shared test fixtures for cutezimage tests."""

import torch
import pytest

from cutezimage.model import CuteZImageConfig, CuteZImageTransformerBlock, SiLUGatedFFN, RMSNorm


# Use a small config for fast tests
SMALL_CONFIG = CuteZImageConfig(
    dim=256,
    n_layers=2,
    n_refiner_layers=1,
    n_heads=4,
    n_kv_heads=4,
    hidden_dim=512,
    cap_feat_dim=128,
    axes_dims=[16, 24, 24],
    axes_lens=[256, 128, 128],
)


@pytest.fixture
def small_config():
    return SMALL_CONFIG


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"
