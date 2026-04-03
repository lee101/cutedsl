"""Tests for the learned latent step forecaster."""

import torch

from latentteleport.step_forecaster import LatentStepForecaster


def test_step_forecaster_output_shape():
    model = LatentStepForecaster()
    latent = torch.randn(2, 16, 64, 64)
    timestep = torch.tensor([6.0, 7.0])
    text_embedding = torch.randn(2, 2560)
    output = model(latent, timestep, text_embedding)
    assert output.shape == latent.shape
