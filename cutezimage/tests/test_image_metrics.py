"""Tests for image comparison metrics."""

import torch
import pytest

from cutezimage.image_metrics import (
    pixel_mae,
    pixel_mse,
    max_pixel_error,
    psnr,
    ssim,
    compare_images,
)


class TestPixelMetrics:
    def test_identical_images(self):
        img = torch.randint(0, 256, (64, 64, 3), dtype=torch.uint8)
        assert pixel_mae(img, img) == 0.0
        assert pixel_mse(img, img) == 0.0
        assert max_pixel_error(img, img) == 0.0

    def test_different_images(self):
        img1 = torch.zeros(64, 64, 3, dtype=torch.uint8)
        img2 = torch.full((64, 64, 3), 128, dtype=torch.uint8)
        assert pixel_mae(img1, img2) == 128.0
        assert pixel_mse(img1, img2) == 128.0 * 128.0
        assert max_pixel_error(img1, img2) == 128.0


class TestPSNR:
    def test_identical_infinite(self):
        img = torch.randint(0, 256, (64, 64, 3), dtype=torch.uint8)
        assert psnr(img, img) == float("inf")

    def test_known_value(self):
        # All-0 vs all-1: MSE=1, PSNR = 10*log10(255^2/1) = ~48.13
        img1 = torch.zeros(64, 64, 3, dtype=torch.uint8)
        img2 = torch.ones(64, 64, 3, dtype=torch.uint8)
        result = psnr(img1, img2)
        assert abs(result - 48.13) < 0.1

    def test_higher_for_more_similar(self):
        img1 = torch.randint(0, 256, (64, 64, 3), dtype=torch.uint8)
        img2_close = (img1.float() + torch.randn_like(img1.float()) * 5).clamp(0, 255).to(torch.uint8)
        img2_far = (img1.float() + torch.randn_like(img1.float()) * 50).clamp(0, 255).to(torch.uint8)
        assert psnr(img1, img2_close) > psnr(img1, img2_far)


class TestSSIM:
    def test_identical_is_one(self):
        img = torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8)
        result = ssim(img, img)
        assert abs(result - 1.0) < 1e-4

    def test_different_is_less(self):
        img1 = torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8)
        img2 = torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8)
        result = ssim(img1, img2)
        assert result < 1.0

    def test_hwc_format(self):
        """Should handle (H, W, C) images too."""
        img = torch.randint(0, 256, (64, 64, 3), dtype=torch.uint8)
        result = ssim(img, img)
        assert abs(result - 1.0) < 1e-4


class TestCompareImages:
    def test_returns_all_keys(self):
        img = torch.randint(0, 256, (64, 64, 3), dtype=torch.uint8)
        result = compare_images(img, img)
        assert "mae" in result
        assert "mse" in result
        assert "psnr_db" in result
        assert "ssim" in result
        assert "max_pixel_error" in result
        assert "images_identical" in result
        assert result["images_identical"] is True

    def test_not_identical(self):
        img1 = torch.zeros(64, 64, 3, dtype=torch.uint8)
        img2 = torch.ones(64, 64, 3, dtype=torch.uint8)
        result = compare_images(img1, img2)
        assert result["images_identical"] is False
        assert result["mae"] > 0
