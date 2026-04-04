"""Conditioning image helpers for Z-Image ControlNet datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps


@dataclass(frozen=True)
class LineExtractionConfig:
    blur_radius: float = 1.2
    edge_percentile: float = 0.82
    line_strength: float = 1.0


def _to_grayscale_array(image: Image.Image, blur_radius: float) -> np.ndarray:
    gray = ImageOps.exif_transpose(image).convert("L")
    if blur_radius > 0:
        gray = gray.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return np.asarray(gray, dtype=np.float32) / 255.0


def extract_line_art(
    image: Image.Image,
    *,
    blur_radius: float = 1.2,
    edge_percentile: float = 0.82,
    line_strength: float = 1.0,
) -> Image.Image:
    """Convert an RGB image into a high-contrast RGB line-art control image."""
    gray = _to_grayscale_array(image, blur_radius=blur_radius)
    padded = np.pad(gray, ((1, 1), (1, 1)), mode="edge")

    gx = (
        padded[:-2, 2:]
        + 2.0 * padded[1:-1, 2:]
        + padded[2:, 2:]
        - padded[:-2, :-2]
        - 2.0 * padded[1:-1, :-2]
        - padded[2:, :-2]
    )
    gy = (
        padded[2:, :-2]
        + 2.0 * padded[2:, 1:-1]
        + padded[2:, 2:]
        - padded[:-2, :-2]
        - 2.0 * padded[:-2, 1:-1]
        - padded[:-2, 2:]
    )
    magnitude = np.hypot(gx, gy)
    scale = max(float(np.quantile(magnitude, 0.995)), 1e-6)
    magnitude = np.clip(magnitude / scale, 0.0, 1.0)

    threshold = float(np.quantile(magnitude, edge_percentile))
    line_mask = magnitude >= threshold
    line_values = np.where(line_mask, 1.0 - np.clip(magnitude * line_strength, 0.0, 1.0), 1.0)
    line_uint8 = np.clip(line_values * 255.0, 0.0, 255.0).astype(np.uint8)
    rgb = np.repeat(line_uint8[:, :, None], 3, axis=2)
    return Image.fromarray(rgb, mode="RGB")


def extract_canny(
    image: Image.Image,
    *,
    low_threshold: int = 100,
    high_threshold: int = 200,
    blur_ksize: int = 5,
    invert: bool = True,
) -> Image.Image:
    """Convert an RGB image into a Canny edge map as an RGB control image."""
    import cv2

    arr = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    if blur_ksize > 0:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    if invert:
        edges = 255 - edges
    rgb = np.repeat(edges[:, :, None], 3, axis=2)
    return Image.fromarray(rgb, mode="RGB")


@dataclass(frozen=True)
class CannyExtractionConfig:
    low_threshold: int = 100
    high_threshold: int = 200
    blur_ksize: int = 5
    invert: bool = True


def drop_line_patches(
    image: Image.Image,
    *,
    patch_size: int = 32,
    drop_prob: float = 0.18,
    seed: int | None = None,
    min_drops: int = 1,
    fill_value: int = 255,
) -> Image.Image:
    """Randomly remove tiles from a line-art image to improve ControlNet robustness."""
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if not 0.0 <= drop_prob <= 1.0:
        raise ValueError("drop_prob must be in [0, 1]")

    rgb = np.asarray(image.convert("RGB")).copy()
    height, width = rgb.shape[:2]
    tiles_y = max(1, (height + patch_size - 1) // patch_size)
    tiles_x = max(1, (width + patch_size - 1) // patch_size)

    rng = np.random.default_rng(seed)
    drop_mask = rng.random((tiles_y, tiles_x)) < drop_prob
    if min_drops > 0 and not drop_mask.any():
        flat_index = int(rng.integers(0, tiles_y * tiles_x))
        drop_mask.flat[flat_index] = True

    for tile_y in range(tiles_y):
        for tile_x in range(tiles_x):
            if not drop_mask[tile_y, tile_x]:
                continue
            y0 = tile_y * patch_size
            y1 = min(height, y0 + patch_size)
            x0 = tile_x * patch_size
            x1 = min(width, x0 + patch_size)
            rgb[y0:y1, x0:x1] = fill_value

    return Image.fromarray(rgb, mode="RGB")


def save_conditioning_triplet(
    target_image: Image.Image,
    *,
    target_path: Path,
    line_path: Path,
    sparse_line_path: Path,
    line_config: LineExtractionConfig,
    sparse_patch_size: int,
    sparse_drop_prob: float,
    sparse_seed: int,
    canny_path: Path | None = None,
    canny_config: CannyExtractionConfig | None = None,
) -> dict[str, str]:
    """Write target, conditioning images and return manifest paths."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    line_path.parent.mkdir(parents=True, exist_ok=True)
    sparse_line_path.parent.mkdir(parents=True, exist_ok=True)

    line_image = extract_line_art(
        target_image,
        blur_radius=line_config.blur_radius,
        edge_percentile=line_config.edge_percentile,
        line_strength=line_config.line_strength,
    )
    sparse_line = drop_line_patches(
        line_image,
        patch_size=sparse_patch_size,
        drop_prob=sparse_drop_prob,
        seed=sparse_seed,
    )

    target_image.save(target_path)
    line_image.save(line_path)
    sparse_line.save(sparse_line_path)
    result = {
        "target_image_path": str(target_path),
        "line_image_path": str(line_path),
        "sparse_line_image_path": str(sparse_line_path),
    }

    if canny_path is not None:
        canny_path.parent.mkdir(parents=True, exist_ok=True)
        cfg = canny_config or CannyExtractionConfig()
        canny_image = extract_canny(
            target_image,
            low_threshold=cfg.low_threshold,
            high_threshold=cfg.high_threshold,
            blur_ksize=cfg.blur_ksize,
            invert=cfg.invert,
        )
        canny_image.save(canny_path)
        result["canny_image_path"] = str(canny_path)

    return result

