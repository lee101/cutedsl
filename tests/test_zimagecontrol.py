from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from zimagecontrol.conditioning import drop_line_patches, extract_line_art
from zimagecontrol.dataset import ZImageControlDataset, collate_controlnet_examples
from zimagecontrol.runtime import parse_index_spec


def test_extract_line_art_returns_rgb_image_with_dark_edges():
    image = Image.new("RGB", (64, 64), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((16, 16, 48, 48), fill="black")

    line_image = extract_line_art(image, blur_radius=0.0, edge_percentile=0.6)

    assert line_image.mode == "RGB"
    assert int(np.asarray(line_image).min()) == 0


def test_drop_line_patches_whitens_part_of_image():
    image = Image.new("RGB", (64, 64), "black")

    sparse = drop_line_patches(image, patch_size=16, drop_prob=1.0, seed=0)

    assert sparse.getpixel((8, 8)) == (255, 255, 255)


def test_parse_index_spec_supports_ranges():
    assert parse_index_spec("0,2-4", 6) == [0, 2, 3, 4]
    assert parse_index_spec("all", 3) == [0, 1, 2]


def test_control_dataset_loads_and_collates(tmp_path: Path):
    target = Image.new("RGB", (32, 32), "red")
    line = Image.new("RGB", (32, 32), "white")
    sparse = Image.new("RGB", (32, 32), "gray")
    target_path = tmp_path / "target.png"
    line_path = tmp_path / "line.png"
    sparse_path = tmp_path / "sparse.png"
    target.save(target_path)
    line.save(line_path)
    sparse.save(sparse_path)

    metadata_path = tmp_path / "metadata.jsonl"
    metadata_path.write_text(
        (
            '{"prompt":"p1","target_image_path":"%s","line_image_path":"%s","sparse_line_image_path":"%s","width":32,"height":32}\n'
            '{"prompt":"p2","target_image_path":"%s","line_image_path":"%s","width":32,"height":32}\n'
        )
        % (target_path, line_path, sparse_path, target_path, line_path)
    )

    dataset = ZImageControlDataset(metadata_path, control_mode="sparse_or_full", sparse_control_prob=1.0)
    item = dataset[0]

    assert item["prompt"] == "p1"
    assert tuple(item["pixel_values"].shape) == (3, 32, 32)
    assert tuple(item["control_values"].shape) == (3, 32, 32)

    batch = collate_controlnet_examples([dataset[0], dataset[1]])
    assert batch["prompts"] == ["p1", "p2"]
    assert isinstance(batch["pixel_values"], torch.Tensor)
    assert tuple(batch["pixel_values"].shape) == (2, 3, 32, 32)
