"""Manifest-backed dataset helpers for Z-Image ControlNet training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from zimagecontrol.conditioning import drop_line_patches


@dataclass(frozen=True)
class ControlNetRecord:
    prompt: str
    target_image_path: str
    line_image_path: str
    sparse_line_image_path: str | None = None
    canny_image_path: str | None = None
    content_prompt: str | None = None
    style_prompt: str | None = None
    width: int | None = None
    height: int | None = None
    seed: int | None = None
    metadata: dict[str, Any] | None = None


def load_records(metadata_path: str | Path) -> list[ControlNetRecord]:
    records: list[ControlNetRecord] = []
    for line in Path(metadata_path).read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        known = {
            "prompt",
            "target_image_path",
            "line_image_path",
            "sparse_line_image_path",
            "canny_image_path",
            "content_prompt",
            "style_prompt",
            "width",
            "height",
            "seed",
        }
        extra = {key: value for key, value in payload.items() if key not in known}
        records.append(
            ControlNetRecord(
                prompt=payload["prompt"],
                target_image_path=payload["target_image_path"],
                line_image_path=payload["line_image_path"],
                sparse_line_image_path=payload.get("sparse_line_image_path"),
                canny_image_path=payload.get("canny_image_path"),
                content_prompt=payload.get("content_prompt"),
                style_prompt=payload.get("style_prompt"),
                width=payload.get("width"),
                height=payload.get("height"),
                seed=payload.get("seed"),
                metadata=extra,
            )
        )
    return records


def pil_to_tensor(image: Image.Image, *, normalize: bool = True) -> torch.Tensor:
    rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
    if normalize:
        tensor = tensor * 2.0 - 1.0
    return tensor


def resize_image(image: Image.Image, width: int, height: int, *, nearest: bool = False) -> Image.Image:
    resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BICUBIC
    return ImageOps.exif_transpose(image).resize((width, height), resample=resample)


class ZImageControlDataset(Dataset):
    """Loads target/control image pairs from a JSONL manifest."""

    def __init__(
        self,
        metadata_path: str | Path,
        *,
        conditioning_type: str = "line",
        control_mode: str = "sparse_or_full",
        sparse_control_prob: float = 0.7,
        regenerate_sparse: bool = False,
        regenerate_patch_size: int = 32,
        regenerate_drop_prob: float = 0.18,
    ):
        self.records = load_records(metadata_path)
        self.conditioning_type = conditioning_type
        self.control_mode = control_mode
        self.sparse_control_prob = sparse_control_prob
        self.regenerate_sparse = regenerate_sparse
        self.regenerate_patch_size = regenerate_patch_size
        self.regenerate_drop_prob = regenerate_drop_prob

    def __len__(self) -> int:
        return len(self.records)

    def _choose_control_image(self, record: ControlNetRecord) -> Image.Image:
        if self.conditioning_type == "canny":
            if record.canny_image_path and Path(record.canny_image_path).exists():
                return Image.open(record.canny_image_path).convert("RGB")
            from zimagecontrol.conditioning import extract_canny
            return extract_canny(Image.open(record.target_image_path).convert("RGB"))

        use_sparse = self.control_mode == "sparse" or (
            self.control_mode == "sparse_or_full"
            and record.sparse_line_image_path is not None
            and torch.rand(1).item() < self.sparse_control_prob
        )
        path = record.sparse_line_image_path if use_sparse else record.line_image_path
        if path is not None and Path(path).exists():
            return Image.open(path).convert("RGB")

        base = Image.open(record.line_image_path).convert("RGB")
        if use_sparse and self.regenerate_sparse:
            seed = record.seed if record.seed is not None else 0
            return drop_line_patches(
                base,
                patch_size=self.regenerate_patch_size,
                drop_prob=self.regenerate_drop_prob,
                seed=seed,
            )
        return base

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        target = Image.open(record.target_image_path).convert("RGB")
        control = self._choose_control_image(record)

        if record.width is not None and record.height is not None:
            target = resize_image(target, record.width, record.height)
            control = resize_image(control, record.width, record.height, nearest=True)

        return {
            "prompt": record.prompt,
            "pixel_values": pil_to_tensor(target, normalize=True),
            "control_values": pil_to_tensor(control, normalize=True),
            "record": record,
        }


def collate_controlnet_examples(examples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "prompts": [item["prompt"] for item in examples],
        "pixel_values": torch.stack([item["pixel_values"] for item in examples]),
        "control_values": torch.stack([item["control_values"] for item in examples]),
        "records": [item["record"] for item in examples],
    }

