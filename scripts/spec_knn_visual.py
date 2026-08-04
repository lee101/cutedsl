#!/usr/bin/env python3
"""Decode a held-out long-warp comparison for the kNN residual forecaster."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import textwrap
from pathlib import Path

import torch
from diffusers import AutoencoderKL
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cutezimage.image_metrics import compare_images, pil_to_tensor  # noqa: E402
from scripts.spec_knn_ablation import (  # noqa: E402
    _apply_plan,
    _dot,
    _fit_two_input_gate,
    _motion_descriptor,
    _neighbor_plan,
    _normalize,
)


def _decode(vae: AutoencoderKL, latent: torch.Tensor) -> Image.Image:
    value = (latent / vae.config.scaling_factor) + vae.config.shift_factor
    with torch.inference_mode():
        image = vae.decode(value.float(), return_dict=False)[0]
    pixels = ((image[0] / 2 + 0.5).clamp(0, 1) * 255).byte()
    return Image.fromarray(pixels.permute(1, 2, 0).cpu().numpy(), mode="RGB")


def _font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(str(path), size=size) if path.exists() else ImageFont.load_default()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(
    trajectory_dir: Path,
    output_dir: Path,
    figure_path: Path,
    fold: int,
    folds: int,
    step: int,
    horizon: int,
    top_k: int,
    pool_k: int,
    examples: int,
    reuse_images: bool = False,
) -> dict:
    files = sorted(trajectory_dir.glob("*.pt"))
    records = [torch.load(path, map_location="cpu", weights_only=False) for path in files]
    latents = torch.stack([record["latents"] for record in records]).float()
    embeddings = _normalize(torch.stack([record["text_emb"] for record in records]))
    all_indices = torch.arange(len(records))
    test_mask = all_indices.remainder(folds) == fold
    train_indices = all_indices[~test_mask]
    test_indices = all_indices[test_mask]

    delta = latents[:, step] - latents[:, step - 1]
    train_delta = delta[train_indices]
    test_delta = delta[test_indices]
    train_motion = _motion_descriptor(train_delta)
    test_motion = _motion_descriptor(test_delta)
    train_plan = _neighbor_plan(
        embeddings[train_indices],
        train_motion,
        embeddings[train_indices],
        train_motion,
        top_k=top_k,
        pool_k=pool_k,
        temperature=0.1,
        motion_weight=0.5,
        exclude_diagonal=True,
    )
    test_plan = _neighbor_plan(
        embeddings[test_indices],
        test_motion,
        embeddings[train_indices],
        train_motion,
        top_k=top_k,
        pool_k=pool_k,
        temperature=0.1,
        motion_weight=0.5,
    )
    train_target = (
        latents[train_indices, step + horizon] - latents[train_indices, step]
    ).flatten(1)
    test_target = (
        latents[test_indices, step + horizon] - latents[test_indices, step]
    ).flatten(1)
    train_local = horizon * train_delta.flatten(1)
    test_local = horizon * test_delta.flatten(1)
    alpha = _dot(train_local, train_target) / _dot(
        train_local, train_local
    ).clamp_min(1e-12)
    train_residual = train_target - float(alpha) * train_local
    train_neighbor_residual = _apply_plan(train_residual, *train_plan)
    test_neighbor_residual = _apply_plan(train_residual, *test_plan)
    local_weight, residual_weight = _fit_two_input_gate(
        float(alpha) * train_local,
        train_neighbor_residual,
        train_target,
    )

    anchors = latents[test_indices, step]
    target_latents = anchors + test_target.reshape_as(anchors)
    local_latents = anchors + (float(alpha) * test_local).reshape_as(anchors)
    retrieval_latents = anchors + (
        local_weight * float(alpha) * test_local
        + residual_weight * test_neighbor_residual
    ).reshape_as(anchors)

    output_dir.mkdir(parents=True, exist_ok=True)
    expected_paths = [
        output_dir / f"fold{fold}_example{index}_{method}.png"
        for index in range(examples)
        for method in ("teacher", "local", "retrieval")
    ]
    can_reuse = reuse_images and all(path.exists() for path in expected_paths)
    vae = None
    if not can_reuse:
        vae = AutoencoderKL.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            subfolder="vae",
            local_files_only=True,
        ).eval()
    rows = []
    panels = []
    for display_index, data_index in enumerate(test_indices[:examples].tolist()):
        paths = {
            method: output_dir / f"fold{fold}_example{display_index}_{method}.png"
            for method in ("teacher", "local", "retrieval")
        }
        if can_reuse:
            images = {method: Image.open(path).convert("RGB") for method, path in paths.items()}
        else:
            assert vae is not None
            images = {
                "teacher": _decode(vae, target_latents[display_index : display_index + 1]),
                "local": _decode(vae, local_latents[display_index : display_index + 1]),
                "retrieval": _decode(vae, retrieval_latents[display_index : display_index + 1]),
            }
        teacher_tensor = pil_to_tensor(images["teacher"])
        metrics = {}
        path_strings = {}
        for method, image in images.items():
            path = paths[method]
            if not can_reuse:
                image.save(path)
            path_strings[method] = str(path)
            if method != "teacher":
                values = compare_images(teacher_tensor, pil_to_tensor(image))
                metrics[method] = {
                    "psnr_db": round(float(values["psnr_db"]), 4),
                    "ssim": round(float(values["ssim"]), 6),
                }
        rows.append(
            {
                "trajectory_index": data_index,
                "prompt": records[data_index]["prompt"],
                "metrics": metrics,
                "paths": path_strings,
                "sha256": {
                    method: _sha256(Path(path)) for method, path in path_strings.items()
                },
            }
        )
        panels.append(images)

    panel_size, header, row_label = 256, 62, 260
    canvas = Image.new(
        "RGB",
        (row_label + 3 * panel_size, header + examples * panel_size),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    labels = ("Teacher endpoint", "Calibrated momentum", f"Pruned {top_k}-NN residual")
    for column, label in enumerate(labels):
        draw.text(
            (row_label + column * panel_size + panel_size // 2, header // 2),
            label,
            font=_font(17, bold=True),
            fill="black",
            anchor="mm",
        )
    for row_index, (row, images) in enumerate(zip(rows, panels, strict=True)):
        y = header + row_index * panel_size
        label = "\n".join(textwrap.wrap(row["prompt"], width=30)[:5])
        draw.multiline_text(
            (10, y + panel_size // 2),
            label,
            font=_font(13),
            fill="black",
            anchor="lm",
            spacing=3,
        )
        for column, method in enumerate(("teacher", "local", "retrieval")):
            image = images[method].resize((panel_size, panel_size), Image.Resampling.LANCZOS)
            canvas.paste(image, (row_label + column * panel_size, y))
            if method != "teacher":
                metric = row["metrics"][method]
                text = f"{metric['psnr_db']:.2f} dB | {metric['ssim']:.3f}"
                box_y = y + panel_size - 25
                draw.rectangle(
                    (row_label + column * panel_size, box_y, row_label + (column + 1) * panel_size, y + panel_size),
                    fill=(0, 0, 0),
                )
                draw.text(
                    (row_label + column * panel_size + panel_size // 2, box_y + 12),
                    text,
                    font=_font(13, bold=True),
                    fill="white",
                    anchor="mm",
                )
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(figure_path)

    summary = {
        method: {
            metric: round(
                statistics.mean(row["metrics"][method][metric] for row in rows), 5
            )
            for metric in ("psnr_db", "ssim")
        }
        for method in ("local", "retrieval")
    }
    result = {
        "protocol": {
            "fold": fold,
            "folds": folds,
            "step": step,
            "horizon": horizon,
            "target_step": step + horizon,
            "top_k": top_k,
            "candidate_pool": pool_k,
            "examples": examples,
            "scope": "held-out endpoint-latent VAE decode; no branched denoiser correction",
        },
        "coefficients": {
            "momentum_scale": round(float(alpha), 6),
            "local_weight": round(local_weight, 6),
            "residual_weight": round(residual_weight, 6),
        },
        "summary": summary,
        "rows": rows,
        "figure": str(figure_path),
        "figure_sha256": _sha256(figure_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode held-out kNN endpoint forecasts")
    parser.add_argument(
        "--trajectory-dir",
        type=Path,
        default=Path("/sdb-disk/latentteleport-spec/trajs-16step-512"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/speculative/knn-visual"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("paper/figures/knn_visual_grid.png"),
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--step", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--pool-k", type=int, default=16)
    parser.add_argument("--examples", type=int, default=6)
    parser.add_argument("--reuse-images", action="store_true")
    args = parser.parse_args()
    result = run(
        args.trajectory_dir,
        args.output_dir,
        args.figure,
        args.fold,
        args.folds,
        args.step,
        args.horizon,
        args.top_k,
        args.pool_k,
        args.examples,
        args.reuse_images,
    )
    print(json.dumps(result["summary"], indent=2))
    print(f"wrote {args.figure} and {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
