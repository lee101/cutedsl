#!/usr/bin/env python3
"""Build the Latent Teleportation paper's image ablation grids.

The script only resizes and labels existing experiment PNGs.  It does not use
an image model or modify panel contents.  A SHA-256 manifest records every
source panel so the committed contact sheets remain auditable.

Run from the repository root::

    python scripts/paper_ablation_grids.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import textwrap
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cutezimage.image_metrics import compare_images, pil_to_tensor


PROMPTS = [
    "a lighthouse on a cliff at golden hour, crashing waves",
    "portrait of an old fisherman with a pipe, dramatic lighting",
    "a cozy cabin in a snowy forest at night, warm windows",
    "cyberpunk street market in the rain, neon signs",
    "a fox curled up on autumn leaves, soft light",
    "isometric tiny island with a waterfall and windmill",
]

STYLE_LABELS = [
    "Golden-hour landscape",
    "Dramatic portrait",
    "Snowy night scene",
    "Cyberpunk / neon",
    "Soft-light wildlife",
    "Isometric illustration",
]

METHOD_LABELS = {
    "base": "Baseline\n16 real steps",
    "spec": "Learned walker\n+ interpolator",
    "taylor": "Taylor momentum",
    "skip": "Skip control",
}


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for base in (
        Path("/usr/share/fonts/truetype/dejavu"),
        Path("/usr/share/fonts/dejavu"),
    ):
        path = base / name
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_summaries(raw_root: Path) -> dict[int, dict]:
    summaries = {}
    for draft_k in (1, 2, 3):
        path = raw_root / f"e2e-16step-k{draft_k}" / "summary.json"
        summary = json.loads(path.read_text())
        if summary.get("draft_k") != draft_k or len(summary.get("rows", [])) != len(PROMPTS):
            raise ValueError(f"unexpected summary contents: {path}")
        summaries[draft_k] = summary
    return summaries


def _source_path(raw_root: Path, prompt_index: int, method: str, draft_k: int) -> Path:
    return raw_root / f"e2e-16step-k{draft_k}" / f"{prompt_index}_{method}.png"


def _measure_all(raw_root: Path) -> dict[str, dict[str, float | str | int]]:
    metrics: dict[str, dict[str, float | str | int]] = {}
    with torch.inference_mode():
        for draft_k in (1, 2, 3):
            for prompt_index in range(len(PROMPTS)):
                base_path = _source_path(raw_root, prompt_index, "base", draft_k)
                base = pil_to_tensor(Image.open(base_path).convert("RGB"))
                for method in ("base", "spec", "taylor", "skip"):
                    source = _source_path(raw_root, prompt_index, method, draft_k)
                    candidate = pil_to_tensor(Image.open(source).convert("RGB"))
                    values = compare_images(base, candidate)
                    key = f"k{draft_k}/p{prompt_index}/{method}"
                    metrics[key] = {
                        "draft_k": draft_k,
                        "prompt_index": prompt_index,
                        "method": method,
                        "source": str(source),
                        "sha256": _sha256(source),
                        "psnr_db": round(float(values["psnr_db"]), 3),
                        "ssim": round(float(values["ssim"]), 5),
                    }
    return metrics


def _multiline_center(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
    width: int,
) -> None:
    lines = []
    for paragraph in text.splitlines():
        lines.extend(textwrap.wrap(paragraph, width=max(8, width // max(8, getattr(font, "size", 12) // 2))))
    rendered = "\n".join(lines)
    draw.multiline_text(xy, rendered, font=font, fill=fill, anchor="mm", align="center", spacing=4)


def _panel(
    source: Path,
    size: int,
    metric: dict[str, float | str | int],
    reference: bool,
) -> Image.Image:
    panel = Image.open(source).convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    panel = panel.convert("RGBA")
    overlay = Image.new("RGBA", panel.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    footer_h = max(25, size // 9)
    draw.rectangle((0, size - footer_h, size, size), fill=(0, 0, 0, 176))
    if reference:
        label = "reference"
    else:
        label = f"{float(metric['psnr_db']):.1f} dB  ·  SSIM {float(metric['ssim']):.3f}"
    draw.text(
        (size // 2, size - footer_h // 2),
        label,
        font=_font(max(13, size // 17), bold=True),
        fill="white",
        anchor="mm",
    )
    return Image.alpha_composite(panel, overlay).convert("RGB")


def _make_grid(
    *,
    raw_root: Path,
    output: Path,
    metrics: dict[str, dict[str, float | str | int]],
    columns: list[tuple[str, int, str]],
    cell: int,
    title: str,
) -> None:
    row_label_w = 250
    header_h = 104
    gap = 7
    width = row_label_w + len(columns) * (cell + gap) + gap
    height = header_h + len(PROMPTS) * (cell + gap) + gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    draw.text((16, 18), title, font=_font(28, bold=True), fill="#111111")
    for col_index, (method, draft_k, label) in enumerate(columns):
        x0 = row_label_w + gap + col_index * (cell + gap)
        _multiline_center(
            draw,
            (x0 + cell // 2, 70),
            label,
            _font(19, bold=True),
            "#111111",
            cell - 10,
        )

    for prompt_index, (prompt, style) in enumerate(zip(PROMPTS, STYLE_LABELS)):
        y0 = header_h + gap + prompt_index * (cell + gap)
        draw.rectangle((0, y0, row_label_w - 1, y0 + cell), fill="#f2f3f5")
        _multiline_center(
            draw,
            (row_label_w // 2, y0 + cell // 2 - 18),
            style,
            _font(18, bold=True),
            "#111111",
            row_label_w - 24,
        )
        _multiline_center(
            draw,
            (row_label_w // 2, y0 + cell // 2 + 43),
            prompt,
            _font(13),
            "#41454a",
            row_label_w - 28,
        )

        for col_index, (method, draft_k, _) in enumerate(columns):
            source = _source_path(raw_root, prompt_index, method, draft_k)
            key = f"k{draft_k}/p{prompt_index}/{method}"
            rendered = _panel(source, cell, metrics[key], reference=method == "base")
            x0 = row_label_w + gap + col_index * (cell + gap)
            canvas.paste(rendered, (x0, y0))
            draw.rectangle((x0, y0, x0 + cell, y0 + cell), outline="#202124", width=2)

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True, dpi=(220, 220))


def _aggregate(metrics: dict[str, dict[str, float | str | int]], draft_k: int, method: str) -> dict[str, float]:
    rows = [metrics[f"k{draft_k}/p{i}/{method}"] for i in range(len(PROMPTS))]
    return {
        "psnr_db": sum(float(row["psnr_db"]) for row in rows) / len(rows),
        "ssim": sum(float(row["ssim"]) for row in rows) / len(rows),
    }


def _write_tables(output_dir: Path, metrics: dict[str, dict[str, float | str | int]]) -> None:
    hyper_lines = [
        r"\begin{tabular}{@{}crrrrrrr@{}}",
        r"\toprule",
        r"$k$ & real calls & \multicolumn{2}{c}{learned} & \multicolumn{2}{c}{Taylor} & \multicolumn{2}{c}{skip} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(l){7-8}",
        r" & & PSNR & SSIM & PSNR & SSIM & PSNR & SSIM \\",
        r"\midrule",
    ]
    real_calls = {1: 9, 2: 6, 3: 5}
    for draft_k in (1, 2, 3):
        values = [_aggregate(metrics, draft_k, method) for method in ("spec", "taylor", "skip")]
        cells = " & ".join(f"{value['psnr_db']:.2f} & {value['ssim']:.3f}" for value in values)
        hyper_lines.append(f"{draft_k} & {real_calls[draft_k]} & {cells} \\\\")
    hyper_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output_dir / "table_image_hyperparams.tex").write_text("\n".join(hyper_lines) + "\n")

    style_lines = [
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        r"style slice & \multicolumn{2}{c}{learned} & \multicolumn{2}{c}{Taylor} & \multicolumn{2}{c}{skip} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}",
        r" & PSNR & SSIM & PSNR & SSIM & PSNR & SSIM \\",
        r"\midrule",
    ]
    for prompt_index, style in enumerate(STYLE_LABELS):
        cells = []
        for method in ("spec", "taylor", "skip"):
            row = metrics[f"k3/p{prompt_index}/{method}"]
            cells.extend([f"{float(row['psnr_db']):.2f}", f"{float(row['ssim']):.3f}"])
        safe_style = style.replace("/", r"/")
        style_lines.append(f"{safe_style} & {' & '.join(cells)} \\\\")
    style_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output_dir / "table_image_styles.tex").write_text("\n".join(style_lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper image ablation grids")
    parser.add_argument("--raw-root", type=Path, default=Path("/sdb-disk/latentteleport-spec"))
    parser.add_argument("--figure-dir", type=Path, default=Path("paper/figures"))
    parser.add_argument("--generated-dir", type=Path, default=Path("paper/generated"))
    parser.add_argument("--result-dir", type=Path, default=Path("results/speculative"))
    args = parser.parse_args()

    summaries = _load_summaries(args.raw_root)
    metrics = _measure_all(args.raw_root)
    for draft_k, summary in summaries.items():
        for prompt_index, row in enumerate(summary["rows"]):
            for method in ("spec", "taylor", "skip"):
                measured = float(metrics[f"k{draft_k}/p{prompt_index}/{method}"]["psnr_db"])
                recorded = float(row[f"psnr_{method}"])
                if abs(measured - recorded) > 0.02:
                    raise ValueError(
                        f"panel/summary mismatch at k={draft_k}, prompt={prompt_index}, "
                        f"method={method}: {measured:.3f} vs {recorded:.3f} dB"
                    )

    method_columns = [
        ("base", 3, METHOD_LABELS["base"]),
        ("spec", 3, METHOD_LABELS["spec"] + "\nk=3 / 5 calls"),
        ("taylor", 3, METHOD_LABELS["taylor"] + "\nk=3 / 5 calls"),
        ("skip", 3, METHOD_LABELS["skip"] + "\nk=3 / 5 calls"),
    ]
    _make_grid(
        raw_root=args.raw_root,
        output=args.figure_dir / "method_style_grid.png",
        metrics=metrics,
        columns=method_columns,
        cell=270,
        title="Method × prompt/style ablation (seed 7)",
    )

    draft_columns = [("base", 3, "Baseline\n16 calls")]
    draft_columns.extend(("spec", k, f"Learned\nk={k}") for k in (1, 2, 3))
    draft_columns.extend(("taylor", k, f"Taylor\nk={k}") for k in (1, 2, 3))
    _make_grid(
        raw_root=args.raw_root,
        output=args.figure_dir / "draft_length_grid.png",
        metrics=metrics,
        columns=draft_columns,
        cell=214,
        title="Draft-length hyperparameter ablation (seed 7)",
    )

    skip_columns = [("base", 3, "Baseline\n16 calls")]
    skip_columns.extend(("skip", k, f"Skip control\nk={k}") for k in (1, 2, 3))
    _make_grid(
        raw_root=args.raw_root,
        output=args.figure_dir / "skip_length_grid.png",
        metrics=metrics,
        columns=skip_columns,
        cell=270,
        title="Skip-control draft-length ablation (seed 7)",
    )

    args.generated_dir.mkdir(parents=True, exist_ok=True)
    _write_tables(args.generated_dir, metrics)

    args.result_dir.mkdir(parents=True, exist_ok=True)
    summary_outputs = {}
    for draft_k, summary in summaries.items():
        output = args.result_dir / f"e2e-16step-k{draft_k}.json"
        output.write_text(json.dumps(summary, indent=2) + "\n")
        summary_outputs[str(draft_k)] = {"path": str(output), "sha256": _sha256(output)}

    manifest = {
        "description": "Losslessly labelled/resized grids of existing experiment PNGs; no synthesized panels.",
        "raw_root": str(args.raw_root),
        "prompts": PROMPTS,
        "style_labels": STYLE_LABELS,
        "metrics": metrics,
        "summaries": summary_outputs,
        "outputs": {
            "method_style_grid": {
                "path": str(args.figure_dir / "method_style_grid.png"),
                "sha256": _sha256(args.figure_dir / "method_style_grid.png"),
            },
            "draft_length_grid": {
                "path": str(args.figure_dir / "draft_length_grid.png"),
                "sha256": _sha256(args.figure_dir / "draft_length_grid.png"),
            },
            "skip_length_grid": {
                "path": str(args.figure_dir / "skip_length_grid.png"),
                "sha256": _sha256(args.figure_dir / "skip_length_grid.png"),
            },
        },
    }
    manifest_path = args.generated_dir / "ablation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {args.figure_dir / 'method_style_grid.png'}")
    print(f"wrote {args.figure_dir / 'draft_length_grid.png'}")
    print(f"wrote {args.figure_dir / 'skip_length_grid.png'}")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
