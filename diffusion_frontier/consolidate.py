"""Consolidate frontier result files without misusing cross-model pixel metrics."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from PIL import Image


def _resolution(row: dict) -> str | None:
    if row.get("width") and row.get("height"):
        return f"{row['width']}x{row['height']}"
    image_path = row.get("image_path")
    if image_path:
        try:
            with Image.open(image_path) as image:
                return f"{image.width}x{image.height}"
        except OSError:
            pass
    return None


def consolidate_reports(paths: list[str | Path]) -> dict:
    groups: dict[tuple[str, str, str, str], list[dict]] = {}
    sources = []
    target = 1.0
    for raw_path in paths:
        path = Path(raw_path)
        report = json.loads(path.read_text())
        sources.append(str(path))
        target = min(target, float(report.get("target_latency_s", target)))
        for row in report["rows"]:
            runtime = report.get("runtime", {}).get(row["model"], {})
            backend = (
                row.get("backend")
                or runtime.get("quantization")
                or runtime.get("dtype")
                or runtime.get("offload")
                or "unspecified"
            )
            profile = row.get("profile") or f"{row.get('steps', 'unknown')}-step"
            resolution = _resolution(row) or "unknown"
            groups.setdefault((row["model"], backend, profile, resolution), []).append(row)

    profiles = []
    for (model, backend, profile, resolution), rows in sorted(groups.items()):
        walls = [float(row["wall_s"]) for row in rows]
        ssims = [
            float(row["quality_retention"]["ssim"])
            for row in rows
            if row.get("quality_retention", {}).get("ssim") is not None
        ]
        physical_steps = sorted(
            {
                int(row.get("physical_steps", row.get("steps")))
                for row in rows
                if row.get("physical_steps", row.get("steps")) is not None
            }
        )
        passes = sum(wall <= target for wall in walls)
        resolutions = [] if resolution == "unknown" else [resolution]
        profiles.append(
            {
                "model": model,
                "backend": backend,
                "profile": profile,
                "physical_steps": physical_steps,
                "resolutions": resolutions,
                "n": len(rows),
                "wall_s_min": min(walls),
                "wall_s_median": statistics.median(walls),
                "wall_s_max": max(walls),
                "latency_passes": passes,
                "strict_latency_pass": passes == len(rows),
                "ssim_median_within_model_reference": statistics.median(ssims) if ssims else None,
                "ssim_min_within_model_reference": min(ssims) if ssims else None,
            }
        )
    return {
        "schema_version": 1,
        "target_latency_s": target,
        "sources": sources,
        "metric_warning": (
            "SSIM is only comparable to the reference used inside each base-model report; "
            "do not rank different base models by SSIM. Use independent blind ratings."
        ),
        "profiles": profiles,
    }


def markdown(report: dict) -> str:
    lines = [
        "# Consolidated Diffusion Frontier",
        "",
        report["metric_warning"],
        "",
        "| Model | Backend | Profile | Resolution | Physical steps | N | Median s | Max s | <= target | Median within-model SSIM |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for item in report["profiles"]:
        ssim = item["ssim_median_within_model_reference"]
        lines.append(
            f"| {item['model']} | {item['backend']} | {item['profile']} | {','.join(item['resolutions'])} | "
            f"{','.join(map(str, item['physical_steps']))} | "
            f"{item['n']} | {item['wall_s_median']:.3f} | {item['wall_s_max']:.3f} | "
            f"{item['latency_passes']}/{item['n']} | {'' if ssim is None else f'{ssim:.3f}'} |"
        )
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    report = consolidate_reports(args.reports)
    rendered = markdown(report)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)
        output.with_suffix(".json").write_text(json.dumps(report, indent=2))
        print(output)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
