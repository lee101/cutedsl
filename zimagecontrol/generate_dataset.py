"""Generate a line-conditioned style-transfer dataset with Z-Image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from zimagecontrol.conditioning import CannyExtractionConfig, LineExtractionConfig, save_conditioning_triplet
from zimagecontrol.runtime import parse_dtype


DEFAULT_CONTENT_PROMPTS = [
    "a portrait of a traveler looking over a futuristic harbor",
    "a dense old city street with hanging lanterns and wet cobblestones",
    "a mountain shrine on a cliff above a cloud sea",
    "a botanical study of an unusual flower arrangement",
    "a racing motorcycle parked under a brutalist overpass",
    "a quiet cafe interior with rain on the windows",
    "a giant library hall with ladders and skylight beams",
    "a fox in a workshop full of tools and sketches",
    "a solitary lighthouse on a rocky coast during rough weather",
    "a floating island farm suspended above a valley",
]

DEFAULT_STYLE_PROMPTS = [
    "rendered as a delicate sumi-e ink wash painting",
    "in vivid art nouveau poster style with ornate borders",
    "as a detailed ukiyo-e woodblock print",
    "as a moody charcoal and graphite illustration",
    "in bold cel-animated key art style",
    "as a cinematic matte painting with dramatic light",
    "in retro-futurist industrial design illustration style",
    "as a richly textured oil painting on canvas",
]


def _load_lines(path: str) -> list[str]:
    if not path:
        return []
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def _build_transformer(choice: str, orig_transformer, dtype: torch.dtype, device: str):
    if choice == "diffusers":
        return orig_transformer.to(device=device, dtype=dtype).eval()
    if choice == "cute":
        from cutezimage.model import CuteZImageTransformer

        return CuteZImageTransformer.from_diffusers(orig_transformer).to(device=device, dtype=dtype).eval()
    if choice == "accelerated":
        from zimageaccelerated.model import AcceleratedZImageTransformer

        return AcceleratedZImageTransformer.from_diffusers(orig_transformer).to(device=device, dtype=dtype).eval()
    raise ValueError(f"Unsupported transformer choice: {choice}")


def _make_pairs(contents: list[str], styles: list[str], cross_product: bool) -> list[tuple[str, str]]:
    if not contents or not styles:
        raise ValueError("Need at least one content prompt and one style prompt")
    if cross_product:
        return [(content, style) for content in contents for style in styles]
    return [(content, styles[idx % len(styles)]) for idx, content in enumerate(contents)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Z-Image ControlNet line dataset")
    parser.add_argument("--model-id", default="Tongyi-MAI/Z-Image-Turbo")
    parser.add_argument("--transformer", default="diffusers", choices=["diffusers", "cute", "accelerated"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--content-prompt", action="append", default=[])
    parser.add_argument("--style-prompt", action="append", default=[])
    parser.add_argument("--content-prompts-file", default="")
    parser.add_argument("--style-prompts-file", default="")
    parser.add_argument("--num-contents", type=int, default=10)
    parser.add_argument("--num-styles", type=int, default=8)
    parser.add_argument("--cross-product", action="store_true")
    parser.add_argument("--output-dir", default="zimagecontrol/generated_dataset")
    parser.add_argument("--blur-radius", type=float, default=1.2)
    parser.add_argument("--edge-percentile", type=float, default=0.82)
    parser.add_argument("--line-strength", type=float, default=1.0)
    parser.add_argument("--sparse-patch-size", type=int, default=32)
    parser.add_argument("--sparse-drop-prob", type=float, default=0.18)
    parser.add_argument("--conditioning-type", default="all", choices=["line", "canny", "all"])
    parser.add_argument("--canny-low", type=int, default=100)
    parser.add_argument("--canny-high", type=int, default=200)
    parser.add_argument("--canny-blur-ksize", type=int, default=5)
    parser.add_argument("--cpu-offload", action="store_true")
    args = parser.parse_args()

    dtype = parse_dtype(args.dtype)
    contents = list(args.content_prompt) + _load_lines(args.content_prompts_file)
    styles = list(args.style_prompt) + _load_lines(args.style_prompts_file)
    if not contents:
        contents = DEFAULT_CONTENT_PROMPTS[: args.num_contents]
    if not styles:
        styles = DEFAULT_STYLE_PROMPTS[: args.num_styles]
    contents = contents[: args.num_contents]
    styles = styles[: args.num_styles]
    pairs = _make_pairs(contents, styles, cross_product=args.cross_product)
    seeds = [args.seed_start + index for index in range(args.num_seeds)]

    output_dir = Path(args.output_dir)
    target_dir = output_dir / "target"
    line_dir = output_dir / "line"
    sparse_dir = output_dir / "line_sparse"
    canny_dir = output_dir / "canny"
    metadata_path = output_dir / "metadata.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    from diffusers import ZImagePipeline

    pipe = ZImagePipeline.from_pretrained(args.model_id, torch_dtype=dtype)
    pipe.transformer = _build_transformer(args.transformer, pipe.transformer, dtype, args.device)
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(args.device)

    line_config = LineExtractionConfig(
        blur_radius=args.blur_radius,
        edge_percentile=args.edge_percentile,
        line_strength=args.line_strength,
    )
    canny_config = CannyExtractionConfig(
        low_threshold=args.canny_low,
        high_threshold=args.canny_high,
        blur_ksize=args.canny_blur_ksize,
    )
    use_canny = args.conditioning_type in ("canny", "all")

    records: list[dict[str, object]] = []
    total = len(pairs) * len(seeds)
    item_index = 0
    for pair_index, (content_prompt, style_prompt) in enumerate(pairs):
        prompt = f"{content_prompt}, {style_prompt}"
        for seed in seeds:
            generator = torch.Generator(device=args.device if args.device == "cuda" else "cpu").manual_seed(seed)
            result = pipe(
                prompt=prompt,
                width=args.width,
                height=args.height,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            )
            image = result.images[0]
            stem = f"c{pair_index:04d}_s{seed:06d}"
            paths = save_conditioning_triplet(
                image,
                target_path=target_dir / f"{stem}.png",
                line_path=line_dir / f"{stem}.png",
                sparse_line_path=sparse_dir / f"{stem}.png",
                line_config=line_config,
                sparse_patch_size=args.sparse_patch_size,
                sparse_drop_prob=args.sparse_drop_prob,
                sparse_seed=seed,
                canny_path=(canny_dir / f"{stem}.png") if use_canny else None,
                canny_config=canny_config if use_canny else None,
            )
            record = {
                "index": item_index,
                "pair_index": pair_index,
                "seed": seed,
                "content_prompt": content_prompt,
                "style_prompt": style_prompt,
                "prompt": prompt,
                "transformer": args.transformer,
                "steps": args.steps,
                "guidance_scale": args.guidance_scale,
                "width": args.width,
                "height": args.height,
                **paths,
            }
            records.append(record)
            item_index += 1
            print(f"[{item_index}/{total}] wrote {paths['target_image_path']}")

    with metadata_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    print(
        json.dumps(
            {
                "model_id": args.model_id,
                "transformer": args.transformer,
                "num_samples": len(records),
                "metadata_path": str(metadata_path),
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

