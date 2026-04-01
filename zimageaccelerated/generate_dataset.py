"""Generate a prompt/seed image dataset for Z-Image acceleration experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cutezimage.model import CuteZImageTransformer
from zimageaccelerated.model import AcceleratedZImageTransformer


DEFAULT_ART_PROMPTS = [
    "an ultradetailed oil painting of a lighthouse in a storm, dramatic sky, cinematic light",
    "a fashion editorial portrait in neon rain, reflective streets, moody film grain",
    "a serene ukiyo-e inspired mountain lake at sunrise, intricate linework, soft mist",
    "a brutalist megastructure in a desert, tiny travelers for scale, high contrast shadows",
    "a botanical illustration of impossible flowers, scientific precision, pastel paper texture",
    "a cozy fantasy tavern interior, warm candlelight, carved wood, bustling atmosphere",
    "a retro-futurist race car concept render, studio lighting, glossy materials, 1980s palette",
    "a gothic cathedral grown from black crystal, volumetric fog, moonlit night",
    "an underwater ballet scene with flowing fabric, bioluminescent coral, ethereal lighting",
    "a playful claymation market street with fruit stalls, shallow depth of field",
    "a monumental sci-fi archive library, endless shelves, golden dust in sunbeams",
    "a painterly portrait of an astronaut with wildflowers in the visor reflection",
    "a minimalist Japanese courtyard in winter, precise geometry, quiet atmosphere",
    "a surreal collage of birds and sheet music, handmade paper edges, museum print quality",
    "an Art Nouveau poster of a moon goddess, ornate borders, emerald and gold inks",
    "a high-detail concept art scene of a volcanic forge city, rivers of lava below bridges",
    "a dreamy analog photograph of a coastal road at blue hour, soft halation",
    "a children's book illustration of a fox inventor workshop, whimsical gadgets everywhere",
    "a cinematic matte painting of floating islands over farmland, crisp midday light",
    "a monochrome ink drawing of a crowded cyberpunk alley, dense signage and cables",
]


def _parse_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[key]


def _load_prompts(args: argparse.Namespace) -> list[str]:
    prompts = list(args.prompt)
    if args.prompts_file:
        raw = Path(args.prompts_file).read_text().splitlines()
        prompts.extend(line.strip() for line in raw if line.strip())
    if not prompts:
        prompts = DEFAULT_ART_PROMPTS[: args.num_prompts]
    return prompts[: args.num_prompts]


def _build_transformer(choice: str, orig_transformer, dtype: torch.dtype, device: str):
    if choice == "diffusers":
        return orig_transformer.to(device=device, dtype=dtype).eval()
    if choice == "cute":
        return CuteZImageTransformer.from_diffusers(orig_transformer).to(device=device, dtype=dtype).eval()
    if choice == "accelerated":
        return AcceleratedZImageTransformer.from_diffusers(orig_transformer).to(device=device, dtype=dtype).eval()
    raise ValueError(f"Unsupported transformer choice: {choice}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Z-Image prompt/seed dataset")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--transformer", default="accelerated", choices=["diffusers", "cute", "accelerated"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--num-prompts", type=int, default=20)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--prompts-file", default="")
    parser.add_argument("--output-dir", default="zimageaccelerated/generated_dataset")
    parser.add_argument("--cpu-offload", action="store_true")
    args = parser.parse_args()

    dtype = _parse_dtype(args.dtype)
    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    prompts = _load_prompts(args)
    seeds = [args.seed_start + idx for idx in range(args.num_seeds)]

    from diffusers import ZImagePipeline

    pipe = ZImagePipeline.from_pretrained(args.model_id, torch_dtype=dtype)
    pipe.transformer = _build_transformer(args.transformer, pipe.transformer, dtype, args.device)
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(args.device)

    metadata_path = output_dir / "metadata.jsonl"
    records: list[dict[str, object]] = []
    total = len(prompts) * len(seeds)

    for prompt_idx, prompt in enumerate(prompts):
        for seed_idx, seed in enumerate(seeds):
            item_idx = prompt_idx * len(seeds) + seed_idx
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
            image_name = f"p{prompt_idx:03d}_s{seed:06d}.png"
            image_path = image_dir / image_name
            image.save(image_path)

            record = {
                "index": item_idx,
                "prompt_index": prompt_idx,
                "seed": seed,
                "prompt": prompt,
                "image_path": str(image_path),
                "transformer": args.transformer,
                "steps": args.steps,
                "guidance_scale": args.guidance_scale,
                "width": args.width,
                "height": args.height,
            }
            records.append(record)
            print(f"[{item_idx + 1}/{total}] wrote {image_path}")

    with metadata_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    summary = {
        "model_id": args.model_id,
        "transformer": args.transformer,
        "device": args.device,
        "dtype": str(dtype),
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "width": args.width,
        "height": args.height,
        "num_prompts": len(prompts),
        "num_seeds": len(seeds),
        "num_images": len(records),
        "metadata_path": str(metadata_path),
        "output_dir": str(output_dir),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
