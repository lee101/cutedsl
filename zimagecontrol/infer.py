"""Run a trained Z-Image ControlNet against a prompt and control image."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from zimagecontrol.conditioning import extract_line_art
from zimagecontrol.runtime import parse_dtype


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Z-Image ControlNet inference")
    parser.add_argument("--model-id", default="Tongyi-MAI/Z-Image-Turbo")
    parser.add_argument("--controlnet-dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--control-image", required=True)
    parser.add_argument("--extract-lines", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--conditioning-scale", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-path", default="zimagecontrol/output.png")
    args = parser.parse_args()

    dtype = parse_dtype(args.dtype)
    control_image = Image.open(args.control_image).convert("RGB")
    if args.extract_lines:
        control_image = extract_line_art(control_image)
    control_image = control_image.resize((args.width, args.height), Image.Resampling.BICUBIC)

    from diffusers import ZImageControlNetModel, ZImageControlNetPipeline

    controlnet = ZImageControlNetModel.from_pretrained(args.controlnet_dir, torch_dtype=dtype)
    pipe = ZImageControlNetPipeline.from_pretrained(args.model_id, controlnet=controlnet, torch_dtype=dtype)
    pipe = pipe.to(args.device)

    generator = torch.Generator(device=args.device if args.device == "cuda" else "cpu").manual_seed(args.seed)
    result = pipe(
        prompt=args.prompt,
        control_image=control_image,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.conditioning_scale,
        generator=generator,
    )
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.images[0].save(output_path)
    print(output_path)


if __name__ == "__main__":
    main()

