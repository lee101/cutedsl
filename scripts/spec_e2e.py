"""End-to-end speculative walking eval: full-step baseline vs drafted runs.

Compares wall-clock and image similarity (PSNR + latent relL2) of:
  baseline   all N steps real
  identity   real steps only at anchors, drafts skipped with no correction
  spec       walker rollout + interpolator teleport between real steps
  scaled     schedule-calibrated momentum teleport

Run: .venv/bin/python scripts/spec_e2e.py --steps 16 --draft-k 3 --n 6
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from latentteleport.speculative import (  # noqa: E402
    GapInterpolator,
    LatentWalker,
    SpecConfig,
    speculative_denoise,
)

BASE = Path("/sdb-disk/latentteleport-spec")
REPO = Path(__file__).resolve().parent.parent
MODEL = "Tongyi-MAI/Z-Image-Turbo"

PROMPTS = [
    "a lighthouse on a cliff at golden hour, crashing waves",
    "portrait of an old fisherman with a pipe, dramatic lighting",
    "a cozy cabin in a snowy forest at night, warm windows",
    "cyberpunk street market in the rain, neon signs",
    "a fox curled up on autumn leaves, soft light",
    "isometric tiny island with a waterfall and windmill",
]


def psnr(a, b):
    a = np.asarray(a, dtype=np.float32) / 255.0
    b = np.asarray(b, dtype=np.float32) / 255.0
    mse = ((a - b) ** 2).mean()
    return 99.0 if mse == 0 else float(-10 * np.log10(mse))


def load_momentum_scales(path: Path, steps: int) -> dict[tuple[int, int], float]:
    raw = json.loads(path.read_text())
    fitted_steps = int(raw.get("protocol", {}).get("n_steps", -1))
    if fitted_steps != steps:
        raise ValueError(
            f"momentum scales were fit for {fitted_steps} steps, requested {steps}; "
            "collect and fit trajectories for the requested schedule first"
        )
    return {
        (int(row["step"]), int(row["draft_k"])): float(row["momentum_scale"])
        for row in raw.get("deployment_coefficients", [])
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--draft-k", type=int, default=3)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cute", action="store_true", help="cutezimage accelerated transformer for real steps")
    ap.add_argument("--compile-nets", action="store_true", help="torch.compile walker+interp (reduce-overhead)")
    ap.add_argument("--offload", action="store_true", help="cpu-offload pipeline modules (shared-GPU safety)")
    ap.add_argument("--warmup", type=int, default=1, help="untimed full-schedule warm-up generations")
    ap.add_argument("--repeats", type=int, default=2, help="timed repeats per prompt and arm")
    ap.add_argument(
        "--cache-prompt-embeds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="encode each prompt once, while still reporting cold end-to-end time",
    )
    ap.add_argument("--continue-on-oom", action="store_true", help="record an OOM arm and continue the sweep")
    ap.add_argument(
        "--modes",
        nargs="+",
        choices=("spec", "taylor", "scaled", "skip"),
        default=("spec", "taylor", "scaled", "skip"),
    )
    ap.add_argument(
        "--momentum-scales",
        type=Path,
        default=REPO / "results/speculative/forecaster-ablation.json",
    )
    args = ap.parse_args()
    momentum_scales = (
        load_momentum_scales(args.momentum_scales, args.steps)
        if "scaled" in args.modes
        else None
    )

    if args.cute:
        from cutezimage.pipeline import get_zimage_pipelines

        pipe, _ = get_zimage_pipelines(MODEL, torch_dtype=torch.bfloat16, use_cute=True)
    else:
        from diffusers import ZImagePipeline

        pipe = ZImagePipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
        if args.offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    walker = interp = None
    if "spec" in args.modes:
        ckpt = torch.load(
            BASE / f"ckpt-{args.steps}step-{args.size}" / "spec.pt",
            map_location="cuda",
            weights_only=False,
        )
        cfg = SpecConfig(**ckpt["cfg"])
        walker = LatentWalker(cfg).to("cuda").eval()
        walker.load_state_dict(ckpt["walker"])
        interp = GapInterpolator(cfg).to("cuda").eval()
        interp.load_state_dict(ckpt["interp"])
        if args.compile_nets:
            walker.forward = torch.compile(walker.forward, mode="reduce-overhead")
            interp.forward = torch.compile(interp.forward, mode="reduce-overhead")

    out_dir = Path(args.out or BASE / f"e2e-{args.steps}step-k{args.draft_k}")
    out_dir.mkdir(parents=True, exist_ok=True)

    for _ in range(max(0, args.warmup)):
        warm_prompt = PROMPTS[0]
        warm_embeds, _ = pipe.encode_prompt(prompt=warm_prompt, device="cuda")
        speculative_denoise(
            pipe, warm_prompt, None, None, total_steps=args.steps, draft_k=0,
            height=args.size, width=args.size, seed=7, mode="baseline",
            prompt_embeds=warm_embeds,
        )

    rows = []
    for i, prompt in enumerate(PROMPTS[: args.n]):
        torch.cuda.synchronize()
        encode_started = time.perf_counter()
        encoded_prompt, _ = pipe.encode_prompt(prompt=prompt, device="cuda")
        torch.cuda.synchronize()
        encode_ms = (time.perf_counter() - encode_started) * 1000.0

        arm_names = ["baseline", *args.modes]
        measurements = {name: [] for name in arm_names}
        images = {}
        errors = {}
        for repeat in range(max(1, args.repeats)):
            shift = (i + repeat) % len(arm_names)
            for mode in arm_names[shift:] + arm_names[:shift]:
                try:
                    image, stats = speculative_denoise(
                        pipe, prompt, walker, interp,
                        total_steps=args.steps,
                        draft_k=args.draft_k,
                        height=args.size,
                        width=args.size,
                        seed=7,
                        mode=mode,
                        momentum_scales=momentum_scales,
                        prompt_embeds=encoded_prompt if args.cache_prompt_embeds else None,
                    )
                except torch.cuda.OutOfMemoryError as exc:
                    errors[mode] = f"{type(exc).__name__}: {exc}"
                    torch.cuda.empty_cache()
                    if not args.continue_on_oom:
                        raise
                    continue
                measurements[mode].append(stats)
                images[mode] = image

        row = {
            "prompt": prompt[:40],
            "encode_ms": round(encode_ms, 2),
            "repeats": args.repeats,
            "errors": errors,
        }
        for mode in arm_names:
            samples = measurements[mode]
            if not samples:
                continue
            cached_ms = statistics.median(s["total_ms"] for s in samples)
            cold_ms = cached_ms + (encode_ms if args.cache_prompt_embeds else 0.0)
            row[f"t_{mode}"] = round(cold_ms / 1000.0, 3)
            row[f"t_cached_{mode}"] = round(cached_ms / 1000.0, 3)
            row[f"t_denoise_{mode}"] = round(statistics.median(s["denoise_ms"] for s in samples) / 1000.0, 3)
            row[f"t_decode_{mode}"] = round(statistics.median(s["decode_ms"] for s in samples) / 1000.0, 3)
            row[f"peak_vram_mb_{mode}"] = round(max(s.get("peak_vram_mb", 0.0) for s in samples), 1)
            if mode != "baseline" and "baseline" in images:
                row[f"psnr_{mode}"] = round(psnr(images["baseline"][0], images[mode][0]), 2)
            if mode != "baseline":
                row["big_steps"] = samples[0]["big_steps"]
            if mode == "scaled":
                row["scale_fallbacks_scaled"] = max(s["scale_fallbacks"] for s in samples)
            images[mode][0].save(out_dir / f"{i}_{'base' if mode == 'baseline' else mode}.png")

        t_base = row.get("t_baseline")
        denoise_base = row.get("t_denoise_baseline")
        for mode in args.modes:
            if t_base is not None and row.get(f"t_{mode}"):
                row[f"speedup_{mode}"] = round(t_base / row[f"t_{mode}"], 3)
            if denoise_base is not None and row.get(f"t_denoise_{mode}"):
                row[f"denoise_speedup_{mode}"] = round(
                    denoise_base / row[f"t_denoise_{mode}"], 3,
                )
        rows.append(row)
        print(row, flush=True)

    summary = {
        "steps": args.steps,
        "draft_k": args.draft_k,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "cache_prompt_embeds": args.cache_prompt_embeds,
        "rows": rows,
    }
    for mode in args.modes:
        valid = [r for r in rows if f"speedup_{mode}" in r]
        if not valid:
            continue
        summary[f"median_speedup_{mode}"] = round(statistics.median(r[f"speedup_{mode}"] for r in valid), 3)
        summary[f"median_denoise_speedup_{mode}"] = round(
            statistics.median(r[f"denoise_speedup_{mode}"] for r in valid), 3,
        )
        summary[f"mean_psnr_{mode}"] = round(statistics.mean(r[f"psnr_{mode}"] for r in valid), 2)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}))
    print(f"images + summary in {out_dir}")


if __name__ == "__main__":
    main()
