"""End-to-end speculative walking eval: full-step baseline vs drafted runs.

Compares wall-clock and image similarity (PSNR + latent relL2) of:
  baseline   all N steps real
  identity   real steps only at anchors, drafts skipped with no correction
  spec       walker rollout + interpolator teleport between real steps

Run: .venv/bin/python scripts/spec_e2e.py --steps 16 --draft-k 3 --n 6
"""

from __future__ import annotations

import argparse
import json
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
    zimage_big_step,
    zimage_decode,
    zimage_prepare,
)

BASE = Path("/sdb-disk/latentteleport-spec")
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


@torch.no_grad()
def baseline(pipe, prompt, steps, size, seed):
    pe, lat, ts = zimage_prepare(pipe, prompt, steps, size, size, seed)
    for t in ts:
        lat = zimage_big_step(pipe, lat, t, pe)
    return zimage_decode(pipe, lat), lat


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
    args = ap.parse_args()

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

    ckpt = torch.load(BASE / f"ckpt-{args.steps}step-{args.size}" / "spec.pt", map_location="cuda", weights_only=False)
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

    rows = []
    for i, prompt in enumerate(PROMPTS[: args.n]):
        torch.cuda.synchronize(); t0 = time.time()
        img_base, lat_base = baseline(pipe, prompt, args.steps, args.size, seed=7)
        torch.cuda.synchronize(); t_base = time.time() - t0

        row = {"prompt": prompt[:40], "t_baseline": round(t_base, 2)}
        img_base[0].save(out_dir / f"{i}_base.png")
        for mode in ("spec", "taylor", "skip"):
            t0 = time.time()
            img_m, stats = speculative_denoise(
                pipe, prompt, walker, interp, total_steps=args.steps, draft_k=args.draft_k,
                height=args.size, width=args.size, seed=7, mode=mode,
            )
            torch.cuda.synchronize(); t_m = time.time() - t0
            row[f"t_{mode}"] = round(t_m, 2)
            row[f"speedup_{mode}"] = round(t_base / t_m, 2)
            row[f"psnr_{mode}"] = round(psnr(img_base[0], img_m[0]), 2)
            row["big_steps"] = stats["big_steps"]
            img_m[0].save(out_dir / f"{i}_{mode}.png")
        rows.append(row)
        print(row, flush=True)

    summary = {"steps": args.steps, "draft_k": args.draft_k, "rows": rows}
    for mode in ("spec", "taylor", "skip"):
        summary[f"mean_speedup_{mode}"] = round(sum(r[f"speedup_{mode}"] for r in rows) / len(rows), 2)
        summary[f"mean_psnr_{mode}"] = round(sum(r[f"psnr_{mode}"] for r in rows) / len(rows), 2)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}))
    print(f"images + summary in {out_dir}")


if __name__ == "__main__":
    main()
