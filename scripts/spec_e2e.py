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
    args = ap.parse_args()

    from diffusers import ZImagePipeline

    pipe = ZImagePipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    ckpt = torch.load(BASE / f"ckpt-{args.steps}step-{args.size}" / "spec.pt", map_location="cuda", weights_only=False)
    cfg = SpecConfig(**ckpt["cfg"])
    walker = LatentWalker(cfg).to("cuda").eval()
    walker.load_state_dict(ckpt["walker"])
    interp = GapInterpolator(cfg).to("cuda").eval()
    interp.load_state_dict(ckpt["interp"])

    out_dir = Path(args.out or BASE / f"e2e-{args.steps}step-k{args.draft_k}")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, prompt in enumerate(PROMPTS[: args.n]):
        torch.cuda.synchronize(); t0 = time.time()
        img_base, lat_base = baseline(pipe, prompt, args.steps, args.size, seed=7)
        torch.cuda.synchronize(); t_base = time.time() - t0

        t0 = time.time()
        img_spec, stats = speculative_denoise(
            pipe, prompt, walker, interp, total_steps=args.steps, draft_k=args.draft_k,
            height=args.size, width=args.size, seed=7,
        )
        torch.cuda.synchronize(); t_spec = time.time() - t0

        t0 = time.time()
        img_id, stats_id = speculative_denoise(
            pipe, prompt, None, None, total_steps=args.steps, draft_k=args.draft_k,
            height=args.size, width=args.size, seed=7,
        )
        torch.cuda.synchronize(); t_id = time.time() - t0

        row = {
            "prompt": prompt,
            "t_baseline": round(t_base, 2),
            "t_spec": round(t_spec, 2),
            "speedup": round(t_base / t_spec, 2),
            "big_steps": stats["big_steps"],
            "psnr_spec": round(psnr(img_base[0], img_spec[0]), 2),
            "psnr_identity_skip": round(psnr(img_base[0], img_id[0]), 2),
        }
        rows.append(row)
        print(row, flush=True)
        img_base[0].save(out_dir / f"{i}_base.png")
        img_spec[0].save(out_dir / f"{i}_spec.png")
        img_id[0].save(out_dir / f"{i}_skip.png")

    summary = {
        "steps": args.steps, "draft_k": args.draft_k,
        "mean_speedup": round(sum(r["speedup"] for r in rows) / len(rows), 2),
        "mean_psnr_spec": round(sum(r["psnr_spec"] for r in rows) / len(rows), 2),
        "mean_psnr_skip": round(sum(r["psnr_identity_skip"] for r in rows) / len(rows), 2),
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}))
    print(f"images + summary in {out_dir}")


if __name__ == "__main__":
    main()
