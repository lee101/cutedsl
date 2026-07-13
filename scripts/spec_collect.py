"""Collect Z-Image denoising trajectories for speculative-walking research.

Saves one .pt per (prompt, seed): all per-step latents + pooled text embedding.
Default output on /sdb-disk (nvme is full).

Run: .venv/bin/python scripts/spec_collect.py --steps 16 --n 200
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from latentteleport.dataset import capture_intermediates, extract_text_embedding  # noqa: E402

MODEL = "Tongyi-MAI/Z-Image-Turbo"
OUT_BASE = Path("/sdb-disk/latentteleport-spec")
PROMPTS = Path("/sdb-disk/cutedsl-images/prompts.jsonl")


def load_prompts(n: int) -> list[str]:
    seen, out = set(), []
    with open(PROMPTS) as f:
        for line in f:
            try:
                p = json.loads(line).get("prompt", "").strip()
            except Exception:
                continue
            if 8 < len(p) < 400 and p not in seen:
                seen.add(p)
                out.append(p)
            if len(out) >= n:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seeds", type=int, default=1)
    args = ap.parse_args()

    out_dir = OUT_BASE / f"trajs-{args.steps}step-{args.size}"
    out_dir.mkdir(parents=True, exist_ok=True)

    from diffusers import ZImagePipeline

    pipe = ZImagePipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    prompts = load_prompts(args.n)
    t0 = time.time()
    done = 0
    for pi, prompt in enumerate(prompts):
        for seed in range(args.seeds):
            key = hashlib.sha1(f"{prompt}|{seed}".encode()).hexdigest()[:16]
            out = out_dir / f"{key}.pt"
            if out.exists():
                continue
            _, inter = capture_intermediates(
                pipe, prompt, args.size, args.size, args.steps, seed, "cuda", guidance_scale=0.0
            )
            emb = extract_text_embedding(pipe, prompt)
            pooled = emb.reshape(-1, emb.shape[-1]).mean(dim=0)
            lat = torch.stack([inter[i] for i in sorted(inter)], dim=0).squeeze(1).to(torch.float16)
            torch.save({"prompt": prompt, "seed": seed, "steps": args.steps, "latents": lat, "text_emb": pooled.half()}, out)
            done += 1
            if done % 20 == 0:
                print(f"{done} trajs, {(time.time()-t0)/done:.1f}s each", flush=True)
    print(f"done: {done} new trajectories in {out_dir} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
