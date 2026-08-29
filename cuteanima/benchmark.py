"""Benchmark CuteAnima variants against the reference Anima pipeline.

    python -m cuteanima.benchmark --variants reference batched fused fused_compile \
        --steps 28 --out results/cuteanima

Every variant writes its image so quality can be reviewed by eye; the report also
carries pixel metrics against the reference run.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

PROMPTS = [
    "masterpiece, best quality, 1girl, wind-swept cape, luminous city at dusk, detailed background, dramatic illustration",
    "a lone samurai standing in falling snow, red maple leaves, cinematic lighting, highly detailed anime key visual",
]
NEGATIVE = "low quality, blurry, malformed hands"
VARIANTS = ("reference", "batched", "fused", "fused_compile", "reference_compile", "batched_compile")


def _metrics(reference, candidate):
    import numpy as np

    left = np.asarray(reference, dtype=np.float64)
    right = np.asarray(candidate, dtype=np.float64)
    if left.shape != right.shape:
        return {"shape_mismatch": True}
    mse = float(((left - right) ** 2).mean())
    return {
        "max_abs_diff": float(np.abs(left - right).max()),
        "mean_abs_diff": float(np.abs(left - right).mean()),
        "psnr_db": float("inf") if mse == 0 else 10.0 * float(np.log10(255.0**2 / mse)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=1216)
    parser.add_argument("--guidance", type=float, default=4.0)
    parser.add_argument("--prompts", type=int, default=1)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--repeat", type=int, default=3, help="timed runs per job; the minimum is reported")
    parser.add_argument("--compile-mode", default="max-autotune-no-cudagraphs")
    parser.add_argument("--out", default="results/cuteanima")
    args = parser.parse_args()

    import torch

    from cuteanima.loader import load_pipeline
    from cuteanima.patch import apply_fused_blocks, remove_fused_blocks
    from cuteanima.runner import AnimaRunner

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    pipe = load_pipeline(torch, fused=False)
    load_seconds = time.perf_counter() - started
    print(f"loaded in {load_seconds:.1f}s", flush=True)

    eager_transformer = pipe.transformer
    compiled_transformer = None
    runner = AnimaRunner(pipe, torch)
    report = {"load_s": load_seconds, "config": vars(args), "runs": []}
    references = {}

    for variant in args.variants:
        if variant not in VARIANTS:
            raise SystemExit(f"unknown variant {variant}; choose from {VARIANTS}")
        remove_fused_blocks(eager_transformer)
        if variant in {"fused", "fused_compile"}:
            apply_fused_blocks(eager_transformer)
        if variant.endswith("_compile"):
            if compiled_transformer is None:
                compiled_transformer = torch.compile(
                    eager_transformer, mode=args.compile_mode, fullgraph=False, dynamic=False
                )
            pipe.transformer = compiled_transformer
        else:
            pipe.transformer = eager_transformer
        runner.batch_cfg = not variant.startswith("reference")
        runner._embeds.clear()

        for prompt_index, prompt in enumerate(PROMPTS[: args.prompts]):
            for seed in args.seeds:
                timings = []
                image = None
                for repeat in range(args.repeat):
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    image = runner(
                        prompt=prompt,
                        negative_prompt=NEGATIVE,
                        width=args.width,
                        height=args.height,
                        num_inference_steps=args.steps,
                        guidance_scale=args.guidance,
                        generator=torch.Generator(device="cuda").manual_seed(seed),
                    )
                    torch.cuda.synchronize()
                    timings.append(time.perf_counter() - start)
                path = out / f"{variant}-p{prompt_index}-s{seed}.png"
                image.save(path)
                entry = {
                    "variant": variant,
                    "prompt_index": prompt_index,
                    "seed": seed,
                    "seconds_min": min(timings),
                    "seconds_all": timings,
                    "s_per_step": min(timings) / args.steps,
                    "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "image": str(path),
                }
                if variant == "reference":
                    references[(prompt_index, seed)] = image
                elif (prompt_index, seed) in references:
                    entry["vs_reference"] = _metrics(references[(prompt_index, seed)], image)
                report["runs"].append(entry)
                print(json.dumps(entry), flush=True)

    (out / "results.json").write_text(json.dumps(report, indent=2))
    baseline = next((run["seconds_min"] for run in report["runs"] if run["variant"] == "reference"), None)
    for run in report["runs"]:
        speedup = f" {baseline / run['seconds_min']:.2f}x" if baseline else ""
        print(f"{run['variant']:<14} {run['seconds_min']:.2f}s{speedup}")


if __name__ == "__main__":
    main()
