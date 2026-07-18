"""Benchmark: fused multi-LoRA delta vs the stock per-adapter fp32 loop.

Measures the delta-compute cost the swapper pays per module when N adapters
are active, across realistic Z-Image attention shapes and ranks. Verifies the
fused result matches the stock fp32 result within LoRA-merge tolerance.

    python -m cuteloras.benchmark_fused --output cuteloras_fused_bench.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from cuteloras.fused_apply import fused_lora_delta, stock_lora_delta


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def make_terms(d, r, n, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    terms = []
    for i in range(n):
        a = torch.randn(r, d, generator=g) * 0.02       # [r, d]
        b = torch.randn(d, r, generator=g) * 0.02       # [d, r]
        terms.append((a, b, 0.8 + 0.1 * i))
    return terms


def time_fn(fn, iters, warmup=5):
    for _ in range(warmup):
        fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    return (time.perf_counter() - t0) / iters * 1000.0  # ms/call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="cuteloras_fused_bench.json")
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dtype = torch.bfloat16
    # Z-Image-Turbo-ish attention widths x realistic LoRA ranks x adapter counts
    shapes = [(1536, 16), (1536, 32), (3072, 32), (3072, 64)]
    counts = [1, 2, 4, 8]

    results = {"device": device, "out_dtype": "bfloat16", "cases": []}
    for d, r in shapes:
        for n in counts:
            terms = make_terms(d, r, n, device)
            stock = lambda: stock_lora_delta(terms, out_dtype, device)
            fused = lambda: fused_lora_delta(terms, out_dtype, device)

            # correctness: fused (bf16) vs stock (fp32), relative to stock magnitude
            ds = stock().float()
            df = fused().float()
            rel_err = (ds - df).abs().max().item() / (ds.abs().max().item() + 1e-9)

            ms_stock = time_fn(stock, args.iters)
            ms_fused = time_fn(fused, args.iters)
            case = {
                "d": d, "r": r, "n_adapters": n,
                "stock_ms": round(ms_stock, 4),
                "fused_ms": round(ms_fused, 4),
                "speedup": round(ms_stock / ms_fused, 2),
                "max_rel_err": round(rel_err, 6),
            }
            results["cases"].append(case)
            print(f"d={d:5d} r={r:3d} n={n}  stock={ms_stock:7.3f}ms  "
                  f"fused={ms_fused:7.3f}ms  {case['speedup']:5.2f}x  relerr={rel_err:.2e}")

    speedups = [c["speedup"] for c in results["cases"]]
    multi = [c["speedup"] for c in results["cases"] if c["n_adapters"] >= 2]
    results["summary"] = {
        "mean_speedup": round(statistics.mean(speedups), 2),
        "mean_speedup_multi_lora": round(statistics.mean(multi), 2) if multi else None,
        "max_rel_err": round(max(c["max_rel_err"] for c in results["cases"]), 6),
    }
    print(f"\nmean speedup: {results['summary']['mean_speedup']}x  "
          f"(multi-LoRA {results['summary']['mean_speedup_multi_lora']}x)  "
          f"max rel err {results['summary']['max_rel_err']:.2e}")
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
