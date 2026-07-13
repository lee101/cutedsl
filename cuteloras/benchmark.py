"""Benchmark LoRA hot-swap latency and end-to-end generation throughput.

Usage:
    python -m cuteloras.benchmark --loras-dir /path/to/safetensors --output cuteloras_bench.json
    python -m cuteloras.benchmark --registry registry.json --e2e --prompts "a cat" "cyberpunk city"
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from cuteloras.registry import LoRARegistry
from cuteloras.swapper import LoRASwapper


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bench_swaps(swapper: LoRASwapper, lora_ids: list[str], cycles: int = 5) -> dict:
    results: dict = {"loras": lora_ids, "cycles": cycles}

    cold_ms = {}
    for lora_id in lora_ids:
        start = time.perf_counter()
        swapper.get_factors(lora_id)
        cold_ms[lora_id] = (time.perf_counter() - start) * 1000
    results["cold_load_ms"] = cold_ms

    start = time.perf_counter()
    swapper.activate(lora_ids[0])
    _sync()
    results["first_apply_ms"] = (time.perf_counter() - start) * 1000

    swap_times = []
    for _ in range(cycles):
        for lora_id in lora_ids:
            start = time.perf_counter()
            swapper.activate(lora_id)
            _sync()
            swap_times.append((time.perf_counter() - start) * 1000)
    results["warm_swap_ms"] = {
        "avg": statistics.mean(swap_times),
        "p50": statistics.median(swap_times),
        "min": min(swap_times),
        "max": max(swap_times),
        "n": len(swap_times),
    }

    start = time.perf_counter()
    swapper.deactivate()
    _sync()
    results["restore_ms"] = (time.perf_counter() - start) * 1000

    if torch.cuda.is_available():
        results["peak_gpu_memory_mb"] = torch.cuda.max_memory_allocated() / (1 << 20)
    return results


def bench_e2e(server, lora_ids: list[str], prompts: list[str], steps: int | None = None) -> dict:
    per_gen = []
    server.generate(prompts[0], lora_id=lora_ids[0], auto_route=False, steps=steps)
    for i, prompt in enumerate(prompts):
        lora_id = lora_ids[i % len(lora_ids)]
        start = time.perf_counter()
        result = server.generate(prompt, lora_id=lora_id, auto_route=False, steps=steps)
        total_ms = (time.perf_counter() - start) * 1000
        per_gen.append(
            {
                "prompt": prompt,
                "lora": lora_id,
                "total_ms": total_ms,
                "gen_ms": result["gen_ms"],
                "swap_ms": result["swap"].get("ms", 0.0),
            }
        )

    start = time.perf_counter()
    server.generate(prompts[0], auto_route=False, steps=steps)
    baseline_ms = (time.perf_counter() - start) * 1000

    totals = [g["total_ms"] for g in per_gen]
    return {
        "per_gen": per_gen,
        "avg_total_ms": statistics.mean(totals),
        "baseline_no_lora_ms": baseline_ms,
        "swap_overhead_pct": (statistics.mean(totals) - baseline_ms) / baseline_ms * 100,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loras-dir")
    parser.add_argument("--registry")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--e2e", action="store_true")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--prompts", nargs="*", default=["a portrait of a woman", "a cyberpunk city at night"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else None)
    parser.add_argument("--output", default="cuteloras_bench.json")
    args = parser.parse_args()

    if args.registry:
        registry = LoRARegistry.from_json(args.registry)
    elif args.loras_dir:
        registry = LoRARegistry.from_directory(args.loras_dir)
    else:
        parser.error("provide --loras-dir or --registry")

    lora_ids = [r.id for r in registry.all()][: args.limit]
    print(f"benchmarking {len(lora_ids)} loras: {lora_ids}")

    from cuteloras.server import LoRAServer, ZImageBackend

    backend = ZImageBackend(device=args.device, enable_cpu_offload=False if args.device == "cuda" else None)
    server = LoRAServer(backend, registry)

    results = {
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "swap": bench_swaps(server.swapper, lora_ids, cycles=args.cycles),
    }
    if args.e2e:
        results["e2e"] = bench_e2e(server, lora_ids, args.prompts, steps=args.steps)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({k: v for k, v in results["swap"].items() if k != "cold_load_ms"}, indent=2))
    if args.e2e:
        print(
            f"e2e avg {results['e2e']['avg_total_ms']:.0f}ms, swap overhead {results['e2e']['swap_overhead_pct']:.1f}%"
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
