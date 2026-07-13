"""Swap-latency benchmark against a random-init transformer with the real Z-Image
architecture — no checkpoint IO, measures pure swap mechanics.

Usage: python -m cuteloras.benchmark_swap_only --loras-dir ... --limit 6 --cycles 3
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from cuteloras.registry import LoRARegistry
from cuteloras.swapper import LoRASwapper


def build_model(device: str, accelerated: bool = True, n_layers: int | None = None):
    from cutezimage.model import CuteZImageConfig

    cfg = CuteZImageConfig()
    if n_layers:
        cfg.n_layers = n_layers
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device(device):
            if accelerated:
                from zimageaccelerated.model import AcceleratedZImageTransformer

                model = AcceleratedZImageTransformer(cfg)
            else:
                from cutezimage.model import CuteZImageTransformer

                model = CuteZImageTransformer(cfg)
    finally:
        torch.set_default_dtype(prev_dtype)
    return model.eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loras-dir", required=True)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eager", action="store_true", help="use non-fused CuteZImageTransformer")
    parser.add_argument("--layers", type=int, default=None, help="override n_layers (fit beside a resident server)")
    parser.add_argument("--pin", action="store_true", help="pin snapshot memory (slower setup, faster restore)")
    parser.add_argument("--output", default="cuteloras_swap_bench.json")
    args = parser.parse_args()

    registry = LoRARegistry.from_directory(args.loras_dir)
    lora_ids = [r.id for r in registry.all()][: args.limit]
    print(f"model: {'eager' if args.eager else 'accelerated/fused'} on {args.device}")
    print(f"loras: {lora_ids}")

    print("building model...")
    model = build_model(args.device, accelerated=not args.eager, n_layers=args.layers)
    swapper = LoRASwapper(model, registry, pin_snapshots=args.pin)
    results_layers = args.layers or 30
    print("model ready")

    def sync():
        if args.device == "cuda":
            torch.cuda.synchronize()

    results: dict = {"device": args.device, "loras": lora_ids, "cycles": args.cycles, "n_layers": results_layers}

    cold = {}
    for lora_id in lora_ids:
        t0 = time.perf_counter()
        f = swapper.get_factors(lora_id)
        cold[lora_id] = {"ms": (time.perf_counter() - t0) * 1000, "modules": len(f.factors)}
    results["cold_load"] = cold

    t0 = time.perf_counter()
    info = swapper.activate(lora_ids[0])
    sync()
    results["first_apply_ms"] = (time.perf_counter() - t0) * 1000
    results["first_apply_params"] = info["params"]
    print(f"first apply: {results['first_apply_ms']:.0f}ms ({info['params']} params)")

    swap_ms = []
    for _ in range(args.cycles):
        for lora_id in lora_ids:
            t0 = time.perf_counter()
            swapper.activate(lora_id)
            sync()
            swap_ms.append((time.perf_counter() - t0) * 1000)
    results["warm_swap_ms"] = {
        "avg": statistics.mean(swap_ms),
        "p50": statistics.median(swap_ms),
        "min": min(swap_ms),
        "max": max(swap_ms),
        "n": len(swap_ms),
        "extrapolated_30_layers": statistics.median(swap_ms) * 30 / results_layers,
    }

    t0 = time.perf_counter()
    swapper.deactivate()
    sync()
    results["restore_ms"] = (time.perf_counter() - t0) * 1000

    stack = [(lid, 1.0) for lid in lora_ids[:3]]
    t0 = time.perf_counter()
    swapper.activate(stack)
    sync()
    results["stack3_ms"] = (time.perf_counter() - t0) * 1000
    swapper.deactivate()

    if args.device == "cuda":
        results["peak_gpu_memory_mb"] = torch.cuda.max_memory_allocated() / (1 << 20)
    snap_bytes = sum(t.numel() * t.element_size() for t in swapper._snapshots.values())
    results["snapshot_cpu_mb"] = snap_bytes / (1 << 20)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(
        json.dumps(
            {k: results[k] for k in ("first_apply_ms", "warm_swap_ms", "restore_ms", "stack3_ms", "snapshot_cpu_mb")},
            indent=2,
        )
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
