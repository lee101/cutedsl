"""Micro-benchmark for the fused adaLN modulation and gated residual kernels.

    python -m cuteanima.benchmark_kernels --batch 2 --seq-len 3952 --dim 2048

Reports the eager PyTorch op chain, the fused Triton kernel, and a `torch.compile`d
version of the eager chain, so the block-level numbers in the CuteAnima benchmark
can be attributed to the right layer.
"""

from __future__ import annotations

import argparse
import json

import torch

from .triton_kernels import (
    adaln_modulate,
    gated_residual,
    reference_adaln_modulate,
    reference_gated_residual,
)


def _time(function, *args, iterations: int = 50, warmup: int = 10) -> float:
    for _ in range(warmup):
        function(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=3952, help="832x1216 latent tokens after 2x2 patching")
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    torch.manual_seed(0)
    hidden = torch.randn(args.batch, args.seq_len, args.dim, device="cuda", dtype=torch.bfloat16)
    delta = torch.randn_like(hidden)
    shift = torch.randn(args.batch, args.dim, device="cuda", dtype=torch.bfloat16)
    scale = torch.randn(args.batch, args.dim, device="cuda", dtype=torch.bfloat16) * 0.1
    gate = torch.randn(args.batch, args.dim, device="cuda", dtype=torch.bfloat16)

    compiled_modulate = torch.compile(reference_adaln_modulate, dynamic=False)
    compiled_residual = torch.compile(reference_gated_residual, dynamic=False)

    results = {
        "shape": [args.batch, args.seq_len, args.dim],
        "modulate_eager_ms": _time(reference_adaln_modulate, hidden, shift, scale, iterations=args.iterations),
        "modulate_fused_ms": _time(adaln_modulate, hidden, shift, scale, iterations=args.iterations),
        "modulate_compiled_ms": _time(compiled_modulate, hidden, shift, scale, iterations=args.iterations),
        "residual_eager_ms": _time(reference_gated_residual, hidden, delta, gate, iterations=args.iterations),
        "residual_fused_ms": _time(gated_residual, hidden, delta, gate, iterations=args.iterations),
        "residual_compiled_ms": _time(compiled_residual, hidden, delta, gate, iterations=args.iterations),
    }
    bytes_moved = hidden.numel() * hidden.element_size()
    results["modulate_fused_gbps"] = 2 * bytes_moved / (results["modulate_fused_ms"] * 1e-3) / 1e9
    results["per_forward_saving_ms"] = 40 * 3 * (
        results["modulate_eager_ms"] - results["modulate_fused_ms"]
        + results["residual_eager_ms"] - results["residual_fused_ms"]
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
