"""Microbenchmarks for Z-Image transformer hot paths.

Benchmarks the attention subgraph, FFN, and full transformer block with
selectable SDPA backends so backend changes can be measured independently of
the rest of the model.

Usage:
    python -m cutezimage.benchmark_blocks --device cuda --dtype bfloat16 --seq-len 2048
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import time

import torch

from cutezimage.model import ADALN_EMBED_DIM, CuteZImageTransformerBlock


def _parse_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[key]


def _parse_backend(name: str | None):
    if not name or name == "auto":
        return None
    return {
        "flash": torch.nn.attention.SDPBackend.FLASH_ATTENTION,
        "math": torch.nn.attention.SDPBackend.MATH,
        "efficient": torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
        "cudnn": torch.nn.attention.SDPBackend.CUDNN_ATTENTION,
    }[name]


def _sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_call(fn, device: torch.device, warmup: int, runs: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    _sync_if_needed(device)

    latencies_ms: list[float] = []
    for _ in range(runs):
        _sync_if_needed(device)
        start = time.perf_counter()
        fn()
        _sync_if_needed(device)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    return {
        "avg_ms": sum(latencies_ms) / len(latencies_ms),
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
    }


def _make_inputs(
    batch_size: int,
    seq_len: int,
    dim: int,
    n_heads: int,
    device: torch.device,
    dtype: torch.dtype,
    modulation: bool,
) -> dict[str, torch.Tensor | None]:
    x = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype)
    ids = torch.stack(
        [
            torch.arange(seq_len, device=device) % 1536,
            torch.arange(seq_len, device=device) % 512,
            torch.arange(seq_len, device=device) % 512,
        ],
        dim=-1,
    ).to(torch.int32)
    head_dim = dim // n_heads
    freqs_real = torch.randn(1, seq_len, head_dim // 2, device=device, dtype=torch.float32)
    freqs_imag = torch.randn(1, seq_len, head_dim // 2, device=device, dtype=torch.float32)
    freqs_cis = torch.complex(freqs_real, freqs_imag)

    result: dict[str, torch.Tensor | None] = {
        "x": x,
        "attn_mask": None,
        "freqs_cis": freqs_cis,
        "ids": ids,
        "adaln_input": None,
    }
    if modulation:
        result["adaln_input"] = torch.randn(
            batch_size, min(dim, ADALN_EMBED_DIM), device=device, dtype=dtype
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Z-Image block microbenchmarks")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--dim", type=int, default=3840)
    parser.add_argument("--n-heads", type=int, default=30)
    parser.add_argument("--n-kv-heads", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--sdpa-backend", default="auto", choices=["auto", "flash", "math", "efficient", "cudnn"])
    parser.add_argument("--modulation", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--compile-mode",
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = _parse_dtype(args.dtype)
    block = CuteZImageTransformerBlock(
        layer_id=0,
        dim=args.dim,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        modulation=args.modulation,
    ).eval().to(device=device, dtype=dtype)

    if args.compile and hasattr(torch, "compile"):
        block.forward = torch.compile(block.forward, mode=args.compile_mode, fullgraph=False)  # type: ignore[assignment]

    inputs = _make_inputs(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        dim=args.dim,
        n_heads=args.n_heads,
        device=device,
        dtype=dtype,
        modulation=args.modulation,
    )
    x = inputs["x"]
    attn_mask = inputs["attn_mask"]
    freqs_cis = inputs["freqs_cis"]
    adaln_input = inputs["adaln_input"]

    backend = _parse_backend(args.sdpa_backend)
    sdpa_context = (
        torch.nn.attention.sdpa_kernel(backend) if backend is not None else contextlib.nullcontext()
    )

    result = {
        "device": str(device),
        "dtype": str(dtype),
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "dim": args.dim,
        "n_heads": args.n_heads,
        "n_kv_heads": args.n_kv_heads,
        "sdpa_backend": args.sdpa_backend,
        "modulation": args.modulation,
        "compile": args.compile,
        "compile_mode": args.compile_mode if args.compile else None,
        "triton_rope_enabled": os.environ.get("CUTEZIMAGE_USE_TRITON_ROPE") == "1",
    }

    with torch.no_grad(), sdpa_context:
        result["attention_only"] = _time_call(
            lambda: block._apply_attention(x, attn_mask, freqs_cis),
            device,
            args.warmup,
            args.runs,
        )

        ffn_input = block.ffn_norm1(x)
        result["ffn_only"] = _time_call(
            lambda: block.feed_forward(ffn_input),
            device,
            args.warmup,
            args.runs,
        )

        if args.modulation:
            result["full_block"] = _time_call(
                lambda: block(x, attn_mask=attn_mask, freqs_cis=freqs_cis, adaln_input=adaln_input),
                device,
                args.warmup,
                args.runs,
            )
        else:
            result["full_block"] = _time_call(
                lambda: block(x, attn_mask=attn_mask, freqs_cis=freqs_cis),
                device,
                args.warmup,
                args.runs,
            )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
