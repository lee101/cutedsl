"""End-to-end benchmark comparing CuteZImageTransformer vs diffusers ZImageTransformer2DModel.

Usage::

    # Transformer-level comparison (synthetic inputs)
    python -m cutezimage.benchmark

    # Full pipeline comparison with a specific prompt
    python -m cutezimage.benchmark --prompt "anime fairy cute winged fairy" --pipeline

    # Custom model path
    python -m cutezimage.benchmark --model-id Tongyi-MAI/Z-Image-Turbo --prompt "anime fairy cute winged fairy"

Measures latency, peak GPU memory, and output matching (max abs error).
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def measure_gpu_memory() -> float:
    """Return peak GPU memory allocated in MB (0 if no CUDA)."""
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def reset_gpu_memory_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()


# ---------------------------------------------------------------------------
# Transformer-level benchmark (synthetic inputs)
# ---------------------------------------------------------------------------

def create_synthetic_inputs(
    batch_size: int = 1,
    height: int = 128,
    width: int = 128,
    in_channels: int = 16,
    cap_feat_dim: int = 2560,
    cap_seq_len: int = 77,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
):
    """Create synthetic inputs matching Z-Image pipeline format."""
    torch.manual_seed(seed)

    # Latent images: (C, F, H, W) where F=1 for images
    x = [torch.randn(in_channels, 1, height, width, device=device, dtype=dtype) for _ in range(batch_size)]

    # Timesteps
    t = torch.tensor([0.5] * batch_size, device=device, dtype=dtype)

    # Caption features
    cap_feats = [torch.randn(cap_seq_len, cap_feat_dim, device=device, dtype=dtype) for _ in range(batch_size)]

    return x, t, cap_feats


def compare_outputs(
    output_orig: list[torch.Tensor],
    output_cute: list[torch.Tensor],
) -> dict:
    """Compare two lists of output tensors."""
    max_err = 0.0
    mean_err = 0.0
    for orig, cute in zip(output_orig, output_cute):
        diff = (orig.float() - cute.float()).abs()
        max_err = max(max_err, diff.max().item())
        mean_err += diff.mean().item()
    mean_err /= len(output_orig)

    return {
        "max_abs_error": max_err,
        "mean_abs_error": mean_err,
        "matches": max_err < 0.01,
    }


def benchmark_transformer(
    model,
    x: list[torch.Tensor],
    t: torch.Tensor,
    cap_feats: list[torch.Tensor],
    n_warmup: int = 2,
    n_runs: int = 10,
    label: str = "model",
) -> dict:
    """Benchmark a single transformer model, return results dict."""
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x, t, cap_feats, return_dict=False)

    # Timed runs
    reset_gpu_memory_stats()
    latencies = []

    with torch.no_grad():
        for _ in range(n_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x, t, cap_feats, return_dict=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append(t1 - t0)

    peak_mem = measure_gpu_memory()

    return {
        "label": label,
        "avg_latency_ms": float(np.mean(latencies)) * 1000,
        "std_latency_ms": float(np.std(latencies)) * 1000,
        "min_latency_ms": float(np.min(latencies)) * 1000,
        "peak_gpu_memory_mb": peak_mem,
    }


# ---------------------------------------------------------------------------
# Pipeline-level benchmark
# ---------------------------------------------------------------------------

def benchmark_pipeline_e2e(
    pipeline,
    prompt: str,
    width: int,
    height: int,
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    n_warmup: int = 1,
    n_runs: int = 5,
    label: str = "pipeline",
) -> dict:
    """Benchmark a full ZImage pipeline end-to-end."""
    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)

    # Warmup
    for _ in range(n_warmup):
        generator.manual_seed(seed)
        pipeline(
            prompt=prompt, width=width, height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

    # Timed runs
    reset_gpu_memory_stats()
    latencies = []

    for _ in range(n_runs):
        generator.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = pipeline(
            prompt=prompt, width=width, height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies.append(t1 - t0)

    peak_mem = measure_gpu_memory()
    last_image = result.images[0]

    return {
        "label": label,
        "avg_latency_ms": float(np.mean(latencies)) * 1000,
        "std_latency_ms": float(np.std(latencies)) * 1000,
        "min_latency_ms": float(np.min(latencies)) * 1000,
        "peak_gpu_memory_mb": peak_mem,
        "image": last_image,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CuteZImage benchmark")
    parser.add_argument("--model-id", default="Tongyi-MAI/Z-Image-Turbo", help="HF model id or local path")
    parser.add_argument("--prompt", default="anime fairy cute winged fairy",
                        help="Prompt for pipeline-level comparison")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--num-inference-steps", type=int, default=9)
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pipeline", action="store_true", help="Run full pipeline comparison")
    parser.add_argument("--n-warmup", type=int, default=2)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="benchmark_results.json")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile on CuteZImage transformer")
    parser.add_argument("--compile-mode", default="reduce-overhead",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="torch.compile mode (default: reduce-overhead)")
    parser.add_argument(
        "--transformer-variant",
        default="cute",
        choices=["cute", "accelerated"],
        help="Cute transformer implementation to benchmark against diffusers",
    )
    parser.add_argument("--sdpa-backend", default=None,
                        choices=["flash", "math", "efficient", "cudnn"],
                        help="Force specific SDPA backend (default: auto)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    results = {}

    print(f"Model: {args.model_id}")
    print(f"Device: {args.device}")
    print(f"Prompt: {args.prompt}")
    print()

    # ------------------------------------------------------------------
    # 1. Load diffusers model
    # ------------------------------------------------------------------
    print("--- Loading diffusers ZImagePipeline ---")
    try:
        from diffusers import ZImagePipeline
    except ImportError:
        print("ERROR: diffusers with Z-Image support not installed.")
        print("Install with: pip install 'diffusers>=0.36.0' transformers accelerate")
        return

    pipe = ZImagePipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    orig_transformer = pipe.transformer
    print(f"  Loaded. Parameters: {sum(p.numel() for p in orig_transformer.parameters()):,}")

    # ------------------------------------------------------------------
    # 2. Create CuteZImage from diffusers weights
    # ------------------------------------------------------------------
    print(f"\n--- Creating {args.transformer_variant} Z-Image transformer from diffusers weights ---")
    if args.transformer_variant == "accelerated":
        from zimageaccelerated.model import AcceleratedZImageTransformer

        if args.compile:
            cute_transformer = AcceleratedZImageTransformer.from_diffusers_compiled(
                orig_transformer, compile_mode=args.compile_mode
            )
        else:
            cute_transformer = AcceleratedZImageTransformer.from_diffusers(orig_transformer)
    else:
        from cutezimage.model import CuteZImageTransformer

        if args.compile:
            cute_transformer = CuteZImageTransformer.from_diffusers_compiled(
                orig_transformer, compile_mode=args.compile_mode
            )
        else:
            cute_transformer = CuteZImageTransformer.from_diffusers(orig_transformer)
    cute_transformer = cute_transformer.to(args.device, torch.bfloat16)
    print(f"  Created. Parameters: {cute_transformer.parameter_count():,}")

    # Move orig to same device for comparison
    orig_transformer = orig_transformer.to(args.device, torch.bfloat16).eval()

    # ------------------------------------------------------------------
    # 2b. Set up SDPA backend context and compile warmup adjustments
    # ------------------------------------------------------------------
    sdpa_context = contextlib.nullcontext()
    if args.sdpa_backend:
        try:
            backend_map = {
                "flash": torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                "math": torch.nn.attention.SDPBackend.MATH,
                "efficient": torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                "cudnn": torch.nn.attention.SDPBackend.CUDNN_ATTENTION,
            }
            sdpa_context = torch.nn.attention.sdpa_kernel(backend_map[args.sdpa_backend])
            print(f"  SDPA backend: {args.sdpa_backend}")
        except (AttributeError, RuntimeError) as e:
            print(f"  Warning: SDPA backend '{args.sdpa_backend}' not available ({e}), using auto")

    # Auto-increase warmup when torch.compile is enabled
    n_warmup = args.n_warmup
    if args.compile:
        n_warmup = max(args.n_warmup, 3)
        if n_warmup != args.n_warmup:
            print(f"  Auto-increased warmup from {args.n_warmup} to {n_warmup} for torch.compile")

    # ------------------------------------------------------------------
    # 3. Transformer-level comparison (synthetic inputs)
    # ------------------------------------------------------------------
    print("\n--- Transformer-level comparison (synthetic inputs) ---")

    # Get config from cute model
    cfg = cute_transformer.config
    latent_h = args.height // 8  # VAE downscale
    latent_w = args.width // 8

    x_synth, t_synth, cap_synth = create_synthetic_inputs(
        batch_size=1,
        height=latent_h,
        width=latent_w,
        in_channels=cfg.in_channels,
        cap_feat_dim=cfg.cap_feat_dim,
        cap_seq_len=77,
        device=args.device,
        dtype=torch.bfloat16,
        seed=args.seed,
    )

    with sdpa_context:
        # Run both models with identical inputs
        with torch.no_grad():
            out_orig = orig_transformer(x_synth, t_synth, cap_synth, return_dict=False)
            out_cute = cute_transformer(x_synth, t_synth, cap_synth, return_dict=False)

        # Compare outputs
        comparison = compare_outputs(out_orig[0], out_cute[0])
        print(f"  Max absolute error: {comparison['max_abs_error']:.6e}")
        print(f"  Mean absolute error: {comparison['mean_abs_error']:.6e}")
        print(f"  Outputs match (< 0.01): {comparison['matches']}")
        results["output_comparison"] = comparison

        # ------------------------------------------------------------------
        # 4. Latency benchmark
        # ------------------------------------------------------------------
        print("\n--- Latency benchmark (transformer only) ---")

        result_orig = benchmark_transformer(
            orig_transformer, x_synth, t_synth, cap_synth,
            n_warmup=n_warmup, n_runs=args.n_runs, label="original_zimage",
        )
        results["original_zimage"] = result_orig
        print(f"  Original: {result_orig['avg_latency_ms']:.1f} ms "
              f"(std={result_orig['std_latency_ms']:.1f}, min={result_orig['min_latency_ms']:.1f})")

        candidate_label = f"{args.transformer_variant}_zimage"
        result_cute = benchmark_transformer(
            cute_transformer, x_synth, t_synth, cap_synth,
            n_warmup=n_warmup, n_runs=args.n_runs, label=candidate_label,
        )
        results[candidate_label] = result_cute
        print(f"  {args.transformer_variant} ZImage: {result_cute['avg_latency_ms']:.1f} ms "
              f"(std={result_cute['std_latency_ms']:.1f}, min={result_cute['min_latency_ms']:.1f})")

        speedup = result_orig["avg_latency_ms"] / max(result_cute["avg_latency_ms"], 1e-9)
        print(f"  Speedup: {speedup:.2f}x")
        results["transformer_comparison"] = {
            "speedup": speedup,
            "max_abs_error": comparison["max_abs_error"],
            "outputs_match": comparison["matches"],
        }

    # Free transformer-only resources before pipeline test
    del orig_transformer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 5. Full pipeline comparison (optional)
    # ------------------------------------------------------------------
    if args.pipeline:
        print(f"\n--- Full pipeline comparison: \"{args.prompt}\" ---")

        # Benchmark original pipeline
        print("  Running original pipeline...")
        result_orig_pipe = benchmark_pipeline_e2e(
            pipe, args.prompt, args.width, args.height, args.seed,
            args.num_inference_steps, args.guidance_scale,
            n_warmup=1, n_runs=3, label="original_pipeline",
        )
        orig_image = result_orig_pipe.pop("image")
        results["original_pipeline"] = result_orig_pipe
        print(f"  Original pipeline: {result_orig_pipe['avg_latency_ms']:.0f} ms")

        # Replace transformer with CuteZImage
        print(f"  Replacing transformer with {args.transformer_variant} ZImage...")
        pipe.transformer = cute_transformer

        result_cute_pipe = benchmark_pipeline_e2e(
            pipe, args.prompt, args.width, args.height, args.seed,
            args.num_inference_steps, args.guidance_scale,
            n_warmup=1, n_runs=3, label=f"{args.transformer_variant}_pipeline",
        )
        cute_image = result_cute_pipe.pop("image")
        results[f"{args.transformer_variant}_pipeline"] = result_cute_pipe
        print(f"  {args.transformer_variant} ZImage pipeline: {result_cute_pipe['avg_latency_ms']:.0f} ms")

        pipe_speedup = result_orig_pipe["avg_latency_ms"] / max(result_cute_pipe["avg_latency_ms"], 1e-9)
        print(f"  Pipeline speedup: {pipe_speedup:.2f}x")

        # Compare images
        import numpy as np_img
        orig_arr = np_img.array(orig_image).astype(np_img.float32)
        cute_arr = np_img.array(cute_image).astype(np_img.float32)
        pixel_diff = np_img.abs(orig_arr - cute_arr)
        max_pixel_err = float(pixel_diff.max())
        mean_pixel_err = float(pixel_diff.mean())
        print(f"  Max pixel error: {max_pixel_err:.1f}/255")
        print(f"  Mean pixel error: {mean_pixel_err:.4f}/255")

        results["pipeline_comparison"] = {
            "speedup": pipe_speedup,
            "max_pixel_error": max_pixel_err,
            "mean_pixel_error": mean_pixel_err,
            "images_identical": max_pixel_err == 0.0,
        }

        # Save images
        output_dir = repo_root / "benchmark_images"
        output_dir.mkdir(exist_ok=True)
        orig_image.save(output_dir / "original.png")
        cute_image.save(output_dir / f"{args.transformer_variant}.png")
        print(f"  Images saved to {output_dir}/")

    # ------------------------------------------------------------------
    # Config info
    # ------------------------------------------------------------------
    results["config"] = {
        "compile": args.compile,
        "compile_mode": args.compile_mode if args.compile else None,
        "sdpa_backend": args.sdpa_backend or "auto",
        "transformer_variant": args.transformer_variant,
    }

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("BENCHMARK RESULTS")
    print("=" * 72)
    compile_info = f"compile={args.compile}"
    if args.compile:
        compile_info += f" (mode={args.compile_mode})"
    sdpa_info = f"sdpa_backend={args.sdpa_backend or 'auto'}"
    print(f"Config: {compile_info}, {sdpa_info}")
    print()
    header = f"{'Model':<25} {'Latency(ms)':<15} {'GPU(MB)':<15}"
    print(header)
    print("-" * 72)
    for key in ["original_zimage", f"{args.transformer_variant}_zimage"]:
        if key in results:
            res = results[key]
            print(f"{res['label']:<25} {res['avg_latency_ms']:<15.1f} {res['peak_gpu_memory_mb']:<15.1f}")

    if "transformer_comparison" in results:
        tc = results["transformer_comparison"]
        print(f"\nTransformer output max error: {tc['max_abs_error']:.6e}")
        print(f"Transformer speedup: {tc['speedup']:.2f}x")
        print(f"Outputs match: {tc['outputs_match']}")

    # Save results
    output_path = repo_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
