from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from cutecosyvoice.model import CosyVoiceVCModel
from cutecosyvoice.runtime import (
    CosyVoicePaths,
    MethodTimer,
    compare_audio,
    configure_inductor_env,
    content_token_metrics,
    extract_audio_tokens,
    flatten_audio,
    timed,
)


def run_split(cosyvoice, source: str, prompt: str, stream: bool, speed: float):
    model_input, frontend_s = timed(lambda: cosyvoice.frontend.frontend_vc(source, prompt, cosyvoice.sample_rate))
    chunks, model_s = timed(lambda: list(cosyvoice.model.tts(**model_input, stream=stream, speed=speed)))
    return model_input, chunks, frontend_s, model_s


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark external CosyVoice VC from the CuteDSL workspace.")
    parser.add_argument("--cosyvoice-root", default=str(CosyVoicePaths.root))
    parser.add_argument("--model-dir", default=CosyVoicePaths.model_dir)
    parser.add_argument("--source", default=CosyVoicePaths.source)
    parser.add_argument("--prompt", default=CosyVoicePaths.prompt)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load-trt", action="store_true")
    parser.add_argument("--trt-concurrent", type=int, default=1)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--reference-tensor", default=str(CosyVoicePaths.reference_tensor))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile-model", action="store_true")
    parser.add_argument("--compile-flow-estimator", action="store_true")
    parser.add_argument("--compile-backend", default=None)
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--flow-steps", type=int, default=None)
    parser.add_argument("--hift-f0-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--content-token-eval", action="store_true")
    parser.add_argument("--skip-alltrue-dit-mask", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_inductor_env()
    args = parse_args()

    paths = CosyVoicePaths(
        root=Path(args.cosyvoice_root).expanduser().resolve(),
        model_dir=args.model_dir,
        source=args.source,
        prompt=args.prompt,
        reference_tensor=Path(args.reference_tensor),
    )
    model = CosyVoiceVCModel(paths=paths, fp16=args.fp16, load_trt=args.load_trt, trt_concurrent=args.trt_concurrent)
    cosyvoice = model.cosyvoice

    from cosyvoice.utils.common import set_all_random_seed

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(json.dumps({
        "event": "env",
        "python": sys.executable,
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "cuda_home": __import__("os").environ.get("CUDA_HOME"),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "cosyvoice_root": str(model.root),
        "load_trt": args.load_trt,
    }))

    print(json.dumps({"event": "load", "seconds": model.load_seconds}))

    if args.flow_steps is not None:
        model.set_flow_steps(args.flow_steps)
        print(json.dumps({"event": "flow_steps", "steps": args.flow_steps}))

    if args.hift_f0_device != "cpu":
        model.set_hift_f0_device(args.hift_f0_device)
        print(json.dumps({"event": "hift_f0_device", "device": args.hift_f0_device}))

    if args.skip_alltrue_dit_mask:
        model.set_skip_alltrue_dit_mask(True)
        print(json.dumps({"event": "skip_alltrue_dit_mask", "enabled": True}))

    if args.compile_flow_estimator:
        if isinstance(model.flow_estimator, torch.nn.Module):
            model.compile_flow_estimator(mode=args.compile_mode, backend=args.compile_backend)
            print(json.dumps({
                "event": "compile_enabled",
                "target": "flow.decoder.estimator",
                "backend": args.compile_backend or "inductor",
                "mode": args.compile_mode,
            }))
        else:
            print(json.dumps({"event": "compile_skipped", "reason": "flow decoder estimator is not a torch module"}))

    source = paths.resolved_source()
    prompt = paths.resolved_prompt()
    for idx in range(args.warmup):
        set_all_random_seed(args.seed)
        _, chunks, frontend_s, model_s = run_split(cosyvoice, source, prompt, args.stream, args.speed)
        audio = flatten_audio(chunks)
        print(json.dumps({
            "event": "warmup",
            "iter": idx,
            "frontend_s": frontend_s,
            "model_s": model_s,
            "audio_s": audio.numel() / cosyvoice.sample_rate,
        }))

    reference_path = paths.reference_tensor
    reference = torch.load(reference_path, map_location="cpu") if reference_path.exists() else None
    if reference is not None:
        reference = reference.float().reshape(-1)
    reference_token_path = reference_path.with_suffix(reference_path.suffix + ".tokens.pt")
    reference_tokens = torch.load(reference_token_path, map_location="cpu") if reference_token_path.exists() else None

    rows = []
    for idx in range(args.iters):
        set_all_random_seed(args.seed)
        if args.profile_model:
            with MethodTimer(cosyvoice.model.flow, "inference") as flow_timer, MethodTimer(cosyvoice.model.hift, "inference") as hift_timer:
                model_input, chunks, frontend_s, model_s = run_split(cosyvoice, source, prompt, args.stream, args.speed)
            profile = {
                "flow_s": flow_timer.seconds,
                "flow_calls": flow_timer.calls,
                "hift_s": hift_timer.seconds,
                "hift_calls": hift_timer.calls,
            }
        else:
            model_input, chunks, frontend_s, model_s = run_split(cosyvoice, source, prompt, args.stream, args.speed)
            profile = {}

        audio = flatten_audio(chunks)
        if reference is None:
            reference_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(audio, reference_path)
            reference = audio
            metrics = compare_audio(audio, reference)
            reference_event = "created"
        else:
            metrics = compare_audio(audio, reference)
            reference_event = "compared"
        content_metrics = {}
        if args.content_token_eval:
            if reference_tokens is None:
                reference_tokens = extract_audio_tokens(cosyvoice, reference, cosyvoice.sample_rate)
                reference_token_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(reference_tokens, reference_token_path)
            content_metrics = content_token_metrics(cosyvoice, audio, cosyvoice.sample_rate, reference_tokens)

        audio_s = audio.numel() / cosyvoice.sample_rate
        total_s = frontend_s + model_s
        row = {
            "event": "iter",
            "iter": idx,
            "frontend_s": frontend_s,
            "model_s": model_s,
            "total_s": total_s,
            "audio_s": audio_s,
            "rtf": total_s / audio_s,
            "reference": reference_event,
            "metrics": metrics,
            "content_metrics": content_metrics,
            "profile": profile,
        }
        rows.append(row)
        print(json.dumps(row))

    if rows:
        print(json.dumps({
            "event": "summary",
            "iters": len(rows),
            "frontend_s_avg": sum(row["frontend_s"] for row in rows) / len(rows),
            "model_s_avg": sum(row["model_s"] for row in rows) / len(rows),
            "total_s_avg": sum(row["total_s"] for row in rows) / len(rows),
            "rtf_avg": sum(row["rtf"] for row in rows) / len(rows),
            "max_abs_max": max(row["metrics"]["max_abs"] for row in rows),
            "snr_db_min": min(row["metrics"]["snr_db"] for row in rows),
        }))


if __name__ == "__main__":
    main()
