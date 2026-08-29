from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from cutecosyvoice.model import CosyVoiceVCModel
from cutecosyvoice.runtime import CosyVoicePaths, MethodTimer, compare_audio, configure_inductor_env, flatten_audio, timed


def parse_steps(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep CosyVoice flow Euler steps in one loaded process.")
    parser.add_argument("--cosyvoice-root", default=str(CosyVoicePaths.root))
    parser.add_argument("--model-dir", default=CosyVoicePaths.model_dir)
    parser.add_argument("--source", default=CosyVoicePaths.source)
    parser.add_argument("--prompt", default=CosyVoicePaths.prompt)
    parser.add_argument("--steps", default="10,9,8,7,6,5,4")
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load-trt", action="store_true")
    parser.add_argument("--trt-concurrent", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference-tensor", default=str(CosyVoicePaths.reference_tensor))
    parser.add_argument("--compile-flow-estimator", action="store_true")
    parser.add_argument("--compile-backend", default=None)
    parser.add_argument("--compile-mode", default="reduce-overhead")
    return parser.parse_args()


def run_once(model: CosyVoiceVCModel, source: str, prompt: str):
    model_input, frontend_s = timed(lambda: model.frontend_vc(source, prompt))
    with MethodTimer(model.cosyvoice.model.flow, "inference") as flow_timer, MethodTimer(model.cosyvoice.model.hift, "inference") as hift_timer:
        chunks, model_s = timed(lambda: model.token2wav(model_input))
    return flatten_audio(chunks), frontend_s, model_s, {
        "flow_s": flow_timer.seconds,
        "flow_calls": flow_timer.calls,
        "hift_s": hift_timer.seconds,
        "hift_calls": hift_timer.calls,
    }


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

    from cosyvoice.utils.common import set_all_random_seed

    if args.compile_flow_estimator:
        model.compile_flow_estimator(mode=args.compile_mode, backend=args.compile_backend)

    source = paths.resolved_source()
    prompt = paths.resolved_prompt()

    print(json.dumps({
        "event": "env",
        "python": sys.executable,
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "cosyvoice_root": str(model.root),
        "load_s": model.load_seconds,
        "load_trt": args.load_trt,
        "compiled": args.compile_flow_estimator,
        "compile_backend": args.compile_backend or "inductor",
    }))

    reference_path = paths.reference_tensor
    reference = torch.load(reference_path, map_location="cpu").float().reshape(-1) if reference_path.exists() else None

    for steps in parse_steps(args.steps):
        model.set_flow_steps(steps)
        for warmup_idx in range(args.warmup):
            set_all_random_seed(args.seed)
            audio, frontend_s, model_s, profile = run_once(model, source, prompt)
            print(json.dumps({
                "event": "warmup",
                "steps": steps,
                "iter": warmup_idx,
                "frontend_s": frontend_s,
                "model_s": model_s,
                "audio_s": audio.numel() / model.sample_rate,
                "profile": profile,
            }))

        rows = []
        for iter_idx in range(args.iters):
            set_all_random_seed(args.seed)
            audio, frontend_s, model_s, profile = run_once(model, source, prompt)
            if reference is None:
                reference_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(audio, reference_path)
                reference = audio
                reference_event = "created"
            else:
                reference_event = "compared"
            metrics = compare_audio(audio, reference)
            total_s = frontend_s + model_s
            row = {
                "event": "iter",
                "steps": steps,
                "iter": iter_idx,
                "frontend_s": frontend_s,
                "model_s": model_s,
                "total_s": total_s,
                "audio_s": audio.numel() / model.sample_rate,
                "rtf": total_s / (audio.numel() / model.sample_rate),
                "reference": reference_event,
                "metrics": metrics,
                "profile": profile,
            }
            rows.append(row)
            print(json.dumps(row))

        print(json.dumps({
            "event": "summary",
            "steps": steps,
            "iters": len(rows),
            "rtf_avg": sum(row["rtf"] for row in rows) / len(rows),
            "model_s_avg": sum(row["model_s"] for row in rows) / len(rows),
            "flow_s_avg": sum(row["profile"]["flow_s"] for row in rows) / len(rows),
            "max_abs_max": max(row["metrics"]["max_abs"] for row in rows),
            "snr_db_min": min(row["metrics"]["snr_db"] for row in rows),
            "cosine_min": min(row["metrics"]["cosine"] for row in rows),
        }))


if __name__ == "__main__":
    main()
