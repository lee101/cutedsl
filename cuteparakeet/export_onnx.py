"""Export Parakeet subnets to ONNX and optionally simplify them."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch


DEFAULT_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v2"


def _load_model(model_id: str, device: str) -> Any:
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)
    model.freeze()
    model.eval()
    return model.to(device)


def _dummy_input(batch_size: int, audio_seconds: float, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    num_samples = int(round(audio_seconds * 16000))
    audio_signal = torch.randn(batch_size, num_samples, device=device, dtype=torch.float32)
    length = torch.full((batch_size,), num_samples, device=device, dtype=torch.int64)
    return audio_signal, length


def _maybe_simplify(path: Path) -> dict[str, Any]:
    try:
        import onnx
        from onnxsim import simplify
    except ImportError as exc:
        return {"simplified": False, "reason": f"onnxsim unavailable: {exc}"}

    model = onnx.load(str(path))
    start = time.perf_counter()
    simplified, check = simplify(model)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if not check:
        return {"simplified": False, "reason": "onnxsim check returned false", "elapsed_ms": elapsed_ms}

    simplified_path = path.with_name(f"{path.stem}.sim{path.suffix}")
    onnx.save(simplified, str(simplified_path))
    return {
        "simplified": True,
        "elapsed_ms": elapsed_ms,
        "output": str(simplified_path),
        "size_bytes": simplified_path.stat().st_size,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Parakeet ONNX subnets")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--audio-seconds", type=float, default=4.0)
    parser.add_argument("--output-dir", default="/tmp/parakeet_onnx")
    parser.add_argument("--simplify", action="store_true")
    parser.add_argument("--use-dynamo", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = _load_model(args.model_id, args.device)
    input_example = _dummy_input(args.batch_size, args.audio_seconds, args.device)
    output_base = output_dir / "parakeet.onnx"

    start = time.perf_counter()
    exported, descriptions = model.export(
        str(output_base),
        input_example=input_example,
        check_trace=False,
        use_dynamo=args.use_dynamo,
    )
    export_elapsed_ms = (time.perf_counter() - start) * 1000.0

    outputs: list[dict[str, Any]] = []
    for output_path, description in zip(exported, descriptions):
        path = Path(output_path)
        item = {
            "path": str(path),
            "description": description,
            "size_bytes": path.stat().st_size if path.exists() else None,
        }
        if args.simplify and path.exists():
            item["onnxsim"] = _maybe_simplify(path)
        outputs.append(item)

    print(
        json.dumps(
            {
                "model_id": args.model_id,
                "device": args.device,
                "batch_size": args.batch_size,
                "audio_seconds": args.audio_seconds,
                "use_dynamo": args.use_dynamo,
                "export_elapsed_ms": export_elapsed_ms,
                "outputs": outputs,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
