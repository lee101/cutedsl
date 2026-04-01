"""Benchmark NVIDIA Parakeet ASR configurations.

This is focused on reproducible throughput / latency sweeps while keeping
transcript drift visible via WER / CER against either ground truth or the
first experiment in the sweep.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import statistics
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
DEFAULT_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v2"


def normalize_text(text: str) -> str:
    """Normalize text before computing transcript quality metrics."""
    normalized = text.lower().strip()
    normalized = re.sub(r"[^\w\s]", "", normalized)
    return " ".join(normalized.split())


def _edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    curr = [0] * (len(b) + 1)
    for i, token_a in enumerate(a, start=1):
        curr[0] = i
        for j, token_b in enumerate(b, start=1):
            substitution = 0 if token_a == token_b else 1
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + substitution,
            )
        prev, curr = curr, prev
    return prev[-1]


def compute_word_error_rate(reference: str, hypothesis: str) -> float:
    ref_tokens = normalize_text(reference).split()
    hyp_tokens = normalize_text(hypothesis).split()
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0
    return _edit_distance(ref_tokens, hyp_tokens) / len(ref_tokens)


def compute_char_error_rate(reference: str, hypothesis: str) -> float:
    ref_chars = list(normalize_text(reference).replace(" ", ""))
    hyp_chars = list(normalize_text(hypothesis).replace(" ", ""))
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    return _edit_distance(ref_chars, hyp_chars) / len(ref_chars)


def collect_audio_files(audio_files: Sequence[str], audio_dir: str | None) -> list[Path]:
    candidates = [Path(path).expanduser().resolve() for path in audio_files]

    if audio_dir:
        directory = Path(audio_dir).expanduser().resolve()
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
                candidates.append(path)

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return deduped


def _load_reference_map(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {str(key): str(value) for key, value in data.items()}


def _resample_audio(audio: np.ndarray, sample_rate: int, target_rate: int = 16000) -> np.ndarray:
    if sample_rate == target_rate:
        return audio.astype(np.float32, copy=False)

    duration = len(audio) / float(sample_rate)
    if duration <= 0.0:
        return np.zeros((0,), dtype=np.float32)

    source_times = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    target_length = max(1, int(round(duration * target_rate)))
    target_times = np.linspace(0.0, duration, num=target_length, endpoint=False)
    return np.interp(target_times, source_times, audio).astype(np.float32)


def _load_audio_arrays(paths: Sequence[Path], target_rate: int = 16000) -> list[np.ndarray]:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("soundfile is required for --input-mode array") from exc

    loaded: list[np.ndarray] = []
    for path in paths:
        audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        loaded.append(_resample_audio(audio, sample_rate=sample_rate, target_rate=target_rate))
    return loaded


def _download_urls(urls: Sequence[str]) -> list[Path]:
    downloaded: list[Path] = []
    for url in urls:
        suffix = Path(urllib.parse.urlparse(url).path).suffix or ".wav"
        fd, path = tempfile.mkstemp(prefix="cutedsl-parakeet-", suffix=suffix)
        os.close(fd)
        urllib.request.urlretrieve(url, path)
        downloaded.append(Path(path))
    return downloaded


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _extract_text(entry: Any) -> str:
    if isinstance(entry, tuple) and len(entry) == 1:
        entry = entry[0]
    if isinstance(entry, str):
        return entry
    for key in ("text", "pred_text"):
        value = getattr(entry, key, None)
        if isinstance(value, str):
            return value
        if isinstance(entry, dict) and isinstance(entry.get(key), str):
            return entry[key]
    return str(entry)


def _extract_segments(entry: Any) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    if isinstance(entry, tuple) and len(entry) == 1:
        entry = entry[0]
    if entry is None:
        return segments

    raw = None
    for key in ("segments", "items", "timestamp"):
        value = getattr(entry, key, None) if not isinstance(entry, dict) else entry.get(key)
        if key == "timestamp" and isinstance(value, dict):
            value = value.get("segment")
        if isinstance(value, (list, tuple)):
            raw = value
            break

    if raw is None:
        return segments

    for segment in raw:
        if not isinstance(segment, dict):
            continue
        segments.append(
            {
                "start": float(segment.get("start", 0.0) or 0.0),
                "end": float(segment.get("end", 0.0) or 0.0),
                "text": str(segment.get("text", "") or segment.get("segment", "")).strip(),
            }
        )
    return segments


def _safe_mean(values: Iterable[float]) -> float | None:
    items = [value for value in values if value is not None and not math.isnan(value)]
    if not items:
        return None
    return sum(items) / len(items)


def _get_audio_durations(paths: Sequence[Path]) -> list[float | None]:
    try:
        import soundfile as sf
    except ImportError:
        return [None for _ in paths]

    durations: list[float | None] = []
    for path in paths:
        try:
            info = sf.info(str(path))
        except RuntimeError:
            durations.append(None)
            continue
        durations.append(float(info.frames) / float(info.samplerate) if info.samplerate else None)
    return durations


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    batch_size: int
    timestamps: bool
    input_mode: str
    amp_dtype: str
    compile_targets: tuple[str, ...]


class ParakeetRunner:
    def __init__(
        self,
        model_id: str,
        device: str,
        allow_tf32: bool,
        torch_compile_mode: str,
    ) -> None:
        self.model_id = model_id
        self.device = torch.device(device)
        self.allow_tf32 = allow_tf32
        self.torch_compile_mode = torch_compile_mode
        self.model = None

    def load(self) -> Any:
        if self.model is not None:
            return self.model

        try:
            import nemo.collections.asr as nemo_asr
        except ImportError as exc:
            raise RuntimeError(
                "NeMo ASR is not installed. Install with: uv pip install -e '.[parakeet]'"
            ) from exc

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = self.allow_tf32
            torch.backends.cudnn.allow_tf32 = self.allow_tf32

        model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.model_id)
        model.freeze()
        model.eval()
        model = model.to(self.device)
        self.model = model
        return model

    def maybe_compile(self, compile_targets: Sequence[str]) -> list[str]:
        model = self.load()
        compiled: list[str] = []
        if not compile_targets:
            return compiled

        for target in compile_targets:
            module = getattr(model, target, None)
            if module is None or not isinstance(module, torch.nn.Module):
                continue
            try:
                compiled_module = torch.compile(module, mode=self.torch_compile_mode, fullgraph=False)
            except Exception:
                continue
            setattr(model, target, compiled_module)
            compiled.append(target)
        return compiled

    def transcribe(
        self,
        inputs: Sequence[str] | Sequence[np.ndarray],
        batch_size: int,
        timestamps: bool,
        amp_dtype: str,
    ) -> tuple[list[str], list[list[dict[str, Any]]]]:
        model = self.load()
        kwargs: dict[str, Any] = {"batch_size": batch_size}
        if timestamps:
            kwargs["timestamps"] = True

        autocast_enabled = self.device.type == "cuda" and amp_dtype in {"fp16", "bf16"}
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(amp_dtype)

        with torch.inference_mode():
            with torch.autocast(device_type=self.device.type, dtype=dtype, enabled=autocast_enabled):
                try:
                    outputs = model.transcribe(inputs, **kwargs)
                except TypeError:
                    kwargs.pop("timestamps", None)
                    outputs = model.transcribe(inputs, **kwargs)

        texts = [_extract_text(item) for item in outputs]
        segments = [_extract_segments(item) for item in outputs]
        return texts, segments


def _chunked(sequence: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for index in range(0, len(sequence), size):
        yield sequence[index : index + size]


def _run_single_experiment(
    runner: ParakeetRunner,
    config: ExperimentConfig,
    audio_paths: Sequence[Path],
    warmup: int,
    repeats: int,
    reference_map: dict[str, str],
) -> dict[str, Any]:
    compiled_targets = runner.maybe_compile(config.compile_targets)

    if config.input_mode == "array":
        prepared_inputs = _load_audio_arrays(audio_paths)
        audio_durations = [len(audio) / 16000.0 for audio in prepared_inputs]
    else:
        prepared_inputs = [str(path) for path in audio_paths]
        audio_durations = _get_audio_durations(audio_paths)

    def _call() -> tuple[list[str], list[list[dict[str, Any]]]]:
        texts: list[str] = []
        segments: list[list[dict[str, Any]]] = []
        for batch in _chunked(prepared_inputs, config.batch_size):
            batch_texts, batch_segments = runner.transcribe(
                batch,
                batch_size=config.batch_size,
                timestamps=config.timestamps,
                amp_dtype=config.amp_dtype,
            )
            texts.extend(batch_texts)
            segments.extend(batch_segments)
        return texts, segments

    first_call_ms: float | None = None
    first_call_texts: list[str] = []
    first_call_segments: list[list[dict[str, Any]]] = []
    if warmup > 0:
        for warmup_index in range(warmup):
            gc.collect()
            _sync(runner.device)
            start = time.perf_counter()
            texts, current_segments = _call()
            _sync(runner.device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if warmup_index == 0:
                first_call_ms = elapsed_ms
                first_call_texts = texts
                first_call_segments = current_segments

    latencies_ms: list[float] = []
    transcripts: list[str] = first_call_texts
    segments: list[list[dict[str, Any]]] = first_call_segments
    for _ in range(repeats):
        gc.collect()
        _sync(runner.device)
        start = time.perf_counter()
        texts, current_segments = _call()
        _sync(runner.device)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
        if not transcripts:
            transcripts = texts
            segments = current_segments

    per_file: list[dict[str, Any]] = []
    wers: list[float] = []
    cers: list[float] = []
    exact_matches = 0
    total_audio_seconds = 0.0

    for path, text, segment_list, audio_duration in zip(audio_paths, transcripts, segments, audio_durations):
        ref = reference_map.get(path.name)
        if ref is not None:
            wer = compute_word_error_rate(ref, text)
            cer = compute_char_error_rate(ref, text)
            wers.append(wer)
            cers.append(cer)
            if normalize_text(ref) == normalize_text(text):
                exact_matches += 1
        else:
            wer = None
            cer = None

        duration_s = audio_duration
        if segment_list:
            duration_s = max(duration_s or 0.0, max((segment.get("end", 0.0) for segment in segment_list), default=0.0))
        total_audio_seconds += duration_s or 0.0

        per_file.append(
            {
                "file": str(path),
                "reference": ref,
                "transcript": text,
                "wer": wer,
                "cer": cer,
                "segment_count": len(segment_list),
                "duration_s": duration_s,
            }
        )

    avg_latency_ms = statistics.mean(latencies_ms)
    p50_latency_ms = statistics.median(latencies_ms)
    sorted_latencies = sorted(latencies_ms)
    p95_index = min(len(sorted_latencies) - 1, max(0, math.ceil(len(sorted_latencies) * 0.95) - 1))
    rtfx = (total_audio_seconds / (avg_latency_ms / 1000.0)) if total_audio_seconds > 0 else None

    return {
        "experiment": asdict(config),
        "compiled_targets": compiled_targets,
        "latency_ms": {
            "first_call": first_call_ms,
            "avg": avg_latency_ms,
            "min": min(latencies_ms),
            "max": max(latencies_ms),
            "p50": p50_latency_ms,
            "p95": sorted_latencies[p95_index],
        },
        "throughput": {
            "audio_seconds": total_audio_seconds,
            "rtfx": rtfx,
            "files_per_second": (len(audio_paths) / (avg_latency_ms / 1000.0)) if avg_latency_ms > 0 else None,
        },
        "quality": {
            "reference_count": len(wers),
            "avg_wer": _safe_mean(wers),
            "avg_cer": _safe_mean(cers),
            "exact_match_rate": (exact_matches / len(wers)) if wers else None,
        },
        "per_file": per_file,
    }


def _build_experiments(args: argparse.Namespace) -> list[ExperimentConfig]:
    experiments: list[ExperimentConfig] = []
    for batch_size in args.batch_sizes:
        for timestamps in args.timestamps:
            for input_mode in args.input_modes:
                for amp_dtype in args.amp_dtypes:
                    for compile_raw in args.compile_targets:
                        compile_targets = tuple(part for part in compile_raw.split(",") if part and part != "none")
                        name = (
                            f"bs{batch_size}-"
                            f"{'ts' if timestamps else 'no-ts'}-"
                            f"{input_mode}-"
                            f"{amp_dtype}-"
                            f"{compile_raw}"
                        )
                        experiments.append(
                            ExperimentConfig(
                                name=name,
                                batch_size=batch_size,
                                timestamps=timestamps,
                                input_mode=input_mode,
                                amp_dtype=amp_dtype,
                                compile_targets=compile_targets,
                            )
                        )
    return experiments


def _print_summary(results: Sequence[dict[str, Any]]) -> None:
    for result in results:
        quality = result["quality"]
        latency = result["latency_ms"]
        throughput = result["throughput"]
        print(
            json.dumps(
                {
                    "name": result["experiment"]["name"],
                    "first_call_ms": round(latency["first_call"], 2) if latency["first_call"] is not None else None,
                    "avg_ms": round(latency["avg"], 2),
                    "p95_ms": round(latency["p95"], 2),
                    "rtfx": round(throughput["rtfx"], 2) if throughput["rtfx"] is not None else None,
                    "avg_wer": round(quality["avg_wer"], 4) if quality["avg_wer"] is not None else None,
                    "avg_cer": round(quality["avg_cer"], 4) if quality["avg_cer"] is not None else None,
                    "compiled_targets": result["compiled_targets"],
                }
            )
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Parakeet ASR experiment sweeps")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--audio-file", action="append", default=[], help="Audio file to transcribe; repeatable")
    parser.add_argument("--audio-dir", default=None, help="Directory of audio files to benchmark")
    parser.add_argument("--audio-url", action="append", default=[], help="Download and benchmark a remote audio file")
    parser.add_argument("--reference-json", default=None, help="JSON mapping filename to reference text")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--timestamps", nargs="+", choices=["on", "off"], default=["off", "on"])
    parser.add_argument("--input-modes", nargs="+", choices=["path", "array"], default=["path", "array"])
    parser.add_argument("--amp-dtypes", nargs="+", choices=["none", "bf16", "fp16"], default=["none", "bf16"])
    parser.add_argument(
        "--compile-targets",
        nargs="+",
        default=["none"],
        help="Comma-separated module attrs to torch.compile, e.g. encoder or encoder,decoder or none",
    )
    parser.add_argument("--torch-compile-mode", default="reduce-overhead")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--output-json", default=None)
    parser.add_argument(
        "--sort-by",
        choices=["avg_ms", "avg_wer", "rtfx"],
        default="avg_ms",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    downloaded = _download_urls(args.audio_url)
    try:
        audio_paths = collect_audio_files([*args.audio_file, *[str(path) for path in downloaded]], args.audio_dir)
        if not audio_paths:
            print("No audio inputs provided.", file=sys.stderr)
            return 2

        reference_map = _load_reference_map(args.reference_json)
        timestamp_flags = [value == "on" for value in args.timestamps]
        args.timestamps = timestamp_flags

        experiments = _build_experiments(args)
        runner = ParakeetRunner(
            model_id=args.model_id,
            device=args.device,
            allow_tf32=args.allow_tf32,
            torch_compile_mode=args.torch_compile_mode,
        )

        results = [
            _run_single_experiment(
                runner=runner,
                config=experiment,
                audio_paths=audio_paths,
                warmup=args.warmup,
                repeats=args.repeats,
                reference_map=reference_map,
            )
            for experiment in experiments
        ]

        if args.sort_by == "avg_wer":
            results.sort(key=lambda item: float("inf") if item["quality"]["avg_wer"] is None else item["quality"]["avg_wer"])
        elif args.sort_by == "rtfx":
            results.sort(key=lambda item: item["throughput"]["rtfx"] or 0.0, reverse=True)
        else:
            results.sort(key=lambda item: item["latency_ms"]["avg"])

        summary = {
            "model_id": args.model_id,
            "device": args.device,
            "audio_files": [str(path) for path in audio_paths],
            "results": results,
        }

        _print_summary(results)
        if args.output_json:
            with open(args.output_json, "w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2)
        return 0
    finally:
        for path in downloaded:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
