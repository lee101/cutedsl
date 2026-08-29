from __future__ import annotations

import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COSYVOICE_ROOT = (REPO_ROOT.parent / "CosyVoice").resolve()

T = TypeVar("T")


@dataclass(frozen=True)
class CosyVoicePaths:
    root: Path = DEFAULT_COSYVOICE_ROOT
    model_dir: str = "pretrained_models/Fun-CosyVoice3-0.5B"
    source: str = "asset/cross_lingual_prompt.wav"
    prompt: str = "asset/zero_shot_prompt.wav"
    reference_tensor: Path = REPO_ROOT / "results" / "cosyvoice_vc_reference.pt"

    def resolved_model_dir(self) -> str:
        path = Path(self.model_dir)
        return str(path if path.is_absolute() else self.root / path)

    def resolved_source(self) -> str:
        path = Path(self.source)
        return str(path if path.is_absolute() else self.root / path)

    def resolved_prompt(self) -> str:
        path = Path(self.prompt)
        return str(path if path.is_absolute() else self.root / path)


def configure_inductor_env() -> None:
    if os.environ.get("CUDA_HOME") is None and Path("/usr/local/cuda-12.2/bin/nvcc").exists():
        os.environ["CUDA_HOME"] = "/usr/local/cuda-12.2"

    if Path("/usr/include/x86_64-linux-gnu/python3.10/pyconfig.h").exists():
        include_path = os.environ.get("C_INCLUDE_PATH", "")
        include_parts = [part for part in include_path.split(":") if part]
        if "/usr/include/x86_64-linux-gnu" not in include_parts:
            os.environ["C_INCLUDE_PATH"] = ":".join(["/usr/include/x86_64-linux-gnu", *include_parts])

    if Path("/usr/bin/gcc").exists():
        os.environ["CC"] = "/usr/bin/gcc"
        os.environ["CXX"] = "/usr/bin/g++"

    for conda_build_var in ("CONDA_BUILD_SYSROOT", "CFLAGS", "CPPFLAGS", "LDFLAGS"):
        os.environ.pop(conda_build_var, None)


def configure_cosyvoice_imports(cosyvoice_root: str | Path = DEFAULT_COSYVOICE_ROOT) -> Path:
    root = Path(cosyvoice_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"CosyVoice root does not exist: {root}")

    root_str = str(root)
    matcha_str = str(root / "third_party" / "Matcha-TTS")
    for path in (matcha_str, root_str):
        if path not in sys.path:
            sys.path.insert(0, path)
    return root


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn: Callable[[], T]) -> tuple[T, float]:
    sync_cuda()
    start = time.perf_counter()
    value = fn()
    sync_cuda()
    return value, time.perf_counter() - start


def flatten_audio(chunks: list[dict[str, torch.Tensor]]) -> torch.Tensor:
    return torch.cat([chunk["tts_speech"].detach().float().cpu().reshape(-1) for chunk in chunks], dim=0)


def compare_audio(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float | int]:
    n = min(candidate.numel(), reference.numel())
    candidate = candidate[:n]
    reference = reference[:n]
    diff = candidate - reference
    signal_rms = reference.square().mean().sqrt().item()
    diff_rms = diff.square().mean().sqrt().item()
    snr_db = 120.0 if diff_rms == 0 else 20.0 * torch.log10(torch.tensor(signal_rms / diff_rms)).item()
    return {
        "length_delta": int(candidate.numel() - reference.numel()),
        "max_abs": float(diff.abs().max().item()) if n else 0.0,
        "mean_abs": float(diff.abs().mean().item()) if n else 0.0,
        "rms": float(diff_rms),
        "snr_db": float(snr_db),
        "cosine": float(torch.nn.functional.cosine_similarity(candidate, reference, dim=0).item()) if n else 1.0,
    }


def edit_distance(a: list[int], b: list[int]) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, item_a in enumerate(a, 1):
        current = [i]
        for j, item_b in enumerate(b, 1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (item_a != item_b),
            ))
        previous = current
    return previous[-1]


def extract_audio_tokens(cosyvoice, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
    import torchaudio

    with tempfile.NamedTemporaryFile(suffix=".wav") as handle:
        torchaudio.save(handle.name, audio.reshape(1, -1), sample_rate)
        generated_tokens, _ = cosyvoice.frontend._extract_speech_token(handle.name)
    return generated_tokens.detach().cpu().reshape(-1)


def token_metrics(generated_tokens: torch.Tensor, reference_tokens: torch.Tensor) -> dict[str, float | int]:
    src = reference_tokens.detach().cpu().reshape(-1).tolist()
    gen = generated_tokens.detach().cpu().reshape(-1).tolist()
    distance = edit_distance(src, gen)
    denom = max(len(src), 1)
    return {
        "reference_tokens": len(src),
        "generated_tokens": len(gen),
        "edit_distance": distance,
        "token_error_rate": distance / denom,
    }


def content_token_metrics(cosyvoice, audio: torch.Tensor, sample_rate: int, reference_tokens: torch.Tensor) -> dict[str, float | int]:
    return token_metrics(extract_audio_tokens(cosyvoice, audio, sample_rate), reference_tokens)


class MethodTimer:
    def __init__(self, module: object, method_name: str):
        self.module = module
        self.method_name = method_name
        self.original = getattr(module, method_name)
        self.seconds = 0.0
        self.calls = 0

    def __enter__(self) -> "MethodTimer":
        def wrapper(*args, **kwargs):
            sync_cuda()
            start = time.perf_counter()
            value = self.original(*args, **kwargs)
            sync_cuda()
            self.seconds += time.perf_counter() - start
            self.calls += 1
            return value

        setattr(self.module, self.method_name, wrapper)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        setattr(self.module, self.method_name, self.original)
