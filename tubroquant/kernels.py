from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import CUDA_HOME, load as load_extension_module

_LOCK = threading.Lock()
_EXTENSION: Any | None = None
_EXTENSION_ATTEMPTED = False


def _kernel_root() -> Path:
    return Path(__file__).resolve().parent / "csrc"


def _should_build_extension(build_extension: bool) -> bool:
    if build_extension:
        return True
    env = os.getenv("TUBROQUANT_BUILD_EXT", "0").strip().lower()
    return env in {"1", "true", "yes", "on"}


def load_tubroquant_extension(*, build_extension: bool = True, verbose: bool = False) -> Any | None:
    """Compile and load the optional TurboQuant CUDA extension."""
    global _EXTENSION
    global _EXTENSION_ATTEMPTED

    if _EXTENSION is not None:
        return _EXTENSION
    if _EXTENSION_ATTEMPTED and _EXTENSION is None:
        return None
    if not _should_build_extension(build_extension):
        return None

    with _LOCK:
        if _EXTENSION is not None:
            return _EXTENSION
        if _EXTENSION_ATTEMPTED and _EXTENSION is None:
            return None

        _EXTENSION_ATTEMPTED = True
        root = _kernel_root()
        cpp_source = root / "tubroquant_ext.cpp"
        cuda_source = root / "qk_scores.cu"
        header_source = root / "qk_scores.cuh"

        if not cpp_source.exists() or not cuda_source.exists() or not header_source.exists():
            return None

        has_cuda = bool(torch.cuda.is_available() and torch.version.cuda and CUDA_HOME is not None)
        if not has_cuda:
            return None

        build_dir = root / "build"
        build_dir.mkdir(parents=True, exist_ok=True)

        try:
            _EXTENSION = load_extension_module(
                name="tubroquant_cuda",
                sources=[str(cpp_source), str(cuda_source)],
                extra_cflags=["-O3", "-std=c++17"],
                extra_cuda_cflags=[
                    "-O3",
                    "--use_fast_math",
                    "-std=c++17",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                ],
                with_cuda=True,
                build_directory=str(build_dir),
                verbose=verbose,
            )
        except Exception as exc:
            print(f"[tubroquant] Extension build failed, falling back to PyTorch: {exc}")
            _EXTENSION = None

        return _EXTENSION
