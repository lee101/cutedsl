"""
Build the cutezimage CUDA C extension (cutezimage._C).

Targets (fat binary – all arches compiled in one pass):
  sm_80  – A100
  sm_86  – A40, RTX 3090, A6000
  sm_89  – L40, RTX 4090
  sm_90  – H100, H200
  sm_100 – H200 NVL  (CUDA 12.6+)
  sm_120 – RTX 5090, GB200  (CUDA 12.8+)

Usage (from repo root):
    pip install cutezimage/csrc/   [installs in-place]
  or:
    python cutezimage/csrc/setup_cuda.py build_ext --inplace
  or (uv):
    uv pip install -e cutezimage/csrc/
"""

import os
import subprocess
import sys
from pathlib import Path

from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

HERE = Path(__file__).resolve().parent

# Embed torch lib path in the .so RPATH so it loads without LD_LIBRARY_PATH tricks.
_TORCH_LIB = str(Path(torch.__file__).parent / "lib")

# ---------------------------------------------------------------------------
# Detect CUDA version and trim unsupported arches
# ---------------------------------------------------------------------------

def _cuda_version() -> tuple[int, int]:
    try:
        out = subprocess.check_output(
            ["nvcc", "--version"], stderr=subprocess.STDOUT, text=True
        )
        for tok in out.split():
            if tok.startswith("V"):
                parts = tok[1:].split(".")
                return int(parts[0]), int(parts[1])
    except Exception:
        pass
    return (12, 0)


_CUDA_MAJOR, _CUDA_MINOR = _cuda_version()
_CUDA_VER = _CUDA_MAJOR * 100 + _CUDA_MINOR


def _gencode(*arches: tuple[int, int]) -> list[str]:
    """Return -gencode flags for the given (major, minor) SM versions."""
    flags: list[str] = []
    for maj, mn in arches:
        sm = f"sm_{maj}{mn}"
        compute = f"compute_{maj}{mn}"
        flags += ["-gencode", f"arch={compute},code={sm}"]
    return flags


# Always include baseline arches; add newer ones when CUDA supports them.
_ARCHES: list[tuple[int, int]] = [(8, 0), (8, 6), (8, 9), (9, 0)]
if _CUDA_VER >= 1206:
    _ARCHES.append((10, 0))
if _CUDA_VER >= 1208:
    _ARCHES.append((12, 0))

NVCC_FLAGS = [
    "-O3",
    "--use_fast_math",
    "-std=c++17",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    # Reduce register pressure for high-occupancy kernels
    "--maxrregcount=64",
    # Verbose PTX stats (useful when profiling)
    "--ptxas-options=-v",
] + _gencode(*_ARCHES)

# Let torch handle include dirs; add csrc/ so headers resolve.
EXTRA_INCLUDE = [str(HERE)]

setup(
    name="cutezimage_cuda",
    version="0.1.0",
    description="CuteZImage CUDA kernels (rms_norm, silu_gate, qk_norm)",
    ext_modules=[
        CUDAExtension(
            name="cutezimage._C",
            sources=[
                str(HERE / "cute_kernels_ext.cpp"),
                str(HERE / "cute_rms_norm.cu"),
                str(HERE / "cute_silu_gate.cu"),
                str(HERE / "cute_qk_norm.cu"),
            ],
            include_dirs=EXTRA_INCLUDE,
            extra_compile_args={
                "cxx":  ["-O3", "-std=c++17"],
                "nvcc": NVCC_FLAGS,
            },
            extra_link_args=[f"-Wl,-rpath,{_TORCH_LIB}"],
            # Link against cublas for potential future CUTLASS use
            libraries=["cublas"],
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
    zip_safe=False,
)
