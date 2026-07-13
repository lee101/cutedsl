"""No-download feasibility checks for local diffusion benchmarks."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import shutil
from pathlib import Path

from .catalog import ModelSpec


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _available_ram_gib() -> float | None:
    try:
        values = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, raw = line.split(":", 1)
            values[key] = int(raw.strip().split()[0])
        return round(values["MemAvailable"] / 1024**2, 2)
    except (OSError, KeyError, ValueError):
        return None


def _hf_cache_dir() -> Path:
    home = os.getenv("HF_HOME")
    if home:
        return Path(home) / "hub"
    return Path(os.getenv("HUGGINGFACE_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))


def _repo_cache_path(repo_id: str, cache_dir: str | Path | None = None) -> Path:
    root = Path(cache_dir) if cache_dir else _hf_cache_dir()
    return root / f"models--{repo_id.replace('/', '--')}"


def _weights_cached(spec: ModelSpec, repo_path: Path) -> bool:
    """Reject metadata-only and interrupted Hub snapshots as model caches."""
    snapshots = repo_path / "snapshots"
    if not snapshots.exists():
        return False
    if any(repo_path.rglob("*.incomplete")):
        return False
    if spec.loader == "single_file" and spec.source_file:
        return any(path.is_file() for path in snapshots.glob(f"*/{spec.source_file}"))
    weight_suffixes = (".safetensors", ".bin", ".pt", ".pth", ".gguf")
    minimum_bytes = spec.approximate_download_gib * 0.95 * 1024**3
    for revision in snapshots.iterdir():
        if not revision.is_dir():
            continue
        weight_bytes = sum(
            path.stat().st_size
            for path in revision.rglob("*")
            if path.is_file() and path.suffix.lower() in weight_suffixes
        )
        if weight_bytes >= minimum_bytes:
            return True
    return False


def _pipeline_available(name: str) -> tuple[bool, str | None]:
    try:
        module = importlib.import_module("diffusers")
    except Exception as exc:  # optional dependency and binary incompatibilities
        return False, f"diffusers import failed: {type(exc).__name__}: {exc}"
    if not hasattr(module, name):
        return False, f"diffusers {_version('diffusers') or 'unknown'} has no {name}"
    return True, None


def system_snapshot(path: str | Path = ".") -> dict:
    usage = shutil.disk_usage(path)
    cache_dir = _hf_cache_dir()
    try:
        cache_usage = shutil.disk_usage(cache_dir.resolve())
        cache_disk_free_gib = round(cache_usage.free / 1024**3, 2)
    except OSError:
        cache_disk_free_gib = None
    snapshot = {
        "torch": _version("torch"),
        "diffusers": _version("diffusers"),
        "transformers": _version("transformers"),
        "accelerate": _version("accelerate"),
        "disk_free_gib": round(usage.free / 1024**3, 2),
        "cache_disk_free_gib": cache_disk_free_gib,
        "ram_available_gib": _available_ram_gib(),
        "hf_cache": str(_hf_cache_dir()),
        "cuda_available": False,
        "gpu": None,
        "gpu_capability": None,
        "gpu_free_gib": None,
        "gpu_total_gib": None,
    }
    try:
        import torch

        snapshot["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            snapshot.update(
                {
                    "gpu": torch.cuda.get_device_name(0),
                    "gpu_capability": list(torch.cuda.get_device_capability(0)),
                    "gpu_free_gib": round(free / 1024**3, 2),
                    "gpu_total_gib": round(total / 1024**3, 2),
                }
            )
    except Exception as exc:
        snapshot["torch_error"] = f"{type(exc).__name__}: {exc}"
    return snapshot


def inspect_model(
    spec: ModelSpec,
    snapshot: dict | None = None,
    cache_dir: str | Path | None = None,
    allow_download: bool = False,
) -> dict:
    snapshot = snapshot or system_snapshot()
    cached_path = _repo_cache_path(spec.repo_id, cache_dir)
    pipeline_ok, pipeline_error = _pipeline_available(spec.pipeline_class)
    cached = _weights_cached(spec, cached_path)
    required_disk = 0.0 if cached else spec.approximate_download_gib
    cache_disk_free = snapshot.get("cache_disk_free_gib")
    disk_free = cache_disk_free if cache_disk_free is not None else snapshot["disk_free_gib"]
    disk_ok = disk_free >= required_disk + 5.0
    gpu_free = snapshot.get("gpu_free_gib") or 0.0
    full_cuda_fit = bool(snapshot.get("cuda_available") and gpu_free >= spec.approximate_resident_gib + 1.5)
    ram = snapshot.get("ram_available_gib")
    offload_fit = bool(
        snapshot.get("cuda_available") and ram is not None and ram >= spec.approximate_resident_gib + 4.0
    )

    blockers = []
    warnings = []
    if not snapshot.get("cuda_available"):
        blockers.append("CUDA is unavailable")
    if not pipeline_ok:
        blockers.append(pipeline_error)
    if not cached and not allow_download:
        blockers.append("weights are not cached; pass --allow-download after reviewing disk/license requirements")
    if not disk_ok:
        blockers.append("insufficient disk headroom for the estimated download")
    if not full_cuda_fit:
        if offload_fit:
            warnings.append("full CUDA residency is unlikely; use model CPU offload (latency will be much higher)")
        else:
            blockers.append("neither estimated CUDA residency nor CPU offload fits current memory headroom")
    if spec.key == "krea2-turbo" and not pipeline_ok:
        warnings.append(
            "install Diffusers from source in an isolated environment: pip install git+https://github.com/huggingface/diffusers.git"
        )
    if spec.notes:
        warnings.append(spec.notes)

    recommended_offload = "cuda" if full_cuda_fit else "model" if offload_fit else None
    return {
        "model": spec.to_dict(),
        "cached": cached,
        "cache_path": str(cached_path),
        "pipeline_available": pipeline_ok,
        "download_required_gib": required_disk,
        "full_cuda_fit": full_cuda_fit,
        "cpu_offload_fit": offload_fit,
        "recommended_offload": recommended_offload,
        "ready": not blockers,
        "blockers": blockers,
        "warnings": warnings,
    }
