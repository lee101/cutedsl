"""LoRA hot-swapper for accelerated transformers.

Swap strategy:
- Low-rank factors live in a CPU LRU cache (pinned when CUDA is available) and are
  promoted to a GPU LRU cache on first use.
- Warm swaps take the delta-difference fast path: ``W += Σ new_deltas − Σ old_deltas``
  computed entirely on-device from GPU-cached factors — no host↔device weight traffic.
- Pristine base weights are snapshotted to CPU the first time a parameter is touched;
  every ``exact_restore_every`` fast swaps (and on deactivate/error) weights are reset
  from snapshots so bf16 rounding drift never accumulates.
- In-place ``add_``/``copy_`` bump ``weight._version``, invalidating fused-QKV weight
  caches in CuteZImage automatically.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import OrderedDict

import torch

from cuteloras.formats import LoRAFactors, load_lora_factors
from cuteloras.registry import LoRARecord, LoRARegistry

logger = logging.getLogger("cuteloras")


class LoRASwapper:
    def __init__(
        self,
        transformer: torch.nn.Module,
        registry: LoRARegistry | None = None,
        max_cached_loras: int = 64,
        max_gpu_cached_loras: int = 8,
        exact_restore_every: int = 64,
        pin_factors: bool | None = None,
        pin_snapshots: bool | None = None,
    ):
        self.transformer = transformer
        self.registry = registry or LoRARegistry()
        self.max_cached_loras = max_cached_loras
        self.max_gpu_cached_loras = max_gpu_cached_loras
        self.exact_restore_every = exact_restore_every
        self._factor_cache: OrderedDict[str, LoRAFactors] = OrderedDict()
        self._gpu_factors: OrderedDict[str, dict] = OrderedDict()
        self._snapshots: dict[str, torch.Tensor] = {}
        self._params: dict[str, torch.nn.Parameter] = dict(transformer.named_parameters())
        self._active: tuple[tuple[str, float], ...] = ()
        self._fast_swaps = 0
        self._lock = threading.Lock()
        has_cuda = torch.cuda.is_available()
        self.pin_snapshots = has_cuda if pin_snapshots is None else pin_snapshots
        self.pin_factors = has_cuda if pin_factors is None else pin_factors

    @property
    def active(self) -> tuple[tuple[str, float], ...]:
        return self._active

    def get_factors(self, lora_id: str) -> LoRAFactors:
        cached = self._factor_cache.get(lora_id)
        if cached is not None:
            self._factor_cache.move_to_end(lora_id)
            return cached
        record = self.registry.get(lora_id)
        if record is None:
            raise KeyError(f"unknown LoRA id: {lora_id}")
        path = self.registry.resolve_path(record)
        factors = load_lora_factors(path)
        if self.pin_factors:
            for module_path, (a, b, alpha) in factors.factors.items():
                try:
                    factors.factors[module_path] = (a.pin_memory(), b.pin_memory(), alpha)
                except RuntimeError:
                    break
        missing = [m for m in factors.factors if self._resolve_target(m) is None]
        if missing:
            logger.warning(
                "LoRA %s: %d/%d modules unresolvable on model (e.g. %s)",
                lora_id,
                len(missing),
                len(factors.factors),
                missing[0],
            )
        self._factor_cache[lora_id] = factors
        while len(self._factor_cache) > self.max_cached_loras:
            evicted, _ = self._factor_cache.popitem(last=False)
            self._gpu_factors.pop(evicted, None)
            logger.info("evicted LoRA factors: %s", evicted)
        return factors

    def _get_device_factors(self, lora_id: str, device: torch.device) -> dict:
        if device.type != "cuda":
            return self.get_factors(lora_id).factors
        cached = self._gpu_factors.get(lora_id)
        if cached is not None:
            self._gpu_factors.move_to_end(lora_id)
            return cached
        cpu = self.get_factors(lora_id)
        gpu = {
            module_path: (a.to(device, non_blocking=True), b.to(device, non_blocking=True), alpha)
            for module_path, (a, b, alpha) in cpu.factors.items()
        }
        torch.cuda.synchronize()
        self._gpu_factors[lora_id] = gpu
        while len(self._gpu_factors) > self.max_gpu_cached_loras:
            self._gpu_factors.popitem(last=False)
        return gpu

    def _module_candidates(self, module_path: str):
        yield module_path
        from cuteloras.formats import ZIMAGE_MODULE_MAP

        for raw, cute in ZIMAGE_MODULE_MAP.items():
            if module_path.endswith(f".{raw}"):
                yield f"{module_path[: -len(raw)]}{cute}"
            elif raw != cute and module_path.endswith(f".{cute}"):
                yield f"{module_path[: -len(cute)]}{raw}"

    def _resolve_target(self, module_path: str):
        """Map a logical module path to (param_path, param, row_slice).

        Tries the raw (diffusers) name, the CuteZImage name, a fused qkv_proj slice
        (AcceleratedZImageTransformer), and adaLN Sequential index variants (.0/.1).
        """
        for candidate in self._module_candidates(module_path):
            param = self._params.get(f"{candidate}.weight")
            if param is not None:
                return f"{candidate}.weight", param, None
            m = re.match(r"^(.*)\.([qkv])_proj$", candidate)
            if m:
                base, which = m.groups()
                qkv = self._params.get(f"{base}.qkv_proj.weight")
                if qkv is not None:
                    dim = qkv.shape[1]
                    kv = (qkv.shape[0] - dim) // 2
                    start, rows = {"q": (0, dim), "k": (dim, kv), "v": (dim + kv, kv)}[which]
                    return f"{base}.qkv_proj.weight", qkv, slice(start, start + rows)
            if re.search(r"\.adaLN_modulation\.[01]$", candidate):
                base = candidate.rsplit(".", 1)[0]
                for idx in ("1", "0"):
                    p = self._params.get(f"{base}.{idx}.weight")
                    if p is not None:
                        return f"{base}.{idx}.weight", p, None
        return None

    def _snapshot(self, param_path: str, param: torch.nn.Parameter) -> None:
        if param_path in self._snapshots:
            return
        snap = param.detach().to("cpu", copy=True)
        if self.pin_snapshots and snap.device.type == "cpu":
            try:
                snap = snap.pin_memory()
            except RuntimeError:
                pass
        self._snapshots[param_path] = snap

    @torch.no_grad()
    def _apply_contributions(self, contributions: list[tuple[dict, float]]) -> int:
        """Add ``Σ scale·(B@A)`` per module in one in-place update; snapshots first touch."""
        per_module: dict[str, list[tuple[torch.Tensor, torch.Tensor, float]]] = {}
        for factors, scale in contributions:
            for module_path, (a, b, alpha) in factors.items():
                per_module.setdefault(module_path, []).append((a, b, scale * alpha))

        applied = 0
        for module_path, terms in per_module.items():
            resolved = self._resolve_target(module_path)
            if resolved is None:
                continue
            param_path, param, rows = resolved
            n_rows = param.shape[0] if rows is None else rows.stop - rows.start
            delta = None
            for a, b, s in terms:
                if s == 0.0:
                    continue
                if (n_rows, param.shape[1]) != (b.shape[0], a.shape[1]):
                    logger.warning(
                        "shape mismatch for %s: target (%d, %d) vs delta (%d, %d)",
                        module_path,
                        n_rows,
                        param.shape[1],
                        b.shape[0],
                        a.shape[1],
                    )
                    continue
                term = (b.to(param.device, torch.float32) @ a.to(param.device, torch.float32)) * s
                delta = term if delta is None else delta.add_(term)
            if delta is None:
                continue
            self._snapshot(param_path, param)
            target = param if rows is None else param[rows]
            target.add_(delta.to(param.dtype))
            applied += 1
        return applied

    @torch.no_grad()
    def _restore_params(self, param_paths: set[str]) -> None:
        for path in param_paths:
            snap = self._snapshots.get(path)
            if snap is not None:
                self._params[path].copy_(snap, non_blocking=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _stale_param_paths(self) -> set[str]:
        stale: set[str] = set()
        for lora_id, _ in self._active:
            cached = self._factor_cache.get(lora_id)
            if cached is None:
                return set(self._snapshots)
            for module_path in cached.factors:
                resolved = self._resolve_target(module_path)
                if resolved is not None:
                    stale.add(resolved[0])
        return stale

    def _first_param_device(self) -> torch.device:
        for param in self._params.values():
            return param.device
        return torch.device("cpu")

    def activate(self, loras: list[tuple[str, float]] | list[str] | str | None) -> dict:
        """Make exactly the given LoRA set active.

        Warm path: on-device delta-difference update (no weight transfer). Exact path
        (snapshot restore) runs on first touch, non-CUDA devices, deactivation, and
        every ``exact_restore_every`` fast swaps to reset rounding drift.
        """
        if loras is None:
            loras = []
        if isinstance(loras, str):
            loras = [loras]
        resolved = []
        for item in loras:
            lora_id, scale = (item, None) if isinstance(item, str) else (item[0], item[1])
            if scale is None:
                record = self.registry.get(lora_id)
                scale = record.scale if record else 1.0
            resolved.append((lora_id, float(scale)))
        target = tuple(resolved)

        with self._lock:
            if target == self._active:
                return {"swapped": False, "active": list(target)}

            start = time.perf_counter()
            device = self._first_param_device()
            use_fast = (
                device.type == "cuda"
                and target
                and self._fast_swaps < self.exact_restore_every
                and all(lora_id in self._factor_cache for lora_id, _ in self._active)
            )
            try:
                new_map = dict(target)
                old_map = dict(self._active)
                if use_fast:
                    contributions = [
                        (self._get_device_factors(lora_id, device), scale - old_map.get(lora_id, 0.0))
                        for lora_id, scale in target
                        if scale != old_map.get(lora_id, 0.0)
                    ] + [
                        (self._get_device_factors(lora_id, device), -scale)
                        for lora_id, scale in self._active
                        if lora_id not in new_map
                    ]
                    restored_t = time.perf_counter()
                    applied = self._apply_contributions(contributions)
                    self._fast_swaps += 1
                    mode = "fast"
                else:
                    for lora_id, _ in target:
                        self.get_factors(lora_id)
                    self._restore_params(self._stale_param_paths())
                    restored_t = time.perf_counter()
                    applied = self._apply_contributions(
                        [(self._get_device_factors(lora_id, device), scale) for lora_id, scale in target]
                    )
                    self._fast_swaps = 0
                    mode = "exact"
                if target and applied == 0:
                    raise RuntimeError(
                        f"LoRA activation resolved zero parameters for {[lora_id for lora_id, _ in target]}"
                    )
            except Exception:
                self._restore_params(set(self._snapshots))
                self._active = ()
                self._fast_swaps = 0
                raise

            if device.type == "cuda":
                torch.cuda.synchronize()
            self._active = target
            end = time.perf_counter()
            timings = {
                "ms": (end - start) * 1000,
                "restore_ms": (restored_t - start) * 1000,
                "apply_ms": (end - restored_t) * 1000,
            }
            logger.info(
                "activated %s (%s, %d params, %.1fms: pre %.1f apply %.1f)",
                [lora_id for lora_id, _ in target] or "base",
                mode,
                applied,
                timings["ms"],
                timings["restore_ms"],
                timings["apply_ms"],
            )
            return {"swapped": True, "active": list(target), "params": applied, "mode": mode, **timings}

    def deactivate(self) -> dict:
        """Restore exact pristine base weights."""
        with self._lock:
            if not self._active and not self._fast_swaps:
                return {"swapped": False, "active": []}
            start = time.perf_counter()
            self._restore_params(set(self._snapshots))
            self._active = ()
            self._fast_swaps = 0
            elapsed = (time.perf_counter() - start) * 1000
            logger.info("deactivated all loras (%.1fms)", elapsed)
            return {"swapped": True, "active": [], "ms": elapsed, "mode": "exact"}

    def preload(self, lora_ids: list[str], to_gpu: bool = False) -> None:
        device = self._first_param_device()
        for lora_id in lora_ids:
            try:
                if to_gpu and device.type == "cuda":
                    self._get_device_factors(lora_id, device)
                else:
                    self.get_factors(lora_id)
            except Exception as e:
                logger.warning("preload failed for %s: %s", lora_id, e)

    def apply_record(self, record: LoRARecord, prompt: str, scale: float | None = None) -> str:
        """Activate a single record and return the templated prompt."""
        if record.id not in self.registry:
            self.registry.add(record)
        self.activate([(record.id, scale)])
        return record.apply_template(prompt)
