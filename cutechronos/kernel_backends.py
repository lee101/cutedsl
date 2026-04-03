"""Backend selection for CuteChronos inference kernels."""

from __future__ import annotations

import logging
import os

import torch
import torch.nn.functional as F

from cutechronos.modules._fallbacks import rms_layernorm as _torch_rms_layernorm
from cutechronos.modules._fallbacks import unscaled_attention as _torch_unscaled_attention

logger = logging.getLogger(__name__)

try:
    from cutechronos.cutlass_kernels.rms_layernorm import rms_layernorm as _cutlass_rms_layernorm

    _HAS_CUTLASS_RMS = True
except Exception as exc:  # pragma: no cover - import is environment-specific
    _HAS_CUTLASS_RMS = False
    _CUTLASS_IMPORT_ERROR = exc

try:
    from cutechronos.triton_kernels.rms_layernorm import rms_layernorm as _triton_rms_layernorm

    _HAS_TRITON_RMS = True
except Exception:  # pragma: no cover - import is environment-specific
    _HAS_TRITON_RMS = False

try:
    from cutechronos.triton_kernels.attention import unscaled_attention as _triton_unscaled_attention

    _HAS_TRITON_ATTN = True
except Exception:  # pragma: no cover - import is environment-specific
    _HAS_TRITON_ATTN = False

try:
    from cutechronos.triton_kernels.fused_layernorm_linear import fused_rms_norm_qkv as _triton_fused_rms_norm_qkv

    _HAS_TRITON_FUSED_QKV = True
except Exception:  # pragma: no cover - import is environment-specific
    _HAS_TRITON_FUSED_QKV = False

_WARNED_KEYS: set[tuple[str, str]] = set()


def _choice(env_var: str, default: str = "auto") -> str:
    return os.getenv(env_var, default).strip().lower()


def _warn_once(kind: str, exc: Exception) -> None:
    key = (kind, type(exc).__name__)
    if key in _WARNED_KEYS:
        return
    _WARNED_KEYS.add(key)
    logger.warning("%s backend failed; falling back: %s", kind, exc)


def _cuda_major(device: torch.device | int | str | None) -> int | None:
    if not torch.cuda.is_available():
        return None
    major, _ = torch.cuda.get_device_capability(device)
    return major


def _rms_candidates(backend: str, x: torch.Tensor) -> tuple[str, ...]:
    if backend == "torch":
        return ("torch",)
    if backend == "triton":
        return ("triton", "torch")
    if backend == "cutlass":
        return ("cutlass", "torch")
    if backend != "auto":
        raise ValueError(f"Unsupported CUTECHRONOS_RMS_BACKEND={backend!r}")

    major = _cuda_major(x.device if x.is_cuda else None)
    if major is None:
        return ("torch",)
    if major >= 9:
        return ("cutlass", "triton", "torch")
    return ("triton", "cutlass", "torch")


def rms_layernorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if not x.is_cuda:
        return _torch_rms_layernorm(x, weight, eps)

    backend = _choice("CUTECHRONOS_RMS_BACKEND")
    candidates = _rms_candidates(backend, x)

    for candidate in candidates:
        if candidate == "cutlass" and _HAS_CUTLASS_RMS:
            try:
                return _cutlass_rms_layernorm(x, weight, eps)
            except Exception as exc:  # pragma: no cover - runtime fallback
                _warn_once("cutlass_rms", exc)
                continue
        if candidate == "triton" and _HAS_TRITON_RMS:
            try:
                return _triton_rms_layernorm(x, weight, eps)
            except Exception as exc:  # pragma: no cover - runtime fallback
                _warn_once("triton_rms", exc)
                continue
        if candidate == "torch":
            return _torch_rms_layernorm(x, weight, eps)

    if not _HAS_CUTLASS_RMS and backend in {"auto", "cutlass"}:
        logger.debug("CUTLASS RMSNorm unavailable: %s", globals().get("_CUTLASS_IMPORT_ERROR"))
    return _torch_rms_layernorm(x, weight, eps)


def unscaled_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    backend = _choice("CUTECHRONOS_ATTENTION_BACKEND")
    candidates = {
        "auto": ("sdpa", "torch"),
        "sdpa": ("sdpa", "torch"),
        "torch": ("torch",),
        "triton": ("triton", "torch"),
    }.get(backend)
    if candidates is None:
        raise ValueError(f"Unsupported CUTECHRONOS_ATTENTION_BACKEND={backend!r}")

    for candidate in candidates:
        if candidate == "sdpa" and q.is_cuda:
            try:
                return F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=mask,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=1.0,
                )
            except Exception as exc:  # pragma: no cover - runtime fallback
                _warn_once("sdpa_attention", exc)
                continue
        if candidate == "triton" and q.is_cuda and _HAS_TRITON_ATTN:
            try:
                return _triton_unscaled_attention(q, k, v, mask)
            except Exception as exc:  # pragma: no cover - runtime fallback
                _warn_once("triton_attention", exc)
                continue
        if candidate == "torch":
            return _torch_unscaled_attention(q, k, v, mask)

    return _torch_unscaled_attention(q, k, v, mask)


def fused_rms_norm_qkv_available(hidden_states: torch.Tensor) -> bool:
    setting = _choice("CUTECHRONOS_FUSED_QKV")
    if setting in {"0", "false", "no", "off"}:
        return False
    if not (hidden_states.is_cuda and _HAS_TRITON_FUSED_QKV):
        return False
    if setting in {"1", "true", "yes", "on"}:
        return True
    major = _cuda_major(hidden_states.device)
    return major is not None and major >= 9


def fused_rms_norm_qkv(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _triton_fused_rms_norm_qkv(x, norm_weight, wq, wk, wv, eps)
