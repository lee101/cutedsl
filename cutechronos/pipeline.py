"""CuteChronos2Pipeline -- drop-in pipeline for Chronos-2 inference.

Supports two backends:
- ``use_cute=True`` (default): Uses CuteChronos2Model with custom Triton
  kernels and optional torch.compile for maximum performance.
- ``use_cute=False``: Delegates to the original upstream Chronos2Model.
"""

from __future__ import annotations

import logging
import math
import warnings
from typing import List, Optional, Union

import torch

logger = logging.getLogger(__name__)


def _select_quantiles(
    preds: torch.Tensor,
    training_quantiles: list[float],
    requested_quantiles: list[float],
) -> torch.Tensor:
    """Select or interpolate quantile levels from model predictions.

    Args:
        preds: (B, Q_train, H) model quantile predictions
        training_quantiles: quantile levels the model was trained on
        requested_quantiles: desired quantile levels

    Returns:
        (B, H, len(requested_quantiles)) selected/interpolated values
    """
    # (B, Q, H) -> (B, H, Q)
    preds_bhq = preds.permute(0, 2, 1)

    if set(requested_quantiles).issubset(training_quantiles):
        indices = [training_quantiles.index(q) for q in requested_quantiles]
        return preds_bhq[..., indices]

    tq = torch.tensor(training_quantiles, dtype=torch.float32)
    results = []
    for ql in requested_quantiles:
        if ql in training_quantiles:
            results.append(preds_bhq[..., training_quantiles.index(ql)])
        else:
            idx = torch.searchsorted(tq, ql).clamp(1, len(tq) - 1).item()
            lo, hi = idx - 1, idx
            frac = (ql - tq[lo].item()) / max(tq[hi].item() - tq[lo].item(), 1e-9)
            results.append(preds_bhq[..., lo] * (1 - frac) + preds_bhq[..., hi] * frac)
    return torch.stack(results, dim=-1)


def _load_model_original(model_path: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
    """Load via the upstream Chronos2Pipeline (requires chronos-forecasting)."""
    from chronos.chronos2 import Chronos2Pipeline

    pipeline = Chronos2Pipeline.from_pretrained(model_path, dtype=dtype)
    model = pipeline.model
    model = model.to(device)
    model.eval()
    return model


def _load_model_cute(
    model_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    compile_mode: str | None = None,
):
    """Load via CuteChronos2Model (no upstream dependency for inference)."""
    from pathlib import Path
    from cutechronos.model import CuteChronos2Model

    if Path(model_path).is_dir():
        local_path = str(model_path)
    else:
        from huggingface_hub import snapshot_download
        local_path = snapshot_download(
            model_path,
            allow_patterns=["*.json", "*.safetensors", "*.bin"],
        )

    if compile_mode:
        model = CuteChronos2Model.from_pretrained_compiled(local_path, compile_mode=compile_mode)
    else:
        model = CuteChronos2Model.from_pretrained(local_path)

    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class CuteChronos2Pipeline:
    """Lightweight pipeline for Chronos-2 inference.

    Provides a simpler API than the upstream ``Chronos2Pipeline``: the
    caller passes a raw ``torch.Tensor`` (or list of tensors) and gets
    back quantile predictions without needing to deal with ``DataLoader``
    or ``Dataset`` wrappers.

    The pipeline handles:
    * Context truncation to ``model_context_length``.
    * Left-padding variable-length series.
    * Device management and dtype casting.
    * Single-step prediction (non-autoregressive) for
      ``prediction_length <= model_prediction_length``.
    """

    def __init__(self, model, *, device: str = "cuda", _is_cute: bool = False):
        self.model = model
        self._device = device
        self._is_cute = _is_cute

    # -- properties ----------------------------------------------------------

    def _get_config(self):
        """Access the model config, handling both CuteChronos2Model and original."""
        if self._is_cute:
            return self.model.config
        return self.model.chronos_config

    @property
    def device(self) -> str:
        return self._device

    @property
    def model_context_length(self) -> int:
        return self._get_config().context_length

    @property
    def model_output_patch_size(self) -> int:
        return self._get_config().output_patch_size

    @property
    def model_prediction_length(self) -> int:
        cfg = self._get_config()
        max_patches = getattr(cfg, "max_output_patches", 64)
        return max_patches * cfg.output_patch_size

    @property
    def quantiles(self) -> list[float]:
        return self._get_config().quantiles

    @property
    def max_output_patches(self) -> int:
        cfg = self._get_config()
        return getattr(cfg, "max_output_patches", 64)

    # -- offload / onload ----------------------------------------------------

    def offload(self):
        """Offload model to CPU to free GPU VRAM."""
        if self._is_cute and hasattr(self.model, "offload_to_cpu"):
            self.model.offload_to_cpu()
        else:
            self.model.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def onload(self, device: str | None = None):
        """Bring model back to GPU."""
        device = device or self._device
        if self._is_cute and hasattr(self.model, "onload_to_gpu"):
            self.model.onload_to_gpu(device)
        else:
            self.model.to(device)

    # -- factory -------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        use_cute: bool = True,
        compile_mode: str | None = None,
    ) -> "CuteChronos2Pipeline":
        """Load a Chronos-2 model and wrap it in a CuteChronos2Pipeline.

        Parameters
        ----------
        model_path
            HuggingFace model id (e.g. ``"amazon/chronos-2"``) or local path.
        device
            Target device, default ``"cuda"``.
        dtype
            Model dtype, default ``torch.bfloat16``.
        use_cute
            If True (default), use CuteChronos2Model with custom kernels.
            If False, use the original upstream Chronos2Model.
        compile_mode
            If set (e.g. ``"reduce-overhead"``), apply torch.compile to the
            CuteChronos2Model. Only effective when ``use_cute=True``.
        """
        if use_cute:
            model = _load_model_cute(model_path, device=device, dtype=dtype, compile_mode=compile_mode)
            return cls(model, device=device, _is_cute=True)
        else:
            model = _load_model_original(model_path, device=device, dtype=dtype)
            return cls(model, device=device, _is_cute=False)

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _left_pad_and_stack(tensors: List[torch.Tensor]) -> torch.Tensor:
        """Left-pad variable-length 1D tensors and stack into a 2D batch."""
        max_len = max(t.shape[-1] for t in tensors)
        padded = []
        for t in tensors:
            pad_len = max_len - t.shape[-1]
            if pad_len > 0:
                pad = torch.full((pad_len,), float("nan"), dtype=t.dtype, device=t.device)
                t = torch.cat([pad, t])
            padded.append(t)
        return torch.stack(padded)

    def _prepare_context(self, context: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """Normalise *context* to a 2-D float32 tensor on the right device."""
        if isinstance(context, list):
            context = self._left_pad_and_stack(context)
        if context.ndim == 1:
            context = context.unsqueeze(0)
        if context.ndim == 3 and context.shape[1] == 1:
            # (B, 1, T) -> (B, T)  univariate convenience
            context = context.squeeze(1)
        assert context.ndim == 2, f"context must be 2-D, got shape {context.shape}"
        # Truncate to model context length
        if context.shape[-1] > self.model_context_length:
            context = context[..., -self.model_context_length:]
        return context.to(device=self._device, dtype=torch.float32)

    # -- prediction ----------------------------------------------------------

    @torch.inference_mode()
    def predict(
        self,
        context: Union[torch.Tensor, List[torch.Tensor]],
        prediction_length: Optional[int] = None,
        limit_prediction_length: bool = True,
        cross_learning: bool = False,
        batch_size: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """Generate quantile predictions.

        Parameters
        ----------
        context
            A 1-D tensor, list of 1-D tensors, or 2-D tensor (B, T).
        prediction_length
            Horizon.  Defaults to ``model_prediction_length``.
        limit_prediction_length
            If *True*, raise when ``prediction_length`` exceeds the model's
            default prediction length.
        cross_learning
            If True, all series share group attention (group_ids all 0).
        batch_size
            If set, process input in chunks of this size and concatenate.

        Returns
        -------
        List of tensors, each of shape ``(1, n_quantiles, prediction_length)``
        (one element per batch item), matching the upstream API.
        """
        # Auto-onload if model was offloaded
        if self._is_cute and getattr(self.model, "is_offloaded", False):
            self.model.onload_to_gpu(self._device)

        if prediction_length is None:
            prediction_length = self.model_prediction_length

        if prediction_length > self.model_prediction_length:
            msg = (
                f"prediction_length ({prediction_length}) exceeds "
                f"model_prediction_length ({self.model_prediction_length}). "
                "Quality may degrade."
            )
            if limit_prediction_length:
                msg += " Set limit_prediction_length=False to allow this."
                raise ValueError(msg)
            warnings.warn(msg)

        ctx = self._prepare_context(context)
        total_batch = ctx.shape[0]

        num_output_patches = math.ceil(prediction_length / self.model_output_patch_size)
        num_output_patches = min(num_output_patches, self.max_output_patches)

        def _forward_chunk(chunk: torch.Tensor) -> torch.Tensor:
            chunk_size = chunk.shape[0]
            if cross_learning:
                gids = torch.zeros(chunk_size, dtype=torch.long, device=self._device)
            else:
                gids = torch.arange(chunk_size, dtype=torch.long, device=self._device)

            if self._is_cute:
                return self.model(chunk, num_output_patches=num_output_patches, group_ids=gids)
            else:
                output = self.model(
                    context=chunk,
                    group_ids=gids,
                    num_output_patches=num_output_patches,
                )
                return output.quantile_preds

        if batch_size is not None and batch_size < total_batch:
            if cross_learning:
                logger.warning("batch_size chunking disabled for cross_learning=True (would break group attention)")
                preds = _forward_chunk(ctx)
            else:
                chunks = [ctx[i : i + batch_size] for i in range(0, total_batch, batch_size)]
                preds = torch.cat([_forward_chunk(c) for c in chunks], dim=0)
        else:
            preds = _forward_chunk(ctx)

        # preds: (B, Q, H) - truncate to requested length
        preds = preds[..., :prediction_length]
        preds = preds.to(dtype=torch.float32, device="cpu")

        # Return as list of (1, Q, H) tensors for API compatibility
        return [preds[i : i + 1] for i in range(total_batch)]

    @torch.inference_mode()
    def predict_quantiles(
        self,
        context: Union[torch.Tensor, List[torch.Tensor]],
        prediction_length: Optional[int] = None,
        quantile_levels: Optional[List[float]] = None,
        limit_prediction_length: bool = True,
    ) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Generate quantile and mean forecasts.

        Parameters
        ----------
        context
            Same as ``predict``.
        prediction_length
            Same as ``predict``.
        quantile_levels
            Quantile levels to return.  Default uses the model's trained
            quantiles.
        limit_prediction_length
            Same as ``predict``.

        Returns
        -------
        quantiles
            List of tensors, each ``(1, prediction_length, len(quantile_levels))``.
        mean
            List of tensors, each ``(1, prediction_length)``.
        """
        if quantile_levels is None:
            quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        predictions = self.predict(
            context,
            prediction_length=prediction_length,
            limit_prediction_length=limit_prediction_length,
        )

        training_quantile_levels = self.quantiles

        # predictions: list of (1, Q, H) -> select quantiles via shared helper
        quantiles = [_select_quantiles(p, training_quantile_levels, quantile_levels) for p in predictions]

        # median as "mean"
        median_idx = training_quantile_levels.index(0.5)
        mean = [p.permute(0, 2, 1)[..., median_idx] for p in predictions]

        return quantiles, mean

    def predict_df(
        self,
        df,
        future_df=None,
        id_column: str = "item_id",
        timestamp_column: str = "timestamp",
        target: str = "target",
        prediction_length: Optional[int] = None,
        quantile_levels: Optional[List[float]] = None,
        batch_size: int = 256,
        cross_learning: bool = False,
        **kwargs,
    ) -> "pd.DataFrame":
        """Generate forecasts from a pandas DataFrame.

        Matches the upstream Chronos2Pipeline.predict_df API.

        Parameters
        ----------
        df
            Input DataFrame with columns for id, timestamp, and target.
        future_df
            Unused, accepted for API compatibility.
        id_column
            Column identifying individual time series.
        timestamp_column
            Column with timestamps (used for ordering).
        target
            Column with target values.
        prediction_length
            Forecast horizon. Defaults to model_prediction_length.
        quantile_levels
            Quantile levels for output columns. Defaults to model quantiles.
        batch_size
            Process series in chunks of this size.
        cross_learning
            If True, all series share group attention.

        Returns
        -------
        pd.DataFrame with columns: item_id, step, and one column per quantile level.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas is required for predict_df") from exc

        if future_df is not None:
            raise NotImplementedError(
                "future_df (future covariates) is not supported by CuteChronos2Pipeline. "
                "Pass future_df=None or use the upstream Chronos2Pipeline."
            )

        if prediction_length is None:
            prediction_length = self.model_prediction_length
        if quantile_levels is None:
            quantile_levels = list(self.quantiles)

        groups = df.sort_values(timestamp_column).groupby(id_column, sort=False)
        item_ids = list(groups.groups.keys())
        tensors = [torch.tensor(g[target].values, dtype=torch.float32) for _, g in groups]

        preds = self.predict(
            tensors,
            prediction_length=prediction_length,
            cross_learning=cross_learning,
            batch_size=batch_size,
            limit_prediction_length=False,
            **kwargs,
        )

        # preds: list of (1, Q, H) -> stack to (N, Q, H)
        all_preds = torch.cat(preds, dim=0)  # (N, Q, H)
        actual_horizon = all_preds.shape[-1]  # may be < prediction_length if clamped

        # Select/interpolate requested quantile levels vectorized
        selected = _select_quantiles(all_preds, self.quantiles, quantile_levels)  # (N, actual_horizon, len(ql))

        n_series = len(item_ids)
        n_steps = actual_horizon
        n_ql = len(quantile_levels)

        ids_repeated = [iid for iid in item_ids for _ in range(n_steps)]
        steps_repeated = list(range(n_steps)) * n_series

        result = {id_column: ids_repeated, "step": steps_repeated}
        flat = selected.reshape(n_series * n_steps, n_ql)
        for qi, ql in enumerate(quantile_levels):
            result[str(ql)] = flat[:, qi].tolist()

        return pd.DataFrame(result)
