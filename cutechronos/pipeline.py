"""CuteChronos2Pipeline -- drop-in pipeline for Chronos-2 inference.

Supports two backends:
- ``use_cute=True`` (default): Uses CuteChronos2Model with CUTLASS/SDPA/Triton
  kernel dispatch and optional torch.compile for maximum performance.
- ``use_cute=False``: Delegates to the original upstream Chronos2Model.
"""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, List, Optional, Union

import torch

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import pandas as pd


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

    quantile_to_index = {q: idx for idx, q in enumerate(training_quantiles)}
    direct_indices = [quantile_to_index.get(q) for q in requested_quantiles]
    if all(idx is not None for idx in direct_indices):
        return preds_bhq[..., [int(idx) for idx in direct_indices]]

    tq = preds_bhq.new_tensor(training_quantiles, dtype=torch.float32)
    results = []
    for ql, direct_idx in zip(requested_quantiles, direct_indices):
        if direct_idx is not None:
            results.append(preds_bhq[..., direct_idx])
            continue
        q_tensor = tq.new_tensor(ql)
        idx = torch.searchsorted(tq, q_tensor).clamp(1, len(tq) - 1).item()
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


def _to_1d_float_tensor(value: Any) -> torch.Tensor:
    import numpy as np
    if isinstance(value, torch.Tensor):
        return value.to(torch.float32)
    arr = np.asarray(value)
    if not np.issubdtype(arr.dtype, np.number):
        _, encoded = np.unique(arr.astype(str), return_inverse=True)
        arr = encoded.astype("float32")
    return torch.from_numpy(arr.astype("float32", copy=False))


def _prepare_single_rich_task(task: Mapping[str, Any], idx: int, prediction_length: int):
    import numpy as np
    target = task.get("target")
    if target is None:
        raise ValueError(f"Element at index {idx} does not contain required key 'target'")
    if isinstance(target, torch.Tensor):
        target_t = target.to(torch.float32)
    else:
        target_t = torch.from_numpy(np.asarray(target).astype("float32", copy=False))
    history_length = target_t.shape[-1]
    target_t = target_t.reshape(-1, history_length)

    past_covariates = task.get("past_covariates", {}) or {}
    future_covariates = task.get("future_covariates", {}) or {}
    cov_keys = sorted(past_covariates.keys())
    future_keys = sorted(future_covariates.keys())
    if not set(future_keys).issubset(cov_keys):
        raise ValueError(f"future_covariates keys must be a subset of past_covariates keys in element {idx}")
    ordered_keys = [k for k in cov_keys if k not in future_keys] + future_keys

    past_rows = []
    future_rows = [torch.full((prediction_length,), float("nan")) for _ in range(target_t.shape[0])]
    for key in ordered_keys:
        past = _to_1d_float_tensor(past_covariates[key])
        if past.ndim != 1 or past.shape[0] != history_length:
            raise ValueError(f"past_covariate {key!r} must be 1-D with length {history_length}")
        past_rows.append(past)
        if key in future_covariates:
            fut = _to_1d_float_tensor(future_covariates[key])
            if fut.ndim != 1 or fut.shape[0] != prediction_length:
                raise ValueError(f"future_covariate {key!r} must be 1-D with length {prediction_length}")
        else:
            fut = torch.full((prediction_length,), float("nan"))
        future_rows.append(fut)

    context = torch.cat([target_t, torch.stack(past_rows) if past_rows else torch.zeros((0, history_length))], dim=0)
    future = torch.stack(future_rows) if future_rows else torch.zeros((context.shape[0], prediction_length))
    return context, future, target_t.shape[0]


def _left_pad_and_cat_2d(tensors: list[torch.Tensor]) -> torch.Tensor:
    max_len = max(t.shape[-1] for t in tensors)
    padded = []
    for tensor in tensors:
        if tensor.shape[-1] < max_len:
            pad = torch.full((tensor.shape[0], max_len - tensor.shape[-1]), float("nan"), dtype=tensor.dtype)
            tensor = torch.cat([pad, tensor], dim=-1)
        padded.append(tensor)
    return torch.cat(padded, dim=0)


def _convert_df_to_rich_inputs(df, future_df, id_column, timestamp_column, target_columns, prediction_length):
    import pandas as pd
    df = df.copy()
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    df = df.sort_values([id_column, timestamp_column])
    if future_df is not None:
        future_df = future_df.copy()
        future_df[timestamp_column] = pd.to_datetime(future_df[timestamp_column])
        future_df = future_df.sort_values([id_column, timestamp_column])

    original_order = df[id_column].drop_duplicates().to_numpy()
    inputs = []
    prediction_timestamps = {}
    covariate_cols = [c for c in df.columns if c not in [id_column, timestamp_column, *target_columns]]

    for series_id, group in df.groupby(id_column, sort=False):
        group = group.sort_values(timestamp_column)
        target_data = group[target_columns].to_numpy().T
        task: dict[str, Any] = {"target": target_data}
        if covariate_cols:
            task["past_covariates"] = {col: group[col].to_numpy() for col in covariate_cols}
        inferred = pd.infer_freq(group[timestamp_column])
        if inferred is None:
            inferred = "D"
        last_timestamp = group[timestamp_column].iloc[-1]
        prediction_timestamps[series_id] = pd.date_range(last_timestamp, periods=prediction_length + 1, freq=inferred)[1:]
        if future_df is not None and covariate_cols:
            fut_group = future_df[future_df[id_column] == series_id].sort_values(timestamp_column)
            if len(fut_group) != prediction_length:
                raise ValueError(
                    f"Future covariates for series {series_id} must have length {prediction_length}, got {len(fut_group)}"
                )
            task["future_covariates"] = {
                col: fut_group[col].to_numpy() for col in covariate_cols if col in fut_group.columns
            }
        inputs.append(task)
    return inputs, original_order, prediction_timestamps


def _load_model_cute(
    model_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    compile_mode: str | None = None,
    turboquant_bits: int | None = None,
):
    """Load via CuteChronos2Model (no upstream dependency for inference).

    When *turboquant_bits* is set (e.g. 4), TurboQuant KV quantization is
    enabled.  Quantizers are attached **before** torch.compile so the
    encode/decode ops are captured in CUDA graphs.
    """
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

    if turboquant_bits and compile_mode:
        model = CuteChronos2Model.from_pretrained_compiled_tq(
            local_path,
            compile_mode=compile_mode,
            tq_bits=turboquant_bits,
        )
    elif compile_mode:
        model = CuteChronos2Model.from_pretrained_compiled(local_path, compile_mode=compile_mode)
    else:
        model = CuteChronos2Model.from_pretrained(local_path)
        if turboquant_bits:
            model.enable_turboquant_kv(bits=turboquant_bits)

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
        turboquant_bits: int | None = None,
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
        turboquant_bits
            If set (e.g. 4), enable TurboQuant KV quantization at the given
            bit-width. Quantizers are attached before torch.compile so
            encode/decode ops can be captured in CUDA graphs. Only effective
            when ``use_cute=True``.
        """
        if use_cute:
            model = _load_model_cute(
                model_path, device=device, dtype=dtype,
                compile_mode=compile_mode, turboquant_bits=turboquant_bits,
            )
            return cls(model, device=device, _is_cute=True)
        else:
            model = _load_model_original(model_path, device=device, dtype=dtype)
            return cls(model, device=device, _is_cute=False)

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _left_pad_and_stack_rows(rows: List[torch.Tensor]) -> torch.Tensor:
        """Left-pad variable-length 1D rows and stack into a 2D batch."""
        max_len = max(row.shape[-1] for row in rows)
        padded = []
        for row in rows:
            pad_len = max_len - row.shape[-1]
            if pad_len > 0:
                pad = torch.full((pad_len,), float("nan"), dtype=row.dtype, device=row.device)
                row = torch.cat([pad, row])
            padded.append(row)
        return torch.stack(padded)

    @staticmethod
    def _as_tensor(value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value
        return torch.as_tensor(value)

    def _finalize_context_batch(self, context: torch.Tensor) -> torch.Tensor:
        if context.shape[-1] > self.model_context_length:
            context = context[..., -self.model_context_length:]
        return context.to(device=self._device, dtype=torch.float32)

    def _prepare_context(
        self,
        context: Union[torch.Tensor, List[torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
        """Normalize context into a flat batch plus grouping metadata.

        Returns
        -------
        context
            2-D float32 tensor of shape ``(sum_variates, history_length)``.
        group_ids
            Group IDs used for cross-variate attention. Variates from the same
            multivariate task share the same group ID.
        target_idx_ranges
            Ranges used to reshape flat batch predictions back into one output
            tensor per original task.
        """
        if isinstance(context, list):
            rows: list[torch.Tensor] = []
            group_ids: list[int] = []
            target_idx_ranges: list[tuple[int, int]] = []

            for group_idx, item in enumerate(context):
                if isinstance(item, dict):
                    raise NotImplementedError(
                        "Covariate dictionaries are not supported by CuteChronos2Pipeline yet. "
                        "Pass raw targets only or use the upstream Chronos2Pipeline."
                    )
                tensor = self._as_tensor(item)
                if tensor.ndim == 1:
                    tensor = tensor.unsqueeze(0)
                elif tensor.ndim != 2:
                    raise ValueError(f"Each list item must be 1-D or 2-D, got shape {tuple(tensor.shape)}")

                start = len(rows)
                rows.extend(tensor[row_idx] for row_idx in range(tensor.shape[0]))
                end = len(rows)
                target_idx_ranges.append((start, end))
                group_ids.extend([group_idx] * (end - start))

            flat_context = self._left_pad_and_stack_rows(rows)
            flat_context = self._finalize_context_batch(flat_context)
            return (
                flat_context,
                torch.tensor(group_ids, dtype=torch.long, device=self._device),
                target_idx_ranges,
            )

        tensor = self._as_tensor(context)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
            tensor = self._finalize_context_batch(tensor)
            return tensor, torch.tensor([0], dtype=torch.long, device=self._device), [(0, 1)]

        if tensor.ndim == 2:
            batch = self._finalize_context_batch(tensor)
            ranges = [(idx, idx + 1) for idx in range(batch.shape[0])]
            gids = torch.arange(batch.shape[0], dtype=torch.long, device=self._device)
            return batch, gids, ranges

        if tensor.ndim == 3:
            rows = [tensor[batch_idx, variate_idx] for batch_idx in range(tensor.shape[0]) for variate_idx in range(tensor.shape[1])]
            flat_context = self._left_pad_and_stack_rows(rows)
            target_idx_ranges = []
            group_ids = []
            start = 0
            for batch_idx in range(tensor.shape[0]):
                end = start + tensor.shape[1]
                target_idx_ranges.append((start, end))
                group_ids.extend([batch_idx] * tensor.shape[1])
                start = end
            flat_context = self._finalize_context_batch(flat_context)
            return (
                flat_context,
                torch.tensor(group_ids, dtype=torch.long, device=self._device),
                target_idx_ranges,
            )

        raise ValueError(f"context must be 1-D, 2-D, or 3-D, got shape {tuple(tensor.shape)}")

    def _predict_rich_inputs(
        self,
        inputs: Sequence[Mapping[str, Any]],
        *,
        prediction_length: int,
        batch_size: int,
        context_length: int | None = None,
        cross_learning: bool = False,
    ) -> List[torch.Tensor]:
        if context_length is None:
            context_length = self.model_context_length
        context_length = min(context_length, self.model_context_length)
        all_predictions: list[torch.Tensor] = []
        num_output_patches = math.ceil(prediction_length / self.model_output_patch_size)
        num_output_patches = min(num_output_patches, self.max_output_patches)
        tasks = [_prepare_single_rich_task(task, idx, prediction_length) for idx, task in enumerate(inputs)]

        task_idx = 0
        while task_idx < len(tasks):
            current_size = 0
            selected = []
            while task_idx < len(tasks) and (not selected or current_size < batch_size):
                selected.append(tasks[task_idx])
                current_size += tasks[task_idx][0].shape[0]
                task_idx += 1

            contexts = []
            futures = []
            group_ids = []
            target_idx_ranges = []
            start = 0
            for group_id, (ctx_task, fut_task, n_targets) in enumerate(selected):
                ctx_task = ctx_task[..., -context_length:]
                contexts.append(ctx_task)
                futures.append(fut_task)
                group_ids.append(torch.full((ctx_task.shape[0],), group_id, dtype=torch.long))
                target_idx_ranges.append((start, start + n_targets))
                start += ctx_task.shape[0]

            ctx = _left_pad_and_cat_2d(contexts).to(device=self._device, dtype=torch.float32)
            gids = torch.cat(group_ids).to(device=self._device)
            if cross_learning:
                gids = torch.zeros_like(gids)
            future_covariates = torch.cat(futures, dim=0).to(device=self._device, dtype=torch.float32)

            if self._is_cute:
                preds = self.model(
                    ctx,
                    num_output_patches=num_output_patches,
                    group_ids=gids,
                    future_covariates=future_covariates,
                )
            else:
                output = self.model(
                    context=ctx,
                    group_ids=gids,
                    num_output_patches=num_output_patches,
                    future_covariates=future_covariates,
                )
                preds = output.quantile_preds
            preds = preds[..., :prediction_length].to(dtype=torch.float32, device="cpu")
            all_predictions.extend([preds[start:end] for start, end in target_idx_ranges])

        return all_predictions

    # -- prediction ----------------------------------------------------------

    @torch.inference_mode()
    def predict(
        self,
        context: Union[torch.Tensor, List[torch.Tensor], Sequence[Mapping[str, Any]]] = None,
        inputs: Union[torch.Tensor, List[torch.Tensor], Sequence[Mapping[str, Any]], None] = None,
        prediction_length: Optional[int] = None,
        limit_prediction_length: bool = True,
        cross_learning: bool = False,
        batch_size: Optional[int] = None,
        context_length: int | None = None,
        predict_batches_jointly: bool | None = None,
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
        if context is None:
            context = inputs
        if context is None:
            raise ValueError("context or inputs must be provided")
        if predict_batches_jointly is not None:
            cross_learning = predict_batches_jointly

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

        if isinstance(context, Sequence) and not isinstance(context, (torch.Tensor, str, bytes)):
            if all(isinstance(item, Mapping) for item in context):
                return self._predict_rich_inputs(
                    context,
                    prediction_length=prediction_length,
                    batch_size=batch_size or 256,
                    context_length=context_length,
                    cross_learning=cross_learning,
                )

        ctx, group_ids, target_idx_ranges = self._prepare_context(context)
        total_batch = ctx.shape[0]
        has_multivariate = any((end - start) > 1 for start, end in target_idx_ranges)

        num_output_patches = math.ceil(prediction_length / self.model_output_patch_size)
        num_output_patches = min(num_output_patches, self.max_output_patches)

        def _forward_chunk(chunk: torch.Tensor, chunk_group_ids: torch.Tensor) -> torch.Tensor:
            chunk_size = chunk.shape[0]
            if cross_learning:
                gids = torch.zeros(chunk_size, dtype=torch.long, device=self._device)
            else:
                gids = chunk_group_ids

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
            if cross_learning or has_multivariate:
                logger.warning(
                    "batch_size chunking disabled for multivariate inputs or cross_learning=True "
                    "(would break group attention semantics)"
                )
                preds = _forward_chunk(ctx, group_ids)
            else:
                chunks = [ctx[i : i + batch_size] for i in range(0, total_batch, batch_size)]
                gid_chunks = [group_ids[i : i + batch_size] for i in range(0, total_batch, batch_size)]
                preds = torch.cat([_forward_chunk(chunk, gids) for chunk, gids in zip(chunks, gid_chunks)], dim=0)
        else:
            preds = _forward_chunk(ctx, group_ids)

        # preds: (B, Q, H) - truncate to requested length
        preds = preds[..., :prediction_length]
        preds = preds.to(dtype=torch.float32, device="cpu")

        # Return as list of (n_variates, Q, H) tensors for API compatibility
        return [preds[start:end] for start, end in target_idx_ranges]

    @torch.inference_mode()
    def predict_quantiles(
        self,
        context: Union[torch.Tensor, List[torch.Tensor], Sequence[Mapping[str, Any]], None] = None,
        inputs: Union[torch.Tensor, List[torch.Tensor], Sequence[Mapping[str, Any]], None] = None,
        prediction_length: Optional[int] = None,
        quantile_levels: Optional[List[float]] = None,
        limit_prediction_length: bool = True,
        cross_learning: bool = False,
        batch_size: Optional[int] = None,
        context_length: int | None = None,
        predict_batches_jointly: bool | None = None,
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
        cross_learning
            Forwarded to ``predict`` so related series can share group attention.
        batch_size
            Forwarded to ``predict`` for batched univariate inference.

        Returns
        -------
        quantiles
            List of tensors, each ``(1, prediction_length, len(quantile_levels))``.
        mean
            List of tensors, each ``(1, prediction_length)``.
        """
        if quantile_levels is None:
            quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        predict_kwargs = {
            "prediction_length": prediction_length,
            "limit_prediction_length": limit_prediction_length,
            "cross_learning": cross_learning,
            "batch_size": batch_size,
        }
        if inputs is not None:
            predict_kwargs["inputs"] = inputs
        if context_length is not None:
            predict_kwargs["context_length"] = context_length
        if predict_batches_jointly is not None:
            predict_kwargs["predict_batches_jointly"] = predict_batches_jointly

        predictions = self.predict(context, **predict_kwargs)

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
        target: str | list[str] = "target",
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

        if prediction_length is None:
            prediction_length = self.model_prediction_length
        if quantile_levels is None:
            quantile_levels = list(self.quantiles)
        target_columns = target if isinstance(target, list) else [target]

        if future_df is not None or len(target_columns) > 1 or any(
            col not in {id_column, timestamp_column, *target_columns} for col in df.columns
        ):
            inputs, original_order, prediction_timestamps = _convert_df_to_rich_inputs(
                df=df,
                future_df=future_df,
                id_column=id_column,
                timestamp_column=timestamp_column,
                target_columns=target_columns,
                prediction_length=prediction_length,
            )
            quantiles, mean = self.predict_quantiles(
                inputs=inputs,
                prediction_length=prediction_length,
                quantile_levels=quantile_levels,
                limit_prediction_length=False,
                batch_size=batch_size,
                predict_batches_jointly=cross_learning,
                **kwargs,
            )
            quantiles_np = torch.stack(quantiles).numpy()
            mean_np = torch.stack(mean).numpy()
            result_dfs = []
            for i, (series_id, future_ts) in enumerate(prediction_timestamps.items()):
                for target_idx, target_col in enumerate(target_columns):
                    data = {
                        id_column: series_id,
                        timestamp_column: future_ts,
                        "step": range(prediction_length),
                        "target_name": target_col,
                        "predictions": mean_np[i, target_idx],
                    }
                    for q_idx, q_level in enumerate(quantile_levels):
                        data[str(q_level)] = quantiles_np[i, target_idx, :, q_idx]
                    result_dfs.append(pd.DataFrame(data))
            predictions_df = pd.concat(result_dfs, ignore_index=True)
            predictions_df.set_index(id_column, inplace=True)
            predictions_df = predictions_df.loc[original_order]
            predictions_df.reset_index(inplace=True)
            return predictions_df

        groups = df.sort_values(timestamp_column).groupby(id_column, sort=False)
        item_ids = list(groups.groups.keys())
        tensors = [torch.tensor(g[target_columns[0]].values, dtype=torch.float32) for _, g in groups]

        quantiles, mean = self.predict_quantiles(
            inputs=tensors,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
            predict_batches_jointly=cross_learning,
            batch_size=batch_size,
            limit_prediction_length=False,
            **kwargs,
        )
        result_dfs = []
        for item_id, (_, group), q_pred, point_pred in zip(item_ids, groups, quantiles, mean):
            ts = pd.to_datetime(group[timestamp_column])
            inferred = pd.infer_freq(ts)
            if inferred is None:
                inferred = "D"
            future_ts = pd.date_range(start=ts.iloc[-1], periods=prediction_length + 1, freq=inferred)[1:]
            data = {
                id_column: item_id,
                timestamp_column: future_ts,
                "step": range(prediction_length),
                "target_name": target_columns[0],
                "predictions": point_pred[0].numpy(),
            }
            for q_idx, q_level in enumerate(quantile_levels):
                data[str(q_level)] = q_pred[0, :, q_idx].numpy()
            result_dfs.append(pd.DataFrame(data))
        return pd.concat(result_dfs, ignore_index=True)

    def fit(self, *args, **kwargs):
        """Fine-tune through the upstream Chronos-2 trainer.

        CuteChronos is an inference-optimized module. Full and LoRA fine-tuning
        are delegated to the official Chronos2Pipeline, then wrapped back in
        this compatibility class so callers can keep using the same API.
        """
        if self._is_cute:
            raise NotImplementedError(
                "Fine-tuning CuteChronos2Model directly is not implemented yet. "
                "Load with use_cute=False to use upstream Chronos-2 full or LoRA fine-tuning."
            )
        from chronos.chronos2 import Chronos2Pipeline

        upstream = Chronos2Pipeline(self.model)
        finetuned = upstream.fit(*args, **kwargs)
        return type(self)(finetuned.model, device=self._device, _is_cute=False)
