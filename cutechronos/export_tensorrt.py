"""Analyze/export CuteChronos2 for ONNX/TensorRT.

This tool is intentionally conservative:

- It exports via the pure PyTorch fallback path rather than custom Triton/CUDA
  kernels so the resulting graph is more likely to import into ONNX/TensorRT.
- TensorRT build is optional and only attempted when ``tensorrt`` is installed.
- The primary artifact is a JSON report describing graph breaks, ONNX export
  status, ONNX operator mix, and optional TensorRT engine build details.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from cutechronos.model import CuteChronos2Model

try:
    import onnx
except ImportError:  # pragma: no cover - optional dependency at runtime
    onnx = None


def _dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return mapping[name]
    except KeyError as exc:  # pragma: no cover - argparse constrains this
        raise ValueError(f"Unsupported dtype: {name}") from exc


def _tensor_shape_proto_to_list(value_info: Any) -> list[int | str]:
    shape: list[int | str] = []
    tensor_type = getattr(value_info.type, "tensor_type", None)
    if tensor_type is None:
        return shape
    for dim in tensor_type.shape.dim:
        if dim.dim_value:
            shape.append(int(dim.dim_value))
        elif dim.dim_param:
            shape.append(dim.dim_param)
        else:
            shape.append("?")
    return shape


def _serialize_break_reason(reason: Any) -> dict[str, Any]:
    return {
        "reason": getattr(reason, "reason", str(reason)),
        "user_stack": [str(item) for item in getattr(reason, "user_stack", [])],
        "graph_break": bool(getattr(reason, "graph_break", True)),
    }


@contextlib.contextmanager
def _export_fallback_backends() -> Any:
    overrides = {
        "CUTECHRONOS_RMS_BACKEND": "torch",
        "CUTECHRONOS_ATTENTION_BACKEND": "torch",
        "CUTECHRONOS_FUSED_QKV": "0",
    }
    old_values = {key: os.environ.get(key) for key in overrides}
    try:
        os.environ.update(overrides)
        yield overrides
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


class _FixedHorizonChronosWrapper(torch.nn.Module):
    def __init__(self, model: CuteChronos2Model, num_output_patches: int):
        super().__init__()
        self.model = model
        self.num_output_patches = num_output_patches

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.model.predict(context, num_output_patches=self.num_output_patches)


def _resolve_model_path(model_path: str) -> str:
    path = Path(model_path)
    if path.is_dir():
        return str(path)

    from huggingface_hub import snapshot_download

    return snapshot_download(
        model_path,
        allow_patterns=["*.json", "*.safetensors", "*.bin"],
    )


def _build_sample_context(
    batch_size: int,
    context_length: int,
    *,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    base = torch.linspace(1.0, 2.0, steps=context_length, device=device, dtype=torch.float32)
    batch_offsets = torch.arange(batch_size, device=device, dtype=torch.float32).unsqueeze(1) * 0.05
    context = base.unsqueeze(0) + batch_offsets
    return context.to(dtype)


def _top_counter(counter: Counter[str], limit: int = 20) -> list[dict[str, Any]]:
    return [{"name": name, "count": count} for name, count in counter.most_common(limit)]


def _trim_text(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...<truncated>..."


def _summarize_exported_program(exported_program: Any) -> dict[str, Any]:
    graph_module = exported_program.graph_module
    node_ops = Counter(str(node.target) for node in graph_module.graph.nodes if node.op == "call_function")
    return {
        "graph_signature": str(exported_program.graph_signature),
        "range_constraints": {str(key): str(value) for key, value in exported_program.range_constraints.items()},
        "call_function_ops": _top_counter(node_ops),
        "node_count": sum(1 for _ in graph_module.graph.nodes),
    }


def _collect_dynamo_report(module: torch.nn.Module, sample_context: torch.Tensor) -> dict[str, Any]:
    if not hasattr(torch, "_dynamo") or not hasattr(torch._dynamo, "explain"):
        return {"available": False, "reason": "torch._dynamo.explain unavailable"}

    try:
        explain_fn = torch._dynamo.explain(module)
        result = explain_fn(sample_context)
        return {
            "available": True,
            "graph_count": int(getattr(result, "graph_count", 0)),
            "graph_break_count": int(getattr(result, "graph_break_count", 0)),
            "op_count": int(getattr(result, "op_count", 0)),
            "ops_per_graph": [[str(op) for op in ops] for ops in getattr(result, "ops_per_graph", [])],
            "break_reasons": [_serialize_break_reason(reason) for reason in getattr(result, "break_reasons", [])],
        }
    except Exception as exc:  # pragma: no cover - exporter failure is environment-specific
        return {
            "available": True,
            "success": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def _export_program_report(
    module: torch.nn.Module,
    sample_context: torch.Tensor,
    *,
    batch_dim_max: int,
    context_dim_min: int,
    context_dim_max: int,
    patch_size: int,
) -> dict[str, Any]:
    if not hasattr(torch, "export"):
        return {"available": False, "reason": "torch.export unavailable"}

    context_dim, context_shape_meta = _dynamic_context_shape_spec(
        patch_size=patch_size,
        min_context_length=context_dim_min,
        max_context_length=context_dim_max,
    )

    try:
        dynamic_shapes = {
            "context": {
                0: torch.export.Dim("batch", min=1, max=batch_dim_max),
                1: context_dim,
            }
        }
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            exported_program = torch.export.export(
                module,
                args=(sample_context,),
                dynamic_shapes=dynamic_shapes,
                strict=False,
            )
        summary = _summarize_exported_program(exported_program)
        summary.update(
            {
                "available": True,
                "success": True,
                "export_mode": "dynamic",
                "context_shape_constraints": context_shape_meta,
            }
        )
        if stdout_buffer.getvalue():
            summary["stdout"] = _trim_text(stdout_buffer.getvalue())
        if stderr_buffer.getvalue():
            summary["stderr"] = _trim_text(stderr_buffer.getvalue())
        return summary
    except Exception as exc:  # pragma: no cover - exporter failure is environment-specific
        dynamic_error = {
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "context_shape_constraints": context_shape_meta,
        }

    try:
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            exported_program = torch.export.export(
                module,
                args=(sample_context,),
                strict=False,
            )
        summary = _summarize_exported_program(exported_program)
        summary.update(
            {
                "available": True,
                "success": True,
                "export_mode": "static",
                "context_shape_constraints": context_shape_meta,
                "dynamic_export_error": dynamic_error["error"],
            }
        )
        if stdout_buffer.getvalue():
            summary["stdout"] = _trim_text(stdout_buffer.getvalue())
        if stderr_buffer.getvalue():
            summary["stderr"] = _trim_text(stderr_buffer.getvalue())
        return summary
    except Exception as exc:  # pragma: no cover - exporter failure is environment-specific
        return {
            "available": True,
            "success": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "dynamic_export_error": dynamic_error["error"],
            "dynamic_export_traceback": dynamic_error["traceback"],
            "context_shape_constraints": context_shape_meta,
        }


def _summarize_onnx_model(path: Path) -> dict[str, Any]:
    if onnx is None:
        return {"available": False, "reason": "onnx not installed"}

    model = onnx.load(str(path))
    op_histogram = Counter(node.op_type for node in model.graph.node)
    return {
        "available": True,
        "size_bytes": path.stat().st_size,
        "node_count": len(model.graph.node),
        "op_histogram": _top_counter(op_histogram),
        "inputs": [
            {"name": value.name, "shape": _tensor_shape_proto_to_list(value)}
            for value in model.graph.input
        ],
        "outputs": [
            {"name": value.name, "shape": _tensor_shape_proto_to_list(value)}
            for value in model.graph.output
        ],
    }


def _aligned_patch_multiple(value: int, patch_size: int, *, mode: str) -> int:
    if patch_size <= 0:
        return value
    if mode == "floor":
        return max(patch_size, (value // patch_size) * patch_size)
    if mode == "ceil":
        return max(patch_size, ((value + patch_size - 1) // patch_size) * patch_size)
    raise ValueError(f"Unsupported alignment mode: {mode}")


def _dynamic_context_shape_spec(
    *,
    patch_size: int,
    min_context_length: int,
    max_context_length: int,
) -> tuple[Any, dict[str, int]]:
    min_aligned = _aligned_patch_multiple(min_context_length, patch_size, mode="ceil")
    max_aligned = _aligned_patch_multiple(max_context_length, patch_size, mode="floor")
    if max_aligned < min_aligned:
        max_aligned = min_aligned

    if patch_size == 1:
        context_dim = torch.export.Dim("context_length", min=min_aligned, max=max_aligned)
    else:
        min_blocks = max(1, min_aligned // patch_size)
        max_blocks = max(1, max_aligned // patch_size)
        context_blocks = torch.export.Dim("context_blocks", min=min_blocks, max=max_blocks)
        context_dim = patch_size * context_blocks

    return context_dim, {
        "patch_size": patch_size,
        "requested_min": min_context_length,
        "requested_max": max_context_length,
        "effective_min": min_aligned,
        "effective_max": max_aligned,
    }


def _export_onnx(
    module: torch.nn.Module,
    sample_context: torch.Tensor,
    *,
    output_path: Path,
    artifacts_dir: Path,
    batch_dim_max: int,
    context_dim_min: int,
    context_dim_max: int,
    patch_size: int,
    opset_version: int,
) -> dict[str, Any]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    context_dim, context_shape_meta = _dynamic_context_shape_spec(
        patch_size=patch_size,
        min_context_length=context_dim_min,
        max_context_length=context_dim_max,
    )
    dynamic_shapes = {
        "context": {
            0: torch.export.Dim("batch", min=1, max=batch_dim_max),
            1: context_dim,
        }
    }
    dynamic_axes = {
        "context": {0: "batch", 1: "context_length"},
        "quantile_preds": {0: "batch"},
    }

    started_at = time.perf_counter()
    try:
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            torch.onnx.export(
                module,
                args=(sample_context,),
                f=str(output_path),
                input_names=["context"],
                output_names=["quantile_preds"],
                opset_version=opset_version,
                dynamo=True,
                dynamic_shapes=dynamic_shapes,
                optimize=True,
                report=True,
                artifacts_dir=str(artifacts_dir),
                fallback=False,
            )
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        summary = {
            "success": True,
            "path": str(output_path),
            "elapsed_ms": elapsed_ms,
            "artifacts_dir": str(artifacts_dir),
            "opset_version": opset_version,
            "exporter": "dynamo",
            "context_shape_constraints": context_shape_meta,
        }
        if stdout_buffer.getvalue():
            summary["stdout"] = _trim_text(stdout_buffer.getvalue())
        if stderr_buffer.getvalue():
            summary["stderr"] = _trim_text(stderr_buffer.getvalue())
        summary.update(_summarize_onnx_model(output_path))
        return summary
    except Exception as exc:
        dynamo_error = {
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }

    try:
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            torch.onnx.export(
                module,
                args=(sample_context,),
                f=str(output_path),
                input_names=["context"],
                output_names=["quantile_preds"],
                opset_version=opset_version,
                dynamo=False,
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,
            )
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        summary = {
            "success": True,
            "path": str(output_path),
            "elapsed_ms": elapsed_ms,
            "artifacts_dir": str(artifacts_dir),
            "opset_version": opset_version,
            "exporter": "legacy",
            "dynamo_export_error": dynamo_error["error"],
            "context_shape_constraints": context_shape_meta,
        }
        if stdout_buffer.getvalue():
            summary["stdout"] = _trim_text(stdout_buffer.getvalue())
        if stderr_buffer.getvalue():
            summary["stderr"] = _trim_text(stderr_buffer.getvalue())
        summary.update(_summarize_onnx_model(output_path))
        return summary
    except Exception as exc:
        return {
            "success": False,
            "path": str(output_path),
            "elapsed_ms": (time.perf_counter() - started_at) * 1000.0,
            "artifacts_dir": str(artifacts_dir),
            "opset_version": opset_version,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "dynamo_export_error": dynamo_error["error"],
            "dynamo_export_traceback": dynamo_error["traceback"],
            "context_shape_constraints": context_shape_meta,
        }


def _build_tensorrt_engine(
    *,
    onnx_path: Path,
    plan_path: Path,
    input_name: str,
    min_shape: tuple[int, int],
    opt_shape: tuple[int, int],
    max_shape: tuple[int, int],
    workspace_bytes: int,
    fp16: bool,
) -> dict[str, Any]:
    try:
        import tensorrt as trt
    except ImportError as exc:
        return {"requested": True, "available": False, "error": str(exc)}

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    else:  # pragma: no cover - old TRT fallback
        config.max_workspace_size = workspace_bytes

    if fp16 and getattr(builder, "platform_has_fast_fp16", False):
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    profile.set_shape(input_name, min=min_shape, opt=opt_shape, max=max_shape)
    config.add_optimization_profile(profile)

    parse_errors: list[str] = []
    with open(onnx_path, "rb") as handle:
        parsed = parser.parse(handle.read())
    if not parsed:
        for idx in range(parser.num_errors):
            parse_errors.append(str(parser.get_error(idx)))
        return {
            "requested": True,
            "available": True,
            "built": False,
            "error": "TensorRT ONNX parse failed",
            "parse_errors": parse_errors,
        }

    started_at = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    elapsed_ms = (time.perf_counter() - started_at) * 1000.0
    if serialized is None:
        return {
            "requested": True,
            "available": True,
            "built": False,
            "error": "builder.build_serialized_network returned None",
            "parse_errors": parse_errors,
            "elapsed_ms": elapsed_ms,
        }

    plan_path.write_bytes(bytes(serialized))
    return {
        "requested": True,
        "available": True,
        "built": True,
        "path": str(plan_path),
        "size_bytes": plan_path.stat().st_size,
        "elapsed_ms": elapsed_ms,
        "fp16": fp16 and bool(getattr(builder, "platform_has_fast_fp16", False)),
        "workspace_bytes": workspace_bytes,
        "profiles": {
            "input_name": input_name,
            "min": list(min_shape),
            "opt": list(opt_shape),
            "max": list(max_shape),
        },
        "parse_errors": parse_errors,
    }


def analyze_cutechronos_tensorrt(
    *,
    model_path: str,
    output_dir: str,
    adapter_path: str | None = None,
    device: str = "cpu",
    dtype_name: str = "float32",
    batch_size: int = 4,
    context_length: int = 512,
    num_output_patches: int = 1,
    min_batch_size: int = 1,
    opt_batch_size: int | None = None,
    max_batch_size: int | None = None,
    min_context_length: int | None = None,
    opt_context_length: int | None = None,
    max_context_length: int | None = None,
    opset_version: int = 18,
    build_engine: bool = False,
    workspace_gb: float = 4.0,
    fp16: bool = True,
) -> dict[str, Any]:
    resolved_model_path = _resolve_model_path(model_path)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    opt_batch_size = opt_batch_size or batch_size
    max_batch_size = max_batch_size or batch_size
    min_context_length = min_context_length or context_length
    opt_context_length = opt_context_length or context_length
    max_context_length = max_context_length or context_length

    dtype = _dtype_from_name(dtype_name)
    if device == "cpu" and dtype != torch.float32:
        dtype = torch.float32
        dtype_name = "float32"

    with _export_fallback_backends() as backend_overrides:
        model = CuteChronos2Model.from_pretrained(resolved_model_path, adapter_path=adapter_path)
        model._use_fallback_preprocess = True
        model.instance_norm.export_safe_arcsinh = True
        model = model.to(device=device, dtype=dtype).eval()
        wrapper = _FixedHorizonChronosWrapper(model, num_output_patches=num_output_patches).eval()
        sample_context = _build_sample_context(
            batch_size=batch_size,
            context_length=context_length,
            device=device,
            dtype=dtype,
        )

        report: dict[str, Any] = {
            "model_path": model_path,
            "resolved_model_path": resolved_model_path,
            "adapter_path": adapter_path,
            "device": device,
            "dtype": dtype_name,
            "batch_size": batch_size,
            "context_length": context_length,
            "num_output_patches": num_output_patches,
            "input_patch_size": model.config.input_patch_size,
            "backend_overrides": backend_overrides,
            "environment": {
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "onnx_installed": onnx is not None,
                "onnxscript_installed": importlib.util.find_spec("onnxscript") is not None,
                "tensorrt_installed": importlib.util.find_spec("tensorrt") is not None,
            },
        }

        report["dynamo_explain"] = _collect_dynamo_report(wrapper, sample_context)
        report["torch_export"] = _export_program_report(
            wrapper,
            sample_context,
            batch_dim_max=max_batch_size,
            context_dim_min=min_context_length,
            context_dim_max=max_context_length,
            patch_size=model.config.input_patch_size,
        )

        onnx_path = output_dir_path / "cutechronos.onnx"
        report["onnx_export"] = _export_onnx(
            wrapper,
            sample_context,
            output_path=onnx_path,
            artifacts_dir=output_dir_path / "onnx_artifacts",
            batch_dim_max=max_batch_size,
            context_dim_min=min_context_length,
            context_dim_max=max_context_length,
            patch_size=model.config.input_patch_size,
            opset_version=opset_version,
        )

        if build_engine and report["onnx_export"].get("success"):
            report["tensorrt"] = _build_tensorrt_engine(
                onnx_path=onnx_path,
                plan_path=output_dir_path / "cutechronos.plan",
                input_name="context",
                min_shape=(min_batch_size, min_context_length),
                opt_shape=(opt_batch_size, opt_context_length),
                max_shape=(max_batch_size, max_context_length),
                workspace_bytes=int(workspace_gb * (1 << 30)),
                fp16=fp16,
            )
        else:
            report["tensorrt"] = {
                "requested": build_engine,
                "available": None,
                "built": False,
                "reason": "build not requested" if not build_engine else "skipped because ONNX export failed",
            }

    json_path = output_dir_path / "report.json"
    report["report_path"] = str(json_path)
    json_path.write_text(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze/export CuteChronos2 for ONNX/TensorRT")
    parser.add_argument("--model-path", default="amazon/chronos-2")
    parser.add_argument("--adapter-path")
    parser.add_argument("--output-dir", default="/tmp/cutechronos_trt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--num-output-patches", type=int, default=1)
    parser.add_argument("--min-batch-size", type=int, default=1)
    parser.add_argument("--opt-batch-size", type=int)
    parser.add_argument("--max-batch-size", type=int)
    parser.add_argument("--min-context-length", type=int)
    parser.add_argument("--opt-context-length", type=int)
    parser.add_argument("--max-context-length", type=int)
    parser.add_argument("--opset-version", type=int, default=18)
    parser.add_argument("--build-engine", action="store_true")
    parser.add_argument("--workspace-gb", type=float, default=4.0)
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = analyze_cutechronos_tensorrt(
        model_path=args.model_path,
        output_dir=args.output_dir,
        adapter_path=args.adapter_path,
        device=args.device,
        dtype_name=args.dtype,
        batch_size=args.batch_size,
        context_length=args.context_length,
        num_output_patches=args.num_output_patches,
        min_batch_size=args.min_batch_size,
        opt_batch_size=args.opt_batch_size,
        max_batch_size=args.max_batch_size,
        min_context_length=args.min_context_length,
        opt_context_length=args.opt_context_length,
        max_context_length=args.max_context_length,
        opset_version=args.opset_version,
        build_engine=args.build_engine,
        workspace_gb=args.workspace_gb,
        fp16=args.fp16,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
