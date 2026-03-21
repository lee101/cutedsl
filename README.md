# CuteDSL

Accelerated model inference via custom Triton and CUDA kernels. CuteDSL converts popular ML models into optimized versions with fused operations, custom attention kernels, and reduced memory allocations while maintaining output equivalence.

This is not just CuteDSL related the goal is to just build the fastest and best froneir models possible and catalog them, much like transformers/diffusers etc.... doesnt need to be cutedsl only!

![cutedsl.png](./cutedsl.png)

## CuteChronos2

[Amazon Chronos-2](https://github.com/amazon-science/chronos-forecasting) is a state-of-the-art time series forecasting model. CuteChronos2 is a from-scratch reimplementation with:

- **Custom Triton kernels** for unscaled attention, RoPE, RMS LayerNorm, fused LN+Linear, fused MLP, fused preprocessing, and fused output transform
- **C++/CUDA preprocessing** kernels for NaN-aware normalization and patching
- **Memory optimizations**: in-place residual adds, cached position_ids, `.reshape()` instead of `.contiguous().view()`
- **torch.compile support** with `reduce-overhead` mode for additional speedup
- **1.4x+ speedup** over the original implementation with identical outputs (max abs error < 1e-4)

### Architecture

CuteChronos2 reimplements the full Chronos-2 encoder-only transformer:

```
Input Series -> InstanceNorm -> Patching -> InputPatchEmbedding
    -> [EncoderBlock x 12] -> FinalLayerNorm -> OutputPatchEmbedding
    -> Inverse InstanceNorm -> Quantile Predictions

Each EncoderBlock:
    TimeSelfAttention (RoPE, unscaled attention)
    GroupSelfAttention (cross-series attention)
    FeedForward (RMS LN + MLP)
```

### Triton Kernels

| Kernel | What it fuses | Speedup source |
|--------|--------------|----------------|
| `unscaled_attention` | QK^T + mask + softmax + V multiply (no 1/sqrt(d_k) scaling) | Avoids materializing S*S attention matrix |
| `rms_layernorm` | T5-style RMS norm (FP32 variance) | Single kernel, no intermediate tensors |
| `rope` | inv_freq + cos/sin + Q/K rotation | Single kernel for both Q and K |
| `fused_rms_norm_linear` | RMS LayerNorm + linear projection | Eliminates normalized intermediate |
| `fused_rms_norm_qkv` | RMS LayerNorm + Q/K/V projections | One normalization, three projections |
| `fused_linear_relu` | Linear + ReLU activation | Eliminates pre-ReLU intermediate |
| `fused_mlp_relu` | Two-layer MLP (linear + relu + linear) | Avoids materializing 3072-wide hidden |
| `fused_preprocess` | NaN-aware normalize + arcsinh + patch + time_enc | Two-phase: reduce then transform |
| `fused_output_transform` | Rearrange + sinh + unscale | Single pass over output tensor |

## Quick Start

### Requirements

- Python >= 3.10
- PyTorch >= 2.2.0
- NVIDIA GPU with CUDA support
- Triton >= 3.0.0

### Installation

```bash
pip install uv
uv venv
source .venv/bin/activate

# Install the package
uv pip install -e .

# For validation against original model (optional)
uv pip install -e ".[chronos]"

# For development
uv pip install -e ".[dev]"
```

### Download and Convert Chronos-2

```bash
# Convert the default model (amazon/chronos-2)
python -m cutechronos.convert --benchmark

# Convert with validation and compiled benchmark
python -m cutechronos.convert \
    --model-id amazon/chronos-2 \
    --benchmark \
    --benchmark-compiled \
    --output-json results.json

# The model is automatically downloaded from HuggingFace Hub
# and converted to CuteChronos2 format.
```

### Use in Code

```python
import torch
from cutechronos.model import CuteChronos2Model

# Load from HuggingFace checkpoint directory
model = CuteChronos2Model.from_pretrained("path/to/chronos-bolt-base")
model = model.to("cuda", torch.bfloat16)

# Or with torch.compile for even faster inference
model = CuteChronos2Model.from_pretrained_compiled(
    "path/to/chronos-bolt-base",
    compile_mode="reduce-overhead",
)
model = model.to("cuda", torch.bfloat16)

# Run inference
context = torch.randn(1, 512, device="cuda")  # (batch, time_steps)
with torch.inference_mode():
    quantile_preds = model(context)  # (batch, 21_quantiles, prediction_length)
```

### Pipeline API (matches upstream)

```python
from cutechronos.pipeline import CuteChronos2Pipeline

pipe = CuteChronos2Pipeline.from_pretrained(
    "amazon/chronos-bolt-base",
    device="cuda",
    dtype=torch.bfloat16,
)

# Single series
context = torch.randn(512)
predictions = pipe.predict(context, prediction_length=30)
# predictions[0].shape: (1, 21, 30)

# Batch of variable-length series
contexts = [torch.randn(300), torch.randn(400), torch.randn(512)]
predictions = pipe.predict(contexts, prediction_length=30)
```

### Run Benchmarks

```bash
# Quick benchmark with synthetic data
python -m cutechronos.benchmark --device cuda --n-runs 20

# Benchmark with real data
python -m cutechronos.benchmark \
    --data-dir /path/to/csv/data \
    --symbols BTCUSD ETHUSD \
    --context-length 512 \
    --prediction-length 30
```

### Run Tests

```bash
# All tests (requires chronos-forecasting for model comparison tests)
pytest cutechronos/tests/ -v

# Tests that don't require the original model
pytest cutechronos/tests/ -v -m "not model_required"

# Individual kernel tests
pytest cutechronos/tests/test_attention.py -v
pytest cutechronos/tests/test_rms_layernorm.py -v
pytest cutechronos/tests/test_rope.py -v
```

## Benchmark Results

On RTX 5090, Chronos-2 base (768 d_model, 12 layers), B=1 L=512:

| Implementation | Latency (ms) | Speedup | GPU Memory (MB) |
|----------------|-------------|---------|-----------------|
| Original Chronos2Pipeline | 30.9 | baseline | 248 |
| CuteChronos2Model (eager) | 24.0 | **1.3x** | 247 |
| CuteChronos2Model (torch.compile) | 1.3 | **24.4x** | 237 |

The compiled mode uses `torch.compile(mode="reduce-overhead")` which captures CUDA graphs for near-zero kernel launch overhead. This is the recommended mode for production inference with fixed input shapes.

## Second Model: CuteZImage

[Z-Image Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) is a fast text-to-image diffusion model. CuteZImage reimplements the transformer backbone with:

- **Fused SiLU-gated FFN**: Eliminates the 10240-wide intermediate allocation
- **Fused AdaLN + RMS Norm**: Timestep conditioning fused with normalization
- **Complex-valued RoPE kernel**: Fused reshape + complex multiply + flatten
- **RMS LayerNorm kernel**: Triton-accelerated T5-style normalization
- **PyTorch SDPA attention**: Leverages FlashAttention v2 automatically
- **from_diffusers() weight loading**: Load from any HuggingFace Z-Image checkpoint

Architecture: 30 main layers + 2 refiner layers, dim=3840, 30 heads, SiLU-gated FFN (hidden=10240).

## Project Structure

```
cutedsl/
  cutechronos/
    model.py              # Full CuteChronos2Model (encoder + weight loading)
    pipeline.py           # Drop-in pipeline wrapper
    convert.py            # Model download + conversion CLI
    benchmark.py          # End-to-end benchmarking
    kernels.py            # C++/CUDA extension loader
    modules/
      _fallbacks.py       # Pure PyTorch fallback implementations
      time_attention.py   # Fused TimeSelfAttention module
      group_attention.py  # Fused GroupSelfAttention module
      feedforward.py      # Fused FeedForward module
      output.py           # Fused output head module
    triton_kernels/
      attention.py        # Unscaled tiled attention (FlashAttention-style)
      rms_layernorm.py    # T5-style RMS LayerNorm
      rope.py             # Fused RoPE (inv_freq + cos/sin + apply)
      fused_layernorm_linear.py  # LN + Linear / LN + QKV
      fused_mlp.py        # Linear+ReLU / full MLP fusion
      fused_preprocess.py # NaN-aware preprocessing pipeline
      fused_output.py     # Rearrange + sinh + unscale
    cpp/
      preprocessing.cpp   # C++ preprocessing kernels
      preprocessing.cu    # CUDA preprocessing kernels
    tests/
      test_model.py       # Model equivalence tests
      test_pipeline.py    # Pipeline API tests
      test_attention.py   # Attention kernel tests
      ...
  cutezimage/
    model.py              # CuteZImageTransformer (30 layers + weight loading)
    triton_kernels/
      rms_norm.py         # RMS LayerNorm
      fused_silu_gate_ffn.py  # SiLU + gating + FFN fusion
      fused_adaln_norm.py # AdaLN + RMS norm fusion
      rope_complex.py     # Complex-valued RoPE
    tests/
      test_model.py       # Model component tests
      test_kernels.py     # Triton kernel correctness tests
```

## Adding New Models

CuteDSL is designed to accelerate more models beyond Chronos-2. The pattern:

1. Identify the bottleneck operations in the original model
2. Write Triton kernels that fuse multiple operations
3. Create a model class that loads original weights and uses fused kernels
4. Validate output equivalence within tight tolerance
5. Benchmark to confirm speedup

## License

Apache-2.0
