# CuteDSL

Accelerated model inference via custom CuTeDSL and CUDA kernels. CuteDSL converts popular ML models into optimized versions with fused operations, custom attention kernels, and reduced memory allocations while maintaining output equivalence.

This is not just CuteDSL related the goal is to just build the fastest and best froneir models possible and catalog them, much like transformers/diffusers etc.... doesnt need to be cutedsl only!

A lot of this is created by autoresearch style bots that try to maintain evals while speeding up models and fusing kernels with cutedsl seems to be working well right now.

![cutedsl.png](./cutedsl.png)

### memecoin

Assiciated with this memecoin for raising funds for this project [CuTeDSL memecoin](https://bags.fm/launch/D322k7ykdgCmNGUZL5XvsgZXdHU4ks8iGoWtfrnmBAGS)

Hopefully this project can baloon out into a kind of transformers or ggml style repo with many useful accelerated models!

Please any help welcome!

# Models
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

# Batch of multivariate series
multi_context = torch.randn(8, 3, 512)
quantiles, mean = pipe.predict_quantiles(
    multi_context,
    prediction_length=24,
    quantile_levels=[0.1, 0.5, 0.9],
)
# quantiles[0].shape: (3, 24, 3)
# mean[0].shape: (3, 24)
```

CuteChronos2Pipeline now supports:

- `1-D` tensors for a single univariate series
- `2-D` tensors for a batch of univariate series
- `3-D` tensors with shape `(batch, n_variates, history_length)` for multivariate inference
- lists mixing univariate `1-D` items and multivariate `2-D` items

Dictionary-style covariate inputs are still not implemented in Cute. For `past_covariates` and `future_covariates`, use the upstream Chronos-2 pipeline for now.

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

## Parakeet STT Experiments

CuteDSL also includes a Parakeet ASR experiment harness for sweeping latency,
real-time factor, and transcript drift across inference settings.

Install the optional ASR stack:

```bash
uv pip install -e ".[parakeet]"
```

Run a sweep over batch size, timestamp extraction, input mode, and autocast:

```bash
python -m cuteparakeet.benchmark \
    --device cuda \
    --audio-dir /home/lee/code/voicetype/test_audio \
    --reference-json /home/lee/code/voicetype/test_audio/ground_truth.json \
    --batch-sizes 1 4 8 12 16 \
    --timestamps off on \
    --input-modes path array \
    --amp-dtypes none bf16 \
    --allow-tf32 \
    --output-json /tmp/parakeet_results.json
```

Current direction from local experiments on the bundled `voicetype` samples:

- `batch_size=12` with `input_mode=array`, `bf16`, `timestamps=off` was the best throughput point so far.
- Preloading audio arrays was slightly faster and slightly more accurate than path-based input on this corpus.
- Enabling timestamps reduced throughput materially.
- `torch.compile` on the Parakeet encoder was a large regression and is not recommended.
- The first GPU call is materially slower than steady-state because NeMo does decoder graph setup on first use, so benchmark first-call and steady-state separately.

There is also an ONNX export utility:

```bash
python -m cuteparakeet.export_onnx \
    --device cuda \
    --batch-size 1 \
    --audio-seconds 4 \
    --output-dir /tmp/parakeet_onnx \
    --simplify
```

### Cross-Language Wrapper Examples

There is also a wrapper benchmark under [`examples/chronos_wrappers/`](/home/lee/code/cutedsl/examples/chronos_wrappers) that drives the same `CuteChronos2` inference path from:

- C
- Go
- Python

and compares those wrapper timings against upstream Chronos-2 Python.

```bash
make -C examples/chronos_wrappers

/home/lee/code/stock/.venv/bin/python examples/chronos_wrappers/benchmark.py \
    --device cuda \
    --runs 5 \
    --warmup 1 \
    --stock-data-dir /home/lee/code/stock/trainingdatadailybinance \
    --symbols BTCUSD ETHUSD SOLUSD
```

The benchmark includes the toy sequence `[2, 4, 6, 8, 12] -> [14, 16, 18]` plus a few real stock/crypto CSV series, and reports both absolute `MAE` and percentage `MAPE%`.

There is also a tuner at [`examples/chronos_wrappers/tune_cutechronos.py`](/home/lee/code/cutedsl/examples/chronos_wrappers/tune_cutechronos.py) that sweeps pre-augmentation strategy, context length, quantile, inference ensemble strategy, dtype, and compile mode over rolling holdout windows, ranking configs by `MAPE%` by default.

For LoRA fine-tuning, there is a frontier runner at [`examples/lora_frontier.py`](/home/lee/code/cutedsl/examples/lora_frontier.py). It drives the stock repo's Chronos-2 LoRA trainer, measures wall-clock training time, records validation/test MAE%, and emits the Pareto frontier for `train_seconds` versus `val_mae_percent`.

```bash
/home/lee/code/stock/.venv/bin/python examples/lora_frontier.py \
    --symbols BTCUSD \
    --preaugs percent_change log_returns baseline \
    --context-lengths 128 256 \
    --learning-rates 5e-5 1e-4 \
    --num-steps 50 100 250
```

To verify an exported LoRA checkpoint can be loaded back through CuteChronos inference, use [`examples/eval_exported_lora.py`](/home/lee/code/cutedsl/examples/eval_exported_lora.py):

```bash
/home/lee/code/stock/.venv/bin/python examples/eval_exported_lora.py \
    --model-id /path/to/finetuned-ckpt \
    --symbol BTCUSD \
    --preaug percent_change \
    --context-length 128 \
    --prediction-length 24
```

For bulk hourly-crypto tuning, use [`examples/crypto_hourly_frontier.py`](/home/lee/code/cutedsl/examples/crypto_hourly_frontier.py). It stages a broad per-symbol inference sweep first, optionally tries dilation ensembles only on laggards, then launches bounded LoRA frontier runs only where the base model is still weak.

```bash
/home/lee/code/stock/.venv/bin/python examples/crypto_hourly_frontier.py \
    --data-root /home/lee/code/stock/trainingdatahourly/crypto \
    --run-dilation-pass \
    --run-lora-pass
```

To test whether a mixed-symbol crypto LoRA base transfers to harder targets, use [`examples/mixed_crypto_lora_transfer.py`](/home/lee/code/cutedsl/examples/mixed_crypto_lora_transfer.py). It trains a LoRA on a pool of hourly crypto symbols, exports a normal `finetuned-ckpt`, and reports target-symbol validation/test MAPE before and after the mixed training.

```bash
/home/lee/code/stock/.venv/bin/python examples/mixed_crypto_lora_transfer.py \
    --max-train-symbols 8 \
    --eval-symbols SOLUSDT ETHUSDT
```

To test raw multivariate pair forecasting on hourly crypto data, use [`examples/multivariate_pair_eval.py`](/home/lee/code/cutedsl/examples/multivariate_pair_eval.py). It now supports small frontier-style searches over partner pools, context lengths, strides, and `cross_learning`, so it is useful both for quick pair checks and for finding lower-MAE multivariate setups before porting them into native services.

```bash
/home/lee/code/stock/.venv/bin/python examples/multivariate_pair_eval.py \
    --backends cute \
    --target-symbols TAOUSD SOLUSDT BTCUSDT ETHUSDT \
    --partner-symbols ETHUSDT SOLUSDT BTCUSDT BTCUSD ETHUSD SOLUSD \
    --partner-pool-sizes 1 2 \
    --context-lengths 128 256 \
    --strides 12 24 \
    --cross-learning-options off on \
    --windows 64 \
    --warmup-windows 4 \
    --max-multivariate-runs 64
```

This emits both JSON and CSV summaries plus a Pareto-style frontier ranked by `avg_latency_ms` versus `mean_mape_pct`.

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
