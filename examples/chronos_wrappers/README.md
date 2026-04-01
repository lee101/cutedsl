# Chronos Wrapper Examples

This directory benchmarks the same `CuteChronos2` inference path from three host languages:

- `C` via the embedded-Python shared library
- `Go` via `cgo` calling that shared library
- `Python` via a thin `cutechronos.foreign` wrapper

The benchmark also compares those wrapper timings against the upstream `Chronos2Pipeline` Python path.

## Runtime

By default the build targets the local Python env that already has Torch and the upstream Chronos checkout on this machine:

- `PYTHON=/home/lee/code/stock/.venv/bin/python`
- `CHRONOS_SRC=/home/lee/code/stock/chronos-forecasting/src`

Override either variable if your environment lives elsewhere.

## Build

```bash
make -C examples/chronos_wrappers
```

## Run The Cross-Language Benchmark

```bash
/home/lee/code/stock/.venv/bin/python examples/chronos_wrappers/benchmark.py \
  --device cuda \
  --runs 5 \
  --warmup 1 \
  --stock-data-dir /home/lee/code/stock/trainingdatadailybinance \
  --symbols BTCUSD ETHUSD SOLUSD
```

That runs:

- the toy sequence `[2, 4, 6, 8, 12]` with actual `[14, 16, 18]`
- a few real CSV series from the stock dataset

Each result includes the forecast, `MAE`, `MAPE%`, and both:

- `avg_inner_latency_ms`: time spent inside the embedded Python inference call
- `avg_outer_latency_ms`: end-to-end host-wrapper timing for that language

## Tune CuteChronos

Use the tuner when you want to search for the best inference-time setup for the Cute backend itself. It sweeps:

- pre-augmentation strategy
- context length
- output quantile
- inference strategy (`single` or dilation ensembles with aggregation)
- dtype
- compile mode

and scores each configuration over multiple rolling holdout windows from the CSV dataset.

```bash
/home/lee/code/stock/.venv/bin/python examples/chronos_wrappers/tune_cutechronos.py \
  --device cuda \
  --symbols BTCUSD ETHUSD SOLUSD \
  --context-lengths 128 256 512 1024 \
  --quantiles 0.3 0.5 0.7 \
  --inference-strategies single dilation \
  --windows 16 \
  --stride 7
```

The tuner writes:

- `examples/chronos_wrappers/tuning_results.json`
- `examples/chronos_wrappers/tuning_results.csv`

The default selection metric is `mean_mape_pct`, so the top rows are directly comparable across different price scales.

For dilation ensembles, repeated identical passes are not useful because CuteChronos2 inference is deterministic for a fixed input. The tunable analogue of "more samples" here is "more distinct context views", for example `stride=1,2,4` combined with `median` or `trimmed_mean`.
