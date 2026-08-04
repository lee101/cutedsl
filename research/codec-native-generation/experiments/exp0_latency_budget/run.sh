#!/usr/bin/env bash
# Experiment 0: where does end-to-end image generation latency actually go?
#
# Reproduces the budget table in ../../README.md. The point is to establish, before
# any model work, how much of the wall clock a codec-native output path could
# possibly recover. Answer on a 3090 at 512px: about 4%.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SDCPP="${SDCPP:-/media/lee/pcd/code/cutedsl/external/stable-diffusion.cpp/build/bin/sd-cli}"
MODELS="${MODELS:-/media/lee/pcd/models/zimage}"
VAE="${VAE:-/mnt/fast/models/FLUX.1-schnell/ae.safetensors}"
STEPS="${STEPS:-3}"
SIZE="${SIZE:-512}"

echo "== encode cost in isolation =="
cc -O2 -o "$HERE/enc_bench" "$HERE/enc_bench.c" -ldl
"$HERE/enc_bench" 512
"$HERE/enc_bench" 1024

echo
echo "== pipeline stages (${SIZE}px, ${STEPS} steps, TAEF1 decoder) =="
# -v makes stable-diffusion.cpp report per-stage timings; we want the conditioner,
# sampler, and decoder split out rather than one total.
"$SDCPP" \
  --diffusion-model "$MODELS/z_image_turbo-Q8_0.gguf" \
  --tae "$MODELS/taef1.safetensors" \
  --llm "$MODELS/Qwen3-4B-Instruct-2507-Q8_0.gguf" \
  -p "a cinematic portrait of a friendly explorer, soft studio lighting, detailed" \
  --cfg-scale 1.0 --steps "$STEPS" -W "$SIZE" -H "$SIZE" --seed 42 \
  --diffusion-fa -v -o "$HERE/out.png" 2>&1 |
  grep -E "get_learned_condition completed|sampling completed|decode_first_stage completed|generate_image completed"

echo
echo "== same, full VAE instead of TAEF1 (decode cost for comparison) =="
"$SDCPP" \
  --diffusion-model "$MODELS/z_image_turbo-Q8_0.gguf" \
  --vae "$VAE" \
  --llm "$MODELS/Qwen3-4B-Instruct-2507-Q8_0.gguf" \
  -p "a cinematic portrait of a friendly explorer, soft studio lighting, detailed" \
  --cfg-scale 1.0 --steps "$STEPS" -W "$SIZE" -H "$SIZE" --seed 42 \
  --diffusion-fa -v -o "$HERE/out_fullvae.png" 2>&1 |
  grep -E "decode_first_stage completed|generate_image completed"
