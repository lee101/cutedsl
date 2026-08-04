# Z-Image Acceleration Experiments

## Current State (April 2025)

### Infrastructure
- **Daisy**: RTX 3090 24GB at /mnt/fast/code/cutedsl, Python 3.10, uv
- **Local**: development machine for code, synced via scripts/daisy_zimage.sh

### What Exists
- **CuteZImage**: Triton kernels (RoPE, RMS norm, SiLU-gated FFN, AdaLN) for Z-Image transformer
- **ZImageAccelerated**: Fused QKV projections on top of CuteZImage, torch.compile support
- **LatentTeleportation**: Cache-based step skipping with combiners (SLERP/neural/tree), trajectory priors, confidence gating, step forecaster
- **ZImageControlNet**: Line art + canny edge ControlNet training pipeline (newly added)
- **stable-diffusion.cpp**: Subprocess wrapper for GGUF-quantized Z-Image (sdcpp_benchmark.py)
- **External refs**: MeanCache, LeMiCa, cache-dit for caching research

### Baseline Performance
- Z-Image Turbo: 30 layers + 2 refiner, 6.15B params, dim=3840
- Diffusers at 512x512 9-step: ~3 it/s on 3090
- sdcpp GGUF Q3_K with CPU offload: TBD (need to benchmark)

### Served baseline (2026-07-31, images2.netwrck.com, 3090, sdcpp Q8_0)

The omniserve-native art gateway at `:8791`, 1024x1024, 3 steps, cfg 1.0,
TAEF1 decoder. Step count sweep, two runs each:

| steps | wall clock |
|-------|-----------|
| 1     | 1.34s     |
| 2     | 2.35s     |
| 3     | 3.37s     |
| 6     | 6.47s     |

Linear: **~1.03s per transformer step, ~0.31s fixed** (text encode + VAE decode +
webp). Per-step cost is ~91% of a 3-step request, so step count is the only
lever worth pulling — shaving the fixed cost cannot pay.

### GGUF quant below Q8_0 is slower, not faster (negative result)

Same prompt grid, seed-matched, per-iteration times straight from sdcpp (so
HTTP and encode are excluded):

| quant | s/it (median) | VRAM   | note |
|-------|---------------|--------|------|
| Q8_0  | **1.02**      | 10.7GB | fastest and best quality |
| Q4_K  | 1.07          | 8.1GB  | ~5% slower |
| Q6_K  | 1.20          | 9.4GB  | ~18% slower; **crashed** on the 5th generation |

At batch 1 and 1024x1024 Z-Image is compute-bound, not weight-bandwidth-bound,
so K-quant dequantisation costs more than the smaller weights save. Q8_0's
dequant is close to a bare scale, which is why it wins. Q6_K is both the slowest
and the only one that fell over.

The consequence for the "approximate it with a smaller model" track: the win has
to come from a genuinely smaller *architecture* (a distilled student, i.e. the
Track 0 walker) or from fewer real steps. Re-quantising the same 6.15B graph is
a dead end on this hardware. Do not re-run this sweep expecting a different
answer without changing the arithmetic, not the storage format.

### The sdcpp step caches cannot beat just lowering the step count (negative result)

stable-diffusion.cpp ships `cache_dit.hpp` (EasyCache / UCache / DBCache /
TaylorSeer / CacheDiT) and the gateway never enabled it. Now wired to
`OMNISERVE_NATIVE_SD_CACHE` and left **off**, because it loses:

| config                 | steps | median | PSNR vs 20-step ref |
|------------------------|-------|--------|---------------------|
| uncached (deployed)    | 3     | 5.81s  | **15.07**           |
| uncached               | 6     | 11.64s | 17.27               |
| taylorseer, warmup=1   | 6     | 11.87s | 17.27               |
| dbcache, warmup=1      | 6     | 8.16s  | **15.12**           |
| taylorseer, warmup=1   | 10    | 18.73s | 20.25               |

(Absolute times inflated ~40% by thermal throttling late in the run — compare
within the table only, not against the 3.37s figure above.)

- **DBCache works** and is ~30% faster than uncached at the same step count, but
  its skipping costs exactly the fidelity the extra steps bought: 6 steps cached
  lands at PSNR 15.12, statistically the same as 3 steps uncached (15.07), for
  40% more wall clock. Strictly dominated by just running 3 steps.
- **TaylorSeer is a no-op on Z-Image in this build** — byte-identical output to
  uncached at every warmup and skip interval tried, while still paying the
  bookkeeping. Engaged per the startup banner, so this is the sampler not
  calling the hook, not a config error.
- `max_warmup_steps` defaults to **8**, so at 3-6 steps every step is warmup and
  nothing is ever cached. That is why the first pass showed identical images plus
  overhead. Any future attempt has to lower it or nothing happens.

Why this should have been predictable: these schemes exploit redundancy between
adjacent steps, and Turbo is already distilled down to 3 — the redundancy was
consumed at distillation time. Step-skipping a 3-step schedule has nothing to
skip. The fidelity ladder (3: 15.07, 6: 17.27, 8: 18.17, 10: 20.25) is real, so
extra steps do buy quality; they just cannot be had cheaply this way.

Where that leaves the acceleration work: the remaining headroom is not in
skipping steps of the existing schedule. It is in making each step cheaper
(kernels, Track 5) or in a smaller student that replaces steps outright
(Track 0's walker) — not in caching, and not in requantising.

Two traps this sweep hit, both now fixed, worth knowing before re-running it:
- `scripts/run-art-zimage.sh` hard-assigned the model path, so an env override
  was discarded and the first sweep measured Q8_0 three times and reported it as
  three quants (PSNR inf, SSIM 1.0 — which is the tell).
- On SIGTERM the gateway closes its listener and frees VRAM *before* it exits, so
  waiting on the port hands the next block a half-dead predecessor that is still
  answering `/v1/models`. Wait for the pid, then poll until the served model id
  is the expected one.
- Within every block latency creeps monotonically (~1.01 -> 1.05 s/it over ten
  generations) from clock/thermal drift. Anything under ~8% needs the reference
  re-measured last, not a single block comparison.

---

## Experiment Tracks

### Track 0: Speculative Latent Walking (MEASURED — negative end-to-end result)
**Goal**: Speculative decoding for diffusion. Big model takes 1 real step, a small
walker drafts k cheap steps through latent space, a learned gap interpolator
teleports the draft endpoint onto where the big model's trajectory would be,
big model verifies with the next real step. Target: large step-count reduction
with near-zero quality loss on Z-Image Turbo (images.netwrck.com prod model).

**Code**: `latentteleport/speculative.py` (LatentWalker draft net, GapInterpolator
teleport net, `speculative_denoise` manual FlowMatch loop that actually skips
transformer calls), `scripts/spec_collect.py` (trajectory capture →
/sdb-disk/latentteleport-spec), `scripts/spec_gap.py` (predictability analysis:
identity vs taylor1 vs affine per (t,k)), `scripts/spec_train.py` (joint walker+
interp training, relL2 eval), `scripts/spec_e2e.py` (wall-clock + PSNR vs
full-step baseline, saves image triptychs).

**Design notes**:
- relL2 metric = err / actual-movement; 1.0 means predictor is no better than
  not moving. taylor1 is the training-free floor (cgtaylor-style); the walker
  must beat it and the interpolator must close the remaining gap.
- Interpolator is residual-on-draft with zero-init out conv → starts as
  identity-on-walker, can only help.
- Trained on 16-step 512x512 trajectories, guidance 0 (prod turbo settings are
  4-step 1024x1024 — resolution/step-count generalization is a later sweep).
- Combine with Track 2 kNN trajectory priors: warm-start the walker rollout
  with neighbor deltas; and with cgtaylor-style confidence gating for
  accept/reject (fall back to real steps when the interpolator is uncertain).

**Results so far (2026-07-13, 200 trajs, 512x512, seed-matched)**:
- Gap analysis: taylor1 relL2 0.21 (k=1) - 0.35 (k=4) mid-trajectory; scalar affine ~0.88; final scheduler step is a NO-OP (duplicate latent).
- v1 walker (position only) plateaued at relL2 ~0.6 — worse than taylor.
- v2 walker conditioned on momentum (x_t - x_{t-1}) + interp: relL2 ~0.25/0.26 vs taylor 0.30 on same windows (~18% better), one epoch to converge.
- e2e 16-step, draft_k=3 (5 real transformer calls, 3.2x fewer): PSNR vs baseline — spec 16.5, taylor teleport 16.4, skip-no-correction 9.0 (pure noise). Spec/taylor retain recognizable subjects while changing composition and detail; aesthetic equivalence has not been measured. The paper contact sheets expose every recorded prompt/method/k panel.

**Corrected 2026-08-01 — the speedup numbers above the PSNRs were wrong.** The
summary JSON reports mean_speedup_spec 1.50x at k=3. That is warm-up
contamination: `spec_e2e.py` times the baseline arm first with no warm-up
iteration, so prompt 0's baseline absorbed CUDA context creation and kernel
autotuning (140.6 s against a 28.3 s median for identical work). Speedup is
averaged over per-prompt ratios, so that one row contributed a spurious 3.79x
and carried the mean. Same pattern at k=1 (96.8 s) and k=2 (64.1 s).

Steady-state, excluding the contaminated row:

| k | real steps | call red. | spec | taylor | skip | PSNR spec | PSNR taylor |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 9 | 1.8x | 1.02x | 1.08x | 1.03x | 18.9 | 20.8 |
| 2 | 6 | 2.7x | 1.03x | 1.03x | 1.04x | 18.0 | 17.9 |
| 3 | 5 | 3.2x | 1.04x | 1.09x | 1.10x | 17.0 | 16.9 |

The `skip` arm is the diagnostic: it discards 3.2x of the denoiser calls,
corrects nothing, produces noise — and still only reaches 1.10x. So ~90% of
measured wall time at 512x512/16-step is *not* the denoiser (text encode and VAE
decode are inside the timed region), and no drafting scheme can win more than
~10% here regardless of how good the draft is.

Also note the learned walker is *dominated* by training-free taylor at k=1, on
both speed and quality. It wins on relL2, the metric it was trained for, and
loses on the metric that matters.

Image retention over all six saved prompts (including the latency warm-up row):

| k | spec PSNR | spec SSIM | taylor PSNR | taylor SSIM | skip PSNR | skip SSIM |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 18.14 | 0.699 | 19.78 | 0.770 | 11.11 | 0.244 |
| 2 | 17.28 | 0.662 | 17.29 | 0.637 | 9.35 | 0.182 |
| 3 | 16.51 | 0.629 | 16.39 | 0.585 | 8.95 | 0.173 |

These are same-seed reference-retention metrics, not aesthetic or prompt-
adherence scores. At k=3 the PSNR winner reverses across the six prompt slices
(three learned, three Taylor), so the sample does not support a universal
method ranking.

Numbers regenerated by `scripts/paper_analysis.py`; full write-up in
`paper/latent_teleportation.pdf`.

- Degenerate cell: t=14 k=1 reports relL2 5.5e9 because the final scheduler step
  moves the stored float16 latent by ~0 — relL2 divides by actual movement. It
  must be excluded from aggregates; remove it only for this exact schedule and
  only after a runtime equality/tolerance check.
- Training does not converge cleanly: walker best relL2 0.242 at epoch 2, final
  epoch 0.257; interp final 0.298. Saving the last checkpoint ships a worse model
  than was trained — select on validation relL2.

**Next steps**:
- [x] Gap analysis numbers on 200 trajs
- [x] Walker+interp training, relL2 vs taylor1 baseline (momentum conditioning is the key)
- [x] e2e: draft_k=1..3 sweep for the pixel-match knee
- [ ] 4-step prod config: draft_k=1 (2 real steps) — the deployable case
- [ ] Text-conditioning for walker/interp (cfg.text_dim=2560, pooled emb)
- [ ] Distill walker from bigger rollouts; try latent-space consistency loss
- [ ] Confidence gate: accept/reject teleports via predicted error head

### Track 1: ControlNet Training (IN PROGRESS)
**Goal**: Train canny ControlNet for Z-Image on daisy

**Status**: Dataset generated (20 samples, 512x512), training with 4 control layers (0,10,20,29) + gradient checkpointing to fit in 24GB.

**Next steps**:
- Scale dataset to 200+ samples (more content/style prompts, more seeds)
- Sweep control layer counts: 4 vs 8 vs 16 layers
- Try line art conditioning alongside canny
- Inference test: generate images with trained controlnet checkpoint
- Quality eval: compare controlnet output to reference

### Track 2: Latent Teleportation Acceleration
**Goal**: Get Z-Image to near-20-step quality in fewer physical model calls

**Approach**: Pre-compute a vocabulary of latent trajectories, teleport into the right region, then refine.

**Experiments**:
1. **Cache population on daisy**
   - Generate 1000+ latent trajectories for diverse prompts
   - Store all intermediate latents (t0..t20) per trajectory
   - Build bigram cache for common visual unit pairs

2. **Combiner quality sweep**
   - SLERP vs neural vs tree combiner
   - Sequence-aware transformer combiner (train_sequence.py)
   - Measure LPIPS/SSIM/FID vs reference at each step budget

3. **Step forecaster training**
   - Train LatentStepForecaster on cached trajectories
   - Compare "delta" vs "next" prediction modes
   - Measure virtual step quality vs real denoising steps

4. **Trajectory prior tuning**
   - KNN k values: 3, 5, 10, 20
   - Virtual steps: 0, 1, 2, 3
   - Repel embedding (negative prompt) scale

5. **Confidence gate calibration**
   - Error derivative thresholds for early stopping
   - Learned vs heuristic gating
   - Target: skip 30-50% of steps with <5% quality loss

6. **Full ablation** (latentteleport/benchmark.py)
   - Tokenizer: nlp, curated, clip
   - Teleport timestep: 0.1, 0.3, 0.5
   - Refinement steps: 0, 1, 3, 5, 7, 10
   - Vocab sizes: 100, 500, 1000, 5000

### Track 3: Trained Cache / Compressed Latent Recalculation
**Goal**: Learn to predict/recalculate latents from nearby pre-computed latents

**Approach**: Like CGTaylor (compressed recalculation) but trained.

**Experiments**:
1. **Pre-compute latent database**
   - Generate diverse latent trajectories (good and bad outcomes)
   - Store embedding + all step latents + quality metrics
   - Index by CLIP embedding for fast nearest-neighbor lookup

2. **Trained interpolation network**
   - Input: k nearest cached latents + their embeddings + target embedding
   - Output: predicted target latent at desired timestep
   - Architecture: cross-attention over cached latents, conditioned on target text
   - Train on held-out latent trajectories

3. **Delta prediction**
   - Given cached latent L_cached at step t and target embedding e_target
   - Predict delta d such that L_target = L_cached + d
   - Much smaller output space than full latent prediction

4. **Quality-aware routing**
   - Maintain quality scores for cached latents
   - Route through high-quality cached paths preferentially
   - Use bad latents as negative examples for contrastive training

### Track 4: stable-diffusion.cpp Full Integration
**Goal**: Measure and improve C++ Z-Image inference

**Experiments**:
1. **Baseline benchmark** (immediate)
   - Build sdcpp on daisy (scripts/setup_external_zimage.sh --build-sdcpp)
   - Run sdcpp_benchmark.py with Q3_K, Q4_K_M, Q8_0 quants
   - Compare latency/quality vs Python diffusers baseline

2. **Quantization sweep**
   - GGUF quant levels vs image quality (FID, LPIPS)
   - Find sweet spot: Q4_K_M likely best quality/speed tradeoff

3. **Cache integration**
   - Can we use sdcpp for the denoising loop but inject cached latents?
   - Measure overhead of Python<->C++ latent transfer
   - If viable: use sdcpp for cheap steps, Python for controlnet/cache logic

4. **Flash attention + VAE tiling**
   - --diffusion-fa and --vae-tiling flags
   - Measure impact on 3090 (24GB limits)

### Track 5: Kernel-Level Acceleration
**Goal**: Push CuteZImage/ZImageAccelerated further

**Experiments**:
1. **torch.compile end-to-end**
   - Full pipeline compile with reduce-overhead
   - Measure first-call vs steady-state latency
   - Graph breaks analysis

2. **Additional Triton fusions**
   - Fused attention + controlnet injection
   - Fused timestep embedding + modulation
   - Patchify/unpatchify kernels

3. **Memory optimization**
   - Activation checkpointing for large batch training
   - In-place operations where safe
   - Dynamic shape handling vs static shapes

---

## Measurement Plan

All experiments should report:
- **Latency**: wall-clock per image (ms)
- **Throughput**: images/sec at batch=1 and batch=4
- **Quality**: FID, LPIPS, SSIM, PSNR vs 20-step reference
- **VRAM**: peak GPU memory (MB)
- **Steps**: actual model forward passes used

Store results in `experiments/results/` as JSONL files.

## Priority Order
1. Finish canny ControlNet training (validates pipeline on daisy)
2. sdcpp baseline benchmark (low effort, establishes C++ baseline)
3. Cache population + teleportation ablation (biggest potential speedup)
4. Trained cache/compressed recalculation (research frontier)
5. Additional kernel fusions (incremental gains)
