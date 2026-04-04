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

---

## Experiment Tracks

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
