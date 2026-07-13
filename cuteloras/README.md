# CuteLoRAs

Generic LoRA hot-swapping inference server for CuteDSL-accelerated diffusion transformers.

## Why

Serving many LoRA styles from one resident base model usually means merge-on-load and
unmerge-after — two full weight passes per request, plus bf16 drift after repeated
merge/unmerge cycles. CuteLoRAs keeps low-rank factors in a CPU LRU cache, snapshots
pristine base weights the first time a parameter is touched, and swaps styles with a
single fused in-place update. Restores are exact (`torch.equal`) after any number of swaps.

- **Sticky activation** — consecutive requests with the same LoRA pay zero swap cost
- **Exact restore** — CPU-pinned snapshots of touched params, no accumulation error
- **fp32 delta computation** — `B @ A` in fp32 on-device, cast once into the live dtype
- **Fused-kernel safe** — in-place `add_` bumps `weight._version`, invalidating cached
  fused QKV weights in CuteZImage automatically
- **Multi-LoRA stacking** — activate several adapters with per-adapter scales
- **Formats** — Z-Image native, PEFT, diffusers (Flux-style), Kohya (with alpha scaling)
- **Routing** — embedding-similarity prompt routing (trigger words, keywords, negative
  keywords, adult gating) with keyword fallback

## Usage

```python
from cuteloras import LoRARegistry, LoRASwapper

registry = LoRARegistry.from_directory("/path/to/loras")   # or .from_json("registry.json")
swapper = LoRASwapper(pipe.transformer, registry)

swapper.activate("anime_style")            # merge in
swapper.activate([("a", 1.0), ("b", 0.7)]) # stack two, restores "anime_style" first
swapper.deactivate()                       # exact base weights
```

Server:

```bash
CUTELORAS_REGISTRY=registry.json uvicorn cuteloras.server:app --port 8200
curl -X POST localhost:8200/generate -d '{"prompt": "anime girl", "auto_route": true}'
```

Benchmark:

```bash
cutedsl-loras-benchmark --loras-dir /path/to/loras --limit 8 --e2e
```

## Registry format

```json
[{
  "id": "anime_style",
  "name": "Anime Style",
  "base_model": "zimage",
  "path": "/local/path.safetensors",
  "url": "https://.../fallback-download.safetensors",
  "trigger_word": "animestyle",
  "template": "animestyle, {prompt}",
  "keywords": ["anime", "manga"],
  "negative_keywords": ["photo"],
  "is_adult": false,
  "scale": 1.0
}]
```
