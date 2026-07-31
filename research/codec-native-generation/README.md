# Codec-native image generation

Can a text-to-image model generate *in a codec's representation* (WebP/HEIC/JPEG
coefficients) instead of generating RGB pixels and then compressing them?

The prompt for this is [Mage-VL](https://arxiv.org/abs/2607.24904), which does the
analogous thing for *understanding*: it follows the structure of a video codec,
keeps every anchor (I) frame patch, and retains predicted (P) frame patches only
where the codec spends bits. That cuts visual tokens by >75% and gives up to
3.5x wall-clock speedup. The question here is whether the same idea pays off in
the generation direction.

Short answer from the first pass: **the obvious motivation is wrong, but a
weaker version of the idea is well-supported and a stronger version looks
genuinely open.** Details below — read the latency budget before proposing
anything, because it rules out a whole class of ideas.

## Measured latency budget (the constraint everything must respect)

RTX 3090, Z-Image Turbo Q8_0 via stable-diffusion.cpp, 512x512, 3 steps, cfg 1.0,
flash attention, TAEF1 decoder. This is the production config behind
images2.netwrck.com, so it is the budget any proposal has to beat.

| stage | time | share |
| --- | ---: | ---: |
| text encode (Qwen3-4B) | 91 ms | 10.7% |
| diffusion sampling (3 steps) | 790 ms | **92.9%** |
| VAE decode (TAEF1) | 80 ms | 9.4% |
| WebP q85 encode | 37 ms | 4.3% |
| total (end to end over HTTP) | ~850 ms | |

(Shares exceed 100% because the stages overlap slightly with HTTP handling; treat
them as fractions of wall clock, not a strict partition.)

Encode measured in isolation by `experiments/exp0_latency_budget/enc_bench.c`:
36.5 ms at 512px, 136.2 ms at 1024px, on a maximum-entropy buffer, so real
photographic content is at or below this.

**Sampling is 93% of the cost.** Everything else is rounding error.

## Three hypotheses, ranked by what the budget permits

### H1 — "generate natively in the codec to skip the conversion step"

This is the intuitive version of the idea, and the numbers kill it. The RGB->WebP
encode is **37 ms of 850 ms, ~4%**. Even a *free* codec-native output path — zero
encode cost, no quality loss — caps out at a 4% speedup. That does not justify
training a new generative model.

**Status: rejected on measurement.** Do not pursue for latency. (It may still be
worth something for *fidelity* — see H2 — just not for speed.)

### H2 — "generate in frequency/DCT space, eliminating both the VAE decode and the encode"

Stronger, because it removes 80 ms of decode as well as 37 ms of encode: ~14% of
the budget with TAEF1, ~25% against the full VAE (which costs 210 ms). Still not
transformative, but this one has real prior art suggesting it is *free or better
than free* on quality:

[DCTdiff](https://arxiv.org/abs/2412.15032) runs diffusion directly in DCT space
(YCbCr, 2x chroma subsampling, high-frequency coefficients dropped), needs **no
VAE at all**, scales to 512x512, and reports FID 7.07 vs 10.89 for UViT latent
diffusion on FFHQ 512 at 100 NFE — at a claimed 1/4 the training cost. Reported
lossless compression 4x at 256px, 7.11x at 512px.

So "generate in the codec's transform domain" is already demonstrated to work and
to *improve* quality-per-training-dollar. The novel part left is aligning that
space with a *shipping* codec (WebP/VP8 or HEIC/HEVC) so the model's output is
directly serialisable to bytes.

**Status: plausible, partially pre-empted.** The win is architectural (drop the
VAE, drop the encoder) rather than a large latency cut.

### H3 — "use codec bit-allocation as a spatial compute-allocation prior during sampling"

This is the actual analogue of what Mage-VL does, and the only hypothesis that
attacks the 93%.

Mage-VL's saving does not come from operating in codec format — it comes from
**spending fewer tokens where the codec spends fewer bits.** The generation-side
analogue is spatially non-uniform compute: run full transformer depth only on
regions that will carry high bit density (edges, texture, faces), and coarse or
cached compute on regions the codec would flatten anyway (sky, bokeh, walls).

Unlike H1/H2 this targets sampling directly, so its ceiling is large rather than
~14%. The search for prior art turned up plenty of *compressed-domain
compression* work and DCT-space generation, but nothing doing codec-guided
adaptive compute during denoising. That may mean it is open, or that it has been
tried and failed unpublished — establishing which is the first job.

The obvious difficulty: bit allocation is a property of the *finished* image, and
during sampling you do not have it yet. That is the crux. Possible outs — predict
an allocation map from the text embedding and the step-1 latent; or derive it
from the low-frequency structure that stabilises within the first step or two.

**Status: open, highest ceiling, highest risk.** This is where the research
effort belongs.

## Relationship to existing work in this repo

`latentteleport/` already skips transformer calls *temporally* (cache and reuse
across steps). H3 is the *spatial* counterpart — skip work across regions rather
than across steps. The two compose, and the existing speculative-walking harness
is the natural place to prototype H3's measurement rig.

## Experiment ladder

Cheapest first, each with a kill criterion. Do not skip ahead — exp1 decides
whether H3 is worth any model training at all.

| # | experiment | question | kill criterion |
| --- | --- | --- | --- |
| 0 | latency budget | where does the time actually go? | **done** — see table above |
| 1 | bit-density oracle | if we had a perfect bit-allocation map, how much compute could we skip? | <20% of transformer FLOPs skippable at equal quality -> H3 is dead |
| 2 | allocation predictability | can the map be predicted from text embedding + step-1 latent? | correlation with true map below what exp1 needs -> H3 is dead |
| 3 | adaptive-depth sampling | wire a predicted map into per-region depth/caching | quality drop visible in blind A/B at any real speedup -> H3 is dead |
| 4 | DCT-space head (H2) | replace VAE decode with a direct DCT/VP8 coefficient head | fails to match TAEF1 quality at equal latency -> H2 is dead |

Exp1 is an *oracle* study: generate normally, compress the result, read the real
per-region bit density back out of the WebP/HEIC bitstream, then re-run sampling
with compute masked off in the low-density regions and measure quality loss. It
needs no training and answers whether the ceiling is real before anyone builds a
predictor.

## Hardware note

HEIC is HEVC-based, so unlike WebP it *can* be hardware-encoded — NVENC on this
3090 does HEVC. That makes HEIC the more interesting target if output encode ever
does become a bottleneck (it currently is not, per H1). NVENC has no WebP path.

## Sources

- Mage-VL: https://arxiv.org/abs/2607.24904 · https://microsoft.github.io/Mage/vl/
- DCTdiff: https://arxiv.org/abs/2412.15032
- Generative Image Coding with Diffusion Prior: https://arxiv.org/pdf/2509.13768
- CADC (content-adaptive diffusion compression): https://arxiv.org/pdf/2602.21591
