# Local Open-Weight Models and Output Parameters

**Date:** 2026-07-14
**Status:** Reference for the local-model tier of the comparison
**Sources:** `docs/2026-04-02_synthetic-image-generation-model-research.md`
(prior survey), `scripts/synthetic/2-generate_synthetic_images_local.py`
(existing implementation), plus verified 2026 web research (see §5 sources).

---

## 1. Good news: the local path already exists

> **Update (2026-07-30):** the roster below is superseded by the experiment's
> actual local generation script,
> `scripts/synthetic_model_comparison/1g-generate_images_local.py`, which now
> runs **six** models (FLUX.1-schnell was swapped for FLUX.2-klein-9B after
> evaluating FLUX.2-dev and finding it too large for this machine's GPU; SD
> 3.5 Large, SD 3.5 Large-Turbo, and Qwen-Image were added). See §7 for the
> current roster and §8 for measured benchmark numbers.

`scripts/synthetic/2-generate_synthetic_images_local.py` (the production script
this experiment's local generation was originally adapted from) implements
three HuggingFace `diffusers` pipelines, tuned for an RTX 3060 12 GB:

| Key | Model(s) | Precision / steps |
|-----|----------|-------------------|
| `flux-schnell` | `black-forest-labs/FLUX.1-schnell` | NF4-quantized, 4 steps, guidance 0.0 |
| `realvisxl-lightning` | `SG161222/RealVisXL_V5.0` + `ByteDance/SDXL-Lightning` 4-step LoRA | fp16, Euler trailing |
| `sd35m` | `stabilityai/stable-diffusion-3.5-medium` | bf16, 40 steps, CFG 4.5 |

It also defines a `NEGATIVE_PROMPT` and a short `build_prompt()` (same
single-sentence style as the API single-class scripts). So the marginal cost of
the local tier is mostly **compute time**, not new engineering.

## 2. Prompt-length limits — the decisive local constraint

This is the single most important difference from the API models. The incumbent
Gemini prompts are ~1,300 words (and the *test*-set prompts reach ~2,700 words /
16k chars — verbatim Wikipedia text). **No local model can read prompts anywhere
near that length.**

| Model | Text encoder(s) | Hard prompt limit | ≈ words | ≈ chars |
|-------|-----------------|-------------------|---------|---------|
| SDXL / RealVisXL / SDXL-Lightning | CLIP-L + OpenCLIP-bigG | **77 tokens each** (75 usable) | ~50–60 | ~300–350 |
| SD 3.5 (Large/Medium) | 2× CLIP (77) + **T5-XXL** | **256 T5 tokens** (edge artifacts beyond) | ~180–200 | ~1,100–1,300 |
| FLUX.1-schnell | CLIP-L (pooled) + T5-XXL | **256 T5 tokens** | ~180–200 | ~1,100–1,300 |
| FLUX.1-dev | CLIP-L (pooled) + T5-XXL | **512 T5 tokens** | ~350–400 | ~2,200–2,600 |
| **Gemini Flash Image (API)** | — | **32,768-token context** | ~24,000 | effectively unlimited |
| **OpenAI gpt-image-* (API)** | — | **32,000 characters** | ~5,000+ | 32,000 |

Implications:
- **SDXL family: silent truncation at 77 tokens.** Everything past ~55 words is
  ignored (a diffusers warning is emitted). Workarounds — **Compel** / **sd_embed**
  chunking (encode 75-token chunks, concatenate embeddings) and prompt weighting
  (`(feature)1.2`) — help, but CLIP was never trained on long sequences, so
  adherence to content deep in the prompt degrades; the first ~75 tokens dominate.
- **FLUX / SD3.5: T5 carries the prompt** and tolerates ~180–400 words — much
  better, but still ~1/6 to 1/3 of the incumbent prompt.
- Therefore the **`compressed` prompt regime** (see
  [`05_prompt-strategy-and-length-limits.md`](05_prompt-strategy-and-length-limits.md))
  is mandatory for any fair local-vs-API contrast: give the API models the *same*
  ≤77-token prompt in the matched cell.

## 3. Output resolution — can they do 500×500?

> **Update (2026-07-29):** the "downscale to 500×500" recommendation below
> was not adopted. That number doesn't match any other convention in this
> repo — every other generator cell (incumbent Gemini, Nano Banana Lite,
> gpt-image-2) stores images at whatever resolution its API natively
> produces (e.g. 592×448), and this experiment's actual training pipeline
> (`scripts/synthetic_model_comparison/training/constants.py`) letterboxes
> every input image to `IMAGE_SIZE=640` regardless of source resolution — so
> a forced 500×500 resize before saving wouldn't match any real downstream
> consumer. `1g-generate_images_local.py` generates at each model's native
> ~1 MP 4:3 bucket (the exact buckets suggested below) and **saves as-is**,
> consistent with every other cell. The rest of this section is kept for its
> native-resolution research; only the final downscale step is superseded.

Short answer (superseded, see above): **generate at each model's native ~1 MP
and downscale to 500×500.** Native ~500-px generation is either invalid or
off-native (worse quality).

| Model | Native / trained resolution | Off-native behavior | Divisibility |
|-------|-----------------------------|---------------------|--------------|
| SDXL / RealVisXL / Lightning | **1024×1024** (~1 MP), multi-aspect buckets | 512² on SDXL base → degraded/incoherent; >1 MP → duplicated subjects | multiples of 8 (64 for buckets) — **500 is invalid** |
| SDXL-Turbo | **512×512** native | 1024 works but off-native | multiples of 8 |
| SD 3.5 Large | **1 MP** (1024² and equal-area aspects) | off-area degrades | multiples of 16 |
| SD 3.5 Medium | **0.25–2 MP** trained range (most flexible) | tolerant in range | multiples of 16 |
| FLUX.1 dev/schnell | **~0.1–2 MP**, arbitrary aspect | very tolerant | multiples of 16 — nearest to 500 is 496/512 |

**500×500 is not a valid native size** for SDXL/SD3.5/FLUX (not divisible by
16). The detector trains at 500×500, but that is a **downscale target**, not a
generation size.

### Practical rule for the detector (input 500×500)
Generate natively at ~1 MP (best sample quality) → **area/bicubic downscale to
500×500** using the **same resize/letterbox function applied to real training
images**. Downscaling discards only high-frequency detail the 500-px detector
can't use anyway, so it is information-preserving in the relevant direction and
matches what real camera images undergo. Do **not** generate at ~500 px natively
(off-native → worse quality, no benefit).

Incumbent Gemini config for reference: aspect `4:3`, size `"512"` / `"0.5K"`
(OpenRouter offers `0.5K/1K/2K/4K`). To match, generate local images at 4:3 ~1 MP
(e.g. 1152×864, both /16-friendly for FLUX/SD3.5; for SDXL use a 4:3 bucket like
1152×896) and downscale.

## 4. Batch generation

All three families support `num_images_per_prompt=N` and batched prompt lists in
their diffusers pipelines (`StableDiffusionXLPipeline`,
`StableDiffusion3Pipeline`, `FluxPipeline`), bounded only by VRAM. On a 12 GB
3060, expect small batch sizes (1–4) especially for FLUX. **Not implemented**
in this experiment's `1g`/`1i` scripts (one image per pipeline call) — real
throughput upside given the GPU headroom found in §8, documented as a
follow-up rather than built, since it requires reworking the per-image loop
into per-batch (list of prompts, per-item generators/seeds, uneven-remainder
handling) and wasn't necessary to answer the questions asked so far.

## 5. Throughput / cost reality (a finding, not just logistics)

From the prior survey (`docs/2026-04-02_...`), measured on RTX 3060 12 GB:
- **FLUX.1-dev NF4:** ~138 s/image (FP8 ~400 s). FLUX.1-schnell (4-step) is much
  faster but still the heaviest local option.
- SDXL-Lightning (4-step) and SD3.5-Medium are faster per image.

At ~100 s/image, 12 classes × 100 images = 1,200 images ≈ **33 GPU-hours per
local model** — vs minutes-to-an-hour of wall-clock for an API batch. **This
throughput gap is itself a thesis finding:** even if a local model's quality were
adequate, generating the full 12,600-image production set locally would take
weeks on this GPU, which is a large part of *why* an API model was the practical
choice. Record generation wall-clock per model in the results — see §8 for the
actual measured numbers on this experiment's GPU (an A40, not the RTX 3060
this section's estimates assume).

## 6. Licensing of outputs (for a thesis that may feed a product)
- **FLUX.1-schnell:** Apache-2.0 — outputs and weights freely usable, incl.
  commercial. Best-licensed *FLUX* option, but no longer in the active
  roster (replaced by FLUX.2-klein-9B — see §7).
- **FLUX.1-dev / FLUX.2-dev / FLUX.2-klein-9B:** FLUX Non-Commercial License
  (gated) — non-commercial weights; outputs' commercial use is separately
  permitted per BFL terms, but the weights themselves are research/
  non-commercial. FLUX.2-klein-9B (now in the active roster) carries this
  same restriction, unlike schnell.
- **SD 3.5 (Medium/Large/Large-Turbo):** Stability Community License — free
  under a revenue threshold; outputs usable. All three gated on HF (manual
  per-repo license acceptance required — Large-Turbo specifically needed a
  separate acceptance click even after Medium/Large were already approved
  for this account, i.e. gating is per-repo, not per-organization).
- **SDXL / RealVisXL:** open (OpenRAIL++/CreativeML) — usable.
- **Qwen-Image:** **Apache-2.0** — the most permissive license of any model
  actually in the local roster (more permissive than every FLUX/SD3.5
  option). Not gated.
- **HiDream-I1-Full:** **MIT** — the most permissive weights license of any
  model in the roster, but composed with Meta's
  `meta-llama/Llama-3.1-8B-Instruct` (Llama 3.1 Community License) as its
  4th text encoder, since HiDream's own repo doesn't bundle it. Originally
  **not included**: this account's HF token got a live 403 on that repo.
  That access has since been granted — reconfirmed with a real
  `hf_hub_download()` of the repo's `config.json`, not just a passing
  `model_info()` check (which had already misleadingly succeeded before,
  same failure mode doc 13 §3/§4 already flags) — so it's now adopted as
  the roster's 7th model. The transformer and the Llama encoder are
  NF4-quantized at load time (same `BitsAndBytesConfig` pattern as
  Qwen-Image): unquantized bf16, HiDream-I1's total footprint (~63.5GB)
  exceeds this machine's 47GB RAM, and `enable_model_cpu_offload()` loads
  the whole pipeline onto the CPU before moving submodules to GPU, so a
  naive load risked heavy swapping or an OOM before generation even
  started. Unlike Qwen-Image's quantization, no visible artifact graininess
  was observed in the spot-checked images from this benchmark.
- Confirm current terms at generation time and cite them (the thesis should state
  the license under which each generator's images were used as training data).

## 7. Local-model roster (current, as actually implemented)

`scripts/synthetic_model_comparison/1g-generate_images_local.py` /
`1i-generate_images_local_maxlen.py` implement seven models:

| Generator key | Model | Params | License | Offload? |
|---|---|---|---|---|
| `flux2-klein-9b` | FLUX.2-klein-9B (step-distilled, `Flux2KleinPipeline`) | 9B DiT + 8B "Qwen3" text encoder | FLUX Non-Commercial (gated) | Mandatory — ~34GB combined vs. 23.8GB VRAM |
| `realvisxl-lightning` | RealVisXL V5.0 + SDXL-Lightning 4-step LoRA | SDXL-scale | OpenRAIL++ | Not used — fits comfortably |
| `sd35m` | SD 3.5 Medium | 2.5B transformer | Stability Community (gated) | **Removed** — fits in ~16.6GB without it (see §8 finding) |
| `sd35-large` | SD 3.5 Large | 8B transformer | Stability Community (gated) | Mandatory — ~27.6GB vs. 23.8GB |
| `sd35-large-turbo` | SD 3.5 Large Turbo (4-step) | 8B transformer | Stability Community (gated) | Mandatory — same footprint as Large |
| `qwen-image` | Qwen-Image, NF4-quantized at load time from the official repo (no pre-quantized checkpoint exists) | 20.4B DiT + 8.3B Qwen2.5-VL encoder | **Apache-2.0** | Kept as a safety margin even post-quantization |
| `hidream-i1` | HiDream-I1-Full, transformer + Llama text encoder NF4-quantized at load time (bundled CLIP/OpenCLIP/T5-XXL encoders stay bf16) | 17B DiT + CLIP + OpenCLIP + T5-XXL + 8B Llama-3.1 text encoder | MIT + Llama 3.1 Community (composite, gated on the Llama piece) | Mandatory — ~63.5GB bf16 unquantized, still over 23.8GB even NF4-quantized (~25.5GB) |

**FLUX.2-dev (32B) was evaluated and dropped**: even its pre-quantized 4-bit
community checkpoint (`diffusers/FLUX.2-dev-bnb-4bit`) totals ~34GB of
weights — more than this GPU's 23.8GB VRAM even 4-bit — and would need
`enable_model_cpu_offload()` with real OOM risk. FLUX.2-klein-9B (same BFL
family, step-distilled) has the same ~34GB total footprint and also needs
offload, but is dramatically faster (4 steps vs. dev's 28-50) so far fewer
offload swap-cycles occur per image — that's its actual advantage over dev,
not a smaller VRAM footprint.

## 8. Benchmark results (measured on this experiment's GPU)

Measured on this machine — a single NVIDIA A40 vGPU instance (**23.8GB
VRAM**, not the RTX 3060 12GB §5 assumes) — via
`1g-generate_images_local.py --benchmark N` at the `compressed` prompt
regime (5-10 images/model; see `reports/model_comparison_local_generation_benchmark.csv`
for the raw data):

| Generator | s/image (steady-state) | Estimated hours for 1,200 |
|---|---|---|
| `realvisxl-lightning` | 1.14 | 0.38 |
| `sd35m` (offload removed) | 14.84 | 4.95 |
| `sd35-large-turbo` | ~25.5 | 8.5 |
| `flux2-klein-9b` | ~48 | 16.0 |
| `sd35-large` | ~57.8 | 19.3 |
| `qwen-image` | ~101.4 | 33.8 |
| `hidream-i1` | 129.85 | 43.28 |

**Total ≈ 126 GPU-hours (~5.3 days continuous)** to generate all seven
models' full 1,200-image cells back-to-back on this GPU — the same
throughput-gap finding as §5, now with real numbers on real
current-generation models rather than the RTX-3060-era FLUX.1-dev estimate.
`hidream-i1` is the slowest model in the roster despite NF4-quantizing its
two largest components — plausibly the combined cost of 50 non-distilled
steps (like Qwen-Image) plus `enable_model_cpu_offload()` swapping four
separate text-encoder components (vs. Qwen-Image's one) in and out of GPU
memory per image.

**GPU-memory-utilization finding:** while benchmarking, GPU memory
utilization stayed between 23-39% of the 23.8GB card. Investigating this
found `enable_model_cpu_offload()` was applied to *every* model as a
blanket RTX-3060-12GB-era default (from the production script this was
copied from) — deliberately parking most weights on system RAM and moving
only the active submodule to GPU per step, which reads as low average VRAM
usage and costs real per-step PCIe transfer time. Removing it for SD 3.5
Medium specifically (the only model in the original three-model roster
that structurally fits without it — RealVisXL never used it) produced a
**~1.8x speedup** (14.84s vs. the previous 27.33s/image with offload) —
confirmed empirically, not just theorized. This fix does **not** generalize
to the four larger models added since (`flux2-klein-9b`, `sd35-large`,
`sd35-large-turbo`, `qwen-image`): all four structurally exceed 23.8GB
before any offload-avoidance benefit could apply, so they keep offload
regardless. Batching multiple prompts per pipeline call (§4) and
`torch.compile` (the production script's existing opt-in `--compile` flag)
remain documented-but-unimplemented further options, since they weren't
necessary to answer what was actually asked.

**Qwen-Image quality note:** the NF4-quantized Qwen-Image output showed
visible quantization-artifact graininess (blocky texture noise, most
visible in grass/fur) compared to the other five models' clean output —
worth flagging as a real quality/speed tradeoff of the custom on-the-fly
quantization approach (no official or `diffusers`-org pre-quantized
checkpoint exists for Qwen-Image, unlike FLUX.2-dev's), not necessarily
representative of what Qwen-Image looks like at full precision.

## 9. The `maxlen` prompt regime — using each model's real capacity

The `compressed` regime (§ and `05_prompt-strategy-and-length-limits.md`)
deliberately gives every local model the *same* ≤75-token prompt for a
fairness ablation against prompt length as a confound. For the actual
production 1,200-image/model dataset, that leaves most of these models'
real capacity unused — SD 3.5's T5 branch can hold 256 tokens, and
FLUX.2-klein-9B/Qwen-Image's Qwen-family encoders can hold 512, but
`compressed` only ever gives them ~40-57 tokens.
`1h-generate_prompts_maxlen.py` builds two new tiers (256/512 tokens) sized
to actually fill each budget, and `1i-generate_images_local_maxlen.py`
generates from them into a new `<generator>/maxlen/` cell (the `compressed`
cells stay untouched for a possible later ablation against the best
proprietary API model).

**A naive approach was tried and rejected first:** truncating the
already-built `full`-regime prompt (the ~1,300-word incumbent-style
template) down to each tier's budget. Inspecting an actual full prompt
showed its own *structural* sections (SCENE SPECIFICATION + PHOTOGRAPHY
STYLE + CRITICAL REQUIREMENTS — everything that isn't free-text Wikipedia
excerpt) already total ~464 words (~600+ T5 tokens) on their own — more
than the entire 256-token budget before a single word of species
description is added. Free-text section length also varies enormously by
class (696 words for aye-aye's whole full prompt vs. 5,438 for red fox),
so per-class truncation logic is unavoidable regardless of approach.

**What was built instead:** the simpler, already-proven `build_prompt()`
shape from `scripts/synthetic/2-generate_synthetic_images_local.py`
(`"Realistic wildlife photograph of a {class}. Species characteristics:
{excerpt}. {STYLE_SUFFIX}"` — doc 05 §1's "path A" tier), extended to
actually *fill* each budget rather than stop at ~150 tokens, with real
per-image scene variation (pose/environment sentences) pulled directly from
the canonical `train/gemini-3.1-flash-image-preview/full/index.jsonl`
(1,200 already-parsed, clean single-sentence records) rather than a
5-item pose rotation. See `05_prompt-strategy-and-length-limits.md` for the
full construction method.
