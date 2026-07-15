# Local Open-Weight Models and Output Parameters

**Date:** 2026-07-14
**Status:** Reference for the local-model tier of the comparison
**Sources:** `docs/2026-04-02_synthetic-image-generation-model-research.md`
(prior survey), `scripts/synthetic/2-generate_synthetic_images_local.py`
(existing implementation), plus verified 2026 web research (see §5 sources).

---

## 1. Good news: the local path already exists

`scripts/synthetic/2-generate_synthetic_images_local.py` already implements three
HuggingFace `diffusers` pipelines on the RTX 3060 12 GB:

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

Short answer: **generate at each model's native ~1 MP and downscale to 500×500.**
Native ~500-px generation is either invalid or off-native (worse quality).

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
3060, expect small batch sizes (1–4) especially for FLUX.

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
choice. Record generation wall-clock per model in the results.

## 6. Licensing of outputs (for a thesis that may feed a product)
- **FLUX.1-schnell:** Apache-2.0 — outputs and weights freely usable, incl.
  commercial. **Best-licensed local option.**
- **FLUX.1-dev:** non-commercial *weights* license; outputs' commercial use is
  permitted per BFL terms but the weights are research/non-commercial — note the
  distinction.
- **SD 3.5:** Stability Community License — free under a revenue threshold;
  outputs usable. **SDXL / RealVisXL:** open (OpenRAIL++/CreativeML) — usable.
- Confirm current terms at generation time and cite them (the thesis should state
  the license under which each generator's images were used as training data).

## 7. Recommended local tier for the comparison
- **FLUX.1-schnell** — best open photorealism, Apache-2.0, T5 256-token prompt.
- **RealVisXL V5.0 + SDXL-Lightning** — fast SDXL photoreal, but 77-token prompt
  (the sharpest test of the prompt-length handicap).
- **SD 3.5 Medium** (optional) — no-quant baseline, 256-token T5.
