# API Image-Model Landscape and Pricing

**Date:** 2026-07-14
**Status:** Reference for the API tier of the comparison
**Primary sources (scraped verbatim):**
[`scraped_sources/openai-image-pricing.md`](scraped_sources/openai-image-pricing.md),
[`scraped_sources/gemini-api-pricing.md`](scraped_sources/gemini-api-pricing.md).
Landscape research compiled July 2026 — **re-verify all prices/snapshots before
thesis submission** (this market repriced repeatedly in H1 2026, and several
models below are scheduled for shutdown within months).

---

## 0. Direct answer to "are there other models from OpenAI?"

**Yes.** OpenAI is not a single image model. As of July 2026 the API exposes
**three live** image models plus two in shutdown countdown, so you can compare
*within* OpenAI as well as against Gemini and the local models:

| Model | Status | ~Price/image (1024², std) | Resolutions |
|-------|--------|---------------------------|-------------|
| **gpt-image-2** | current flagship | $0.006 (low) / $0.053 (med) / $0.211 (high) | **arbitrary WxH**, ÷16, up to 3840×2160 |
| **gpt-image-1.5** | live, **EOL 2026-12-01** | ~$0.04 | 1024², 1024×1536, 1536×1024 |
| **gpt-image-1-mini** | live budget, **EOL 2026-12-01** | ~$0.005–$0.052 | 3 fixed sizes |
| gpt-image-1 | deprecated, **EOL 2026-10-23** | ~$0.011–$0.25 | 3 fixed sizes |
| DALL·E 2 / 3 | **removed 2026-05-12** | — | — |

Prompt limit for all GPT image models: **32,000 characters**. Batch API: **−50%**.

Likewise **Google is multiple models** (see §2): the Gemini-native "Nano Banana"
family *and* the (being-retired) Imagen line. So both vendors support the
same-prompt, multi-model comparison the professor wants.

## 1. What the incumbent actually is

The production pipeline uses `gemini-3.1-flash-image-preview`, i.e. the **Gemini
3.1 Flash Image** ("Nano Banana 2") model, at aspect `4:3`, size `"512"`/`"0.5K"`.
Pricing: **$0.045–$0.151/image** by resolution ($0.50/M input tokens), **−50%**
in Batch. At 0.5K it is at the low end (~€0.04/img — consistent with the ~€500 /
12,600-image production spend).

## 2. Google image models (Gemini API)

| Model | API id | ~Price/image | Resolutions | Notes |
|-------|--------|--------------|-------------|-------|
| Nano Banana 2 Lite | `gemini-3.1-flash-lite-image` | ~$0.034 (1K) | 0.5K, 1K | cheapest |
| **Nano Banana 2** | `gemini-3.1-flash-image` | ~$0.045–$0.15 | 0.5K–4K | **incumbent family**; up to 14 reference images; Search grounding |
| Nano Banana Pro | `gemini-3-pro-image` | ~$0.13–$0.24 | 1K/2K/4K | highest quality |
| Nano Banana (legacy) | `gemini-2.5-flash-image` | $0.039 flat | 1K | deprecated |
| Imagen 4 Fast/Std/Ultra | — | $0.02/$0.04/$0.06 | ≤2K | **Gemini-API shutdown 2026-08-17** — do not build on it |

Aspect ratios 1:1–21:9. Prompts are ordinary token-billed context (effectively
very long). **All outputs carry an invisible SynthID watermark** (worth a
sentence in the thesis — the training data is watermarked). Batch: −50%.

> Intra-vendor comparison is cheap and informative: the incumbent Nano Banana 2
> vs Nano Banana 2 Lite vs Nano Banana Pro, same prompt, isolates the
> quality/price frontier *within* Google before comparing across vendors.
> **Decision:** both extra Google cells are in the experiment — **Nano Banana 2
> Lite** (`gemini-3.1-flash-lite-image`, cheaper-than-incumbent floor) and
> **Nano Banana Pro** (`gemini-3-pro-image`, quality ceiling) will be run with
> the same prompts alongside the incumbent Nano Banana 2.

## 3. Other API vendors (context / optional extra cells)

| Vendor / model | ~Price/image | Max prompt | Resolutions | Batch | Training-data licensing |
|----------------|--------------|-----------|-------------|-------|-------------------------|
| **Black Forest Labs FLUX.2 [pro]** (BFL API) | ~$0.03 @1MP | not documented | ≤~4MP, MP-scaled | No | outputs OK; **no competing-model training**; separate synthetic-data license sold; **dev weights non-commercial** |
| **Stability Stable Image Ultra / Core** | $0.08 / $0.03 | ~77 *effective* tokens (10k-char field) | 640–1536px, 9 ratios | No | Community License self-host free <$1M rev |
| Stability SD3.5 Large/Turbo/Medium (API) | $0.065/$0.04/$0.035 | ~77 eff. tokens | ~1MP | No | same |
| **Ideogram** v3/4 Turbo→Quality | $0.03–$0.10 | undocumented | ≤~2MP presets | No (multi-image/call) | text-render focus; photoreal secondary |
| **Recraft** V4 / Pro | $0.04 / higher | undocumented | 1024² / 2048² | No | design focus; V4.1 improved photoreal |
| **Adobe Firefly Image 5** | ~$0.02–$0.10 (**~$1k/mo enterprise min**) | n/a | ≥2K | async | **trained on licensed data; IP indemnification — cleanest provenance** |
| **Midjourney** | — | — | — | — | **no public official API**; unofficial APIs breach ToS → not viable for research |
| fal.ai / Replicate (hosts) | FLUX.2 dev ~$0.012/MP (fal); FLUX dev ~$0.025–0.03 (Replicate) | per model | per model | No | host open models per-image; includes commercial license for non-commercial-weight models |

**fal.ai / Replicate matter for the local tier too:** they let you run FLUX/SD3.5
*via API, per-image*, avoiding both GPU purchase and the local generation-time
bottleneck — a middle path between "buy a GPU" and "pay a frontier API." Useful
if the RTX 3060 throughput (see [`04`](04_local-models-and-output-parameters.md) §5)
makes local generation impractical.

## 4. Dimensions, quality, batch — what to request for this study

- **Resolution:** match the production `4:3 ~0.5K–1K`. Generate at ≥1K where
  cheap, downscale to 500×500 for training (see [`04`](04_local-models-and-output-parameters.md) §3).
  gpt-image-2 supports arbitrary ÷16 sizes; Gemini uses `0.5K/1K/2K/4K` presets.
- **Quality tiers:** gpt-image-2 has low/medium/high ($0.006/$0.053/$0.211 at
  1024²). **Decision: compare `low` vs `medium`** — that is the cost-relevant
  frontier (~9× price gap), since "quality" directly trades cost against
  fidelity and is part of "can it be used affordably?". `high` (~$0.21/img) is
  skipped: at ~4× medium it exceeds the incumbent's per-image cost by an order
  of magnitude and is out of scope for a training-data budget.
- **Prompt regimes:** all five API cells run the `full` Wikipedia-description
  prompt; the **compressed-vs-full ablation runs only on the incumbent NB 2 and
  gpt-image-2 low** — the two cheapest ablation sites, spanning the capability
  range so the cheap-vs-strong interaction is measurable. Rationale in
  [`01`](01_experiment-design.md) §4b; prompt construction in
  [`05`](05_prompt-strategy-and-length-limits.md).
- **Batch:** both OpenAI and Google give **−50%** via their Batch APIs — use
  Batch for all API generation to halve cost. (Local, FLUX-BFL, Stability,
  Ideogram, Recraft have **no** batch discount.)

## 5. Licensing summary for using outputs as training data
- **OpenAI:** you own outputs; ToS forbids training *competing* (generative)
  models — a wildlife detector is non-competing, but document the call.
- **Google:** no ownership claim; SynthID watermark present; no restriction found
  on training a downstream non-generative model.
- **BFL FLUX:** outputs commercial-OK but **explicitly restricts training AI
  models on outputs** and sells a separate synthetic-data license — **read
  carefully before using FLUX-API images as training data** (self-hosted
  FLUX.1-schnell under Apache-2.0 avoids this entirely).
- **Stability / SDXL / SD3.5:** open weights, outputs usable.
- **Adobe Firefly:** cleanest provenance (licensed training data + IP
  indemnification) but enterprise-gated.

## 6. Thesis-hygiene warnings (important)
- **Snapshot churn:** gpt-image-1 (Oct 23), gpt-image-1.5 & -mini (Dec 1), Imagen
  4 in Gemini API (Aug 17) all shut down in 2026. **Pin exact model-id snapshots
  and record retrieval dates**; a model you benchmark may be gone by submission.
- **Derived prices:** per-image figures for token-billed models (OpenAI, Gemini)
  are computed from token rates and vary with quality/size — cite the token rate
  *and* your computed per-image number, with the request params.
- **Watermarks:** Gemini outputs carry SynthID; note that the training data is
  watermarked (unlikely to matter for detection, but disclose it).
