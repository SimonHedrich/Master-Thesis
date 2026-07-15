# Synthetic-Generation Model Comparison

**Created:** 2026-07-14
**Owner:** Simon Hedrich

This subdirectory collects the planning and reference material for a **controlled
comparison of image-generation models** used to produce synthetic wildlife
training data.

## Why this exists (the short version)

The production synthetic dataset (~12,600 images, 76 rare classes) was generated
with a single model — `gemini-3.1-flash-image-preview` (Gemini "Nano Banana 2")
— chosen after informally judging that its images *looked* more realistic than
the local diffusion models'. The supervising professor flagged this as
**methodologically insufficient**: a thesis-grade model choice must be
**qualitatively evaluated** against alternatives under a stated, repeatable
criterion, even though the thesis's core question is *"can current image models
be used to generate synthetic training data?"* rather than *"which is best?"*.

The budget is largely spent, so we **cannot** regenerate the whole dataset per
model. Instead we run a **rigorous comparison on a deliberately chosen subset of
classes** — spanning rare/unusual species, species with robust real test sets,
and a fine-grained look-alike group (the three zebra species) — across a few API
and local generators, evaluated on qualitative, automatic, and downstream axes.

## Documents in this directory

| File | What it covers |
|------|----------------|
| [`00_motivation-and-research-question.md`](00_motivation-and-research-question.md) | The professor's critique, the scoped research question, budget constraint, scope of the comparison |
| [`01_experiment-design.md`](01_experiment-design.md) | The controlled protocol: what varies (generator, prompt regime) vs what's held fixed; tractable model grid; cost |
| [`02_class-selection.md`](02_class-selection.md) | The 12 proposed classes and rationale (rare vs robust-test vs zebra look-alike), with test-count/GBIF data and the central tension |
| [`03_api-models-landscape-and-pricing.md`](03_api-models-landscape-and-pricing.md) | OpenAI (multiple models!), Google (Nano Banana + Imagen), FLUX/Stability/Ideogram/Recraft/Firefly/Midjourney/fal/Replicate; pricing, dimensions, quality, batch, licensing |
| [`04_local-models-and-output-parameters.md`](04_local-models-and-output-parameters.md) | Local diffusion models (already implemented), **prompt-length limits**, output resolutions, 500×500 downscaling, throughput/cost, licensing |
| [`05_prompt-strategy-and-length-limits.md`](05_prompt-strategy-and-length-limits.md) | How to build fair prompts across models with wildly different limits; the ≤75-token compressed prompt; worked examples |
| [`06_evaluation-methodology.md`](06_evaluation-methodology.md) | The three evaluation axes (blind multi-rater rubric; automatic proxies; downstream real-test mAP); statistics; the final results table |
| [`07_open-questions-and-what-to-reconsider.md`](07_open-questions-and-what-to-reconsider.md) | Confounds, gaps, framing risks, and decisions to make before generating |
| [`scraped_sources/`](scraped_sources/) | Verbatim scrapes of the OpenAI and Gemini pricing pages (primary sources) |

## Key facts at a glance

- **Incumbent model:** `gemini-3.1-flash-image-preview` (Gemini 3.1 Flash Image /
  "Nano Banana 2"), aspect `4:3`, size `0.5K` (~512 px), ~€0.04/image.
- **"Other OpenAI models?" → Yes:** `gpt-image-2` (flagship), `gpt-image-1.5`,
  `gpt-image-1-mini` (last two EOL 2026-12-01). DALL·E removed from the API.
- **Prompt-length wall:** SDXL = **77 tokens (~55 words)**; FLUX.1-schnell / SD3.5
  = **256 T5 tokens (~180 words)**; FLUX.1-dev = **512 (~350 words)**; API models
  = effectively unlimited (Gemini 32k-token context; OpenAI 32k chars). The
  incumbent prompts were ~1,300 words — hence a mandatory *compressed* prompt
  regime for fair comparison.
- **Output size:** generate at each model's native ~1 MP and **downscale to
  500×500** (native 500 px is invalid/off-native for the local models).
- **Class subset (proposed 12):** 3 zebras (fine-grained); red fox / American
  black bear / lion (robust-test anchors); kinkajou / water deer / ringtail
  (rare **and** >100 test); saiga / aye-aye / pangolin (iconic rare, test-limited).
- **Local generation already exists:** `scripts/synthetic/2-generate_synthetic_images_local.py`
  (FLUX.1-schnell, RealVisXL+SDXL-Lightning, SD 3.5 Medium).

## Related existing docs

- `docs/2026-04-02_synthetic-image-generation-model-research.md` — prior local-model survey
- `docs/plans/2026-05-12_synthetic-image-generation-strategy.md` — production generation strategy & prompt template
- `docs/plans/2026-06-10_model-evaluation-strategy.md` — the thesis evaluation strategy (real-only anchor, domain-shift watchdog)
- `docs/plans/2026-06-11_lookalike-groups-review.md` — frozen look-alike groups (zebra, panthera_rosette, gazelle, …)
- `reports/class_split_counts.csv`, `reports/lookalike_groups_v2.csv`, `data/gbif/metadata/GBIF_image_counts_v1.csv` — the data behind class selection

## Status / next decisions

Open decisions are collected in [`07`](07_open-questions-and-what-to-reconsider.md)
§5 — euro cap, final class count, rater setup, detector-vs-classifier, and the
name-only prior-knowledge probe. Nothing has been generated yet; this is the
planning stage.
