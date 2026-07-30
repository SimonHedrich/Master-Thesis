# Experiment Design

**Date:** 2026-07-14
**Status:** Draft protocol for the reduced-set generator comparison
**Depends on:** [`00_motivation`](00_motivation-and-research-question.md),
[`02_class-selection`](02_class-selection.md),
[`06_evaluation-methodology`](06_evaluation-methodology.md)

---

## 1. Design in one sentence

Hold **everything** fixed — the same classes, the same per-class image count,
the same labeling pipeline, the same detector/classifier architecture and
hyperparameters, and the same **real** held-out test set — and vary **only the
synthetic-image source**, so that any difference in downstream accuracy or
qualitative score is attributable to the generator.

## 2. The factors being varied

| Factor | Levels | Notes |
|--------|--------|-------|
| **Generator** | **API (5, decided):** gpt-image-2 low, gpt-image-2 medium, Nano Banana 2 Lite, Nano Banana 2 (incumbent), Nano Banana Pro — all can read the full Wikipedia-description prompt. **Local (2–3):** FLUX.1-schnell, RealVisXL+Lightning, SD 3.5 Medium | The primary factor. API selection in [`03`](03_api-models-landscape-and-pricing.md) §0/§2; keep the count tractable — see §6. |
| **Prompt regime** | `full` (long structured prompt) vs `compressed` (≤77-token-safe) | Secondary factor. Needed to **decouple model quality from prompt capacity** — see §4. |

Everything else is a **fixed control**: class set, images/class, splits, labeling,
augmentation, model architecture, training schedule, evaluation set and metrics.

## 3. Pipeline (per generator × prompt-regime cell)

```
select classes (fixed) ──► build prompt (per regime) ──► generate N images/class
      │                                                          │
      │                                                          ▼
      │                                    auto-label with MegaDetector + review
      │                                                          │
      ▼                                                          ▼
  fixed REAL test set  ◄──── evaluate ◄──── train detector/classifier (fixed arch/HPs)
                                                    on this cell's synthetic images
```

Reuse the existing production tooling wherever possible:
- Prompt/index construction: `scripts/synthetic/1-generate_image_list.py`
- Generation: `scripts/synthetic/2-generate_images.py` (Gemini),
  `1-generate_synthetic_images_openrounter.py` (OpenRouter → many models),
  `2-generate_synthetic_images_local.py` (local diffusion).
- Labeling: `scripts/synthetic_model_comparison/2-run_megadetector.py`
  through `5-export_coco.py` — per-cell adaptations of the production
  MegaDetector → triage-review → bbox-labeling → COCO-export chain
  (`scripts/synthetic/{3,4,5,6}-*.py`).
- Training/eval: `scripts/synthetic_model_comparison/training/` — a
  self-contained YOLO26n pipeline copied and adapted from
  `scripts/training/yolo26n/` (12 classes, per-cell data, internal
  train/val split) — see
  [`11_detector-architecture-selection.md`](11_detector-architecture-selection.md)
  for why YOLO26n rather than YOLOv5s, and that package's own README for
  usage.

## 4. Why the prompt regime is a *factor*, not an afterthought

The incumbent images were made with ~1,300-word prompts (full Wikipedia
description + scene spec). Local diffusion models **cannot read prompts that
long** — SDXL truncates at 77 CLIP tokens (~50–60 words); FLUX/SD3.5 tolerate
more via T5 but still far less than 1,300 words (see
[`05_prompt-strategy-and-length-limits.md`](05_prompt-strategy-and-length-limits.md)).

If we simply give Gemini the long prompt and SDXL a short one, a quality gap
**conflates two different causes**: the model's raw generative quality *and* the
information it was allowed to condition on. To separate them:

- Generate a **`compressed`** condition from an identical ≤77-token prompt for
  *every local* model, plus **two API models** (see §4b — not all five, for cost).
- Generate a **`full`** condition for **all five API models** (and FLUX/SD3.5 at
  their longer-but-still-limited ceiling).

This yields the clean contrasts:
- **API-full vs API-compressed** → "how much does the long prompt actually buy?"
  (If little, the local models' prompt handicap barely matters.)
- **API-compressed vs local-compressed** → **fair, apples-to-apples model
  quality** at matched conditioning.
- **API-full vs local-best** → the *practical* comparison (each model at its
  best achievable prompt).

A minimal version drops the full grid and runs just the three cells above.

### 4b. Which API models get both prompt regimes — decision and rationale

**Decision: the compressed-vs-full ablation runs on exactly two of the five API
models — the incumbent Nano Banana 2 and gpt-image-2 low.** The other three API
cells (gpt-image-2 medium, NB 2 Lite, NB Pro) run `full` only.

Why two models, and why these two:

1. **The underlying question is an interaction effect.** The debate "should the
   ablation use a cheap or a powerful model?" *is* the research question: does
   prompt detail matter more for weak or for strong models? One model can never
   measure an interaction — a single result would be ambiguous in exactly the
   way the debate fears (a cheap model's gain might not transfer to strong
   models; a strong model's indifference might not transfer to cheap ones).
   Two models at different capability points resolve this. The direction is
   genuinely uncertain: the intuitive hypothesis is that cheap models benefit
   more from spelled-out detail, but prompt-following results generally show
   the opposite — *stronger* models exploit long prompts better, while weaker
   ones ignore or garble the extra detail. Either outcome is a reportable
   thesis finding, which is precisely why two data points are needed.
2. **These are the two cheapest possible ablation sites.** The incumbent's
   `full` cell already exists from production (sunk cost) — its ablation costs
   only the one new compressed cell (~€50 at 0.5K for 1,200 images). For
   gpt-image-2 low, the *pair* of cells costs ~$14 total ($0.006/img). The
   budget objection to a wider ablation does not apply to these two.
3. **The incumbent ablation is the most production-relevant question in the
   whole study:** did the ~1,300-word Wikipedia prompts (already paid for
   across ~12,600 images) actually buy anything over a 50-word prompt?
4. **Nano Banana Pro is excluded deliberately:** it is the most expensive site
   (~$0.13–0.24/img → ~$150–290 for one extra compressed cell) and the least
   decision-relevant — no one will generate training data at Pro prices
   regardless of prompt length, and Pro's `full` cell already establishes the
   quality ceiling.
5. **The gpt-image-2 low result transfers to medium for free:** low and medium
   are the same model at different sampling compute and share the same text
   encoder, so prompt *understanding* is identical across the two tiers.
6. **Nothing is wasted:** the two API compressed cells double as the
   **API-compressed anchor** required by §4 for the apples-to-apples
   comparison against the local models, which can only read short prompts.

## 5. Fixed controls (must not vary across cells)

1. **Images per class.** Same N for every generator (proposal: 100/class for
   detector training + a fixed real test set already in the repo). Under-provisioning
   one generator would bias downstream results.
2. **Labeling.** Same MegaDetector pass + same review rules. **Report the
   auto-label yield per generator** (fraction of images where MD finds exactly
   one box of adequate size) — a low yield is itself a quality signal (malformed
   or ambiguous animals).
3. **Splits.** Synthetic images are **train-only**. The **test set is REAL
   images only**, reused unchanged from the existing dataset split — this is the
   whole point and aligns with the thesis evaluation strategy (real-only is the
   anchor metric; never judge on synthetic). See `CLAUDE.md` and
   `docs/plans/2026-06-10_model-evaluation-strategy.md`.
4. **Architecture + hyperparameters.** One detector (**YOLO26n**, existing
   config — see
   [`11_detector-architecture-selection.md`](11_detector-architecture-selection.md)
   for why not YOLOv5s) and/or one classifier, identical schedule/seed policy
   for every cell.
5. **Augmentation.** Identical.

## 6. Keeping it tractable (and cheap)

Do **not** compare ten models. A defensible, budget-safe grid:

| Tier | Model | Prompt regime(s) | Rationale |
|------|-------|------------------|-----------|
| Incumbent | Nano Banana 2 (`gemini-3.1-flash-image`) | full (already generated) + **compressed** | baseline, full cell already paid for; production-relevant ablation (§4b) |
| API cross-vendor, floor | gpt-image-2 **low** | full + **compressed** | price floor ($0.006/img); cheap-model ablation endpoint + cross-vendor replication (§4b) |
| API cross-vendor, mid | gpt-image-2 **medium** | full | quality-tier contrast vs `low` at matched prompt (~9× price) |
| API intra-vendor, floor | Nano Banana 2 Lite | full | Google price floor vs incumbent |
| API intra-vendor, ceiling | Nano Banana Pro | full | quality ceiling; **no ablation** — most expensive, least decision-relevant (§4b) |
| Local | FLUX.1-schnell | compressed (T5 ceiling) | best open photoreal, Apache-2.0 |
| Local | RealVisXL + SDXL-Lightning | compressed (77 tok) | fast SDXL photoreal |
| (optional) Local | SD 3.5 Medium | compressed (T5 ceiling) | no-quant baseline |

**Rough generation cost** for the API cells (12 classes × 100 images = 1,200
images/cell, before the −50% Batch discount): gpt-image-2 low ≈ $0.006/img →
~$7/cell; gpt-image-2 medium ≈ $0.053/img → ~$64/cell; NB 2 Lite ≈ $0.034/img →
~$41/cell; NB 2 (incumbent) ≈ €0.04–0.15/img → €50–180/cell by resolution;
NB Pro ≈ $0.13–0.24/img → ~$156–288/cell. The prompt ablation adds only two
compressed cells: NB 2 (~€50 at 0.5K) + gpt-image-2 low (~$7).
Local cells cost **only GPU time** (but that time is large — see the throughput
note in [`04`](04_local-models-and-output-parameters.md)). Full pricing in
[`03`](03_api-models-landscape-and-pricing.md) and the scraped sources.

> Recommendation: fix N per class *after* pricing all cells, so the whole grid
> fits a stated euro cap (propose ≤€250 for the API side). Prefer more classes
> at fewer images/class over the reverse — species coverage matters more than
> per-class depth for the qualitative rubric.

## 7. Outputs of the experiment

- Per-cell **downstream mAP on the real test set** (mixed and real-only
  breakouts), plus per-species AP for the fine-grained group (zebras).
- Per-cell **qualitative rubric scores** (blind, multi-rater) and
  **auto-proxy** scores (teacher recognition confidence, FID, CLIP-score).
- A **cost/throughput** table (€/image, images/hour, GPU feasibility).
- A single ranking table that the thesis can cite (see
  [`06`](06_evaluation-methodology.md) §7).
