# Motivation and Research Question

**Date:** 2026-07-14
**Status:** Framing document for the synthetic-generation model-comparison sub-study

---

## 1. Where this task comes from

In the current pipeline the entire synthetic training set (≈12,600 images across
76 rare "Tier C" classes) was generated with a **single** model —
`gemini-3.1-flash-image-preview` (the Gemini "Flash Image" family, colloquially
"Nano Banana"). The model was chosen after a qualitative look at candidate
outputs: the Gemini images simply *looked* more realistic than what the local
diffusion models produced.

In a supervision meeting the professor flagged this as **methodologically
insufficient for a Master's thesis**. Her point, restated precisely:

> Eyeballing a handful of images and declaring "Gemini looks better, so we used
> Gemini" is not a defensible model-selection argument. A thesis-grade choice
> must be **qualitatively evaluated** against alternatives under a stated,
> repeatable criterion — even when picking the "best generator" is not itself
> the research question.

This is correct and it is the reason for this sub-study. The fix is **not** to
re-run the whole pipeline on every model (the budget is spent — see §3), but to
build a **small, rigorous, controlled comparison** on a deliberately chosen
subset of classes that can stand in for the whole.

## 2. The actual research question (and what this sub-study adds)

The thesis's core question about synthetic data is **not** "which image
generator is best?" It is:

> **Can current image-generation models be used to produce useful synthetic
> training data for wildlife object detection / classification — particularly
> for rare, long-tail species where real imagery is scarce?**

For that question a full head-to-head is not strictly required. But the
professor's critique still binds, for two reasons:

1. **The single-model choice is a load-bearing assumption of the whole thesis.**
   If a cheaper/free/local model would have worked as well, or if Gemini's
   apparent realism does *not* translate into downstream detection accuracy,
   that changes the thesis's conclusions. The choice must be *earned*, not
   asserted.
2. **"Can they be used" is only meaningful relative to alternatives.** "Useful"
   is a comparative claim. A controlled comparison across a few models is what
   turns "the images look good" into "model X's synthetic data yields Y mAP on a
   held-out **real** test set, versus Z for model W."

So this sub-study answers a **scoped, secondary** question that supports the
primary one:

> **On a deliberately chosen subset of classes, how do synthetic images from
> different generators (API and local) compare — both qualitatively (a
> structured rubric, not vibes) and quantitatively (downstream detection/
> classification on a real test set) — and does the ranking justify the
> production choice of Gemini?**

## 3. Hard constraint: the budget is (mostly) spent

The production run consumed most of the synthetic-image budget (≈€500–630 for
the 12,600-image set at ≈€0.04/image). **We cannot regenerate the full dataset
per model for a 1:1 whole-pipeline comparison.**

Consequence: the comparison must be run on a **reduced set of classes** (see
[`02_class-selection.md`](02_class-selection.md)) with a modest images/class
budget, chosen so the subset is *representative of the phenomena that matter* —
not a random sample, but a purposive one (rare species, robust-test species,
fine-grained look-alike groups).

## 4. Scope of the comparison

Three model tiers are in scope (details in
[`03_api-models-landscape-and-pricing.md`](03_api-models-landscape-and-pricing.md)
and [`04_local-models-and-output-parameters.md`](04_local-models-and-output-parameters.md)):

1. **The incumbent** — `gemini-3.1-flash-image-preview` (already have production
   images and full-length prompts for these classes).
2. **Other API models, same prompt** — this directly answers the user's
   question *"are there other models from OpenAI?"* Yes: OpenAI exposes
   **`gpt-image-2`, `gpt-image-1.5`, `gpt-image-1-mini`** (plus legacy DALL·E 3),
   and Google exposes **multiple** image models (Gemini Flash Image variants +
   the Imagen line). So we can compare *within* a vendor (different OpenAI
   models, different Gemini models) and *across* vendors on an identical prompt.
3. **Local open-weight models** — FLUX.1-schnell, an SDXL-based photoreal model
   (RealVisXL + SDXL-Lightning), SD 3.5 Medium. A local generation script
   already exists (`scripts/synthetic/2-generate_synthetic_images_local.py`), so
   the marginal engineering cost is low. **Key caveat:** these have *severe
   prompt-length limits* (CLIP's 77-token ceiling for SDXL) versus the ~1,300-word
   prompts used for Gemini — this is both a fairness issue and a finding in its
   own right (see [`05_prompt-strategy-and-length-limits.md`](05_prompt-strategy-and-length-limits.md)).

## 5. What "qualitatively evaluated" will mean here (preview)

To satisfy the professor, "qualitative" is operationalised, not left to
impression (full method in [`06_evaluation-methodology.md`](06_evaluation-methodology.md)):

- A **written rubric** scoring anatomical correctness, diagnostic-feature
  fidelity, habitat plausibility, pose realism, and artifact rate.
- **Blind** rating (model identity hidden), **multiple raters**, and an
  **inter-rater agreement** statistic so the qualitative scores are auditable.
- Backed by **automatic proxies** (teacher/SpeciesNet recognition confidence on
  generated images; FID/CLIP-score) and the **downstream** signal (mAP on the
  real test set), so no single soft judgment carries the conclusion.

## 6. Deliverable of the sub-study

A section of the thesis that can state, defensibly:
> "We compared N generators on K purposively chosen classes under a fixed
> protocol. Gemini ranked [rank] on the qualitative rubric and [rank] on
> downstream real-test mAP; the gap to [next model] was [size]. The production
> choice of Gemini is therefore justified / should be revisited because […]."
