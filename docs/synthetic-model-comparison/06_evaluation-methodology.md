# Evaluation Methodology

**Date:** 2026-07-14
**Status:** How to evaluate the generators rigorously (the professor's requirement)
**Depends on:** [`00_motivation`](00_motivation-and-research-question.md),
[`01_experiment-design`](01_experiment-design.md)

---

## 1. Principle: three independent axes, no single soft judgment decides

The professor's objection was to a **single, informal** judgment ("Gemini looks
better"). The remedy is not to drop qualitative evaluation but to make it
**structured, blind, and multi-rater**, and to triangulate it with two
quantitative axes that don't depend on human impression. A generator's standing
is reported on all three; a claim is strong only when they agree.

```
Axis A  Qualitative rubric   (human, blind, multi-rater)    ← the professor's requirement
Axis B  Automatic proxies    (teacher recognition, FID, CLIP-score)  ← cheap, many samples
Axis C  Downstream utility    (detector/classifier mAP on REAL test) ← the thesis's actual goal
```

## 2. Axis A — structured qualitative rubric

Score each generated image (or a fixed random sample per class × model) on a
fixed rubric. Proposed 1–5 scales:

| Criterion | What it measures |
|-----------|------------------|
| **Anatomical correctness** | limb count, proportions, no fused/duplicated parts |
| **Diagnostic-feature fidelity** | are the species-defining markers present and correct? (stripe type, nose, tail) |
| **Species identity** | would a knowledgeable observer call it the intended species (vs a generic/related animal)? |
| **Habitat plausibility** | is the background ecologically consistent with the species? |
| **Photorealism** | does it read as a real field photograph vs illustration/CGI? |
| **Artifact rate** (binary/count) | text, watermarks, extra animals, impossible geometry |

Rigor requirements (this is what makes it thesis-grade):
- **Blind:** raters see images with model identity hidden and order randomised.
- **Multiple raters:** ≥2 (ideally 3), independently.
- **Inter-rater agreement:** report **Cohen's / Fleiss' κ** (or weighted κ for the
  ordinal scales). Low agreement → tighten rubric definitions and re-rate.
- **Pre-registered rubric:** freeze the criteria and anchor descriptions *before*
  looking at outputs, so scoring isn't reverse-engineered to a preferred model.
- **Domain check (optional but strong):** have one rater with biology background,
  or cross-check diagnostic features against a reference field guide.

## 3. Axis B — automatic proxies (cheap, high sample size)

These scale to all images and give the statistical power the rare classes lack:

1. **Teacher-recognition confidence (the key proxy).** Run the project's teacher
   / SpeciesNet classifier on each generated image. *Does the model that will act
   as teacher recognise the intended species, and with what confidence?* This is
   a near-direct measure of "are the diagnostic features there," and it is
   exactly the signal that matters for downstream distillation. Report mean
   confidence and top-1 accuracy of teacher-on-synthetic, per model × class.
2. **FID / KID** against the **real** images of each class (per-class, because
   global FID hides per-species failure). Measures distributional realism.
   Caveat: FID is unreliable at small n — use as corroboration, report n.
3. **CLIP-score** (image–prompt alignment) for prompt adherence, computed with a
   CLIP model *not* used by any generator.
4. **Auto-label yield** from MegaDetector: fraction of images with exactly one
   adequately-sized box. Low yield = malformed/ambiguous subjects — a quality
   signal and also a data-efficiency cost.

## 4. Axis C — downstream utility (the thesis's real target)

Train the fixed detector/classifier on each cell's synthetic images and evaluate
on the **held-out REAL test set** (never synthetic — per `CLAUDE.md` and
`docs/plans/2026-06-10_model-evaluation-strategy.md`).

- **Headline:** mAP on the real test set (mixed and **real-only** breakouts).
- **Fine-grained metric (the zebra test):** per-species AP and the
  within-`zebra`-group confusion matrix. The decisive question: does model X's
  synthetic data let the classifier separate Grévy's / plains / mountain zebra?
  Tie to the existing Δ_fine machinery in `eval_suite/`.
- **Rare-species metric:** for classes with ≥100 real test images (kinkajou,
  water deer, ringtail) report AP directly; for test-limited rare classes, lean
  on Axes A/B instead and say so.

### Statistical care
- Few classes × modest images → downstream mAP differences may be within noise.
  Use **multiple training seeds** per cell (≥3) and report **mean ± CI**; don't
  declare a winner on a single run.
- Where downstream power is weak, the **teacher-recognition proxy** (Axis B) is
  the higher-sample fallback that still speaks to utility.

## 5. Controlling confounds (recap of what must be held fixed)

- Same images/class, splits, labeling rules, architecture, HPs, augmentation,
  real test set. Only the generator (and, deliberately, the prompt regime) vary.
- Report **auto-label yield** and **generation wall-clock/cost** alongside quality
  so "usable" reflects practicality, not just picture quality.
- Zero-shot text-to-image only — do **not** use reference-image conditioning for
  some models and not others (that would confound prior-knowledge with few-shot
  help). If you want to study reference conditioning, make it its own labelled
  axis (see [`07`](07_open-questions-and-what-to-reconsider.md)).

## 6. Reproducibility

- Local models: fix seeds (already `42 + index`), log everything — fully
  reproducible.
- API models: **seeds are generally not exposed** (esp. Gemini) → exact images
  aren't reproducible. Record model id, date, all request params, and archive the
  generated images as the artifact. State this asymmetry in the thesis.

## 7. The single results table the thesis will cite

| Model | Prompt regime | Rubric (mean, κ) | Teacher top-1 | FID↓ | CLIP-score | Real-test mAP | Zebra fine AP | €/img | img/hr |
|-------|---------------|------------------|---------------|------|------------|---------------|---------------|-------|--------|
| Gemini 3.1 Flash Image | full | | | | | | | | |
| Gemini 3.1 Flash Image | compressed | | | | | | | | |
| OpenAI gpt-image-2 | full | | | | | | | | |
| OpenAI gpt-image-2 | compressed | | | | | | | | |
| FLUX.1-schnell | compressed | | | | | | | | |
| RealVisXL+Lightning | compressed | | | | | | | | |
| SD 3.5 Medium | compressed | | | | | | | | |

A generator "can be used" if it lands within a stated margin of the incumbent on
Axes A–C **and** is practical on cost/throughput. The production choice of Gemini
is justified iff it is at/near the top on the axes that matter (real-test mAP and
diagnostic fidelity), not merely on raw photorealism.
