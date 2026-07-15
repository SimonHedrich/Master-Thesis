# Open Questions and What to Reconsider

**Date:** 2026-07-14
**Status:** Critical review of the plan — gaps, confounds, and framing risks

This document is the "think harder" companion: things the original framing
under-weighted, and decisions to make consciously before generating a single
image.

---

## 1. Confounds that can invalidate the comparison

### 1.1 Prompt length is a confound, not just a nuisance
The incumbent images used ~1,300-word prompts; SDXL reads ~55 words. If you give
each model "the same prompt" naively, a quality gap conflates **model quality**
with **conditioning capacity**. This is handled by the two-regime design
([`01`](01_experiment-design.md) §4, [`05`](05_prompt-strategy-and-length-limits.md)),
but it must be stated explicitly in the thesis or a reviewer will raise it. The
`API-full vs API-compressed` contrast is what proves the long prompt actually
matters (or not).

### 1.2 "Unusual = not in pretraining" is unverifiable
We cannot inspect proprietary training sets. **Reframe honestly:** low GBIF count
is a *proxy* for likely under-representation, not proof. A cleaner, testable
probe: generate each rare species from a **name-only** prompt (no description).
If the model produces a correct saiga/aye-aye from the name alone, it *had* prior
knowledge; if it produces a generic antelope/lemur, it didn't. That name-only
probe is a direct, publishable measure of prior knowledge — stronger than citing
GBIF.

### 1.3 Labeling is a hidden variable
Different generators produce different styles; MegaDetector may find boxes at
different rates. If model A's images auto-label at 95% and model B's at 60%,
you're comparing datasets of different *sizes* and *label quality*, not just
image quality. **Fix:** hold the labeling pipeline identical, and **report
auto-label yield per model as a first-class result** (low yield is a real usability
cost, not something to silently paper over).

### 1.4 Downstream statistical power is thin
12 classes × modest images, small real test sets for rare species → mAP
differences may be inside the noise band. **Mitigations:** ≥3 training seeds with
CIs; lean on the higher-sample teacher-recognition proxy for rare classes; don't
over-claim a downstream winner on a single run. Consider adding a couple more
robust-test classes purely to buy power if the API budget allows.

## 2. Design choices that need a conscious decision

### 2.1 Text-to-image only, or allow reference images?
Gemini Nano Banana takes **up to 14 reference images**; gpt-image-2 supports
edits/references. A single real reference photo could hugely improve rare-species
fidelity. But that turns the task from *zero-shot generation* into *few-shot*,
and not all models support it equally — a confound. **Recommendation:** keep the
main comparison **zero-shot text-to-image** for fairness, and if reference
conditioning is interesting, run it as a **separate, explicitly-labelled axis**
(it's arguably a more practical production method and worth a sub-experiment).

### 2.2 Detector vs classifier as the downstream probe
The thesis targets *detection*, but the fine-grained zebra question is really a
*classification* question (localisation is easy; telling Grévy's from plains is
hard). **Recommendation:** run **both** — a detector for the headline mAP and a
crop classifier (or the fine-granularity head of the eval suite) for the
species-discrimination result. Reporting only detection mAP would bury the most
interesting finding.

### 2.3 How many images/class, and depth vs breadth
For the qualitative rubric, breadth (more species) beats depth. For downstream
power, depth helps. Given the budget cap, prefer **more classes at ~100
images/class** over fewer classes at 300. Fix N only after pricing the whole grid
against a stated euro cap.

### 2.4 Which quality tier per API model
gpt-image-2 low vs high is a 35× price swing ($0.006 → $0.211). "Can it be used"
includes "at what quality tier is it good enough and still affordable?" Include
≥2 tiers for at least one model so the cost/quality frontier is visible, not
assumed.

## 3. Things the original framing under-weighted

### 3.1 Throughput/cost is a first-class result, not logistics
On the RTX 3060, FLUX is ~100–400 s/image → the full 12,600-image set would take
**weeks** locally. That practical fact is a large part of *why* an API model was
chosen, and it belongs in the thesis as a finding — "usable" means usable at
scale, on the available hardware, within budget. Report €/image and images/hour
for every model. Consider fal.ai/Replicate as the middle path (open models,
per-image, no GPU purchase, no weeks-long runs).

### 3.2 Model churn threatens reproducibility
Several benchmarked models shut down in 2026 (gpt-image-1 Oct 23; gpt-image-1.5 &
-mini Dec 1; Imagen 4 in Gemini API Aug 17). **Pin snapshot ids + dates, archive
generated images**, and note that API seeds are generally not exposed (esp.
Gemini) so exact regeneration is impossible. Local models are fully reproducible
(fixed seed) — a genuine advantage worth stating.

### 3.3 Licensing of training data
Using generated images to *train* a model has model-specific restrictions —
notably **BFL FLUX API prohibits training on outputs** (and sells a separate
license), while self-hosted FLUX.1-schnell (Apache-2.0) is clean. Gemini adds
SynthID watermarks. The thesis should state, per generator, the license under
which its images were used. This is easy to overlook and awkward to fix after the
fact.

### 3.4 Domain-shift watchdog already exists — reuse it
`CLAUDE.md` and the evaluation-strategy doc define a real-vs-synthetic
domain-shift watchdog. This comparison is a natural place to exercise it: if a
generator's synthetic-only score diverges sharply from its real-test score, that
divergence is a per-generator finding (some models may look great but transfer
poorly). Report the real-only breakout for every cell.

## 4. Scope discipline (what NOT to do)
- Don't compare 10+ models. 2 API vendors (≤3 models) + 2–3 local is enough.
- Don't regenerate the full 12,600-image dataset for any model — that's what
  blew the budget and is unnecessary given a purposive subset.
- Don't let the study drift into "which generator is best overall" — the question
  is "can these be used, and is the incumbent choice justified." Keep the framing
  scoped ([`00`](00_motivation-and-research-question.md)).

## 5. Concrete open questions to resolve before starting
1. Euro cap for the API side? (proposal: ≤€250) → fixes N/class.
2. Final class count: 12 (this doc's proposal) or +tapirs (+3) for a second
   fine-grained group?
3. Number of raters for the qualitative rubric, and can a biologist be involved?
4. Detector-only, or detector + classifier downstream? (recommend both)
5. Include the name-only prior-knowledge probe (§1.2)? (recommend yes — cheap,
   high-value)
6. Include reference-image conditioning as a separate axis? (recommend: only if
   time allows)
7. Which local host — own RTX 3060, or fal.ai/Replicate per-image? (depends on
   throughput tolerance)
