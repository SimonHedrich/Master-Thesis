# Model Evaluation Strategy

**Date:** 2026-06-10
**Status:** Draft for discussion — no implementation yet
**Scope:** How to evaluate trained detectors so they can be compared (a) to each
other across the thesis experiments and (b) to public benchmarks, given the
peculiarities of this wildlife dataset.

---

## 1. Problem Statement

After training, every candidate model needs an evaluation that is **precise
enough to draw scientific conclusions** yet **compact enough to report without
drowning the reader in tables**. Two features of this use case make standard
COCO reporting insufficient on its own:

1. **Visually near-identical classes.** Several species are barely
   distinguishable in a single still frame (the three zebra species; African vs.
   Asian elephant; Canada / Eurasian lynx + bobcat + caracal; the *Equus*
   asses; multiple gazelles and antelopes). A fine-grained mistake here is a
   fundamentally different (and more forgivable) error than missing the animal
   entirely or calling a wolf a squirrel.
2. **Heterogeneous training conditions.** Classes were trained under four
   different data regimes (Bands A–D: synthetic-only, mixed, real-200,
   real-large) and the dataset ships **two test domains** (the uneven real test
   set and a balanced 225×50 synthetic test set). Aggregate numbers blur these
   apart.

The COCO 12-metric vector (AP@[.5:.95], AP50, AP75, AP_S/M/L, AR@1/10/100,
AR_S/M/L) is the right *baseline* instrument and the bridge to public numbers,
but it is too coarse to localise *why* a model is good or bad here.

The user's stated assessment levels are:

- **Detection only** — is there an animal, ignore the species label.
- **Look-alikes merged** — coarse classification where confusable species
  collapse into one group.
- **Per training band** (A/B/C/D), at both fine and merged granularity.
- **Real vs. synthetic test set**, separately.

Crossed naively, these produce **dozens** of AP/AR vectors. This document
analyses that explosion and recommends a way to tame it.

---

## 2. The Evaluation Axes (and why they are not equal)

It helps to name the independent axes precisely. Everything below is a point in
this space:

| Axis | Levels | What it is |
|------|--------|-----------|
| **G — Granularity** | `detect` (class-agnostic) · `coarse` (look-alikes merged) · `fine` (full 225-way) | The label resolution at which a TP/FP is judged |
| **D — Test domain** | `mixed` (real + synthetic union) · `real` · `synthetic` | Which held-out set the metric is computed on |
| **B — Training band** | `A` · `B` · `C` · `D` · `all` | A *per-class* property of how those classes were trained |
| **M — Metric vector** | AP@[.5:.95], AP50, AP75, AR@1/10/100, area-split | The COCO statistics themselves |

The **`mixed`** domain is the union of the real test images and the balanced
225×50 synthetic test images, scored together as a single set. `real` and
`synthetic` are its two components, evaluated on their own.

Naive full cross: `3 (G) × 3 (D) × 5 (B) × 12 (M) = 540` numbers, before
per-class tables. This is the "overwhelming and confusing" problem.

**Key reframe — the axes serve different roles:**

- **B (band) and D (domain) are the scientific independent variables.** They are
  what the thesis is *about* (does synthetic training data work? does it
  generalise to real photos?). Crossing them is the experiment.
- **G (granularity) is not a separate experiment — it is an error
  decomposition.** `detect`, `coarse`, and `fine` are not three rival numbers;
  they are nested upper bounds. The *gaps between them* are the interesting
  quantity (see §4).
- **M (metric vector) is a fixed instrument, not an axis to cross.** Report the
  full 12-metric vector only in the few "primary" cells; elsewhere carry just
  `mAP@[.5:.95]` and `mAP50`.

Once you stop treating all four axes as co-equal, the explosion collapses.

---

## 3. Core Principle: Slices, Not the Cube

> **Define one default cell. Every diagnostic varies exactly one axis away from
> it and reports a compact 2-number metric. Fill the full 12-metric vector only
> in the default cell and per band.**

**Default cell** (the headline, directly comparable across all thesis models):

```
G = fine   ·   D = mixed (real + synthetic)   ·   B = all   ·   M = full COCO-12 vector
```

This is the single number that ranks models against each other. The `mixed`
default reports performance over both test domains at once — a fuller, more
robust picture than either domain alone, especially for classes with few or
low-quality real photos (rationale in §3.1). The **real-only breakout** (one of
the `D` perturbations below) is always reported alongside it: it is the
public-comparison anchor (the cleaner analog to a published real-image COCO mAP)
and the watchdog against synthetic inflation; see §3.1 and §7. Everything else is
a *one-axis perturbation* of the default cell, chosen because it answers a
specific question:

| Vary… | Holding… | Answers | Reports |
|-------|----------|---------|---------|
| **G** → detect / coarse / fine | D=mixed, B=all | *Where do errors come from — localisation, look-alike confusion, or fine ID?* | 3× (mAP, mAP50) + 2 gaps |
| **B** → A / B / C / D | D=mixed, G∈{fine,coarse} | *Does synthetic / mixed / real training change accuracy?* (the dataset's core comparison) | band × granularity grid of (mAP, mAP50) |
| **D** → mixed / real / synthetic | B=all, G∈{fine,coarse} | *Domain shift — does the model hold up on real photos, and does synthetic training generalise?* | mixed headline + paired real−synth Δ per class |

We deliberately **do not fill the full 4-D cube** (e.g. band×domain×granularity
× all 12 metrics). The cube's interior cells answer no question we are asking and
are relegated to machine-readable appendix artifacts (§9), not the narrative.

This turns "540 numbers" into **~4 small tables carrying ~40 numbers** in the
body of the thesis.

### 3.1 Why `mixed` is the default (and when to revisit it)

The default test domain was deliberately moved from `real` to `mixed`
(real + synthetic union) on 2026-06-10, superseding the earlier "test on real"
rule. The reasoning:

- **Robustness on thin / low-quality real classes.** Band A classes have as few
  as 5–30 real test images, some of low quality. Real-only per-class AP there is
  high-variance and easily dominated by a handful of hard frames. Folding in a
  fixed 50 synthetic images/class gives every class a consistent, controlled
  probe and stabilises the metric exactly where the real test set is weakest.
- **Negligible distortion where real data is plentiful.** For Band D (up to ~500
  real test images/class), the 50 synthetic images are ~9–25 % of the class's
  test mass — small enough not to move the number much, while keeping the *same*
  50-image synthetic contribution present for every class. The synthetic probe is
  consistent across all 225 classes regardless of band.
- **Never synthetic-only.** The model is never judged on synthetic images alone;
  `mixed` always contains the real evidence, and the `real`-only breakout is
  reported alongside the headline (§7, §9).

**Acknowledged property — synthetic weight is uneven by design.** Because the
synthetic contribution is a fixed 50 images while the real contribution varies,
the *synthetic fraction* of each class's score is inversely related to real
availability: dominant for thin Band A classes (the stabilisation we want),
negligible for Band D. This is intentional, not a bug, but it means the `mixed`
number's composition differs per class — hence the per-class real/synthetic split
sizes are reported (§8) so the headline is transparent.

**Revision trigger (watchdog).** The real-vs-synthetic domain-shift delta (§6b)
and the `real`-only breakout are monitored on every run. **If a clear discrepancy
between the `mixed` and `real` (or `synthetic` and `real`) results emerges** —
i.e. the synthetic component is materially inflating or deflating the headline
relative to real-world performance — **the default axes will be revised** (likely
reverting the default to `real`, keeping `mixed` as a secondary figure). Until
that signal appears, `mixed` is the default.

---

## 4. Granularity as Error Decomposition (peculiarity #1)

The three granularity levels are a ladder of nested upper bounds:

```
mAP_detect   ≥   mAP_coarse   ≥   mAP_fine
   │                │                │
   │                │                └─ localisation + coarse ID + fine look-alike ID
   │                └─ localisation + coarse ID
   └─ localisation only ("is there an animal?")
```

Report them as a **gap decomposition** rather than three independent rows:

```
mAP_detect                         = X
  − Δ_coarse  (coarse-ID cost)     = a      ← how much accuracy is lost just by
  − Δ_fine    (look-alike cost)    = b        having to name a group, vs. naming
                                              the exact species among look-alikes
= mAP_fine                         = X−a−b
```

- A large **Δ_fine** with a small **Δ_coarse** is the signature of the look-alike
  problem: the model finds and roughly categorises the animal but cannot pick
  the right species within a confusable group. This is the *expected* and most
  defensible failure mode for this dataset.
- A large **Δ_coarse** means cross-group confusion (wolf↔squirrel) — a real
  failure.
- A low **mAP_detect** means the detector cannot even localise — orthogonal to
  classification entirely.

### 4.1 Implementation is a label remap, not new metric code

All three levels run through the **same** `torchmetrics` /
`pycocotools` evaluator. Only the category IDs are rewritten before scoring,
for **both predictions and ground truth**:

- `detect`: every `category_id → 1` (single "animal" class).
- `coarse`: every fine `category_id → group_id` via a label→group table (§5).
- `fine`: identity (the current behaviour).

This means the entire granularity ladder is ~30 lines of remapping on top of the
existing `evaluation.py`, run three times. No bespoke metric logic.

> **Caveat to note in the writeup:** merging classes turns what were
> different-class predictions into same-class duplicates on one animal. Standard
> COCO greedy matching counts the best one as TP and the rest as FP duplicates,
> so coarse mAP is *slightly* penalised by duplicates rather than inflated — a
> conservative, honest direction. A `max_det`/NMS sanity pass is worth a footnote
> but does not change the method.

### 4.2 Complement: TIDE + within-group confusion

The gap decomposition tells you *how much* accuracy each level costs; two
companion instruments tell you *which errors*:

- **TIDE** (Bolya et al., 2020) decomposes detector error into
  classification / localisation / duplicate / background / missed in one pass.
  This is an off-the-shelf, citable way to corroborate the granularity gaps and
  is recommended as the primary error-typing tool.
- **Within-group confusion rate** — for each look-alike group, of the detections
  that are correctly localised *and* in the right group, what fraction get the
  wrong fine species? This is the single most interpretable number for "does it
  confuse the zebras?" Report it as a small per-group table + a block-diagonal
  confusion matrix restricted to the look-alike groups (the full 225×225 matrix
  is appendix-only).

---

## 5. Defining the "Look-Alike" Groups (the key design decision)

`coarse` granularity needs a fixed **label → group** table. The 225 labels are
at *mixed* taxonomic ranks already (178 species, 35 genus, 12 family), so this
table has to be authored once and frozen. Three options:

| Option | How groups are formed | Pros | Cons |
|--------|----------------------|------|------|
| **(a) Taxonomic rollup** | Map each label to its **genus** (then family for orphans) using `reports/genus_species_mapping.csv` + `family_species_mapping.csv` | Reproducible, a-priori, zero model leakage, defensible in writing | Taxonomy ≠ visual similarity (some genera are visually distinct; some look-alikes span genera, e.g. lynx vs. caracal) |
| **(b) Empirical / confusion-driven** | Cluster classes the model actually confuses | Reflects real visual confusability | **Circular** — derived from the model being evaluated; not valid as an a-priori grouping |
| **(c) Curated visual-similarity sets** | Hand-list the notorious look-alikes (zebras, elephants, the lynx/caracal cluster, asses, gazelles…) from domain knowledge | Targets exactly what "look-alike" means | Subjective; must be documented and frozen before evaluation |

**Recommendation: (a) as the backbone, refined by (c).**
Use genus-level taxonomic rollup as the default, reproducible coarse grouping,
then apply a **small, frozen, documented override list** for the handful of
visually-confusable clusters that taxonomy splits or merges wrongly (e.g. group
lynx + bobcat + caracal even across genera; keep visually-distinct
same-genus pairs split if warranted). Freeze this table in
`reports/lookalike_groups.csv` **before** running any evaluation, and treat it
as a fixed experimental artifact. Use (b) only descriptively, *after the fact*,
to comment on whether the model's empirical confusions line up with the a-priori
groups — never to define them.

The label→group table is the one genuine prerequisite that must be built and
reviewed before evaluation can start.

---

## 6. Bands and Domain — the Two Real Cross-Tabs (peculiarity #2)

Band (B) and test domain (D) are **orthogonal**: band is *how a class was
trained*; domain is *which test images we score on*. The two cross-tabs worth
reporting:

### 6a. Band × Granularity

The dataset's headline scientific comparison (synthetic-only A vs. mixed B vs.
real-200 C, at equal 200-image budget; D as the well-resourced reference),
reported on the default `mixed` test domain:

|        | mAP_fine | mAP50_fine | mAP_coarse | mAP50_coarse |
|--------|----------|-----------|-----------|--------------|
| Band A (synth-only) | … | … | … | … |
| Band B (mixed)      | … | … | … | … |
| Band C (real-200)   | … | … | … | … |
| Band D (real-large) | … | … | … | … |

Because A vs. B vs. C are *different species* that happened to fall in different
bands (not a controlled within-class experiment — see the dataset strategy doc),
the **coarse column is especially important here**: it strips out the
fine-grained difficulty differences between the species sets so the
training-regime effect is less confounded by which look-alikes happen to live in
each band.

> **Domain caveat for the band comparison.** On the `mixed` domain, Band A/B
> classes are partly scored on synthetic test images that share their *training*
> domain, which flatters them; Band C/D (real-trained) are scored partly
> off-domain. For the band comparison specifically, also show the **real-only
> breakout** — it is the more honest synthetic-vs-real-training read because it
> removes the home-field advantage. The `mixed` grid is the consistent default;
> the real-only grid is the scientific tie-breaker.

### 6b. Domain shift: real vs. synthetic test

Score the *same classes* on both test domains and report the **paired delta**,
not two independent matrices:

```
Δ_domain(class) = mAP_real(class) − mAP_synthetic(class)
```

- Reported as a distribution (mean ± CI, and a per-band breakdown).
- This delta is the **watchdog** for the `mixed` default (§3.1): a large or
  systematic real−synth gap is the signal to revise the default axes.
- The synthetic test set is **balanced** (50/class), so it is the better
  instrument for class-balanced and per-class comparison; the `real`-only
  breakout remains the public-comparison anchor (§7).
- Crucial framing: Band A/B classes were *trained on synthetic data*, so a small
  real−synth gap for them is the success criterion (synthetic training
  generalised). For Band C/D (real-trained), the synthetic test set probes the
  *reverse* shift. Interpret the sign per band.

We do **not** also produce band × domain × all-12-metrics; the per-band domain
delta at fine+coarse is sufficient.

---

## 7. Comparability to Public Benchmarks

Be explicit in the thesis about what is and is not a fair comparison:

- **Compare against the `real`-only breakout, not the `mixed` default.** Public
  benchmarks are real photographs; the `mixed` headline blends in synthetic test
  images and is therefore *not* the figure to put next to a published number. Use
  the real-only breakout (the thesis's primary-evaluation figure) for every
  external comparison, and reserve `mixed` for internal cross-model ranking.
- **Not directly comparable in absolute mAP:** public YOLOv5 numbers are 80-class
  COCO. Ours is **225-class fine-grained wildlife** — more classes, many of them
  near-duplicates — so a lower fine mAP is expected and is *not* evidence of a
  worse detector. State this plainly.
- **The closest honest analog is `mAP_detect`** (class-agnostic): it measures
  pure detection/localisation capability on the same footing as a generic
  object detector, independent of the fine-grained taxonomy. Lead the
  public-comparison paragraph with this number, then explain the granularity gap.
- **Use COCO-standard settings** so the bridge is valid: AP averaged over
  IoU=.5:.05:.95 with `max_det=100` (`constants.EVAL_MAX_DET=100`, used
  everywhere — model selection and reporting). The dual-cap idea was dropped:
  the busiest test image has 28 GT boxes, so 100 ≫ enough and 300 made no
  meaningful difference (Q2).
- **The balanced synthetic test set** gives a class-balanced accuracy that is
  more comparable across classes than the imbalanced real test set — useful as a
  secondary, standardized figure.

---

## 8. Statistical Hygiene

These caveats must accompany the numbers or the fine-grained tables will
mislead:

- **Test-limited classes.** ~17 Band-A classes have <30 *real* test images (some
  <10: giant armadillo 11, domestic pig 15, hog badger 18). Per-class AP for
  these is high-variance on the real component. Flag them in every per-class
  table, and **report the headline mAP both with and without** the <30-image
  classes so the reader sees the sensitivity. The fixed 50 synthetic images/class
  in the `mixed` set partly cushions this, but the real component stays thin.
- **`mixed` re-weights the per-class test mass.** Adding 50 synthetic images per
  class to an uneven real test set shifts each class's effective weight (and
  *partially balances* the otherwise wildly imbalanced real set). Report the
  real/synthetic split sizes per class so the headline `mixed` number's
  composition is transparent and reproducible.
- **Macro vs. micro averaging.** COCO mAP is **macro** (unweighted mean over
  classes) — class-imbalance-robust and the standard; keep it as primary. Also
  report a **count-weighted** variant as a sensitivity check, since the test set
  is imbalanced and the two can diverge.
- **Confidence intervals.** For the headline mAP and the band/domain
  deltas, report bootstrap CIs (resample images) so cross-model and cross-band
  differences can be judged as significant or not — important because Band A/B
  test sets are small.
- **Single fixed seed & fixed sampled subsets** for any sampled evaluation, so
  re-running against a new checkpoint is directly comparable (the existing
  `run_inference.py` already does this with `SEED=42`).

---

## 9. Recommended Reporting Structure

A three-tier hierarchy. Tiers 1–2 live in the thesis body; Tier 3 is
appendix / machine-readable artifacts.

**Tier 1 — Headline (1 table per model).**
Default cell: `G=fine, D=mixed, B=all`, full COCO-12 vector — the cross-model
ranking number. Alongside it, the **real-only breakout** (same cell, `D=real`)
as the primary-evaluation figure and public-comparison anchor, plus the single
`mAP_detect` analog (§7).

**Tier 2 — Diagnostic (3 small tables, the analytical core).**
1. **Granularity gap decomposition** (§4): detect/coarse/fine + Δ_coarse, Δ_fine,
   on the mixed test set, all classes. Backed by a TIDE error-type chart.
2. **Band × granularity grid** (§6a): 4 bands × {fine, coarse} × {mAP, mAP50}.
3. **Domain-shift delta** (§6b): real−synth paired Δ, per band, fine + coarse;
   plus the look-alike within-group confusion table (§4.2).

**Tier 3 — Appendix / artifacts (not in narrative).**
- Full per-class AP table (225 rows), with test-image count and band flag.
- Block-diagonal look-alike confusion matrix (and full 225² matrix as a file).
- Full COCO-12 vector for each band cell (the cube interior).
- AR@1/10/100 and area-split (S/M/L) breakdowns.
- All of the above emitted as JSON/CSV (and logged to MLflow as tables) so the
  numbers exist and are auditable without cluttering the prose.

### What we deliberately do NOT report (and why)

- The full `band × domain × granularity × 12-metric` cube → no question needs its
  interior; appendix file only.
- AR / area-split crossed with band or granularity → diagnostic noise; appendix.
- Empirical confusion-derived groupings as a *primary* metric → circular.

---

## 10. Implementation Sketch (for a later plan — not now)

To keep the analysis honest, the eventual build is small and reuses the existing
engine:

1. **`reports/lookalike_groups.csv`** — frozen label→group table (§5). The only
   real design artifact; build and review first.
2. **A remap layer** over `evaluation.py`: given the eval's pred/GT dicts, run
   the same `MeanAveragePrecision` three times with category IDs rewritten
   (`fine` / `coarse` / `detect`). Same for a pycocotools path if the full
   12-vector + AR is wanted.
3. **A band/domain stratifier**: filter the test annotations by the per-class
   band (from the dataset manifest) and run the evaluator per stratum. Run
   inference once on `data/real/annotations_test.json` (real, 63,822 imgs) and
   once on `data/synthetic/annotations_test.json` (synthetic, 11,250 imgs); the
   **`mixed` domain is the concatenation of the two**, so it needs no extra
   inference — just score the union. **The two files share `image_id=1…`, so
   offset the synthetic image + annotation IDs past the real max before merging**
   (category IDs already match and need no remap).
4. **TIDE** as an optional add-on consuming the same COCO predictions JSON that
   `run_inference.py` already emits.
5. **Reporting glue**: assemble Tiers 1–3 tables to CSV/Markdown + MLflow.

A single `predictions.json` per (model, real-or-synthetic domain) feeds *all*
granularity, band, and domain slices via remap/filter; `mixed` is just the two
prediction sets concatenated. Predictions are computed once per model per
component domain, then scored many ways. This is what makes the dozen views
cheap.

---

## 11. Open Questions / Decisions Needed

1. **Look-alike grouping (§5):** approve "genus rollup + frozen curated override"
   as the coarse definition? If so, who reviews the override list, and at what
   rank do orphan single-species genera sit (own group vs. family)?
2. **Headline `max_det` — RESOLVED (2026-06-11):** use the COCO-standard
   `max_det=100` **everywhere** (model selection + reporting). Empirically
   verified the dual-cap idea is unnecessary: the busiest real test image has only
   28 GT boxes (synthetic: 4), 0 images exceed 30, so 100 already captures every
   possible TP and 300 only adds a low-confidence FP tail that barely moves AP.
   `constants.EVAL_MAX_DET` changed 300 → 100.
3. **Synthetic test set readiness — RESOLVED (2026-06-11).** The 225×50 set is
   generated (`scripts/synthetic/2-generate_test_images.py` →
   `data/synthetic/images/test/`, 11,250 images) and annotated in COCO format at
   `data/synthetic/annotations_test.json` (225 categories, 11,250 images, 11,264
   boxes). Verified: its category IDs are **identical** (id→name) to
   `data/real/annotations_test.json`, so the `mixed` union is label-safe. **Build
   note:** image IDs collide (both files start at `id=1`), so constructing the
   `mixed` set requires offsetting the synthetic `image_id`/annotation `id` past
   the real max before concatenating (see §10). The `mixed` default is therefore
   unblocked; no fallback to `real` needed.
4. **`mixed` default vs. the "test on real" constraint — RESOLVED (2026-06-10).**
   The CLAUDE.md "test on real" constraint has been changed: `mixed` is now the
   primary/default evaluation, with the **`real`-only breakout always reported
   alongside** it as the primary-evaluation figure + public-comparison anchor,
   and the model **never** judged on synthetic alone. Rationale and revision
   trigger are recorded in §3.1 below. CLAUDE.md has been updated to match.
5. **CIs:** is bootstrap-CI reporting worth the compute for every model, or only
   for the final shortlisted models?
6. **Per-class macro vs. count-weighted:** report both, or pick macro (COCO) as
   the single primary?
