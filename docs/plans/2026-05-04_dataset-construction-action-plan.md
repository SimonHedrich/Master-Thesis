# Dataset Construction — Action Plan

**Date:** 2026-05-04
**Companion to:** `docs/plans/2026-05-04_dataset-construction-strategy.md`
**Inputs:** `docs/images-per-class-analysis.md` · `reports/speciesnet_filter.md`

This document does **not** restate the strategy. Read the strategy first, then use this as the practical roadmap: what to analyse, what to decide, and in what order.

---

## 1. Evaluation of the Strategy Draft

The trust-level split and the four-tier framework are sound. The analysis document (`images-per-class-analysis.md`) broadly confirms the tier boundaries, adds empirically grounded thresholds per training track, and surfaces a class-count recommendation that the strategy does not yet address. The tensions are reconcilable rather than contradictory.

### Confirmed

- **100 images is the right hard minimum for real-data-supported classes.** The analysis independently arrives at the same value from QAT activation-statistics requirements and anchor-box regression behaviour on nano models.
- **All 225 classes are included in both tracks.** Tier 1 classes (<100 real images) participate in Track A and Track B; their training signal comes entirely from synthetic images, and the small real-image pool is held out for evaluation. The analysis doc's suggestion to "exclude <100 from Track A/B" is an optimisation recommendation for the comparison design, not a pipeline constraint — the strategy intentionally retains all classes with synthetic supplementation.
- **Trusted-source manual review for Tiers 1–2 is appropriate.** Below 500 images the teacher is not reliable enough to act as a label arbiter; human review is the only safe path.

### Gaps and adjustments

| Topic | Strategy | Analysis finding | Action |
|---|---|---|---|
| Track-specific thresholds | Not distinguished | KD student: 300 post-filter; KD teacher: 500 post-filter; Track B: 1,000 post-filter | Useful as quality benchmarks when reading results; not a gating criterion for training inclusion |
| Raw-to-post-filter multiplier | Implicit ≈20% buffer | Empirically 65.9% failure → effective multiplier is **3.0×** | Revise any planning numbers that assumed a smaller buffer |
| Copy-paste augmentation | Not mentioned | 2–3× effective variance multiplier; does not substitute for the 100-image hard floor | Add to Tier 2 assembly pipeline |
| QAT implications of sparse data | Not mentioned | <100 diverse images → activation statistics narrow for INT8 calibration | Noted as a likely finding from the Tier 1 evaluation, not a blocker |

---

## 2. Track Threshold Overlay on the Four Tiers

The strategy's tiers are defined by post-filter image counts. The analysis adds track-level viability on top of those count bands.

All classes from all tiers are included in both Track A and Track B. The table below describes expected model quality and what to watch for per tier — not whether a class is included.

| Tier | Effective real pool | Training signal | Expected model quality | Key risk |
|---|---|---|---|---|
| **1** | <100 real images | Synthetic only (real → eval) | Low; useful for studying synth-only ceiling | Activation statistics narrow for INT8 QAT |
| **2a** | 100–299 | Real + synthetic supplement | Marginal; KD soft labels help vs hard labels | Teacher quality uncertain at this count |
| **2b** | 300–499 | Real + possible small supplement | Acceptable with augmentation | Teacher reaches reliable threshold near 500 |
| **3** | 500–1,499 | Real; capped subset for comparison | Good; primary KD vs direct fine-tune comparison zone | Verify cap is applied consistently |
| **4** | ≥1,500 | Real, capped at 1,500 | Strong; Track B also viable | Ensure surplus goes to eval set, not training |

The existing tier boundaries (100 / 500 / 1,500) do not need to change. A soft internal split of Tier 2 at 300 is useful for reading results but does not require a different pipeline.

---

## 3. What Is Missing from `reports/speciesnet_filter.md`

The current report runs Script 7 (`7-filter_speciesnet.py`) in stats-only mode and counts **SpeciesNet pass/fail across all sources combined**. This is the wrong counting unit for the strategy.

### Why the current numbers understate the effective pool

The strategy distinguishes:
- **Trusted sources** (iNaturalist, GBIF, Wikimedia): SpeciesNet confidence is used as a *flag for manual review*, not a discard gate. An iNaturalist image that fails SpeciesNet is still available for training after review.
- **Unverified sources** (ImageCV, OpenImages): SpeciesNet pass is required before the image can be used.

The report collapses these into one "pass" column. The ~65.9% failure rate includes a large number of trusted-source images that the strategy would retain (subject to review). The **effective training pool is larger** than 158,667.

### What is needed: trust-aware per-class counts

For each class, three numbers are required before tier assignment is possible:

```
trusted_available(class)   = images from iNat + GBIF + Wikimedia that passed quality filter
                             (stages 1–5: metadata, heuristics, MegaDetector, caption VLM, LLM eval)
                             regardless of SpeciesNet result

unverified_available(class) = images from ImageCV + OpenImages that passed SpeciesNet threshold

effective_pool(class)       = trusted_available(class) + unverified_available(class)
```

`effective_pool` is the number that maps a class to its tier.

The current report also does not show the breakdown of trusted-source images by their SpeciesNet match level. Knowing how many trusted-source images have `match_level = species/genus` vs `order/class/no_match` is important for prioritising manual review workload: a class whose trusted images mostly fail with `match_level_no_match` needs heavier review than one failing with `low_speciesnet_confidence`.

### How to produce these numbers

Extend Script 7 or write a small analysis script that reads `speciesnet_results.jsonl` for each source and `filter_results.jsonl` (for quality-filter pass/fail) and outputs per-class:

| Column | Definition |
|---|---|
| `class` | Common name |
| `trusted_quality_pass` | Trusted-source images passing quality stages (stages 1–5 in `filter_results.jsonl`) |
| `trusted_sn_pass` | Subset of above that also pass SpeciesNet |
| `trusted_sn_fail_reason` | Most common SpeciesNet fail reason among trusted-source failures |
| `unverified_sn_pass` | Unverified-source images passing SpeciesNet |
| `effective_pool` | `trusted_quality_pass + unverified_sn_pass` |
| `tier` | Assigned tier based on `effective_pool` |

This is a read-only analytics script (no writes to `filter_results.jsonl`). It can be run before Script 7 `--write` is committed.

---

## 4. Prioritised Next Steps

Steps are ordered by dependency. Steps 1–3 must complete before any tier assignment or downstream decision.

### Step 1 — Produce the trust-aware class distribution

Write `scripts/dataset_quality/8-class_distribution_report.py` (or extend Script 7 with a `--trust-split` flag). Output: a CSV with the columns from §3 above, plus a `tier` column.

Run against all five sources. The result replaces the current filter report as the authoritative input for all downstream decisions.

**Blockers:** None — all input files (`speciesnet_results.jsonl`, `filter_results.jsonl`) already exist.

### Step 2 — Assign classes to tiers and flag manual-review queue

With the CSV from Step 1:
1. Apply the tier thresholds (100 / 500 / 1,500 on `effective_pool`).
2. Identify classes in Tiers 1–2 where `trusted_quality_pass > 0` — these go to the **manual review queue**.
3. Identify classes with `effective_pool = 0` — these will be Tier 1 with no real training images at all; synthetic generation is the only option and real images for evaluation will need to be sourced separately or noted as absent.
4. Identify classes with `trusted_sn_fail_reason` dominated by `match_level_no_match` — flag for deeper review (possible label error at source).

### Step 3 — Manual review of Tier 1 and Tier 2 images

For each class in Tiers 1–2, review the trusted-source images that passed quality filtering. Confirm labels are correct. Images confirmed as correct can be used in training; images that fail review are discarded.

Scope estimate: sum of `trusted_quality_pass` across all Tier 1 and Tier 2 classes. From the current filter report, many sparse classes have <100 pre-filter images total, so the manual review queue should be manageable.

### Step 4 — Decide per-class synthetic counts for Tiers 1–2 (see Q1 in §5)

With the confirmed real-image counts from Step 3, compute how many synthetic images are needed per class and verify the total stays within the 10,000-image / €500 budget.

Constraint from the analysis: Copy-paste augmentation (Step 6) acts as a 2–3× multiplier, so the synthetic generation target can be lower than the raw deficit would suggest.

### Step 5 — Integrate copy-paste augmentation into the Tier 2/3 assembly pipeline

Implement copy-paste augmentation for Tiers 2 and 3:
- Extract animal crops using the MegaDetector bboxes already in `filter_results.jsonl`.
- Paste onto background images from the same-source pool (camera-trap backgrounds).
- Apply during training as an online augmentation step, not as static pre-generated images.

This is a training-pipeline task, not a data-collection task, but it must be accounted for before finalising the real-image targets.

### Step 6 — Write Script 7 `--write` and assemble per-class YOLO datasets

After all decisions are made:
1. Run `python scripts/dataset_quality/7-filter_speciesnet.py --source all --write` to commit the SpeciesNet eval block into `filter_results.jsonl`.
2. Write the YOLO dataset assembly script that respects trust levels (keep all trusted-quality-pass images for trusted sources; keep only SpeciesNet-pass images for unverified sources).
3. Apply per-tier caps (Tier 4: cap at 1,500; Tier 3 comparison set: cap at the Tier 2 maximum for cross-tier comparability).

---

## 5. Open Questions

### Q1 — Per-class synthetic image count for Tiers 1 and 2

**Background:** The strategy caps budget at ~a few hundred per class, but the exact number is unresolved. The answer depends on how many Tier 1/2 classes exist (from Step 2) and what the real-image count per class is after manual review.

**Framing:** With copy-paste augmentation providing a 2–3× multiplier, the synthetic target for a Tier 2 class with 150 real images might be as low as 50–100 synthetic images to bring effective variance toward the 300–500 image range. For a Tier 1 class, synthetic images are the only training signal — a larger number is needed, but diminishing returns kick in quickly.

**Suggested approach:** Set a fixed synthetic count per tier rather than per class (e.g., 200 images/class for Tier 1, 100 images/class for Tier 2). Verify total stays within budget after Step 2.

### Q2 — Tier 3 training cap for real-vs-synthetic comparison

**Background:** The strategy says the Tier 3 training set "may be capped to the same image count used in Tiers 1 and 2" for cross-tier comparability. The exact cap cannot be set until Q1 is resolved (the Tier 2 total = real + synthetic is the reference).

**Dependency:** Resolve Q1 first, then set the Tier 3 cap to the same total.

### Q3 — Trusted-source SpeciesNet failures: which warrant manual review?

**Background:** The strategy flags trusted-source SpeciesNet failures for manual review rather than automatic discard. But not all failure reasons carry the same weight:

| Fail reason | Interpretation | Review priority |
|---|---|---|
| `low_speciesnet_confidence` | Classifier uncertain, label may still be correct | Low: likely fine |
| `family_mismatch_high_confidence` | Classifier is confident it sees a *different genus* | High: likely label error or wrong image |
| `match_level_order` / `match_level_class` | Very coarse match | Medium: possible label error |
| `match_level_no_match` | Classifier sees nothing in the 225-class universe | High for iNat; could be unusual species representation |

**Decision:** Should manual review cover all trusted-source failures or only high-priority ones? Limiting to `family_mismatch_high_confidence` and `match_level_no_match` may be more practical if the queue is large.

### Q4 — OpenImages bbox handling

**Background:** Script 6 produced 7,405 zero-detection records for OpenImages because the OpenImages bboxes were stored differently (not in `filter_results.jsonl["detections"]`). Script 7 will mark all of these as `no_animal_detection` failures. OpenImages is an unverified source, so these images cannot be used unless MegaDetector is re-run with the pre-annotated bboxes injected into the detections field.

**Decision:** Is it worth patching the pipeline to recover OpenImages images, or is the source small enough (7,499 records, ~1.6% of total) to simply exclude?
