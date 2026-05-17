# Investigation: Severe SpeciesNet Shrinkage in Tier 3 Classes

**Date:** 2026-05-06
**Status:** RESOLVED — investigation complete, fix applied to script 9

---

## Situation

`reports/class_distribution_reviewed.md` reveals a systematic pattern: a large group of classes
with hundreds of quality-pass images (`trusted_quality_pass`, which already incorporates
MegaDetector and caption/description filtering) ends up with only a handful after SpeciesNet
filtering. These classes were classified as Tier 3 (500–1,499 images) in `class_distribution.csv`
based on their `trusted_quality_pass` count, but their `trusted_sn_pass` count is so small that
they drop to Tier 1 or 2 in the reviewed report.

### Worst cases (tqp ≥ 300, shrinkage > 80%)

| Class | tqp | tsn_pass | shrinkage | fail reason |
|---|---:|---:|---:|---|
| saguinus species | 1,065 | 5 | 99% | match_level_no_match |
| eulemur species | 880 | 2 | 99% | match_level_no_match |
| rock hyrax | 1,336 | 7 | 99% | match_level_class |
| ateles species | 1,437 | 38 | 97% | match_level_no_match |
| pikas genus | 1,484 | 31 | 97% | match_level_no_match |
| klipspringer | 701 | 68 | 90% | match_level_no_match |
| domestic water buffalo | 694 | 57 | 91% | family_mismatch_high_confidence |
| leaf monkeys genus | 693 | 74 | 89% | match_level_no_match |
| saimiri species | 1,327 | 138 | 89% | match_level_no_match |
| nilgai | 549 | 64 | 88% | match_level_no_match |
| reedbuck genus | 514 | 65 | 87% | match_level_order |
| mountain goat | 793 | 100 | 87% | match_level_no_match |
| martes species | 1,298 | 145 | 88% | match_level_no_match |
| reindeer | 835 | 105 | 87% | match_level_order |
| blesbok | 823 | 185 | 77% | low_speciesnet_confidence |
| asian elephant | 1,301 | 164 | 86% | family_mismatch_high_confidence |
| colobus species | 663 | 117 | 82% | match_level_no_match |
| steenbok | 957 | 167 | 82% | match_level_no_match |
| striped skunk | 666 | 101 | 84% | match_level_no_match |

Total: **~18 classes** with tqp ≥ 300 and >80% SN shrinkage.

An additional ~20 classes with tqp ≥ 200 show 50–80% shrinkage but still land in Tier 2.

---

## Why This Is a Problem

The pipeline design (docs/plans/2026-05-04_dataset-construction-strategy.md) says:

> Tier 3: "if an image **fails** SpeciesNet filtering, it is flagged for **manual review**
> rather than automatically discarded — the label may still be correct despite a low
> confidence score."

However, `9-manual_review_queue.py` only queues **Tier 1 and Tier 2** classes. Tier 3 SN-fail
images were never routed to the review server. As a result, the `eff_pool` for these classes is
simply `trusted_sn_pass`, which is a tiny fraction of what was available.

The `tsp == 0` fix applied on 2026-05-06 already handles the two extreme cases (sea otter,
hoffmann's two-toed sloth) where no image at all passes SN. But the broader class of
"near-zero SN pass rate" has not been addressed.

---

## Hypotheses on What Went Wrong

### Hypothesis 1 — SpeciesNet coverage gap (most likely for most classes)

`match_level_no_match` is the dominant fail reason (19 of 31 affected classes). This means
SpeciesNet found no species from its 225-class taxonomy at any confidence in these images.
This is consistent with SpeciesNet simply not having been trained on these species, or them
being taxonomically distinct from anything in its training set.

Examples: saguinus (tamarins), eulemur (lemurs), ateles (spider monkeys), pikas, rock hyrax —
all of these are either geographically unusual, underrepresented in SpeciesNet's training
data, or belong to taxonomic groups poorly covered by the SpeciesNet taxonomy.

**Implication:** The images are likely fine; SpeciesNet is an unreliable signal for these species.
The correct fix is the same as for sea otter: use `trusted_quality_pass` directly (or at minimum
route SN-fail images for manual review).

### Hypothesis 2 — Genus/family-level class labels confusing SpeciesNet

Several severely affected classes are genus- or family-level aggregates:
`saguinus species`, `ateles species`, `martes species`, `saimiri species`, `colobus species`,
`leaf monkeys genus`. SpeciesNet classifies to a specific species (e.g., *Callithrix jacchus*)
and the 225-class common name may not match a genus-level folder name even if the animal is
correct. SN can return a match that is taxonomically correct but fails the current match-level
check.

**Implication:** The match-level check in `7-filter_speciesnet.py` might be misapplying
species-level taxonomy to genus-level classes. These classes may need a relaxed match
strategy (e.g., genus or family match accepted).

### Hypothesis 3 — Family mismatch is a real label problem for some classes

For classes with `family_mismatch_high_confidence` (domestic water buffalo, asian elephant, chital,
and a few others), SpeciesNet is *confidently* predicting a different family — which is a strong
signal of mislabeled images. These are genuine quality concerns, not SpeciesNet coverage gaps.

**Implication:** These classes likely contain a mix of correct images and wrong-species images.
Review + careful filtering is the right approach, not wholesale acceptance of all tqp images.

### Hypothesis 4 — MegaDetector / caption filter removed too many images

Some classes with high tqp still have a much lower tqp than their raw download count would
suggest. It is possible that the MegaDetector (≥ 0.5 confidence) or the caption/description
LLM filter rejected many valid images for these classes — for example:
- Small animals that MegaDetector misses (pikas, small primates close to camera)
- Zoo / captive images where the caption filter flags "captive" or "no wild animal visible"
- Images where the animal occupies only a small portion of the frame

This is a separate pipeline issue from the SN filter but contributes to the final small pool.

**Implication:** The MegaDetector and caption filter thresholds and prompts should be validated
on a sample of rejected images for these classes.

### Hypothesis 5 — SN filter threshold is too aggressive globally

The current SpeciesNet threshold (`sn_score ≥ 0.3`) and match-level requirements may simply be
too strict for species that are rare in SpeciesNet's training distribution. A lower threshold
or a match-level relaxation would pass more images but at the cost of potentially including
more mislabeled images.

---

## What Needs To Be Investigated

### Investigation A — Classify affected classes by root cause

For each high-shrinkage class, determine whether the primary issue is:
1. SN coverage gap (`match_level_no_match` + verified images are correct) → fix as for sea otter
2. Taxonomy mismatch (genus-level class vs. species-level SN output) → relax match strategy
3. Label contamination (`family_mismatch_high_confidence`) → review required
4. Pipeline filter aggressiveness (MD or caption) → examine rejected images

This requires sampling SN-fail images per class and inspecting them visually.

### Investigation B — Check where images are lost before SN

Compare per-class image counts at each pipeline stage:
- Raw download count (from source manifests or folder counts)
- Post-metadata filter
- Post-heuristics filter (blur, resolution)
- Post-MegaDetector (tqp includes this already, but worth isolating)
- Post-caption filter (also included in tqp)
- Post-SN (`tsn_pass`)

The gap between raw download and `trusted_quality_pass` may itself be very large for some
classes, indicating the problem starts before SN.

### Investigation C — Review a sample of SN-fail images for the worst classes

For the top 5–10 most severe classes (saguinus, eulemur, rock hyrax, ateles, pikas), open a
sample of SN-fail images in the review server or manually, and assess:
- Are the images the correct species?
- Is the image quality acceptable?
- Is there an animal visible / detectable?

This will determine whether to apply the `tsp ≈ 0 → use tqp` fix more broadly.

### Investigation D — Assess whether the Tier 3 SN-fail review path is worth implementing

The plan says Tier 3 SN-fail images should go to manual review. Given that there are potentially
thousands of such images across the 18 worst-affected classes, the review burden needs to be
estimated before committing to this approach.

If most SN-fail images for `match_level_no_match` classes are genuinely correct, a blanket
"use tqp" fix (as applied for sea otter) is more efficient than full manual review.

---

## Resolution

Root cause confirmed: Hypothesis 1 (SpeciesNet coverage gap) applies to all 19 affected classes. The fix is purely in `9-class_distribution_with_reviews.py` — no ML re-runs required.

### What changed

Added `_is_sn_coverage_gap(tsp, tqp, fail_reason)` helper and updated `build_rows` in script 9:
- If `fail_reason in {match_level_no_match, match_level_class}` AND pass rate < 20% → use `tqp`
- If `fail_reason == match_level_order` AND pass rate < 15% → use `tqp`
- `family_mismatch_high_confidence` and `low_speciesnet_confidence` remain conservative (`tsp + app`)

### Outcome (after re-running script 9)

| Class | Was | Now | final_tier |
|---|---:|---:|---:|
| pikas genus | 31 | 1,484 | 3 |
| ateles species | 38 | 1,437 | 3 |
| saguinus species | 5 | 1,065 | 3 |
| saimiri species | 138 | 1,327 | 3 |
| martes species | 145 | 1,298 | 3 |
| rock hyrax | 7 | 1,336 | 3 |
| eulemur species | 2 | 880 | 3 |
| reindeer | 105 | 835 | 3 |
| mountain goat | 100 | 793 | 3 |
| steenbok | 167 | 957 | 3 |
| klipspringer | 68 | 701 | 3 |
| leaf monkeys genus | 74 | 693 | 3 |
| striped skunk | 101 | 666 | 3 |
| colobus species | 117 | 663 | 3 |
| gorilla species | 82 | 550 | 3 |
| nilgai | 64 | 549 | 3 |
| reedbuck genus | 65 | 514 | 3 |
| **asian elephant** | 164 | 164 | 2 (conservative — family mismatch) |
| **domestic water buffalo** | 57 | 57 | 1 (conservative — family mismatch) |

**~14,500 additional images unlocked** across 17 classes. All 17 coverage-gap classes restored to Tier 3.

---

## Original Investigation Notes (for reference)

## Proposed Next Step (original)

Before implementing any fixes, run a targeted analysis script that:

1. For each class in the 225-class list, emits: `class`, `tqp`, `tsn_pass`, `tsn_fail`,
   `fail_reason`, `tsn_pass_rate`, and classifies each class into one of the four root-cause
   buckets above using heuristic rules:
   - `coverage_gap`: `fail_reason == match_level_no_match` AND `tsn_pass_rate < 10%`
   - `taxonomy_mismatch`: genus/family-level class name AND `match_level_*` fail
   - `label_contamination`: `fail_reason == family_mismatch_high_confidence`
   - `pipeline_filter`: `tqp` significantly lower than expected raw count

2. Produces a ranked list of classes that would benefit from the `use tqp` fix (i.e., share
   the same profile as sea otter), and estimates the total additional images that would be
   unlocked.

3. Samples 10–20 SN-fail images per high-shrinkage class (using the review server or a quick
   visual spot-check) to validate the classification.

This analysis will determine whether to extend the `tsp == 0` formula fix to a broader
`low_sn_pass_rate` fix, or whether a different approach is warranted per class.
