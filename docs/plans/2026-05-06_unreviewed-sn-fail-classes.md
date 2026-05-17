# Unreviewed Classes with Large SN-Fail Shrinkage

**Date:** 2026-05-06  
**Status:** Resolved — manual reviews completed 2026-05-07  
**Relates to:** `docs/plans/2026-05-06_sn-shrinkage-investigation.md` (coverage-gap fix, resolved)

---

## Situation

After the coverage-gap fix applied to script 9 (2026-05-06), 17 classes with `match_level_no_match`
/ `match_level_order` pass rates below threshold were correctly restored to Tier 3 by using
`trusted_quality_pass` directly.

However, two categories of classes were intentionally left conservative and not fixed:

- **`family_mismatch_high_confidence`** — SpeciesNet is confident the animal in the image belongs
  to a *different family* from the expected class. Wholesale acceptance would likely admit mislabeled
  images.
- **`low_speciesnet_confidence`** — SpeciesNet's top prediction is below the confidence threshold.
  Images may be correct but SN simply cannot confirm them.

These classes currently use `trusted_sn_pass` as their effective trusted count. For many of them,
`trusted_sn_pass` is a tiny fraction of `trusted_quality_pass`, and **no manual review has been
performed** to determine whether the SN-rejected images are valid.

The user also flagged two specific cases (`domestic water buffalo` tq=694→eff=57, `asian elephant`
tq=1301→eff=181) as unexpectedly low, triggering this broader audit.

---

## Problem

For these classes, the effective pool is based solely on SN-pass images. Without manual review of
the SN-fail images, there is no way to know whether:

1. The SN rejections are correct (images are genuinely mislabeled) → accept low count as ground
   truth.
2. SN is systematically wrong for this species (similar to coverage-gap classes) → the rejected
   images are usable and should be unlocked.
3. The pool is a mix of both → partial recovery via review.

The consequence is that several classes with hundreds of valid-looking `trusted_quality_pass` images
are sitting at Tier 1 or 2 with counts far too low for reliable training, even though those images
passed MegaDetector, caption filtering, and human-uploaded iNaturalist curation.

---

## Affected Classes

### `family_mismatch_high_confidence` — reviewed 2026-05-07

SpeciesNet predicts a *different family* with high confidence. Strongest signal of potential label
contamination, but also known to misfire on confusable species pairs
(e.g., Asian elephant vs. African elephant, water buffalo vs. African buffalo / cattle).

| Class | tq_pass | eff (before review) | eff (after review) | Band | Outcome |
|---|---:|---:|---:|---|---|
| domestic water buffalo | 694 | 57 | 81 | A | Marginal — still severely restricted |
| asian elephant | 1,301 | 181 | 207 | B | Marginal — still restricted |
| chital | 1,169 | 263 | 360 | C | Improved within band |
| sambar | 851 | 274 | 342 | C | Improved within band |
| sika deer | 1,136 | 354 | 471 | **D** | Crossed 400 threshold |
| grant's gazelle | 535 | 294 | 341 | C | Improved within band |

### `low_speciesnet_confidence` — reviewed 2026-05-07

SN prediction below threshold; could be correct images SN simply cannot verify, or could be
ambiguous / wrong-species images.

| Class | tq_pass | eff (before review) | eff (after review) | Band | Outcome |
|---|---:|---:|---:|---|---|
| blesbok | 823 | 185 | 439 | **D** | Crossed 400 threshold |
| nyala | 957 | 277 | 554 | **D** | Crossed 400 threshold |
| golden jackal | 781 | 304 | 551 | **D** | Crossed 400 threshold |
| gemsbok | 818 | 438 | 563 | D | Increased within band |

### High absolute loss, still no review (other fail reasons but large drop)

These classes use `trusted_sn_pass` as eff_trusted because they did not meet the coverage-gap
threshold (pass rate was not low enough). The rejected images have never been examined.

| Class | tq_pass | eff_pool | Retained | final_tier | Fail reason |
|---|---:|---:|---:|---:|---|
| cercopithecus species | 1,420 | 337 | 24% | 2 | match_level_no_match |
| northern chamois | 1,269 | 315 | 25% | 2 | match_level_no_match |
| waterbuck | 1,444 | 437 | 30% | 2 | match_level_order |
| swamp wallaby | 1,377 | 420 | 31% | 2 | match_level_no_match |
| alpine ibex | 1,294 | 418 | 32% | 2 | match_level_no_match |
| domestic donkey | 1,200 | 398 | 33% | 2 | match_level_class |
| muntjac genus | 1,049 | 250 | 24% | 2 | match_level_no_match |
| springbok | 759 | 247 | 33% | 2 | match_level_no_match |
| african wild dog | 907 | 229 | 25% | 2 | match_level_class |
| hartebeest | 877 | 386 | 44% | 2 | match_level_no_match |
| common eland | 962 | 385 | 40% | 2 | match_level_no_match |
| south american coati | 1,014 | 411 | 41% | 2 | match_level_no_match |
| common wombat | 614 | 241 | 39% | 2 | match_level_class |
| hippopotamus | 1,319 | 418 | 32% | 2 | match_level_no_match |
| grey fox | 1,116 | 425 | 38% | 2 | match_level_no_match |
| spotted hyaena | 1,233 | 640 | 52% | 3 | match_level_no_match |
| collared peccary | 1,419 | 521 | 37% | 3 | match_level_no_match |

Note: `cercopithecus species`, `muntjac genus`, `african wild dog`, `common wombat`, `domestic donkey`
failed the coverage-gap threshold (their pass rates were 20–40%, just above the <20% cutoff), so
the conservative path was applied. A lower threshold (e.g., <40%) would have restored them
automatically, but visual inspection is the safer approach given the sample sizes.

---

## Resolution (2026-05-07)

Manual review of SN-fail images was completed for all 10 priority classes. Key outcomes:

- **4 classes crossed the Band D threshold (≥ 400 eff pool):**
  - sika deer: 354 → 471 (Band C → D)
  - blesbok: 185 → 439 (Band B → D)
  - nyala: 277 → 554 (Band C → D)
  - golden jackal: 304 → 551 (Band C → D)
  These classes no longer need synthetic supplementation and are now treated as full Band D
  classes in `2026-05-06_dataset-caps-and-synthetic-counts.md`.

- **3 classes improved but remained in Band C:** chital (263→360), sambar (274→342),
  grant's gazelle (294→341).

- **2 classes still severely restricted by family_mismatch:**
  - domestic water buffalo: 57 → 81 (Band A, still requires 200 synthetic images)
  - asian elephant: 181 → 207 (Band B, still requires 100 synthetic + 100 real)
  The SN rejections for these two appear to be largely genuine mismatches; little further
  recovery expected from additional review.

- gemsbok increased from 438 → 563, remaining in Band D.

The `2026-05-06_dataset-caps-and-synthetic-counts.md` band tables and synthetic budget have
been updated to reflect these changes.

---

## Possible Solutions

### Option A — Manual review of SN-fail images (recommended for family_mismatch classes)

Route the `trusted_sn_fail` images for these classes into the existing batch review server
(`scripts/dataset_quality/11-batch_review_server.py`). Currently the review queue only covers
Tier 1/2 classes; it needs to be extended to accept SN-fail images from Tier 3 classes.

**Pros:** Ground-truth labels, recovers usable images, catches label noise.  
**Cons:** Review burden — the 10 highest-priority classes have ~6,000 rejected images combined.
At the current review speed, this is a significant time investment.

### Option B — Extend coverage-gap threshold to include near-threshold classes

Lower the coverage-gap pass-rate threshold from 20% → 35% for `match_level_no_match` and
`match_level_class` fail reasons. This would automatically restore cercopithecus species,
muntjac genus, african wild dog, common wombat, and several others.

**Pros:** Zero review effort, consistent with the logic already applied to coverage-gap classes.  
**Cons:** Less conservative — some mislabeled images will enter the pool. Only applicable to
`match_level_*` fails, not to `family_mismatch_high_confidence` or `low_speciesnet_confidence`
where SN's signal is stronger.

### Option C — Accept current conservative counts

Leave the pool as-is. These classes have enough images for Tier 2 training. Accept that SN
filtering was the correct decision and move on.

**Pros:** No work required.  
**Cons:** Several species that could reach Tier 3 are arbitrarily held at Tier 2 because SN
happens to perform poorly on them. For `family_mismatch` cases specifically, the SN fail reason
may be a model artifact, not a true labeling problem.

---

## Recommendation

Apply a two-track approach based on fail reason:

### Track 1 — `family_mismatch_high_confidence` → manual review (high priority)

These classes cannot be auto-fixed without evidence because SN's family-level rejection is a
genuine quality signal. However, the rejection rate is suspicious for well-photographed species
(>80% rejection of iNaturalist curator-approved images is implausible for Asian elephant or
water buffalo). Review will quickly reveal whether SN is systematically wrong.

**Queue for review (SN-fail images only):**

| Class | Images to review | Priority |
|---|---:|---|
| domestic water buffalo | 637 | critical (currently Tier 1) |
| asian elephant | ~1,100 | high |
| chital | ~900 | high |
| sambar | ~570 | high |
| sika deer | ~780 | high |
| blesbok | ~630 | high (low_confidence, same priority) |
| nyala | ~670 | high (low_confidence) |

Estimated total: ~5,300 images. If even 50% pass review, all seven classes move to Tier 3.

### Track 2 — `match_level_no_match` / `match_level_class` near-threshold → lower threshold

For classes where `match_level_no_match` or `match_level_class` is the fail reason and the pass
rate is between 20% and 40%, lower the coverage-gap threshold in script 9 from 20% → 40%.
This restores cercopithecus species, muntjac genus, african wild dog, common wombat, collared
peccary, and domestic donkey without requiring review.

If any of these turn out to be quality concerns, they can be manually flagged later; but
`match_level_no_match` at 25–40% pass rates is almost certainly a SpeciesNet coverage issue
rather than genuine label contamination.

### Deprioritise

Classes already at Tier 2 eff_pool ≥ 400 (grey fox, south american coati, hartebeest,
common eland, hippopotamus, gemsbok, alpine ibex, swamp wallaby, waterbuck) have enough images
for their current tier. Address only after the critical classes above are resolved.

---

## Implementation Notes

The batch review server (`scripts/dataset_quality/11-batch_review_server.py`) serves images from
the `trusted_quality_pass` pool. To review SN-fail images, the server needs a mode that serves
from `trusted_sn_fail` images for a given class. The review decisions are recorded in
`reports/review_decisions.jsonl` and are already picked up by script 9 via `review_approved` /
`review_declined` counts. No changes to the downstream pipeline are required once the images are
correctly routed through the review server.
