# Dataset Pipeline Fixes: Ghost Classes and Tier 3 All-SN-Fail Species

**Date:** 2026-05-06
**Status:** Planned — implementation pending

---

## Situation

`reports/class_distribution_reviewed.md` shows 9 classes with `eff_pool = 0`. These split into three groups:

| Group | Classes | Root cause |
|---|---|---|
| Ghost classes | 6 | Apostrophe-stripped folder names; images unreviewed |
| Tier 3 all-SN-fail | 2 | All trusted images fail SpeciesNet; never reviewed |
| Human pseudo-class | 1 | All images declined; expected — separate dataset used |

---

## Fix 1 — Ghost Classes

### Root cause

When images were downloaded, directory names had apostrophes stripped (OS/filesystem artifact). The script 8 path-to-class extractor reads the folder name literally, so these images appear under ghost class names in the distribution:

| Ghost name (in CSV/folder) | Canonical name | Trusted quality-pass images |
|---|---|---:|
| grevys zebra | grevy's zebra | 16 |
| kirks dik-dik | kirk's dik-dik | 11 |
| bairds tapir | baird's tapir | 9 |
| thomsons gazelle | thomson's gazelle | 9 |
| grants gazelle | grant's gazelle | 7 |
| hoffmanns two-toed sloth | hoffmann's two-toed sloth | 7 |

Total ghost images that passed all quality filters: **~59** across `data/wikimedia/` and `data/openimages/`.

### Were ghost images already reviewed?

**No.** The canonical folder images (e.g., `data/wikimedia/images/grevy's_zebra/`) were reviewed and decisions recorded in `review_decisions.jsonl`. The ghost folder images (e.g., `data/wikimedia/images/grevys_zebra/`) appear in `reports/class_distribution.csv` as separate zero-reviewed rows and have never been shown in the review server.

The canonical versions are in good shape:

| Canonical class | Pre-review tq_pass | rev_app | rev_dec | eff_pool |
|---|---:|---:|---:|---:|
| grevy's zebra | 209 | 188 | 21 | 188 |
| kirk's dik-dik | 215 | 200 | 15 | 200 |
| baird's tapir | 323 | 184 | 139 | 184 |
| thomson's gazelle | 452 | 417 | 35 | 417 |
| grant's gazelle | 528 | 0 | 0 | 294 (SN-pass, no review needed for Tier 3) |
| hoffmann's two-toed sloth | 667 | 0 | 0 | 0 (→ see Fix 2) |

### Code changes required

**`scripts/dataset_quality/8-class_distribution_report.py`**

In `process_source()`, after extracting the class name from the folder path, normalize apostrophes by resolving the ghost name to its canonical form. Add a mapping lookup or use the same `_strip_apostrophe` trick already in `9-manual_review_queue.py` — load `reports/classes_225.csv` once at startup and build a normalized→canonical dict; map every extracted class name through it before use.

**`scripts/dataset_quality/9-class_distribution_with_reviews.py`**

In `_class_from_filepath()`, apply the same normalization so that review decisions recorded under ghost folder paths (e.g., `grevys_zebra`) are attributed to the canonical class (`grevy's zebra`) when merging review counts.

### Review required

After the script 8 fix, re-running the report will fold ghost images into their canonical class counts. The ~59 ghost images will then appear in the review queue for their canonical classes (since those are Tier 2 classes where all trusted_quality_pass images must be reviewed). Run `9-manual_review_queue.py` after the script 8 fix to regenerate the queue, then review via `11-batch_review_server.py`.

---

## Fix 2 — Tier 3 All-SN-Fail Species

### Affected classes

| Class | Tier | trusted_quality_pass | tsn_pass | tsn_fail | fail reason |
|---|---|---:|---:|---:|---|
| sea otter | 3 | 1,404 | 0 | 1,404 | match_level_no_match |
| hoffmann's two-toed sloth | 3 | 667 | 0 | 667 | match_level_no_match |

Both species return `match_level_no_match` on every image — SpeciesNet cannot classify them at any taxonomic level. This is a SpeciesNet coverage gap, not a data quality issue.

### Decision

For these classes: **skip SpeciesNet results entirely and use all images that passed the megadetector and caption filter.** No additional manual review required.

`trusted_quality_pass` is exactly this set — it counts images from trusted sources that cleared stages 1–5:

1. Metadata pre-filter
2. Heuristics (resolution ≥ 256 px, blur, grayscale)
3. MegaDetector animal detection (confidence ≥ 0.5)
4. Florence-2 caption generation
5. LLM caption evaluation (rejects dead animals, non-daytime, no live animal visible)

Stage 6 (SpeciesNet) is what fails for these classes and is what we are choosing to ignore.

### Code change required

**`scripts/dataset_quality/9-class_distribution_with_reviews.py`**

In `build_rows()`, add a special case for Tier 3 classes where `trusted_sn_pass == 0`: treat them like Tier 4 and use `trusted_quality_pass` directly as `eff_trusted`.

```python
if tier in (1, 2):
    eff_trusted = app
elif tier == 3:
    if tsp == 0:
        # All trusted images failed SN (SpeciesNet coverage gap).
        # Use the full quality-pass set; SN result is not a useful signal.
        eff_trusted = tqp
    else:
        eff_trusted = tsp + app
else:  # tier 4
    eff_trusted = tqp
```

Also update the filtering-rules table in `write_md()` to document the new Tier 3 sub-case.

After this change: sea otter eff_pool = 1,404 (Tier 3), hoffmann's two-toed sloth eff_pool = 667 (Tier 3). Both move from final_tier 1 to final_tier 3.

Note: after Fix 1 lands, hoffmann's two-toed sloth will also absorb the 7 ghost images → eff_pool = 674, still Tier 3.

---

## Execution order

1. **Fix `8-class_distribution_report.py`** — normalize apostrophe-stripped class names to canonical form
2. **Fix `9-class_distribution_with_reviews.py`** — normalize `_class_from_filepath()` for review merging; add `tsp == 0` Tier 3 formula branch
3. **Re-run `8-class_distribution_report.py`** — regenerates `reports/class_distribution.csv` with ghost images merged into canonical classes
4. **Re-run `9-manual_review_queue.py`** — regenerates review queue; ~59 ghost images now appear under their canonical classes
5. **Run `11-batch_review_server.py`** — review the ~59 ghost images
6. **Re-run `9-class_distribution_with_reviews.py`** — produces updated `class_distribution_reviewed.md`

### Expected outcome

| Class | Before (eff_pool) | After (eff_pool) | final_tier |
|---|---:|---:|---:|
| sea otter | 0 | 1,404 | 3 |
| hoffmann's two-toed sloth | 0 | ~674 | 3 |
| grevy's zebra | 188 | ~196 (+ghost approvals) | 2 |
| kirk's dik-dik | 200 | ~209 (+ghost approvals) | 2 |
| baird's tapir | 184 | ~192 (+ghost approvals) | 2 |
| thomson's gazelle | 417 | ~424 (+ghost approvals) | 2 |
| grant's gazelle | 294 | ~299 (+ghost approvals) | 2 |
| human | 0 | 0 | 1 (ignored) |
