# Dataset Split Correctness Review

**Reviewed:** `scripts/dataset_quality/12-assign_dataset_splits.py`  
**Outputs reviewed:** `reports/dataset_split_report.md`, `reports/dataset_split_summary.json`, `reports/dataset_split_manifest.json`  
**Plan reviewed against:** `docs/plans/2026-05-25_dataset-split-real-image-selection.md`  
**Date:** 2026-05-25

---

## Summary

| # | Severity | Finding |
|---|----------|---------|
| 1 | **CRITICAL** | Band A test set contains 99 hard-excluded (Q=0) images with invalid bboxes |
| 2 | **MINOR BUG** | `write_summary` double-counts hard-excluded images in `pool_actual` for Band A |
| 3 | **MAJOR** | Band assignment uses stale CSV pool estimates; 47/103 non-D classes are in the wrong (lower) band |
| 4 | Observation | Band D val Q mean (0.905) > all-active Q mean (0.856) — not a bug, explained below |
| 5 | Observation | Band B/C test sets are quality-biased low — per-spec but undocumented in thesis |

---

## Finding 1 — CRITICAL: Band A test set contains 99 hard-excluded images

### What the plan says (§2.3)

> "These hard exclusions are applied **before any split assignment.**"

### What the code does

`_assign_band_a` (lines 370–372) returns `images[:]` without filtering:

```python
def _assign_band_a(images: list[dict]) -> dict:
    # All passed images → test; no quality filtering for Band A (data is too scarce)
    return {"train": [], "val": [], "test": images[:], "surplus": []}
```

`images` is the raw output of `load_all_passed_images`, which includes images with `hard_excluded: True` (Q = 0). Every other band allocator filters first:

```python
active = [img for img in images if not img["score_components"]["hard_excluded"]]
```

Band A never does this.

### Evidence from manifest

```
Band A test images total:              5,808
Band A test images with hard_excluded: 99
```

Examples from the manifest:

| class | area_frac | min_margin | reason |
|-------|-----------|------------|--------|
| aardvark | 0.0583 | −0.0 | bbox clips image boundary |
| aardwolf | 0.7362 | −0.0 | bbox clips image boundary |
| african civet | 0.8211 | −0.0 | bbox clips image boundary |

These images have bboxes that extend outside the image boundary — the annotation is geometrically wrong. The `yolo_to_coco` converter silently clamps the invalid coordinates to the image edge instead of rejecting the annotation. The resulting COCO bboxes are truncated and do not reflect the true detection.

### Classes affected

All 51 Band A classes with any hard-excluded images are affected. The classes with the most contaminated test images:

| class | hard_excluded in test |
|-------|-----------------------|
| domestic water buffalo | 17 |
| eurasian badger | 8 |
| leopardus species | 12 |
| raccoon dog | 5 |
| sun bear | 5 |

### Fix

```python
def _assign_band_a(images: list[dict]) -> dict:
    active = [img for img in images if not img["score_components"]["hard_excluded"]]
    return {"train": [], "val": [], "test": active[:], "surplus": []}
```

This brings Band A in line with the identical pattern used in `_assign_band_b`, `_assign_band_c`, and `_assign_band_d`.

After the fix, Band A test count drops from 5,808 → **5,709** (−99 images with invalid annotations removed).

---

## Finding 2 — MINOR BUG: `pool_actual` double-counts hard-excluded images for Band A

### Location

`write_summary`, line 611:

```python
pool_actual = len(train) + len(val) + len(test) + len(surplus) + hard_excl
```

### Problem

For Band A:
- `test` **includes** the hard-excluded images (due to Finding 1)
- `hard_excl` **also counts** those same images (they are in `all_imgs` and `hard_excluded=True`)

Result: each hard-excluded image is counted twice in `pool_actual`.

Example — `domestic water buffalo`:
- `pool` shown = 711 = test(694) + hard_excl(17)
- Unique images actually in the dataset = 694 (not 711)
- The 17 hard-excluded images are in both `test=694` and in `hard_excl=17`

Total double-counted images across all Band A classes: **99** (exactly the number of hard-excluded images).

This does not affect the COCO annotation files, only the summary JSON and report counts.

### Fix

Automatic: once Finding 1 is fixed and hard-excluded images are removed from Band A test, the formula `pool_actual = test + hard_excl` becomes correct again (hard-excluded images are no longer in `test`).

---

## Finding 3 — MAJOR: Band assignment uses stale CSV estimates; 47 classes are in the wrong band

### Root cause

`load_class_bands` assigns bands from the `effective_pool` column in `class_distribution_reviewed.csv`. This CSV was produced at an earlier stage before quality filtering was complete. The actual number of quality-passed images is substantially higher than the CSV estimates for a large fraction of classes.

### Scale of the problem

After computing `actual_active = actual_pool − hard_excluded` from the summary JSON:

| Band assigned by CSV | Classes in band | Classes whose actual active pool warrants a **higher** band |
|---|---|---|
| A (csv_pool < 150) | 51 | **12 classes (24%)** |
| B (csv_pool 150–249) | 26 | **14 classes (54%)** |
| C (csv_pool 250–399) | 26 | **21 classes (81%)** |

**47 out of 103 non-D classes (46%) are assigned to a lower band than their actual data supports.**

### Impact on training data — worst cases

**Band A classes that should be Band D (get 0 real training images instead of up to 1,500):**

| class | csv_pool | actual_active | correct band | real train allocated | should have | lost |
|-------|----------|--------------|-------------|---------------------|-------------|------|
| domestic water buffalo | 81 | 677 | D | 0 | 1,500 | **1,500** |
| eurasian badger | 141 | 452 | D | 0 | 1,500 | **1,500** |
| sun bear | 78 | 263 | C | 0 | 170 | 170 |
| leopardus species | 132 | 259 | C | 0 | 170 | 170 |
| nine-banded armadillo | 145 | 237 | B | 0 | 85 | 85 |
| genet genus | 132 | 223 | B | 0 | 85 | 85 |

**Band B classes that should be Band D (get 85 real training images instead of 1,500):**

| class | csv_pool | actual_active | correct band | real train allocated | should have | lost |
|-------|----------|--------------|-------------|---------------------|-------------|------|
| asian elephant | 224 | 1,509 | D | 85 | 1,500 | **1,415** |
| common wombat | 241 | 1,353 | D | 85 | 1,500 | **1,415** |
| springbok | 247 | 755 | D | 85 | 1,500 | **1,415** |
| african wild dog | 229 | 885 | D | 85 | 1,500 | **1,415** |
| american mink | 176 | 820 | D | 85 | 1,500 | **1,415** |
| eurasian otter | 247 | 429 | D | 85 | 1,500 | **1,415** |
| giant panda | 169 | 408 | D | 85 | 1,500 | **1,415** |

**Band C classes that should be Band D (get 170 real training images instead of 1,500):**

| class | csv_pool | actual_active | correct band | real train allocated | should have | lost |
|-------|----------|--------------|-------------|---------------------|-------------|------|
| cercopithecus species | 337 | 1,420 | D | 170 | 1,500 | 1,330 |
| domestic donkey | 398 | 1,397 | D | 170 | 1,500 | 1,330 |
| northern chamois | 315 | 1,269 | D | 170 | 1,500 | 1,330 |
| muntjac genus | 250 | 1,049 | D | 170 | 1,500 | 1,330 |
| chital | 360 | 1,169 | D | 170 | 1,500 | 1,330 |
| bornean orangutan | 277 | 1,090 | D | 170 | 1,500 | 1,330 |
| snow leopard | 363 | 755 | D | 170 | 1,500 | 1,330 |
| grey wolf | 348 | 782 | D | 170 | 1,500 | 1,330 |
| sambar | 342 | 851 | D | 170 | 1,500 | 1,330 |
| red kangaroo | 342 | 678 | D | 170 | 1,500 | 1,330 |
| hartebeest | 386 | 877 | D | 170 | 1,500 | 1,330 |
| … (10 more Band C→D) | | | | | | |

Estimated total real training images left allocated to surplus/test instead of train due to wrong band: **~25,000+**

### Practical consequence

- The 12 Band A classes that should be in higher bands receive **zero real training images** — their training relies entirely on 200 synthetic images/class, despite having 155–677 quality-passed real images available.
- For classes like `domestic water buffalo` (677 real, Band A assigned): synthetic-only training when real data is plentiful is the inverse of what the thesis pipeline intends.

### Fix

After `load_all_passed_images`, recompute the effective band from actual active image counts instead of (or in addition to) the CSV:

```python
# After computing per_class_images, override band where actual pool differs
for cls, info in band_info.items():
    actual_active = sum(
        1 for img in per_class_images.get(cls, [])
        if not img["score_components"]["hard_excluded"]
    )
    actual_band = (
        "A" if actual_active < 150 else
        "B" if actual_active < 250 else
        "C" if actual_active < 400 else "D"
    )
    if actual_band != info["band"]:
        # log the promotion
        info["band"] = actual_band
        info["effective_pool"] = actual_active
```

Alternatively, regenerate `class_distribution_reviewed.csv` from actual passed image counts — a single query over the `filter_results.jsonl` files produces accurate per-class counts.

> **Note:** Promoting classes upward changes train/val/test allocations and will produce different COCO files. All downstream training runs must be re-executed against the updated splits.

---

## Finding 4 — Observation: Band D val Q mean (0.905) > all-active Q mean (0.856)

The split report's "Val Set Representativeness" section shows Band D val mean Q (+0.049 above all-active). This is **not a bug**; it is explained by two effects:

1. **Right-skewed Q distribution.** Band D aggregate (manifest): mean=0.892, p25=0.850, p50=0.960, p75=0.984. The distribution has a heavy lower tail of lower-quality images. The 30th–70th percentile window therefore sits above the mean.

2. **The report's "all-active mean" includes 231,107 surplus images.** The manifest-only mean is 0.892. The report's 0.856 pulls in the large surplus (the bottom of the Q-ranked pool, never selected for any split). Comparing val (selected from within-class 30th–70th percentile) to surplus-inclusive all-active (0.856) inflates the delta artificially.

**Caveat:** The aggregate "66.3% of Band D val images fall within aggregate [p30, p70]" is an unreliable summary — per-class percentile windows vary, so the aggregate range does not directly apply. The per-class sampling is operating correctly.

---

## Finding 5 — Observation: Band B/C test sets are quality-biased low

For Bands B and C, the test set is defined as whatever images remain after the top-Q images are taken for training:

```python
test = [img for img in active if img["filepath"] not in train_fps and img["filepath"] not in val_fps]
```

Unlike Band D (which draws a stratified *random* test sample before train is selected), Bands B/C assign test last. The practical result:

- Band B/C test = bottom ~50–65% of the Q distribution
- Band D test = random draw from the full Q distribution

This means per-class test-set difficulty is not uniform across bands. Band B/C classes are evaluated on their hardest images; Band D classes are evaluated on a representative random sample. This is **consistent with the plan** ("remaining real (pool − 85 − 20)") but is not called out anywhere in the documentation.

**Recommendation:** Add a note to the thesis methodology section acknowledging that B/C test images have lower average Q than D test images, and that cross-band performance comparisons should account for this difference in test-set quality composition.

---

## Verification Steps

After applying the fixes for Findings 1 and 3, confirm:

1. **No hard-excluded images in any split:**
   ```bash
   python3 -c "
   import json
   with open('reports/dataset_split_manifest.json') as f:
       m = json.load(f)
   bad = [r for r in m['splits'] if r['score_components'].get('hard_excluded')]
   print('Hard-excluded in splits:', len(bad))  # expected: 0
   "
   ```

2. **No split overlap** — the existing `verify_no_overlap` in the script handles this.

3. **COCO bbox validity:**
   ```bash
   python3 -c "
   import json
   for split in ('train', 'val', 'test'):
       d = json.load(open(f'data/real/annotations_{split}.json'))
       bad = [a for a in d['annotations'] if a['bbox'][2] <= 0 or a['bbox'][3] <= 0]
       print(f'{split}: {len(d[\"images\"])} images, {len(d[\"annotations\"])} anns, {len(bad)} zero-dim bboxes')
   "
   ```

4. **Band promotion spot-check:**
   ```bash
   python3 -c "
   import json
   s = json.load(open('reports/dataset_split_summary.json'))
   # domestic water buffalo should now be Band D with ~677 train images
   v = s['domestic water buffalo']
   print('domestic water buffalo:', v['band'], 'train=', v['train'])
   # asian elephant should now be Band D with ~1500 train images
   v = s['asian elephant']
   print('asian elephant:', v['band'], 'train=', v['train'])
   "
   ```

5. **Pool integrity per class** (no double-counting):
   ```bash
   python3 -c "
   import json
   s = json.load(open('reports/dataset_split_summary.json'))
   for cls, v in s.items():
       expected = v['train'] + v['val'] + v['test'] + v['surplus'] + v['hard_excluded']
       if expected != v['pool']:
           print(f'MISMATCH {cls}: pool={v[\"pool\"]} expected={expected}')
   print('Done')
   "
   ```
   Should print only "Done" with no mismatches.
