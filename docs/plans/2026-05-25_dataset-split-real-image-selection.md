# Dataset Split & Real Image Selection Plan

**Date:** 2026-05-25
**Status:** Implemented 2026-05-25
**Depends on:** docs/plans/2026-05-04_dataset-construction-strategy.md, docs/plans/2026-05-18_synthetic-labeling-pipeline.md

**Implemented by:** `scripts/dataset_quality/12-assign_dataset_splits.py`

**Outputs produced:**
- `reports/dataset_split_manifest.json` — 220,049 image records (train+val+test, all bands)
- `reports/dataset_split_summary.json` — per-class stats with Q distributions and source breakdowns
- `data/real/annotations_train.json` — 144,191 images / 144,142 annotations
- `data/real/annotations_val.json` — 12,419 images / 12,419 annotations
- `data/real/annotations_test.json` — 63,439 images / 63,396 annotations
- `reports/dataset_split_report.md` — auto-generated methodology report

---

## 1. Current Situation

### Completed

| Component | State |
|-----------|-------|
| Real image download (5 sources, ~597k images) | Done |
| Quality filtering — metadata, heuristics, MegaDetector, VLM captions | Done |
| Manual review (17,737 reviewed, 13,827 approved) | Done |
| SpeciesNet filtering for unverified sources (OpenImages, ImagesCV) | Done |
| Band assignment (A/B/C/D) per `reports/class_distribution_reviewed.csv` | Done |
| Synthetic training images generated (12,600: Band A 10k + Band B 2.6k) | Done |
| Synthetic labeling pipeline (stages 3–7, scripts 3–7 in `scripts/synthetic/`) | Done (design), in progress (execution) |
| Synthetic test set concept (225 × 50 images) | Designed, not yet generated |

### Missing

1. **Train/val/test split assignment** for all real quality-passed images — no split manifest exists yet.
2. **Image quality scoring** — quality-passed images have not been scored/ranked; Band D selection is first-come-first-served today.
3. **Validation set strategy** — the existing band plan allocates real images to train and test only; no val split is defined.
4. **COCO export for real data** — only synthetic has a COCO export script; real data remains in internal `filter_results.jsonl` format.
5. **Split methodology report** — the reasoning behind selections must be documented for the thesis.

---

## 2. Image Quality Scoring System

All quality-passed real images (those with `passed=true` in `filter_results.jsonl`) carry MegaDetector detection metadata. A composite quality score Q ∈ [0, 1] is computed from four components.

### 2.1 Input fields (from `filter_results.jsonl`)

```
detections[]: list of {bbox: [xc, yc, w, h], conf: float}  # YOLO-normalized
bbox: [xc, yc, w, h]      # primary detection (highest-conf)
bbox_conf: float           # primary detection confidence
```

### 2.2 Score Components

**Component 1 — Bbox Area Fraction** (weight 0.30)

The animal should fill a meaningful but not overwhelming portion of the frame. An animal covering <2% of the image is barely detectable; >70% typically means the animal is clipped or pressed against the lens.

Target range: 4–40% of image area.

```
area_frac = bbox_w * bbox_h   # normalized area
if area_frac < 0.02:           area_score = 0.0
elif area_frac < 0.04:         area_score = (area_frac - 0.02) / 0.02   # linear ramp 0→1
elif area_frac <= 0.40:        area_score = 1.0                           # optimal zone
elif area_frac <= 0.70:        area_score = (0.70 - area_frac) / 0.30   # linear ramp 1→0
else:                          area_score = 0.0
```

**Component 2 — Edge Proximity** (weight 0.25)

If the bbox is at the image edge the animal is partially cut off and the detection annotation may be wrong. Compute the minimum clearance between any bbox edge and the image boundary.

```
xmin = xc - w/2;  xmax = xc + w/2
ymin = yc - h/2;  ymax = yc + h/2
margin = min(xmin, 1 - xmax, ymin, 1 - ymax)  # ≤ 0 means clipped
if margin < 0.0:    edge_score = 0.0           # bbox clips outside image
elif margin < 0.02: edge_score = margin / 0.02  # near-edge penalty
else:               edge_score = 1.0
```

**Component 3 — Single Animal** (weight 0.20)

Ideally one animal is clearly depicted. Multiple animals complicate labeling (multi-instance annotations were not collected for real data). Zero significant detections means MegaDetector missed the animal (the image passed other filters so may still be valid — soft penalty only).

```
n_sig = len([d for d in detections if d['conf'] >= 0.5])
single_score = {0: 0.50, 1: 1.00, 2: 0.60}.get(n_sig, 0.30)
```

**Component 4 — Detection Confidence** (weight 0.25)

Higher MegaDetector confidence correlates with clean, unambiguous animal images.

```
conf_score = bbox_conf if bbox_conf is not None else 0.30
```

**Composite:**

```
Q = 0.30 * area_score + 0.25 * edge_score + 0.20 * single_score + 0.25 * conf_score
```

### 2.3 Hard Exclusions

Images with any of the following are set to Q = 0 and excluded from all selection pools:

- `area_frac < 0.01` — animal occupies < 1% of image (essentially invisible)
- `margin < 0.0` — bbox extends outside image boundary (annotation would be wrong)

These hard exclusions are applied before any split assignment.

---

## 3. Per-Band Split Strategy

### Band A — pool < 150 (50 classes)

Real images are too scarce for training. All quality-passed real images go to the test set.

| Split | Source | Count |
|-------|--------|-------|
| Train | synthetic | 200/class |
| Val | synthetic | 40/class (already allocated in labeling pipeline plan) |
| Test | real (all passed) | all available (avg ~67/class) |

No quality-based selection needed; all passed real images are used as-is.

### Band B — pool 150–249 (26 classes)

Pool is scarce. A small real validation set is carved out before train/test.

| Split | Source | Count |
|-------|--------|-------|
| Train | 85 real (top-Q) + 100 synthetic | 185 total |
| Val | 20 real (mid-Q, 30th–70th percentile) | 20 |
| Test | remaining real (pool − 85 − 20) | 45–144 |

If `pool − 85 − 20 < 50` (pool < 155): reduce val proportionally: `val = max(10, pool − 85 − 50)`.

### Band C — pool 250–399 (26 classes)

| Split | Source | Count |
|-------|--------|-------|
| Train | 170 real (top-Q) | 170 |
| Val | 30 real (mid-Q, 30th–70th percentile) | 30 |
| Test | remaining real (pool − 170 − 30) | 50–199 |

Minimum test ≥ 50 is guaranteed since pool ≥ 250.

### Band D — pool ≥ 400 (122 classes)

Three sub-ranges to handle the wide spread in pool size:

| Sub-range | Test | Val | Train |
|-----------|------|-----|-------|
| Small (400–999) | max(⌊pool×0.20⌋, 50), cap 200 | max(20, ⌊pool×0.07⌋), cap 70 | min(pool − test − val, 1500) |
| Medium (1000–4999) | ⌊pool×0.15⌋, cap 500 | 100 | min(pool − test − 100, 1500) |
| Large (≥5000) | 500 | 150 | 1500 |

Surplus (images beyond train + val + test budget) are documented but not included in any split; they remain in the pool for potential future use.

**Band D selection order:**
1. Score all quality-passed images with Q.
2. Apply hard exclusions (Q = 0).
3. Assign test set: stratified random sample by source (proportional to source counts), seeded with SEED=42.
4. Assign val set: from remaining images, random sample from the **30th–70th percentile** of Q — ensures val images are representative quality, not cherry-picked.
5. Assign train set: greedy top-Q selection from remaining, subject to source diversity cap (§3.1).

#### 3.1 Source Diversity Cap (Band D training only)

To avoid training sets dominated by a single source:

| Available sources | Max share from any single source |
|-------------------|----------------------------------|
| 3 or more | 60% |
| 2 | 75% |
| 1 | 100% (no alternative) |

Applied as a running constraint during top-Q greedy selection: track per-source counts and skip an image if its source would exceed the cap, continuing to the next image in Q-ranked order.

---

## 4. Validation Set Philosophy

Val images should be **representative, not cherry-picked**:

- **Independent** from training: no image appears in both train and val.
- **Representative quality**: sampling from the 30th–70th percentile of Q avoids both perfect showcase images (those go to training) and borderline images (better suited to stress-test the model, i.e., test set).
- **Deterministic**: fixed seed 42 for all random operations; the split manifest is written once and never regenerated unless explicitly versioned.
- **Source-stratified**: where multiple sources exist, val images should include ≥1 image from each available source if pool allows.

For Band A (no real training data), the 40 synthetic val images per class defined in the labeling pipeline plan serve as validation. No real val set exists for Band A because the real images are too few and all are reserved for test.

---

## 5. Split Manifest and Documentation Format

### 5.1 Primary manifest: `reports/dataset_split_manifest.json`

One record per assigned image:

```json
{
  "metadata": {
    "created_at": "<ISO timestamp>",
    "seed": 42,
    "version": "1.0",
    "scoring_weights": {
      "area": 0.30, "edge": 0.25, "single": 0.20, "conf": 0.25
    },
    "counts": {
      "total_assigned": 0,
      "train": 0, "val": 0, "test": 0
    }
  },
  "splits": [
    {
      "filepath": "data/gbif/images/aardvark/gbif_aardvark_00009.jpg",
      "class": "aardvark",
      "band": "D",
      "source": "gbif",
      "split": "train",
      "quality_score": 0.847,
      "score_components": {
        "area_score": 0.92, "area_frac": 0.18,
        "edge_score": 1.00, "min_margin": 0.08,
        "single_score": 1.00, "n_significant": 1,
        "conf_score": 0.94
      }
    }
  ]
}
```

### 5.2 Per-class summary: `reports/dataset_split_summary.json`

```json
{
  "aardvark": {
    "band": "D", "pool": 2340, "hard_excluded": 12,
    "train": 1500, "val": 100, "test": 500, "surplus": 228,
    "q_stats": {"mean": 0.72, "p25": 0.61, "p50": 0.74, "p75": 0.83},
    "train_sources": {"gbif": 612, "inaturalist": 888},
    "val_sources": {"gbif": 41, "inaturalist": 59},
    "test_sources": {"gbif": 204, "inaturalist": 296}
  }
}
```

### 5.3 COCO exports: `data/real/annotations_{train,val,test}.json`

Standard COCO JSON. The primary detection bbox from MegaDetector (converted to absolute pixels) serves as the annotation bounding box. Each image entry carries band, source, split, and quality_score as custom fields.

```json
{
  "info": {"description": "Wildlife 225-class real images", "date_created": "...", "version": "1.0"},
  "categories": [{"id": 1, "name": "aardvark"}, ...],
  "images": [
    {
      "id": 1,
      "file_name": "data/gbif/images/aardvark/gbif_aardvark_00009.jpg",
      "width": 1024, "height": 768,
      "band": "D", "source": "gbif", "split": "train",
      "quality_score": 0.847
    }
  ],
  "annotations": [
    {
      "id": 1, "image_id": 1, "category_id": 1,
      "bbox": [x1_px, y1_px, w_px, h_px],
      "area": w_px_times_h_px,
      "iscrowd": 0,
      "source": "megadetector",
      "conf": 0.936
    }
  ]
}
```

Images with `n_sig = 0` (no MegaDetector detection) that still passed pipeline quality filters: include as images without annotations by default (they are not usable as positive training examples). An `--include-negatives` flag can opt them in for experiments that benefit from hard negatives.

### 5.4 Methodology report: `reports/dataset_split_report.md`

Auto-generated Markdown report containing:
- Date, seed, software version
- Per-band image counts (train/val/test) with totals
- Q-score distribution summary per band (min/p25/p50/p75/max)
- Source distribution per band and split
- Hard-exclusion count and reasons
- Classes with fewer than 30 real test images (flagged as test-limited)
- Val set representativeness: mean Q of val vs. overall Q distribution per band

---

## 6. Implementation Plan

### New script: `scripts/dataset_quality/12-assign_dataset_splits.py`

**Inputs:**
- `data/{gbif,inaturalist,wikimedia,openimages,images_cv}/filter_results.jsonl`
- `reports/class_distribution_reviewed.csv` (band assignments per class)
- `resources/2026-03-19_student_model_labels.txt` (225 canonical class names)

**Outputs:**
- `reports/dataset_split_manifest.json`
- `reports/dataset_split_summary.json`
- `reports/dataset_split_report.md`
- `data/real/annotations_train.json`
- `data/real/annotations_val.json`
- `data/real/annotations_test.json`

**High-level pseudocode:**

```python
SEED = 42

# 1. Load all quality-passed images across sources, group by class
# 2. For each image, compute Q using scoring functions above
# 3. Apply hard exclusions (Q = 0 → remove from all pools)
# 4. Determine band for each class from class_distribution_reviewed.csv
# 5. For each class, allocate test/val/train per band rules (§3)
#    - Test: stratified random sample by source, seeded
#    - Val: random sample from 30th–70th pct of Q, seeded
#    - Train: top-Q greedy with source diversity cap
# 6. Write manifest JSON, summary JSON, and COCO JSONs
# 7. Generate split_report.md from manifests
```

**Execution:** single run, deterministic. Re-running with the same SEED produces identical output. If the pool changes (new images approved), increment `version` in manifest metadata.

---

## 7. Open Questions

1. **Images with n_sig=0 in COCO export**: default is exclude (no usable annotation). An `--include-negatives` mode can opt them in. Decide per training experiment.

2. **Multi-animal images in real data (n_sig≥2)**: `filter_results.jsonl` stores only the primary (highest-conf) detection bbox. Using such images for training creates false-negatives for unlabeled animals. Recommendation: exclude n_sig≥2 from real training (single_score already penalizes these); keep them in test for robustness evaluation.

3. **Val domain mismatch for Band A/B experiments**: the Band A val set is 40 synthetic images/class. When a training run uses synthetic and real images together, a mixed val set is more informative. Can be addressed post-hoc by creating a merged val COCO JSON that concatenates synthetic and real val annotations.

---

## 8. Verification

After running `12-assign_dataset_splits.py`:

1. Check `reports/dataset_split_summary.json`: every class has `train + val + test ≤ pool`.
2. Confirm no image appears in more than one split: set intersection of all filepaths must be empty.
3. Check val Q-distribution is centered within the [30th, 70th] percentile range of the full Q distribution per class.
4. Spot-check 5 Band D classes in FiftyOne: load `annotations_train.json`, verify bbox sizes look visually reasonable (animal fills a decent but not overwhelming share of frame).
5. Sanity-count the COCO files:
   ```bash
   python -c "import json; d=json.load(open('data/real/annotations_train.json')); print(len(d['images']), len(d['annotations']))"
   ```
   Output should match totals in `dataset_split_summary.json`.
