# Fix: Incomplete Ground-Truth Annotations in Real-Dataset COCO JSONs

**Date:** 2026-06-08

---

## Problem Statement

When viewing inference results via `scripts/evaluation/visualize_fiftyone.py`, many images in `data/real/annotations_test.json` appear to have missing or incomplete ground-truth bounding boxes. Images clearly containing multiple animals of the same species show only a single GT box, making it impossible to evaluate whether the model correctly detected all animals present.

---

## Investigation Results

### Finding 1 — Every annotated image has exactly one GT box

```
Annotations per image distribution (annotations_test.json):
  1 annotation(s): 63,796 images
  0 annotation(s):     69 images
  Max annotations on one image: 1
```

This is the smoking gun. A dataset about wildlife photography with 225 species categories — including herd animals, pack hunters, and colonial species — should naturally have multi-instance images. The uniform 1-per-image distribution is a pipeline artifact, not a property of the data.

### Finding 2 — The source data correctly captures multiple detections

Each entry in the `filter_results.jsonl` files (one per data source) stores:

- `bbox` — the single highest-confidence MegaDetector detection (YOLO-normalized `[xc, yc, w, h]`)
- `bbox_conf` — confidence of that primary detection
- `detections` — **array of ALL detections** above the secondary confidence threshold (conf ≥ 0.2)

Example from `data/openimages/filter_results.jsonl`:

```json
{
  "filepath": "data/openimages/images/sun_bear/oi_sun_bear_00002.jpg",
  "passed": true,
  "bbox": [0.8091, 0.4914, 0.1396, 0.4389],
  "bbox_conf": 0.8477,
  "detections": [
    {"conf": 0.8477, "bbox": [0.8091, 0.4914, 0.1396, 0.4389]},
    {"conf": 0.7604, "bbox": [0.4414, 0.5895, 0.2012, 0.2702]},
    {"conf": 0.7506, "bbox": [0.1162, 0.5731, 0.2012, 0.2719]}
  ]
}
```

Three animals are detected; all three are faithfully stored in `detections`. Only the first ever becomes a GT box in the exported COCO JSON.

**Scale of multi-detection images across sources (conf ≥ 0.5):**

| Source      | Images with 2+ significant detections | Approx. share |
|-------------|---------------------------------------|---------------|
| GBIF        | ~2,605                                | ~26 %         |
| iNaturalist | ~3,079                                | ~15 %         |
| Wikimedia   | ~3,188                                | ~26 %         |
| OpenImages  | ~2,402                                | ~31 %         |

Across train + val + test this amounts to tens of thousands of images with suppressed ground-truth boxes.

### Finding 3 — Root cause: `detections` is discarded in `load_all_passed_images()`

File: `scripts/dataset_quality/12-assign_dataset_splits.py`, lines 157–168

```python
bbox       = entry.get("bbox")
bbox_conf  = entry.get("bbox_conf")
detections = entry.get("detections") or []   # read for quality scoring …
Q, comps   = compute_quality_score(bbox, bbox_conf, detections)
per_class[cls].append({
    "filepath":         fp,
    "source":           source,
    "Q":                Q,
    "score_components": comps,
    "bbox":             bbox,
    "bbox_conf":        bbox_conf,
    # "detections" is NOT stored here  ← bug
})
```

`detections` is passed to `compute_quality_score()` (which uses it to score "single clean animal" vs "crowded multi-animal" images), but it is not forwarded into the image dict. From this point on, the information is gone.

### Finding 4 — `build_coco_split()` can only emit one annotation per image

File: `scripts/dataset_quality/12-assign_dataset_splits.py`, lines 509–527

```python
bbox = img.get("bbox")            # only the primary bbox survived
if bbox is not None:
    cat_id = cat_id_map.get(cls)
    if cat_id is not None:
        coco_bbox = yolo_to_coco(bbox, img_w, img_h)
        annotations.append({
            "id":          ann_id,
            "image_id":    img_id,
            "category_id": cat_id,
            "bbox":        coco_bbox,
            ...
        })
        ann_id += 1
```

Because `detections` is absent, the function has no information about secondary animals and writes exactly one annotation per image regardless of how many animals MegaDetector found.

### Finding 5 — Relevant confidence thresholds

From `scripts/dataset_quality/12-assign_dataset_splits.py`, line 54:
```python
CONF_SIG = 0.5   # significance threshold used in quality scoring
```

From `scripts/dataset_quality/1-filter_dataset_quality.py`, lines 98–104:
```python
MD_CONF_PASS      = 0.5   # min confidence for an image to pass (primary detection)
MD_CONF_SECONDARY = 0.2   # lower bound — all detections above this stored in `detections`
MD_BBOX_MIN_AREA  = 0.01  # min fractional area (enforced before storage)
```

The `detections` array contains all detections with conf ≥ 0.2 and area ≥ 0.01. For GT export, the correct filter is **conf ≥ CONF_SIG (0.5)** — matching the primary-detection pass threshold and the existing quality-scoring cutoff. This keeps confidently detected secondary animals and excludes marginal detections that are likely noise.

### Finding 6 — Secondary issue: 69 images have zero annotations

- **43 Wikimedia images:** MegaDetector returned zero detections (`bbox=null`). The images contain animals, but none were detected. These images are included in the COCO `images` list with no matching entry in `annotations`. For evaluation this is harmful: any model prediction on them is scored as a false positive, while the missed animal is never counted as a false negative, biasing precision downward.
- **26 blank/background images:** Intentional true-negative samples from `data/blanks/`. Having no annotations is correct for these.

---

## Proposed Fixes

### Fix A — Pipeline fix in `12-assign_dataset_splits.py`

**Step A1 — `load_all_passed_images()`**: Forward `detections` into the stored image dict:

```python
per_class[cls].append({
    "filepath":         fp,
    "source":           source,
    "Q":                Q,
    "score_components": comps,
    "bbox":             bbox,
    "bbox_conf":        bbox_conf,
    "detections":       detections,   # add this line
})
```

**Step A2 — `build_coco_split()`**: Replace the single-annotation block with a loop over all significant detections:

```python
bbox = img.get("bbox")
if bbox is not None:
    cat_id = cat_id_map.get(cls)
    if cat_id is not None:
        sig_dets = [d for d in (img.get("detections") or [])
                    if d.get("conf", 0) >= CONF_SIG]
        if not sig_dets:   # fallback: primary bbox always qualifies
            sig_dets = [{"bbox": bbox, "conf": img.get("bbox_conf")}]
        for det in sig_dets:
            coco_bbox = yolo_to_coco(det["bbox"], img_w, img_h)
            w_px, h_px = coco_bbox[2], coco_bbox[3]
            annotations.append({
                "id":          ann_id,
                "image_id":    img_id,
                "category_id": cat_id,
                "bbox":        coco_bbox,
                "area":        w_px * h_px,
                "iscrowd":     0,
                "source":      "megadetector",
                "conf":        det.get("conf"),
            })
            ann_id += 1
else:
    no_annotation += 1
```

**Step A3 — zero-annotation Wikimedia images**: Skip non-blank images with `bbox=None` before adding them to the COCO `images` list (move the `bbox is None` guard before `images.append()`). This prevents unevaluable images from polluting the test set.

**Consequence:** Fix A only takes effect after re-running `12-assign_dataset_splits.py`, which regenerates all three COCO JSONs from the `filter_results.jsonl` source files. The pipeline is deterministic and reproducible.

---

### Fix B — Post-processing script (immediate relief, no pipeline re-run)

Write `scripts/dataset_quality/patch_multi_annotations.py`:

1. Build a lookup `{filepath: [sig_dets]}` from all `data/{source}/filter_results.jsonl` files, filtering conf ≥ 0.5.
2. Load each COCO JSON (`annotations_train.json`, `_val.json`, `_test.json`).
3. For each image that currently has exactly one annotation, look up additional secondary detections and append new annotation entries.
4. Also remove the 43 Wikimedia annotation-free images from the `images` list.
5. Write results back in-place.

This produces correct multi-annotation JSONs immediately without re-running the full dataset quality pipeline.

---

### Fix C — Visualization sampling in `run_inference.py`

Pre-filter the 100-image sampling pool to only include images with at least one annotation (lines 109–111):

```python
annotated_ids = {a["image_id"] for a in coco.get("annotations", [])}
pool = [img for img in coco["images"] if img["id"] in annotated_ids]
n = min(args.num_images, len(pool))
sampled = random.Random(SEED).sample(pool, n)
```

This is independent of Fix A/B and prevents blank/annotation-free images from ever appearing in the FiftyOne visualization subset.

---

## Recommended Implementation Order

| Step | Action | Effort |
|------|--------|--------|
| 1 | Fix C: patch `run_inference.py` sampling | ~5 min |
| 2 | Fix A: patch `12-assign_dataset_splits.py` (A1 + A2 + A3) | ~20 min |
| 3 | Re-run `12-assign_dataset_splits.py` to regenerate JSONs | pipeline runtime |
| 4 | (Fallback) Fix B: post-processing script if re-run is impractical | ~30 min |

Steps 1 and 2 are independent. Step 3 is required for Fix A to propagate into the data files; Fix B is the alternative if re-running the pipeline is too expensive.

---

## Verification

After applying Fix A + re-run (or Fix B), run:

```python
import json
from collections import Counter
with open("data/real/annotations_test.json") as f:
    d = json.load(f)
cnt = Counter(a["image_id"] for a in d["annotations"])
dist = Counter(cnt.values())
for n in sorted(dist):
    print(f"  {n} annotation(s): {dist[n]} images")
print(f"  Max on one image: {max(cnt.values())}")
```

Expected: distribution spreads across 1, 2, 3, 4+ annotations per image. The uniform "exactly 1" result should disappear.

Then re-run `run_inference.py` and re-launch `visualize_fiftyone.py` to visually confirm multiple GT boxes appear on multi-animal images.

---

## Files Affected

| File | Change |
|------|--------|
| `scripts/dataset_quality/12-assign_dataset_splits.py` | Bug fix — pipeline source of truth |
| `data/real/annotations_train.json` | Regenerated or post-processed |
| `data/real/annotations_val.json` | Regenerated or post-processed |
| `data/real/annotations_test.json` | Regenerated or post-processed |
| `scripts/dataset_quality/patch_multi_annotations.py` | New script (Fix B fallback) |
| `scripts/evaluation/run_inference.py` | Minor quality-of-life improvement (Fix C) |
