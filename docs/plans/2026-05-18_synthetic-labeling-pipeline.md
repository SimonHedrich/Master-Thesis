# Synthetic Image Labeling Pipeline

**Date:** 2026-05-18  
**Status:** Planning  
**Goal:** Produce COCO-formatted bounding box annotations for all 12,600 synthetic images.

---

## Overview

The 12,600 synthetic images in `data/synthetic/images/` (band_a: 50 species, band_b: 26 species) are already generated but have no bounding box annotations. We need labels for object detection training. The pipeline has five stages:

```
[MD inference] → [triage review] → [multi-animal labeling] → [COCO export] → [FiftyOne check]
      3                  4                    5                     6               7
```

MegaDetector provides initial bboxes. Human review resolves two ambiguous cases: (1) MD detected one animal but there are actually more; (2) multi-animal images where MD bboxes need correction. All verified bboxes are merged into a COCO JSON.

---

## Pipeline Stages

### Stage 3 — MegaDetector inference (`3-run_megadetector.py`)

**Input:** `data/synthetic/index.jsonl` + all images on disk  
**Output:** `data/synthetic/md_detections.jsonl`

Run MD v5 on every synthetic image. Record **all** animal detections above a secondary conf threshold (default 0.2), not only the top-1. This matches the existing behavior in `1-filter_dataset_quality.py`.

One JSONL line per image:
```json
{
  "filepath": "data/synthetic/images/band_a/walrus/a_walrus_001.png",
  "class": "walrus",
  "band": "a",
  "width": 1024,
  "height": 1024,
  "detections": [{"bbox": [xc, yc, w, h], "conf": 0.87}],
  "n_significant": 1
}
```

- `detections`: all animal dets with `conf >= conf_secondary` (default 0.2), sorted by conf desc, YOLO-normalised `[xc, yc, w, h]`
- `n_significant`: count of dets with `conf >= conf_pass` (default 0.5) — the key for triage routing
- Resumable: skips filepath already present in output file
- Args: `--batch-size 32`, `--num-workers 4`, `--conf-pass 0.5`, `--conf-secondary 0.2`, `--force`
- Reuse the `_FileListDataset` + batched NMS pattern from the existing MD stage

Image dimensions are stored here so later stages don't need to re-open images for COCO conversion.

**Routing after MD:**
| n_significant | Route |
|---|---|
| 0 | → **Stage 5** bbox labeling (MD missed animal; draw bbox from scratch) |
| 1 | → Stage 4 triage review |
| ≥ 2 | → Stage 5 bbox labeling (directly) |

> **Note (actual run 2026-05-18):** Only 2 of 12,600 images had `n_significant == 0`, and both were confirmed to contain animals. These go to Stage 5 for from-scratch annotation. The overwhelmingly dominant case is `n_significant == 1` (98.8%), with 153 images (1.2%) having ≥2 detections.

---

### Stage 4 — Single-detection triage review (`4-single_detect_review.py`)

**Input:** `md_detections.jsonl` (n_significant == 1 entries)  
**Output:** `data/synthetic/single_detect_flags.jsonl`

**Purpose:** Images where MD found exactly one significant animal may actually contain more (model missed them). A human scans these quickly in batch and flags the exceptions.

FastAPI server adapted from `11-batch_review_server.py`. Key differences:
- Source: images from `md_detections.jsonl` where `n_significant == 1`
- Batch grid layout (same as script 11): images grouped by species, 4-column grid
- Default state per image = **single** (only one animal); clicking marks it **multi**
- Committing a batch writes decisions to `single_detect_flags.jsonl`
- Shows the one MD detection bbox as an overlay (green box)
- Port: `--port 8082` (avoid conflict with existing servers)

Output format (one line per image):
```json
{"filepath": "...", "decision": "single"|"multi", "ts": "2026-05-18T..."}
```

After all images are reviewed, images flagged `multi` join the MD-detected multi-animal images in Stage 5.

Keyboard shortcuts: Space = commit batch, Z = undo, B = toggle bbox, ? = help (identical UX to script 11).

---

### Stage 5 — Multi-animal bbox labeling (`5-bbox_labeling_server.py`)

**Input:**
- `md_detections.jsonl` (n_significant ≥ 2 entries — go directly here)
- `single_detect_flags.jsonl` (decision == "multi" entries — redirected from Stage 4)

**Output:** `data/synthetic/manual_labels.jsonl`

**Purpose:** Interactive per-image bbox editor where MD bboxes are pre-loaded as a starting point. The user can add, remove, or adjust bboxes.

FastAPI server adapted from `10-review_server.py`. Backend:
- Builds a queue of unlabeled multi-animal images (those absent from `manual_labels.jsonl`)
- `/api/image` returns image metadata + existing MD bboxes for the current image
- `/api/save` accepts the edited bbox list and appends to `manual_labels.jsonl`
- `/api/skip` records a skip decision (image has no labelable content)
- Serves `/image?path=...` same as existing servers
- Progress tracking: labeled / total in queue
- Port: `--port 8083`

Frontend canvas-based bbox editor (vanilla JS):
- Image renders in a `<canvas>` element, scaled to fit the viewport
- Pre-loaded bboxes rendered as draggable rectangles with resize handles (8 handles per box)
- **Draw mode** (default): click + drag on empty area → creates new bbox
- **Select mode**: click on existing bbox → selects it (highlighted); drag to move; drag corner/edge handles to resize
- **Delete**: Delete/Backspace key or a trash button removes selected bbox
- Mode toggles automatically: clicking near an existing box selects it, clicking empty space starts drawing
- All bboxes for one image share the same class label (the species from `index.jsonl`; no per-box class needed)
- Buttons: **Save & Next** (saves bboxes, advances), **Skip** (no annotation, skip image), **← Prev** (go back, reloads last saved state)
- Keyboard: S = save & next, N = next without saving (= skip), Escape = deselect

Output format (one line per image, written on "Save & Next"):
```json
{
  "filepath": "data/synthetic/images/band_a/walrus/a_walrus_017.png",
  "class": "walrus",
  "skipped": false,
  "bboxes": [
    {"bbox": [0.45, 0.52, 0.30, 0.40], "source": "megadetector", "conf": 0.87},
    {"bbox": [0.20, 0.35, 0.25, 0.33], "source": "manual", "conf": null}
  ],
  "labeled_at": "2026-05-18T12:34:56Z"
}
```

- `bbox`: YOLO normalised `[xc, yc, w, h]` — keeps format consistent with MD output
- `source`: tracks provenance of each box
- `skipped`: true if user pressed Skip (image excluded from annotations in Stage 6)

---

### Stage 6 — COCO export (`6-export_coco.py`)

**Input:**
- `data/synthetic/index.jsonl`
- `data/synthetic/md_detections.jsonl`
- `data/synthetic/single_detect_flags.jsonl`
- `data/synthetic/manual_labels.jsonl`

**Output:** `data/synthetic/annotations_coco.json`

**Merge logic per image:**

| Condition | Bbox source |
|---|---|
| `n_significant == 0`, saved in manual_labels | Use manual_labels bboxes (drawn from scratch) |
| `n_significant == 0`, NOT in manual_labels | Warn + skip |
| `n_significant == 1`, flagged `multi`, saved in manual_labels | Use manual_labels bboxes |
| `n_significant == 1`, flagged `multi`, NOT yet in manual_labels | Warn + skip (labeling incomplete) |
| `n_significant == 1`, flagged `single` | Use the single MD detection |
| `n_significant == 1`, NOT yet reviewed (no flag) | Warn + use MD detection (best-effort) |
| `n_significant >= 2`, saved in manual_labels | Use manual_labels bboxes |
| `n_significant >= 2`, NOT yet in manual_labels | Warn + use all MD detections above conf_pass as best-effort |
| `skipped == true` in manual_labels | Skip — no annotation |

**COCO output structure:**
```json
{
  "info": {"description": "Synthetic wildlife images", "date_created": "2026-05-18"},
  "categories": [
    {"id": 1, "name": "aardvark", "supercategory": "animal"},
    ...
  ],
  "images": [
    {
      "id": 1,
      "file_name": "data/synthetic/images/band_a/walrus/a_walrus_001.png",
      "width": 1024, "height": 1024,
      "band": "a", "split": "val", "shot_type": "eye_level",
      "distance": "medium", "lighting": "overcast", "occlusion": "none"
    }
  ],
  "annotations": [
    {
      "id": 1, "image_id": 1, "category_id": 74,
      "bbox": [410, 470, 205, 204],
      "area": 41820, "iscrowd": 0,
      "source": "megadetector", "conf": 0.87
    }
  ]
}
```

- **Categories**: all unique class names from `index.jsonl`, sorted alphabetically (IDs are stable across runs)
- **bbox format**: COCO absolute pixel coords `[x_topleft, y_topleft, width, height]` — converted from YOLO normalised using `width`/`height` from `md_detections.jsonl`
- **Splits**: `index.jsonl` has `split` field ("train"/"val"); the export can optionally split into separate files via `--split train` / `--split val`
- Prints statistics at end: total images, annotated images, skipped images, annotations per class, mean/median annotations per image

---

### Stage 7 — FiftyOne verification (`7-verify_fiftyone.py`)

**Input:** `data/synthetic/annotations_coco.json`

Loads the annotation file as a FiftyOne dataset using `fo.Dataset.from_dir` with the COCO format importer, launches the app, and prints per-class counts. Provides a quick sanity check that bboxes are visually correct.

```python
import fiftyone as fo

dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path=REPO_ROOT / "data/synthetic/images",
    labels_path=REPO_ROOT / "data/synthetic/annotations_coco.json",
    name="synthetic_wildlife",
)
session = fo.launch_app(dataset)
session.wait()
```

---

## Data Flow Summary

```
data/synthetic/index.jsonl
data/synthetic/images/
        │
        ▼ 3-run_megadetector.py
data/synthetic/md_detections.jsonl
        │
        ├─── n_significant==1 ──▶ 4-single_detect_review.py
        │                               │
        │         ┌──────── single ─────┤
        │         │         multi ──────┤
        │         │                     ▼
        └──────── └──── n_sig>=2 ──▶ 5-bbox_labeling_server.py
                                          │
                                          ▼
                                  data/synthetic/manual_labels.jsonl
                                          │
                              ┌───────────┘
                              │  + md_detections.jsonl
                              │  + single_detect_flags.jsonl
                              ▼
                         6-export_coco.py
                              │
                              ▼
                  data/synthetic/annotations_coco.json
                              │
                              ▼
                       7-verify_fiftyone.py
```

---

## File Locations

| File | Path |
|---|---|
| MD detections | `data/synthetic/md_detections.jsonl` |
| Single-detect review output | `data/synthetic/single_detect_flags.jsonl` |
| Manual bbox labels | `data/synthetic/manual_labels.jsonl` |
| COCO annotations | `data/synthetic/annotations_coco.json` |

---

## Implementation Notes

### Reuse from existing pipeline
- MD batched inference: reuse `_FileListDataset`, NMS + `scale_boxes`, `megadetector_to_yolo()` from `1-filter_dataset_quality.py`
- Server skeleton (FastAPI + single-page HTML): reuse the structure from `10-review_server.py` and `11-batch_review_server.py`
- Image serving endpoint: identical `/image?path=` pattern across all three new servers

### Canvas bbox editor (Stage 5)
The most complex piece. Key implementation choices:
- Use a single `<canvas>` overlay on top of an `<img>` tag (not drawing image onto canvas — avoids cross-origin issues and keeps image rendering to the browser)
- Actually: draw the image onto canvas using `drawImage()` so hit-testing and coordinate math works cleanly in one coordinate space
- Scale factor: track `scaleX = canvas.width / naturalWidth` for coordinate conversion
- Handle minimum bbox size (e.g., 10px) to prevent accidental point-clicks creating empty boxes
- Resize handles: 8 control points (corners + midpoints of edges), 10px radius hit zone

### YOLO ↔ COCO conversion
- YOLO `[xc, yc, w, h]` (normalised) → COCO `[x1, y1, w_px, h_px]` (absolute pixels):
  ```python
  x1 = (xc - w/2) * img_width
  y1 = (yc - h/2) * img_height
  w_px = w * img_width
  h_px = h * img_height
  ```
- Store as integers (round)

### Conf thresholds
- `conf_pass = 0.5` (what counts as "significant" for routing)
- `conf_secondary = 0.2` (lower bound for recording all detections in md_detections.jsonl)
- These match the existing pipeline thresholds

### Partial completion handling
Stage 6 should be runnable before Stage 4 and 5 are 100% complete (use best-effort MD bboxes for unlabeled multi-animal images, with a warning). This lets you start training with partial labels while labeling continues.

### Port assignments
- Port 8080: existing `10-review_server.py`
- Port 8081: existing `11-batch_review_server.py`
- Port 8082: new `4-single_detect_review.py`
- Port 8083: new `5-bbox_labeling_server.py`

---

## Actual Scale (from MD run 2026-05-18)

- Total images: 12,600
- n_significant==0: **2** (0.0%) → Stage 5, draw bbox from scratch
- n_significant==1: **12,445** (98.8%) → Stage 4 triage review
- n_significant>=2: **153** (1.2%) → Stage 5 directly
- Of Stage 4, expect "actually multi" flags: ~1–3% → ~125–375 additional Stage 5 images
- Stage 5 total workload: ~280–530 images (153 MD-multi + 2 MD-zero + ~125–375 Stage-4 flagged)

Manual labeling estimate at ~15–30 sec/image: 1–4 hours.

---

## Implementation Order

1. Script 3 (MD inference) — no UI, run once, parallelizable → implement and run first
2. Script 6 (COCO export) — implement skeleton early so output format is validated
3. Script 7 (FiftyOne check) — short script, implement alongside Script 6
4. Script 4 (single-detect triage) — simpler UI, based directly on Script 11
5. Script 5 (multi-animal labeling) — most complex, canvas bbox editor
