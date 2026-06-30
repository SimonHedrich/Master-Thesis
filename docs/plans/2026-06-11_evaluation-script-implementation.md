# Evaluation Script — Implementation Plan

**Date:** 2026-06-11
**Implements:** `docs/plans/2026-06-10_model-evaluation-strategy.md`
**Goal:** A modular, standalone evaluation suite that scores a saved YOLOv5s
checkpoint across the strategy doc's axes (granularity × domain × band), runnable
both **independently** (`run_evaluation.py --checkpoint best.pt`) and as an
**optional post-training hook**.

---

## 1. Design principles

- **Predict once, score many ways.** Run the model exactly once per *component*
  domain (real, synthetic) → cache a COCO-format `predictions.json`. Every
  granularity/band/domain slice is then a cheap CPU remap+filter over the cached
  predictions. `mixed` = the two prediction sets concatenated (no extra inference).
- **Reuse the proven engine.** Coordinate un-letterboxing reuses the exact math
  in `evaluation.py`; scoring reuses `torchmetrics.MeanAveragePrecision`
  (pycocotools backend) which already yields the full COCO-12 vector + per-class.
- **Granularity = label remap, not new metric code.** `fine`=identity,
  `coarse`=label→group, `detect`=label→1. Applied to **both** preds and GT.
- **Don't disturb training-time `evaluation.py`.** New code lives in a separate
  `eval_suite/` package; the existing light per-epoch eval is untouched.

## 2. Module layout (new `scripts/training/yolov5s/eval_suite/`)

| File | Responsibility | Built by |
|------|---------------|----------|
| `grouping.py` | Load `reports/lookalike_groups.csv` → coarse remap (COCO id→group id); detect remap (→1); class→band map from `dataset_split_summary.json` | Agent 1 |
| `predict.py` | checkpoint + annotation JSON → cached COCO `predictions.json` | Agent 2 |
| `scoring.py` | remap+filter+torchmetrics; band×granularity; domain Δ; within-group confusion; optional bootstrap CI | Agent 3 |
| `report.py` | assemble Tier 1/2/3 tables → CSV + Markdown + MLflow | Orchestrator |
| `run_evaluation.py` | CLI entrypoint wiring it all together | Orchestrator |

Plus builder `scripts/dataset_quality/16-build_lookalike_groups.py` →
`reports/lookalike_groups.csv` (Agent 1), and an optional hook in
`run_training_pipeline.py`.

## 3. Frozen data contracts (so the 3 modules integrate cleanly)

**Predictions JSON** (one per component domain):
```json
{"checkpoint": "...", "annotations": "data/real/annotations_test.json",
 "eval": {"conf_thres":0.001,"iou_thres":0.6,"max_det":100,"image_size":640},
 "predictions": [{"image_id": 1, "category_id": 1, "bbox": [x,y,w,h], "score": 0.9}]}
```
`category_id` = COCO id (1..225); `bbox` = original-image-pixel xywh.

**GT index** (built in-memory from an annotation JSON):
- `images: {image_id: {band, width, height, file_name}}`
- `anns:   {image_id: [{category_id, bbox(xywh), area}]}`
- `cats:   {coco_id: name}`

**Remap**: `dict[int,int]` over COCO category_id. `fine`=identity, `coarse`=group,
`detect`=all→1. Applied to preds and GT before scoring.

**`score(gt_index, predictions, image_ids|None, remap|None, max_det, class_metrics) -> dict`**
returns flat: `map, map_50, map_75, map_small/medium/large, mar_1/10/100,
mar_small/medium/large, map_per_class{label:ap}, n_images, n_dets`.

**Mixed domain**: offset synthetic `image_id` by `max(real_image_id)` in both the
GT index and predictions, then merge dicts/lists. Category ids already aligned.

## 4. Reporting (mirrors strategy §9)

- **Tier 1 — headline:** `G=fine,D=mixed,B=all` full COCO-12 + `D=real` breakout +
  `mAP_detect` analog.
- **Tier 2:** (1) granularity gap decomposition detect/coarse/fine + Δs;
  (2) band×{fine,coarse}×{mAP,mAP50} grid (mixed + real breakout);
  (3) domain-shift paired Δ per band + look-alike within-group confusion.
- **Tier 3 (artifacts):** full 225-row per-class AP (with test-img count + band),
  full COCO-12 per band cell, within-group confusion matrix; all JSON/CSV + MLflow.
- Headline reported with & without <30-real-image classes (statistical hygiene §8).
- `negative`-band real images: counted in real/mixed/all; excluded from A/B/C/D slices.
- Optional: `--bootstrap N` (image-resample CI on headline), TIDE (skipped — `tidecv`
  not installed; guarded import + note).

## 5. Orchestration

Three Sonnet subagents build `grouping.py`, `predict.py`, `scoring.py` in parallel
against the §3 contracts. Orchestrator writes `report.py` + `run_evaluation.py` +
the pipeline hook, then performs a critical senior review (coordinate transforms,
ID offsetting, remap-on-both-sides, negative band, empty-prediction images) and a
CPU smoke test on a tiny subset before declaring done.
