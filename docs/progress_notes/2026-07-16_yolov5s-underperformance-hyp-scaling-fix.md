# yolov5s underperforming yolo26n: missing nc-scaled loss gains

**Date:** 2026-07-16

## What happened

A yolo26n run (`yolo26n-20260715-010031`, 53 epochs: mAP50=0.614,
mAP50_95=0.512) was dramatically outperforming a yolov5s run (74 epochs:
mAP50=0.371, mAP50_95=0.291) — backwards from expectations, since yolov5s is
the more mature/tuned architecture and both fine-tune from COCO weights on
the identical dataset.

## Investigation

Confirmed both pipelines are a fair comparison before looking for a bug:
same dataset (145,809 train images, 225 classes, byte-identical COCO JSON),
same optimizer/LR schedule/early-stopping/seed, no frozen layers on either
side, both fine-tuned from unmodified COCO pretrained weights. yolo26n
deliberately imports its dataset, optimizer, scheduler, and training loop
straight from the yolov5s package for this reason.

### Hypothesis 1 (disproven): anchor mismatch

yolov5s inherits its 9 anchor boxes unmodified from COCO's k-means fit
(`yolov5s.yaml`), and nowhere does the pipeline run YOLOv5's autoanchor
recompute (`check_anchors`) to refit them to the wildlife dataset's actual
box-size distribution — a step YOLOv5's own `train.py` normally does
automatically. This looked like a strong candidate since it's a failure mode
structurally impossible for yolo26n (anchor-free).

Tested numerically against the real training set (best-possible-recall
metric, thr=4.0, imgsz=640):

```
COCO anchors on wildlife train set: BPR=1.0000, anchors/target=3.71
```

BPR=1.0 means the COCO anchors already fit the wildlife box-size distribution
essentially perfectly (median box ~235×214px at 640 scale, well inside the
anchor range 10–373px). **Recomputing anchors would have been a no-op** —
ruled out.

### Hypothesis 2 (confirmed): missing nc/nl/imgsz loss-gain autoscaling

YOLOv5's own official `train.py` autoscales its loss gains for the actual
class count, image size, and detection-layer count before training
(`yolov5/train.py:255-258`):

```python
hyp['box'] *= 3 / nl
hyp['cls'] *= nc / 80 * 3 / nl
hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl
```

This repo's `yolov5s_model.py:_hyp_dict()` built the hyp dict straight from
`constants.py` (`HYP_CLS=0.5`, etc.) with **no such scaling** — those are the
raw COCO (nc=80) reference values. For this project (`nc=225`, `nl=3`,
`imgsz=640`), the box/obj factors both evaluate to 1 (no effective change),
but **the cls gain factor is `225/80 * 3/3 = 2.8125`** — classification loss
was under-weighted by ~2.8x relative to YOLOv5's own tuning recipe. With cls
loss under-weighted, the box/cls loss balance skews toward pure localization
at the expense of species discrimination — right box, wrong species — which
depresses mAP through misclassification specifically, consistent with the
observed gap. This is not applicable to yolo26n, which uses Ultralytics'
newer TAL/DFL loss with no equivalent nc-scaling in its own trainer either
(confirmed by inspecting `ultralytics/utils/loss.py`).

## Fixes (all in `scripts/training/yolov5s/`)

1. **`yolov5s_model.py` — `_hyp_dict()` now applies YOLOv5's own autoscaling.**
   Takes `num_classes` and `nl` (`= model.model[-1].nl`) and scales
   box/cls/obj before returning the dict passed to `ComputeLoss` via
   `model.hyp`. Effective values are logged alongside the model summary.
   For this project: `cls: 0.5 → 1.40625`; `box`/`obj` unchanged.
2. **New `autoanchor.py` — anchor-fit audit.** Builds shapes/labels directly
   from the COCO JSON's `width`/`height` fields (no image decoding — both
   are already present on every image record), wraps them in a minimal
   adapter satisfying `yolov5.utils.autoanchor.check_anchors`'s dataset
   contract, and calls it. `check_anchors` only replaces anchors if the
   current fit is poor (BPR ≤ 0.98) *and* k-means finds a strictly better
   set — safe to always call, and a documented no-op today (BPR=1.0). Wired
   into `run_training_pipeline.py` right after model construction, skipped
   on `--resume-from` (the checkpoint's anchors already reflect a prior
   decision).
3. **`run_training_pipeline.py` — MLflow traceability.** Logs the *effective*
   (scaled) `hyp_box/cls/obj_effective` values and the autoanchor outcome
   (`autoanchor_bpr`, `autoanchor_anchors_changed`) as run params, since
   `constants.as_dict()` only captures the raw pre-scaling constants and
   `check_anchors` only logs to yolov5's own logger.
4. **`README.md`** — module-map entry added for `autoanchor.py`.

## Verification

- Spot-checked `_hyp_dict(225, 3)` directly: `cls == 1.40625`, `box == 0.05`,
  `obj == 1.0` — matches hand calculation.
- Ran `--smoke` end-to-end. Log confirms:
  ```
  hyp (autoscaled for nc=225, nl=3): box=0.05 cls=1.4062 obj=1
  autoanchor: BPR=1.0000 against current anchors (thr=4, imgsz=640)
  ```
  and `autoanchor_bpr=1.0`, `autoanchor_anchors_changed=False`,
  `hyp_cls_effective=1.40625` recorded as MLflow params. Training proceeded
  normally (`loss_cls` decreasing across the first 150 steps).

## Next steps

Kick off a fresh (non-resumed) full yolov5s training run — not
`--resume-from`, since the new loss balance must apply from epoch 0 — and
compare its mAP50/mAP50_95 trajectory over the first ~20-30 epochs against
the previous 74-epoch baseline (mAP50=0.371, mAP50_95=0.291) to confirm the
classification-loss fix closes a meaningful part of the gap to yolo26n.

## Lessons

- When porting a well-established training recipe (YOLOv5) to a class count
  it wasn't tuned for, check the *upstream* `train.py`/CLI for dataset-shape
  autoscaling steps, not just the loss-gain defaults in a hyp YAML — the
  defaults alone are silently wrong outside the reference `nc`.
- Verify a hypothesis against real data before implementing the fix it
  implies. The anchor-mismatch theory was plausible and structurally
  well-motivated, but a five-line numpy check against the actual training
  boxes disproved it before any code was written.
