# Logged Metrics

All metrics are logged to MLflow during training and evaluation.

## Training Losses

Logged at step level (every 50 steps, key prefix `train/step/`) and as epoch averages (prefix `train/epoch_`).

| Metric | What it measures |
|--------|-----------------|
| `loss_box` | Bounding box regression loss (CIoU) — penalizes deviation of predicted box coordinates from ground truth |
| `loss_obj` | Objectness loss (BCE) — penalizes grid cells that should/shouldn't contain an object |
| `loss_cls` | Classification loss (BCE) — penalizes wrong species class assignments for detected objects |
| `loss_total` | Weighted sum of the three losses above; the quantity the optimizer minimizes |
| `train/lr` | Current learning rate; shows scheduler progress (cosine warm-up / decay) |

## Evaluation Metrics

Computed by `torchmetrics.detection.MeanAveragePrecision` on bounding boxes in `xyxy` format.

| Metric | What it measures |
|--------|-----------------|
| `val/mAP50` | Mean Average Precision across all 225 classes at IoU threshold 0.50; primary checkpoint selection criterion (`best.pt`) |
| `val/mAP50_95` | Mean Average Precision averaged over IoU thresholds 0.50–0.95 in steps of 0.05 (COCO standard); stricter localisation quality measure |
| `test/mAP50` | Same as `val/mAP50` but evaluated once on the held-out test set at run end |
| `test/mAP50_95` | Same as `val/mAP50_95` but on the test set |

Logged per epoch for `val`; logged once at training completion for `test`.

## Artifacts

| File | Contents |
|------|----------|
| `val_per_class_ap_step<epoch>.json` | Table of per-class AP50_95 on the validation set; useful for identifying which species are hardest to detect |
| `test_per_class_ap_stepfinal.json` | Same table for the test set, logged at run end |
| `best.pt` | Model checkpoint with the highest `val/mAP50` seen during training |
| `last.pt` | Model checkpoint from the final training epoch |

## Comprehensive Evaluation Suite (`eval_suite/`)

The per-epoch metrics above are the lightweight training signal. The **standalone
evaluation suite** (`eval_suite/`, see its `README.md`) produces the full
granularity × test-domain × training-band report from
`docs/plans/2026-06-10_model-evaluation-strategy.md`. Run it on any checkpoint:

```bash
# defaults to best.pt from the latest training run under model_exports/:
python -m scripts.training.yolov5s.eval_suite.run_evaluation
# or target a specific run / checkpoint:
python -m scripts.training.yolov5s.eval_suite.run_evaluation --run-dir .../model_exports/<run_name>
# or automatically after training:
python -m scripts.training.yolov5s.run_training_pipeline --full-eval
```

It emits a Markdown report + CSV/JSON artifacts (and MLflow scalars under `eval/`):

| Tier | Contents |
|------|----------|
| 1 — headline | `G=fine·D=mixed·B=all` full COCO-12 vector, the `D=real` breakout (public-comparison anchor), the class-agnostic `mAP_detect` analog, count-weighted + test-limited-excluded sensitivity figures |
| 2 — diagnostic | granularity gap decomposition (detect/coarse/fine + Δs), band×granularity grid (mixed + real), real−synth domain-shift Δ per band, within-look-alike-group confusion table |
| 3 — appendix | 225-row per-class AP table (band + real test-image count), per-band COCO-12 vectors, confusion pairs — all as CSV/JSON |

The scorer was validated to reproduce the training-time `evaluate()` exactly
(fine mAP50_95/mAP50 matched to 4 dp on a 400-image sample).
