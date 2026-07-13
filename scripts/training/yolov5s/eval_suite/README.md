# `eval_suite` — comprehensive model evaluation

Implements the evaluation strategy in
`docs/plans/2026-06-10_model-evaluation-strategy.md`: it scores a trained
YOLOv5s checkpoint across **granularity × test-domain × training-band** and emits
the Tier 1/2/3 tables (Markdown + CSV + JSON, optionally MLflow).

It is **independent of training** — give it a checkpoint and the test annotation
files. Inference is run once per component domain (real, synthetic) and cached;
every slice is then a cheap CPU remap+filter over the cached predictions.

## Quick start

```bash
# Full evaluation on the real + synthetic test sets (mixed default headline).
# With no args it uses best.pt from the latest training run under model_exports/.
python -m scripts.training.yolov5s.eval_suite.run_evaluation

# Target a specific run (or pass --checkpoint .../<run_name>/best.pt)
python -m scripts.training.yolov5s.eval_suite.run_evaluation \
    --run-dir scripts/training/yolov5s/model_exports/yolov5s-20260602-233434

# Smoke test: 1500 representative random images, GPU, with bootstrap CI
python -m scripts.training.yolov5s.eval_suite.run_evaluation \
    --limit 1500 --bootstrap 200 --device cuda

# Real-only (skip synthetic)
python -m scripts.training.yolov5s.eval_suite.run_evaluation --synth-ann none
```

Output lands in `--output-dir` (default `<ckpt_dir>/eval_<ckpt_stem>/`):
`evaluation_report.md`, `evaluation_report.json`, `eval_per_class.csv`,
`eval_band_grid.csv`, `eval_confusion_pairs.csv`, and the cached
`predictions_{real,synth}.json`.

## Run automatically after training

```bash
python -m scripts.training.yolov5s.run_training_pipeline --full-eval
```

This runs the suite on `best.pt` once training finishes and logs the report +
scalars to the active MLflow run. It is best-effort (a failure never fails the
training run).

## Prerequisite artifact

`reports/lookalike_groups.csv` defines the **coarse** (look-alike-merged)
granularity. Build/refresh it with:

```bash
python scripts/dataset_quality/16-build_lookalike_groups.py
```

(genus-level taxonomic rollup + a frozen curated override list for the notorious
visual look-alikes; strategy doc §5.)

## Evaluating the MegaDetector + SpeciesNet ensemble

The ensemble is a teacher-class baseline: MegaDetector v5 detects animals; SpeciesNet
classifies each crop; the joint `md_conf × sn_score` is the detection confidence.
Both steps run inside the standard training container (`make execute`).

**Step 1 — generate predictions** (inside the training container):

```bash
# Smoke test (100 images/domain, ~5 min):
PYTHONPATH=/app python -m scripts.training.yolov5s.eval_suite.predict_ensemble \
    --output-dir scripts/training/yolov5s/model_exports/megadet_speciesnet_ensemble/ \
    --limit 100

# Full run (~2–3 hours for 63K real + 11K synth):
PYTHONPATH=/app python -m scripts.training.yolov5s.eval_suite.predict_ensemble \
    --output-dir scripts/training/yolov5s/model_exports/megadet_speciesnet_ensemble/
```

Output: `megadet_speciesnet_ensemble/predictions_real.json` + `predictions_synth.json`.

**Step 2 — score** (same container, after Step 1):

```bash
PYTHONPATH=/app python -m scripts.training.yolov5s.eval_suite.run_evaluation \
    --real-predictions scripts/training/yolov5s/model_exports/megadet_speciesnet_ensemble/predictions_real.json \
    --synth-predictions scripts/training/yolov5s/model_exports/megadet_speciesnet_ensemble/predictions_synth.json \
    --output-dir scripts/training/yolov5s/model_exports/megadet_speciesnet_ensemble/eval/
```

Output: `megadet_speciesnet_ensemble/eval/evaluation_report.md` (same format as a
YOLOv5s eval report). Step 2 takes ~55 min (same scoring workload).

## Modules

| Module | Role |
|--------|------|
| `predict.py` | checkpoint + COCO annotations → cached COCO predictions JSON |
| `predict_ensemble.py` | MegaDetector+SpeciesNet → COCO predictions JSON (runs in the standard training container) |
| `scoring.py` | remap + filter + torchmetrics COCO-12; band/domain/confusion/CI helpers |
| `grouping.py` | load fine/coarse/detect remaps + class→band map from `reports/` |
| `report.py` | assemble Tier 1/2/3 tables → Markdown/CSV/JSON/MLflow |
| `run_evaluation.py` | CLI + `evaluate_checkpoint()` / `evaluate_from_predictions()` entrypoints |

## Validation

The scorer was checked to reproduce the trusted training-time `evaluation.py`
exactly: on a 400-image representative sample the fine mAP50_95 / mAP50 matched
to 4 decimals, confirming the predictions→COCO-id remap and GT coordinate
handling are consistent with the established evaluator.
