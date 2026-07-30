# YOLOv5s Fine-Tuning Pipeline

Custom training loop for fine-tuning YOLOv5s on the 225-class wildlife dataset
(`data/real/`). YOLO-specific machinery (model, loss, NMS) is imported from
the `yolov5` PyPI package; everything else — dataset, dataloader, optimizer,
scheduler, training loop, evaluation, MLflow logging — is owned by this
package so it can be reused for NanoDet / PicoDet runs.

Design rationale and trade-offs:
`docs/plans/2026-06-02_yolov5s-training-pipeline.md`.

## Module map

| File | Role |
|------|------|
| `constants.py` | Single source of truth for paths and hyperparameters. `as_dict()` is logged to MLflow at run start. |
| `transforms.py` | `letterbox` + `to_tensor`. Augmentation stubs are wired but disabled (`AUG_*` flags). |
| `dataset.py` | `CocoYoloDataset` reads COCO JSON, returns `(image, targets[N,6], path, shapes)`. `collate_fn` fills the batch-index column. `Dataloader` wraps the PyTorch `DataLoader`. |
| `yolov5s_model.py` | Factories: `yolov5s_model` (loads pretrained, swaps detect head for `NUM_CLASSES`, builds `model.hyp` with nc/nl/imgsz-autoscaled loss gains per YOLOv5's own `train.py` convention), `model_optimizer` (3 param groups: BN / conv / bias), `model_scheduler` (`OneCycleLR`; stepped every batch in the training loop). |
| `autoanchor.py` | `check_anchor_fit` — best-possible-recall audit of the yolov5 anchors against the actual training-set box distribution (built from COCO JSON `width`/`height`, no image decoding); recomputes via `yolov5.utils.autoanchor.check_anchors` only if the fit is poor. Run once per fresh (non-resumed) training run. |
| `loss.py` | `YoloLoss` — thin wrapper around `yolov5.utils.loss.ComputeLoss`. Returns `(total_loss, parts_dict)` for direct MLflow logging. |
| `evaluation.py` | `evaluate` runs NMS + un-letterboxes preds + computes mAP via `torchmetrics.detection.MeanAveragePrecision`. `eval_log_mlflow` logs scalars and a per-class AP table (as a plain JSON artifact under `per_class_ap/`, best-effort — see the 2026-07-13 progress note). |
| `training_pipeline.py` | `TrainingPipeline` class. Trains, evaluates each epoch, saves `best.pt` / refreshes `last.pt` every epoch, supports `--resume-from`, runs final test eval, logs everything to MLflow. |
| `logging_setup.py` | Configures stdlib `logging` for terminal + file output. Routes console writes through `tqdm.write()` so log lines never break active progress bars. |
| `run_training_pipeline.py` | Entry point. Loads `.env`, sets up logging, sets MLflow URI/experiment, wires datasets/loaders/model/optimizer/scheduler, opens an MLflow run, calls `TrainingPipeline.run_pipeline`. |

## Data contract

`data/real/annotations_{train,val,test}.json` are standard COCO JSON.

- `images[*].file_name` is **relative to the repo root** (e.g.
  `data/inaturalist/images/african_buffalo/...jpg`). The dataset resolves it
  via `constants.IMAGE_ROOT / file_name`.
- `annotations[*].bbox` is pixel `[x, y, w, h]` (COCO).
- `categories[*].id` is 1..225; mapped to YOLO indices 0..224 inside the
  dataset (subtract 1).

Splits: 145 809 train / 12 545 val / 63 865 test. 50 of the 225 classes have
zero training annotations — they remain in the head and will simply never be
predicted; the per-class AP table makes this visible.

## Terminal logging

Every meaningful pipeline event is emitted via stdlib `logging` at `INFO`
level. The same stream is written to a per-run log file at
`model_exports/<run_name>/<run_name>.log` and uploaded to MLflow as an artifact
at run end. Per-batch progress is shown as a live `tqdm` bar (with losses in the
postfix); a full-detail log line is also written every
`MLFLOW_LOG_EVERY_N_STEPS` steps so the file has a permanent per-step
record. Format:

```
2026-06-02 16:42:18 INFO  scripts.training.yolov5s.training_pipeline | epoch 03/50 — train start (lr=0.001, batches=9114)
2026-06-02 16:42:21 INFO  scripts.training.yolov5s.training_pipeline | step 50 | box=0.043 obj=0.022 cls=0.012 total=0.077 lr=0.001
...
2026-06-02 17:08:09 INFO  scripts.training.yolov5s.training_pipeline | epoch 03/50 — val mAP50=0.4123 mAP50_95=0.2287
2026-06-02 17:08:09 INFO  scripts.training.yolov5s.training_pipeline | new best mAP50=0.4123 at epoch 3
```

Noisy library loggers (`PIL`, `matplotlib`, `urllib3`, `mlflow.utils`) are
clamped to `WARNING`.

## MLflow logging contract

| When | What |
|---|---|
| Run start | All `constants.py` values + dataset sizes + git SHA + device, as tags + `mlflow.log_params` |
| Every `MLFLOW_LOG_EVERY_N_STEPS` (default 50) | `train/step/loss_{box,obj,cls,total}`, `train/lr` at `step=global_step` |
| End of each epoch | `train/epoch_loss_*`, `val/mAP50`, `val/mAP50_95` at `step=epoch` |
| End of each epoch (if torchmetrics returns per-class) | Per-class AP table as a JSON artifact under `per_class_ap/` (via `log_artifact`, **never** `mlflow.log_table` — per-epoch `log_table` calls overflow the 8000-char `mlflow.loggedArtifacts` run tag after ~137 epochs and kill the run; see `docs/progress_notes/2026-07-13_mlflow-log-table-crash-and-resume.md`). Best-effort: failures are logged, not raised. |
| End of run | `test/mAP50`, `test/mAP50_95`, `<run_name>/best.pt`, `<run_name>/last.pt`, `<run_name>/<run_name>.log` |

Identical contract will be reused by the NanoDet / PicoDet pipelines for
apples-to-apples comparison in the MLflow UI.

## Setup

### 1. Dependencies

`pyproject.toml` already pins `yolov5>=7.0.13`, `torchmetrics>=1.4.0`,
`pycocotools>=2.0.7`, plus `mlflow`, `python-dotenv`, `torch`, etc.

```bash
uv lock && uv sync
```

Or rebuild the Docker image (`make build`).

### 2. Pretrained weights

Download COCO-pretrained YOLOv5s into `weights/yolov5s.pt` at the repo root.

```bash
mkdir -p weights
curl -L https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt \
  -o weights/yolov5s.pt
```

If `weights/yolov5s.pt` is missing the run still works — training simply
starts from random initialization. The shape-mismatched detect-head tensors
are filtered out automatically when the pretrained checkpoint is loaded
(225 classes ≠ 80 COCO classes).

### 3. MLflow credentials

Copy `.env.example` to `.env` (gitignored via the repo-root `.env` pattern)
and fill in real values:

```bash
cp scripts/training/yolov5s/.env.example scripts/training/yolov5s/.env
# then edit it
```

The four variables read at run start:

```
MLFLOW_TRACKING_URI=http://mlflow-server:5000
MLFLOW_TRACKING_USERNAME=...
MLFLOW_TRACKING_PASSWORD=...
MLFLOW_EXPERIMENT_NAME=yolov5s-wildlife225
```

## Running

### Smoke test (1 epoch on val set — wiring check)

```bash
PYTHONPATH=/home/debian/Master-Thesis \
  uv run -m scripts.training.yolov5s.run_training_pipeline --smoke
```

Uses the val split as a tiny train set, `num_workers=0`, 1 epoch.
Use this before any real run to verify dataset paths, MLflow connectivity,
and gradient flow end to end.

### Full run inside Docker

```bash
make yolov5s-train
```

This runs the entry point inside the training container on the
`mlflow-server_default` network so the tracking URI in `.env` is reachable.
The Makefile target mounts the repo + dataset and passes the `.env` via
`--env-file`.

### Full run on the host

```bash
PYTHONPATH=/home/debian/Master-Thesis \
  python -m scripts.training.yolov5s.run_training_pipeline
```

### Resuming a crashed run

`last.pt` is refreshed at the end of every epoch, so a crash at any point
leaves a current resume checkpoint. To continue:

```bash
make yolov5s-train \
  YOLOV5S_ARGS="--resume-from scripts/training/yolov5s/model_exports/<run_name>/last.pt"
```

(or pass `--resume-from` directly to `run_training_pipeline`; `best.pt` works
too — training then restarts from the best epoch instead of the latest one).

Restored state: model weights, EMA copy + update count, optimizer, OneCycleLR
step counter (incl. current LR position), AMP scaler, best metric, and the
early-stop patience counter. Training continues at the checkpoint's `epoch + 1` under a
**new** run dir and a **new** MLflow run (checkpoint path recorded as the
`resume_from` param); the previous `best.pt` is copied into the new run dir
so the final test eval always has a best checkpoint. Caveat: checkpoints hold
the deployable (EMA) weights, so on resume the raw model also restarts from
the EMA weights — a standard, benign approximation.

Outputs land in a per-run directory `model_exports/<run_name>/` (where
`<run_name>` is e.g. `yolov5s-20260602-233434`), so runs never overwrite each
other:

- `scripts/training/yolov5s/model_exports/<run_name>/best.pt` (highest val `SELECTION_METRIC`; EMA weights when `USE_EMA`)
- `scripts/training/yolov5s/model_exports/<run_name>/last.pt` (final epoch)
- `scripts/training/yolov5s/model_exports/<run_name>/<run_name>.log` (full terminal log)
- `scripts/training/yolov5s/model_exports/<run_name>/evaluation/` (only with `--full-eval`)
- All of the above + per-epoch metrics on the MLflow server

The eval (`eval_suite.run_evaluation`) and inference (`scripts.evaluation.run_inference`)
scripts default to the **latest** run dir, or accept `--run-dir` / `--checkpoint`
(`--weights`) to target a specific run.

## Tuning

Every knob lives in `constants.py`. The common ones:

| Constant | Default | Notes |
|---|---|---|
| `EPOCH_COUNT` | 200 | Safety ceiling only — early stopping is expected to end the run. |
| `BATCH_SIZE` | 16 | Tuned for ~8 GB VRAM at 640×640. Reduce if OOM. |
| `NUM_WORKERS` | 8 | |
| `LEARNING_RATE` | 1e-3 | lr0 — 10× lower than scratch training, standard for fine-tuning. |
| `OPTIMIZER` | `"SGD"` | `"AdamW"` also implemented. |
| `WARMUP_EPOCHS` | 3 | Gates early-stop patience counting; also sets `ONE_CYCLE_PCT_START = WARMUP_EPOCHS / EPOCH_COUNT`. |
| `ONE_CYCLE_MAX_LR` | 1e-2 | Peak LR (10× `LEARNING_RATE`). |
| `ONE_CYCLE_DIV_FACTOR` | 10.0 | `initial_lr = max_lr / div_factor = LEARNING_RATE`. |
| `ONE_CYCLE_FINAL_DIV_FACTOR` | 100.0 | `min_lr = initial_lr / final_div_factor = 1e-5`. |
| `SELECTION_METRIC` | `"mAP50_95"` | One metric drives best.pt + early stop. |
| `EARLY_STOP` / `EARLY_STOP_PATIENCE` / `EARLY_STOP_MIN_DELTA` | `True` / 20 / 1e-3 | Stop after N epochs without improvement. |
| `USE_EMA` / `USE_AMP` | `True` / `True` | EMA weights for eval/checkpoint; AMP mixed precision (CUDA only). |
| `IMAGE_SIZE` | 640 | |
| `AUG_*` | all `False` | Augmentation hooks exist but are off for this baseline. |
| `EVAL_CONF_THRES` / `EVAL_IOU_THRES` / `EVAL_MAX_DET` | 0.001 / 0.6 / 300 | NMS params used during validation/test eval. |

## Limitations / known issues

- **Augmentation disabled** by design for the baseline (only letterbox-resize
  is active). Future ablations can flip `AUG_*` flags and implement the
  bodies of the stubs in `transforms.py`.
- **No multi-GPU / DDP**: single GPU only.
- **MegaDetector pseudo-labels**: annotation `source: megadetector` indicates
  the bboxes are MegaDetector predictions, not human-labeled. Acceptable as
  a baseline but worth flagging in thesis methodology.
- **50/225 classes have zero training annotations** — they stay in the head
  (ID stability across the thesis) and simply never get predicted. The
  per-class AP table surfaces this.
