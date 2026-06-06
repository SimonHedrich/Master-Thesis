# YOLOv5s Training Pipeline — Overview

Custom fine-tuning pipeline for YOLOv5s on the 225-class wildlife dataset. Rather than
using the built-in `yolov5.train()` CLI, the pipeline owns its dataset, training loop,
evaluation, and logging so the same structure can be reused for the NanoDet and PicoDet
baseline runs, enabling apples-to-apples comparison in MLflow.

> Developer reference (setup, running, tuning, limitations):
> `scripts/training/yolov5s/README.md`

---

## Architecture Overview

```
                         ┌─────────────────────────────────────┐
                         │         run_training_pipeline.py     │
                         │  (entry point)                       │
                         │  • load .env → MLflow credentials    │
                         │  • set_seed(42)                      │
                         │  • open MLflow run                   │
                         │  • wire all components               │
                         │  • call TrainingPipeline.run()       │
                         └──────────────┬──────────────────────┘
                                        │
              ┌─────────────────────────┼──────────────────────────┐
              ▼                         ▼                          ▼
   ┌─────────────────┐       ┌─────────────────┐        ┌─────────────────┐
   │   dataset.py    │       │ yolov5s_model.py │        │    loss.py      │
   │                 │       │                  │        │                 │
   │ CocoYoloDataset │       │ • load yolov5s   │        │ YoloLoss wraps  │
   │ (×3 splits)     │       │   .yaml config   │        │ ComputeLoss     │
   │                 │       │ • load pretrained│        │ → (total, parts)│
   │ COCO JSON       │       │   COCO weights   │        └────────┬────────┘
   │ → letterbox     │       │ • swap 80-class  │                 │
   │ → YOLO targets  │       │   head → 225     │                 │
   │ → batch collate │       │ • model_optimizer│                 │
   └────────┬────────┘       │ • model_scheduler│                 │
            │                └────────┬─────────┘                 │
            │                         │                           │
            └────────────┬────────────┘                           │
                         ▼                                        │
              ┌─────────────────────┐                             │
              │  training_pipeline  │◄────────────────────────────┘
              │  TrainingPipeline   │
              │                     │
              │  for epoch 1..50:   │
              │  ├─ train 1 epoch   │────► evaluation.py
              │  ├─ evaluate val    │      • NMS + un-letterbox
              │  ├─ log to MLflow   │      • torchmetrics mAP
              │  ├─ step scheduler  │      • eval_log_mlflow
              │  └─ save best.pt    │
              │                     │
              │  after all epochs:  │
              │  ├─ save last.pt    │
              │  └─ eval test set   │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │      Outputs        │
              │  model_exports/     │
              │  ├─ best.pt         │
              │  ├─ last.pt         │
              │  └─ <run>.log       │
              │                     │
              │  MLflow run         │
              │  ├─ all params      │
              │  ├─ train losses    │
              │  ├─ val/test mAP    │
              │  └─ artifacts       │
              └─────────────────────┘
```

Supporting modules used throughout:

```
constants.py      ← all paths and hyperparameters (imported everywhere)
transforms.py     ← letterbox + to_tensor (used by dataset.py)
logging_setup.py  ← tqdm-aware logger (used by run_training_pipeline.py)
```

---

## Module Responsibilities

| File | Responsibility | Key Public API |
|------|---------------|----------------|
| `constants.py` | Single source of truth for every path and hyperparameter. `as_dict()` dumps all values to MLflow at run start. | `REPO_ROOT`, `NUM_CLASSES`, `EPOCH_COUNT`, `BATCH_SIZE`, `LEARNING_RATE`, `as_dict()` |
| `transforms.py` | Image preprocessing: aspect-ratio preserving resize + tensor conversion. Augmentation stubs exist but are disabled. | `letterbox(img, new_shape)`, `to_tensor(img)` |
| `dataset.py` | Reads COCO JSON, converts to YOLO-format targets, batches them. | `CocoYoloDataset`, `collate_fn`, `Dataloader` |
| `yolov5s_model.py` | Factories for model, optimizer, and scheduler. Handles pretrained weight loading with class-count mismatch. | `yolov5s_model()`, `model_optimizer()`, `model_scheduler()` |
| `loss.py` | Thin wrapper around `yolov5.utils.loss.ComputeLoss`. Extracts individual loss components for MLflow logging. | `YoloLoss(model)`, `__call__(preds, targets) → (loss, parts_dict)` |
| `evaluation.py` | Runs NMS, reverses letterbox padding, computes COCO mAP via torchmetrics, logs per-class AP tables. | `evaluate(model, loader, ...)`, `eval_log_mlflow(result, prefix, step)` |
| `training_pipeline.py` | Orchestrates the epoch loop: trains, validates, checkpoints, logs, and runs final test eval. | `TrainingPipeline`, `run_pipeline()` |
| `logging_setup.py` | Configures stdlib `logging` for terminal + file output without breaking tqdm progress bars. | `setup_logging(log_file, level)` |
| `run_training_pipeline.py` | CLI entry point. Bootstraps everything (env, seeds, MLflow, datasets, model) and calls `run_pipeline()`. | `--smoke` flag, `training_run()` |

---

## Process Walkthrough

### 1. Bootstrap (run_training_pipeline.py)

```
python -m scripts.training.yolov5s.run_training_pipeline [--smoke]
```

- Load `.env` → set `MLFLOW_TRACKING_URI`, credentials, experiment name
- `set_seed(42)` → seeds Python, NumPy, PyTorch, CUDA for reproducibility
- Configure `logging` → console (tqdm-safe) + `model_exports/<run_name>.log`
- Open `mlflow.start_run()` with tags: `model=yolov5s`, `dataset=wildlife225`, `git_sha`
- Log all `constants.as_dict()` values as MLflow params

In `--smoke` mode: uses the val split as train, 1 epoch, `num_workers=0`. Used for
wiring validation before launching a real run.

### 2. Dataset Construction (dataset.py + transforms.py)

Three `CocoYoloDataset` instances are created — one per split:

```
data/real/annotations_train.json  →  145,809 images
data/real/annotations_val.json    →   12,545 images
data/real/annotations_test.json   →   63,865 images
```

Per `__getitem__`:
1. `cv2.imread()` loads the image
2. `letterbox()` resizes to 640×640 preserving aspect ratio, pads gray (114, 114, 114)
3. COCO bbox `[x, y, w, h]` → YOLO normalized `[cx, cy, w, h]` with letterbox offset applied
4. Bboxes clamped to image boundary; zero-size boxes dropped
5. Returns `(image_tensor, targets[N, 6], path, shapes)` — column 0 of targets is the batch-index placeholder

`collate_fn` fills in the batch indices when PyTorch batches samples together, producing
a single `[total_objects, 6]` target tensor with `[batch_idx, cls, cx, cy, w, h]` columns.

### 3. Model Initialization (yolov5s_model.py)

```python
model, preprocess = yolov5s_model(num_classes=225, weights="weights/yolov5s.pt", device=device)
```

1. Load `yolov5s.yaml` architecture from the yolov5 PyPI package → build model with `nc=225`
2. Load `weights/yolov5s.pt` (COCO-pretrained, 80 classes)
3. Filter out shape-mismatched keys (the detection head tensors differ: 80 vs. 225 classes)
4. Load matching backbone + neck weights; detection head initializes from scratch

Optimizer uses **3 parameter groups** following YOLOv5 convention:
- `g0`: BatchNorm weights — no weight decay
- `g1`: Conv weights — with `weight_decay=5e-4`
- `g2`: Biases — no weight decay

Scheduler: **linear warmup** for 3 epochs → **cosine decay** from `lr=1e-3` down to
`lr=1e-5` (`lr0 × LRF=0.01`) over the remaining 47 epochs.

### 4. Training Loop (training_pipeline.py)

For each epoch (1 → 50):

```
┌─ _train_one_epoch(epoch) ──────────────────────────────────────────┐
│  for batch in train_loader:                                         │
│    preds = model(images)                 # forward pass             │
│    loss, parts = YoloLoss(preds, targets) # box + obj + cls        │
│    loss.backward()                        # compute gradients       │
│    optimizer.step(); optimizer.zero_grad()                          │
│    every 50 steps: log step losses + LR to MLflow + log file       │
│  return epoch-averaged loss dict                                    │
└─────────────────────────────────────────────────────────────────────┘
```

Loss components logged at every step:

| Component | Loss function | Measures |
|-----------|--------------|---------|
| `loss_box` | CIoU | Bounding box coordinate accuracy |
| `loss_obj` | BCE | Whether each grid cell contains an object |
| `loss_cls` | BCE | Which species class is present |
| `loss_total` | Weighted sum | What the optimizer minimizes |

### 5. Validation (evaluation.py)

After each training epoch:

```
┌─ evaluate(model, val_loader, conf_thres=0.001, iou_thres=0.6) ─────┐
│  for batch in val_loader:                                            │
│    preds = model(images)          # forward (eval mode, no grad)    │
│    preds = NMS(preds, conf, iou)  # filter overlapping boxes        │
│    preds = un_letterbox(preds)    # remove padding, rescale         │
│    targets = un_letterbox(targets)                                  │
│    metric.update(preds, targets)  # torchmetrics accumulation       │
│  return {mAP50, mAP50_95, per_class_ap, class_names}               │
└─────────────────────────────────────────────────────────────────────┘
```

The un-letterboxing step is critical: predictions and targets must be in original
image coordinates (not 640×640 padded space) for bounding box overlap to be
computed correctly.

### 6. Checkpoint Selection & LR Update

```python
if val_mAP50 > best_mAP50:
    best_mAP50 = val_mAP50
    save_checkpoint("best.pt")   # only updated on improvement

scheduler.step()                 # advance cosine decay
```

### 7. Post-Training Test Evaluation & Artifact Upload

After epoch 50:
1. Load `best.pt` is already saved; `last.pt` is saved from final epoch state
2. Run `evaluate()` on the held-out test set (63,865 images, never seen during training)
3. Log `test/mAP50` and `test/mAP50_95` to MLflow
4. Upload `best.pt`, `last.pt`, `<run_name>.log`, and per-epoch per-class AP tables
   as MLflow artifacts

---

## Data Contract

### Inputs

| Item | Path | Format |
|------|------|--------|
| Train annotations | `data/real/annotations_train.json` | COCO JSON |
| Val annotations | `data/real/annotations_val.json` | COCO JSON |
| Test annotations | `data/real/annotations_test.json` | COCO JSON |
| Images | `data/inaturalist/images/<species>/...` | JPEG/PNG, paths relative to repo root |
| Pretrained weights | `weights/yolov5s.pt` | PyTorch checkpoint (COCO, 80 classes) |
| MLflow credentials | `scripts/training/yolov5s/.env` | `KEY=VALUE` |

COCO JSON conventions:
- `images[*].file_name` — path relative to repo root
- `annotations[*].bbox` — pixel `[x, y, w, h]`
- `categories[*].id` — 1..225 (mapped to YOLO 0..224 in dataset)

### Outputs

| Item | Path | Contents |
|------|------|---------|
| Best checkpoint | `model_exports/best.pt` | `{state_dict, epoch}` — highest val mAP50 |
| Last checkpoint | `model_exports/last.pt` | `{state_dict, epoch}` — final epoch |
| Run log | `model_exports/<run_name>.log` | Full terminal output |
| MLflow run | remote server | Params, metrics, artifacts (see below) |

---

## MLflow Logging Reference

| When | Key(s) | Type |
|------|--------|------|
| Run start | All `constants.py` values + dataset sizes + git SHA + device | params |
| Every 50 steps | `train/step/loss_{box,obj,cls,total}`, `train/lr` | metrics (step=global_step) |
| End of each epoch | `train/epoch_loss_{box,obj,cls,total}` | metrics (step=epoch) |
| End of each epoch | `val/mAP50`, `val/mAP50_95` | metrics (step=epoch) |
| End of each epoch | `val_per_class_ap_step<epoch>.json` | artifact |
| End of run | `test/mAP50`, `test/mAP50_95` | metrics |
| End of run | `test_per_class_ap_stepfinal.json` | artifact |
| End of run | `best.pt`, `last.pt`, `<run_name>.log` | artifacts |

---

## Key Hyperparameters

All live in `constants.py` — one edit, consistently applied everywhere.

| Constant | Default | Notes |
|----------|---------|-------|
| `EPOCH_COUNT` | 50 | |
| `BATCH_SIZE` | 16 | Tuned for ~8 GB VRAM at 640×640 |
| `NUM_WORKERS` | 8 | Parallel data loading |
| `LEARNING_RATE` | `1e-3` | 10× lower than scratch training (fine-tuning) |
| `OPTIMIZER` | `"SGD"` | `"AdamW"` also implemented |
| `WARMUP_EPOCHS` | 3 | Linear warmup, then cosine to `lr0 × LRF` |
| `LRF` | `0.01` | Final LR fraction (cosine end point) |
| `IMAGE_SIZE` | 640 | Letterbox target resolution |
| `NUM_CLASSES` | 225 | Wildlife species |
| `EVAL_CONF_THRES` | `0.001` | NMS confidence cutoff during eval |
| `EVAL_IOU_THRES` | `0.6` | NMS IoU overlap threshold |
| `EVAL_MAX_DET` | 300 | Max detections per image |
| `AUG_*` | `False` | All augmentations disabled for baseline |
| `SEED` | 42 | Reproducibility |
| `MLFLOW_LOG_EVERY_N_STEPS` | 50 | Step-level logging frequency |

---

## Limitations

- **Augmentation disabled** — only letterbox-resize is active for this baseline run.
  Stubs in `transforms.py` exist for mosaic, HSV jitter, and horizontal flip; they can
  be enabled via `AUG_*` flags and implemented without changing any other module.
- **Single GPU only** — no DDP/multi-GPU support.
- **No resume from checkpoint** — a crashed run restarts from epoch 0.
- **50 of 225 classes have zero training annotations** — they remain in the detection
  head for ID stability and simply never get predicted. The per-class AP table in
  MLflow makes these visible.
- **MegaDetector pseudo-labels** — annotations tagged `source: megadetector` are
  MegaDetector predictions, not human-verified bboxes. Acceptable for a baseline;
  flagged in thesis methodology.
