# Progress Notes – 03.06.2026

## YOLOv5s Fine-Tuning Pipeline

**Context:** First model-specific training pipeline built on top of the
infrastructure described in
[`2026-04-24_training-setup-and-model-smoke-test.md`](./2026-04-24_training-setup-and-model-smoke-test.md).
The pipeline structure is intentionally generic so the same skeleton
(`constants.py` / `dataset.py` / `<model>.py` / `loss.py` / `evaluation.py`
/ `training_pipeline.py` / `run_training_pipeline.py` /
`logging_setup.py`) can be reused for NanoDet, PicoDet, and the YOLO-nano
variants when their pipelines are added next.

Plan document: [`docs/plans/2026-06-02_yolov5s-training-pipeline.md`](../plans/2026-06-02_yolov5s-training-pipeline.md).
Code: `scripts/training/yolov5s/`.

---

## 1. Design Decisions

| Question | Decision | Rationale |
|---|---|---|
| Where does YOLOv5-specific code come from? | `yolov5` PyPI package (≥7.0.13) | Avoids the commit-pin (`5cdad89`) of the earlier attempt. We import only `Model`, `ComputeLoss`, and `non_max_suppression`; everything else is ours. |
| How much of the training loop do we own? | All of it. | Cross-model comparability requires identical dataloader / optimizer / scheduler / eval / MLflow logging. The Ultralytics built-in trainer hides too much. |
| Dataset preprocessing | Read COCO JSON at runtime, convert to YOLO targets in `__getitem__`. | No on-disk preprocessing step. Images live across `data/wikimedia/`, `data/inaturalist/`, etc.; the dataset resolves `file_name` against the repo root. |
| Class universe | All 225 classes in the detection head, even the 50 with zero training annotations. | Stable class IDs across thesis. Per-class AP table surfaces empty classes. |
| Pretrained weights | COCO-pretrained `yolov5s.pt`. Detect-head tensors are shape-filtered out automatically (225 ≠ 80). | Standard transfer-learning start. Run still works without the file (random init). |
| Image size | 640 × 640 | YOLOv5s default. Comparable with published numbers; embedded-size ablations can come later. |
| Augmentation | All off (`AUG_MOSAIC=False` etc.). Letterbox-resize only. | Baseline simplicity. Hooks exist; flipping flags is a one-line change later. |
| Logging | stdlib `logging` + `tqdm` progress bars, terminal + per-run log file uploaded to MLflow. | Matches the verbosity required for hours-long training runs without adding a new dependency. |

---

## 2. Package Layout

```
scripts/training/yolov5s/
├── .env.example           # MLflow URI / user / password / experiment
├── constants.py           # Single source of truth — paths, hyp, loop config
├── transforms.py          # letterbox + to_tensor (augmentation stubs disabled)
├── dataset.py             # CocoYoloDataset + collate_fn + Dataloader wrapper
├── yolov5s_model.py       # model / optimizer (3 param groups) / scheduler factories
├── loss.py                # YoloLoss → wraps yolov5.utils.loss.ComputeLoss
├── evaluation.py          # mAP via torchmetrics; un-letterbox preds before metric
├── training_pipeline.py   # TrainingPipeline class — train, val, test, save best/last
├── logging_setup.py       # TqdmLoggingHandler + setup_logging(log_file, level)
├── run_training_pipeline.py  # Entry point — loads .env, configures MLflow + logging
└── README.md              # Per-package documentation
```

Module module names mirror the reference template so the next models drop
into the same skeleton — only `<model>.py` and `loss.py` change.

---

## 3. Dataset Contract

| Field | Value |
|---|---|
| Source files | `data/real/annotations_{train,val,test}.json` |
| Format | Standard COCO JSON |
| Image-path resolution | `file_name` is repo-root-relative, e.g. `data/inaturalist/images/african_buffalo/...jpg` |
| BBox format | Pixel `[x, y, w, h]` (COCO) → normalized `[cx, cy, w, h]` in YOLO target tensor |
| Category IDs | COCO 1..225 → YOLO 0..224 (subtract 1 internally) |
| Train / Val / Test | 145 809 / 12 545 / 63 865 images |
| Per-image targets | `Tensor[(N, 6)]` columns `[batch_idx, cls, cx, cy, w, h]`; `batch_idx` filled by `collate_fn` |

Letterbox math is mirrored on both sides: the dataset applies it to the
ground-truth boxes, the evaluation function un-letterboxes the predictions
back to original-image coords before passing them to torchmetrics.

---

## 4. Training Loop & MLflow Logging Contract

| Phase | What gets logged |
|---|---|
| Run start | All `constants.py` values + dataset sizes + git SHA + device, via `mlflow.log_params`. Identical dump written to terminal at INFO level. |
| Every `MLFLOW_LOG_EVERY_N_STEPS` train steps (default 50) | `train/step/loss_{box,obj,cls,total}`, `train/lr`, plus a full INFO log line |
| End of each train epoch | `train/epoch_loss_*` at `step=epoch`; INFO line `epoch i/N — train done in Xs | avg ...` |
| End of each val epoch | `val/mAP50`, `val/mAP50_95`; per-class AP table (when present); INFO line |
| End of run | `test/mAP50`, `test/mAP50_95`, `best.pt`, `last.pt`, `<run_name>.log` uploaded as artifacts |

Identical contract will be reused by NanoDet / PicoDet — the MLflow runs
view becomes apples-to-apples without extra adapter code.

---

## 5. Terminal Logging (added on top)

Pipeline events are emitted via stdlib `logging` with this format:

```
2026-06-03 11:42:18 INFO  scripts.training.yolov5s.training_pipeline | epoch 03/50 — train start (lr=0.001, batches=9114)
2026-06-03 11:42:21 INFO  scripts.training.yolov5s.training_pipeline | step 50 | box=0.043 obj=0.022 cls=0.012 total=0.077 lr=0.001
```

Implementation details:

- **Console handler** is a custom `TqdmLoggingHandler` that calls
  `tqdm.write(msg)` rather than writing to stderr directly. This prevents
  log lines from scribbling over the active per-batch progress bar.
- **File handler** mirrors every line to
  `scripts/training/yolov5s/model_exports/<run_name>.log`, which is
  uploaded to MLflow as an artifact at run end (covers clean exits and
  exceptions thanks to the `finally: logging.shutdown()` block).
- Noisy library loggers (`PIL`, `matplotlib`, `urllib3`, `mlflow.utils`)
  are clamped to `WARNING`.

Per-batch verbosity is `tqdm` bar **and** a full INFO log line every N
steps — best-of-both: live progress for the operator, permanent record
for post-hoc analysis.

---

## 6. Dependency Changes

`pyproject.toml` gained three pins:

```toml
"torchmetrics>=1.4.0",
"pycocotools>=2.0.7",       # required transitively by torchmetrics' MAP
"yolov5>=7.0.13",
```

`mlflow`, `python-dotenv`, `tqdm`, `pandas`, `opencv-python-headless`,
`torch>=2.11.0` were already pinned. No new Docker image needed —
everything fits inside the existing `wildlife-training` image.

---

## 7. Verification

Smoke runs performed during development (Python 3.13.5 + PyTorch 2.11.0
+ CUDA 13.0 on the RTX 3060):

| Check | Result |
|---|---|
| `yolov5.models.yolo.Model(cfg=yolov5s.yaml, nc=225)` builds | ✓ 7.63 M params, 214 layers, 17.9 GFLOPs |
| `CocoYoloDataset(val)` loads | ✓ 12 545 images, 225 classes |
| `__getitem__` produces tensors | ✓ `(3, 640, 640) float32 [0,1]`; targets in `[0,1]` |
| One full train step (forward + loss + backward + step) | ✓ `loss_total ≈ 0.146` on random init |
| Optimizer param-group split | ✓ g0(BN)=57, g1(conv)=60, g2(bias)=60 |
| Eval pass (NMS + un-letterbox + torchmetrics) | ✓ runs end-to-end; mAP≈0 with random init |
| Per-run `.log` file written and uploaded to MLflow | ✓ |
| tqdm progress bar + log line interleaving | ✓ log lines appear above bar, no scribble |

Two bugs surfaced and were fixed during smoke testing:

1. `torch.load(weights_only=True)` (default since PyTorch 2.6) cannot
   deserialize the pickled `Model` inside `yolov5s.pt`. Fixed by passing
   `weights_only=False`.
2. `torchmetrics.MeanAveragePrecision.map_per_class` returns a scalar
   `tensor(-1.0)` when no classes have detections; `enumerate(float)`
   then crashes. Fixed with an `ndim == 1` guard before treating the
   value as a list.

---

## 8. Running

`.env` must exist at `scripts/training/yolov5s/.env` (copy from
`.env.example`). The repo-root `.gitignore` already covers any nested
`.env` via the bare `.env` pattern.

**Smoke test (1 epoch on val split, wiring check)**:

```bash
uv run -m scripts.training.yolov5s.run_training_pipeline --smoke
```

**Full run via Docker (recommended)**:

```bash
make yolov5s-train
```

`yolov5s-train` mounts the repo + dataset, joins the
`mlflow-server_default` network so the tracking URI in `.env` resolves,
and runs the entry point. Outputs land in
`scripts/training/yolov5s/model_exports/` (`best.pt`, `last.pt`,
`<run_name>.log`).

---

## 9. Known Limitations

- Single-GPU only; no DDP.
- No resume-from-checkpoint. A crashed run restarts from epoch 0.
- Augmentation off (only letterbox-resize). Ablations later.
- 50 / 225 classes have zero training annotations — the head still has
  them; they simply never get predicted and depress the macro-mAP. The
  per-class AP artifact makes this visible.
- BBoxes from MegaDetector are pseudo-labels (`source: megadetector`).
  Acceptable as a baseline; flag in the thesis methodology.
- Pretrained weights need to be downloaded manually to
  `weights/yolov5s.pt`. Without the file the run still works but starts
  from random init.

---

## 10. Next Steps

1. **First real training run** on the GPU box with the MLflow server up.
2. **NanoDet pipeline**: copy `scripts/training/yolov5s/` → `scripts/training/nanodet/`, replace only `<model>.py` + `loss.py`. The dataset, training loop, eval, and logging modules transfer untouched.
3. **PicoDet pipeline**: same pattern, inside the dedicated
   `wildlife-paddle` Docker image.
4. **Augmentation ablation**: flip the `AUG_*` flags in `constants.py` and
   implement the stubs in `transforms.py`. Use a second MLflow experiment
   so the runs view stays clean.
5. **Resume-from-checkpoint**: needed before the first multi-day run.
