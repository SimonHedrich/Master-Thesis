# YOLOv5s Fine-Tuning Pipeline — Implementation Plan

**Date:** 2026-06-02
**Status:** Plan only — implementation deferred
**Deliverable location:** `docs/plans/2026-06-02_yolov5s-training-pipeline.md` (this content)
**Code location:** `scripts/training/yolov5s/`

---

## 1. Context

The thesis compares several object-detection architectures (YOLOv5s, NanoDet, PicoDet, YOLO-nano variants, etc.) fine-tuned on a 225-class wildlife dataset. To make the comparison clean, **every model must be trained inside the same generic pipeline structure** — same dataloader, same optimizer/scheduler factories, same MLflow logging contract, same evaluation code. Only the model-specific bits (architecture, loss, anchor/NMS logic) come from the upstream model implementation.

A previous attempt (deleted in git history: `1-prepare_yolov5_dataset.py`, `mlflow_yolov5_callback.py`, `hyp.finetune-wildlife.yaml`) relied on the legacy YOLOv5 repo pinned to commit `5cdad89` and used Ultralytics' built-in trainer + a callback. That approach hid most of the training loop and made it hard to swap parts in/out for cross-model comparison.

This plan replaces that with a **custom training loop** that imports only YOLO-specific pieces (model definition, `ComputeLoss`, anchor matching, NMS) from the `yolov5` PyPI package, and owns everything else (dataset, augmentation, optimizer, scheduler, eval, MLflow) as project code under `scripts/training/yolov5s/`.

### Goals
- Reproducible YOLOv5s fine-tuning on `data/real/` (225-class COCO JSON).
- Every hyperparameter (lr, optimizer, scheduler, batch size, epochs, image size, augmentation toggles) lives in one configuration module.
- Detailed MLflow logging (params, per-epoch train/val losses, per-epoch mAP, final test mAP, best checkpoint as artifact).
- Pipeline shape mirrors the user's reference template so future models (NanoDet, PicoDet, …) drop into the same skeleton.

### Non-goals (explicit)
- Data augmentation beyond letterbox-resize. Mosaic / HSV / flip / mixup are **off** for this baseline. Hooks exist but disabled.
- Multi-GPU / DDP training (single GPU first; DDP can be added later).
- Quantization, distillation, export (ONNX/TFLite). Handled by separate downstream stages.
- Resume-from-checkpoint mid-run.

---

## 2. Pre-flight checks (assumptions to verify before coding)

| # | Assumption | How to verify |
|---|---|---|
| A1 | `yolov5` PyPI package exposes `Model`, `ComputeLoss`, `non_max_suppression`, and supports loading `yolov5s.pt` COCO weights | `pip install yolov5 && python -c "from yolov5.models.yolo import Model; from yolov5.utils.loss import ComputeLoss; from yolov5.utils.general import non_max_suppression"` |
| A2 | `file_name` in `data/real/annotations_*.json` resolves relative to repo root `/home/debian/Master-Thesis/` | Already confirmed: sample `file_name = "data/inaturalist/images/african_buffalo/inat_african_buffalo_00271.jpg"` exists under repo root |
| A3 | COCO category IDs are 1..225 contiguous | Confirmed via JSON inspection |
| A4 | A GPU is available (`torch.cuda.is_available()`); falls back to CPU otherwise | Smoke check in `run_training_pipeline.py` |

Add `yolov5` to `pyproject.toml` dependencies if not already present.

---

## 3. Directory layout

```
scripts/training/yolov5s/
├── .env.example                  # template (no secrets) — committed
├── .env                          # actual creds — gitignored, copied by user
├── constants.py                  # all paths + hyperparameters (single source of truth)
├── dataset.py                    # CocoYoloDataset + Dataloader wrapper
├── transforms.py                 # letterbox-resize + (disabled) augmentation hooks
├── yolov5s_model.py              # model / optimizer / scheduler factories
├── loss.py                       # thin wrapper around yolov5.utils.loss.ComputeLoss
├── evaluation.py                 # mAP via torchmetrics + MLflow logging helper
├── training_pipeline.py          # TrainingPipeline class (the loop)
└── run_training_pipeline.py      # entry point — wires MLflow, calls TrainingPipeline
```

Module names mirror the user's reference template so the same skeleton can be copied for NanoDet/PicoDet next.

---

## 4. Module-by-module specification

### 4.1 `.env.example` (and `.env`)

```
MLFLOW_TRACKING_URI=https://mlflow.example.invalid
MLFLOW_TRACKING_USERNAME=
MLFLOW_TRACKING_PASSWORD=
MLFLOW_EXPERIMENT_NAME=yolov5s-wildlife225
```

`.env` is loaded via `python-dotenv` in `run_training_pipeline.py`. **Add `scripts/training/yolov5s/.env` to `.gitignore`** (repo root currently does not gitignore `.env` — flag in implementation).

### 4.2 `constants.py`

One module, three sections: paths, model/optim/scheduler hyperparams, runtime hyperparams.

```python
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]          # /home/debian/Master-Thesis
DATA_ROOT = REPO_ROOT / "data"
ANNOTATIONS_TRAIN = REPO_ROOT / "data/real/annotations_train.json"
ANNOTATIONS_VAL   = REPO_ROOT / "data/real/annotations_val.json"
ANNOTATIONS_TEST  = REPO_ROOT / "data/real/annotations_test.json"
PRETRAINED_WEIGHTS = REPO_ROOT / "weights/yolov5s.pt"    # downloaded once, cached
OUTPUT_DIR = REPO_ROOT / "scripts/training/yolov5s/model_exports"

# Model
NUM_CLASSES   = 225
IMAGE_SIZE    = 640
MODEL_CONFIG  = "yolov5s.yaml"   # ships inside the yolov5 PyPI package

# Optimizer (SGD, YOLOv5 reference defaults)
OPTIMIZER       = "SGD"          # alt: "AdamW"
LEARNING_RATE   = 1e-3           # lr0 for fine-tuning (1/10 of training-from-scratch)
MOMENTUM        = 0.937
WEIGHT_DECAY    = 5e-4

# Scheduler (cosine to lr0 * LRF over EPOCH_COUNT)
LRF             = 0.01           # final lr fraction
WARMUP_EPOCHS   = 3

# Data / loop
EPOCH_COUNT     = 50
BATCH_SIZE      = 16
NUM_WORKERS     = 8
SEED            = 42

# Augmentation toggles (all off for baseline)
AUG_MOSAIC      = False
AUG_HSV         = False
AUG_HFLIP       = False

# Evaluation
EVAL_CONF_THRES = 0.001
EVAL_IOU_THRES  = 0.6
EVAL_MAX_DET    = 300
```

### 4.3 `dataset.py` — `CocoYoloDataset`

A `torch.utils.data.Dataset` that reads a COCO JSON once at init and returns YOLOv5-shaped batches.

**Init responsibilities:**
- `json.load` the annotation file.
- Build `images_by_id: dict[int, image_record]`.
- Build `annotations_by_image_id: dict[int, list[ann]]` (handles images with zero annotations — they appear as background examples).
- Build `class_id_remap: dict[coco_cat_id -> yolo_class_idx]` — COCO ids are 1..225, YOLO needs 0..224. Store the inverse so MLflow / evaluation can log human-readable class names.

**`__getitem__(idx)` returns `(image_tensor, targets, path, shapes)`:**
- `image_tensor`: `float32` CHW, normalized to `[0,1]`, letterboxed to `(IMAGE_SIZE, IMAGE_SIZE)`.
- `targets`: `torch.Tensor` of shape `(N, 6)` per YOLOv5 convention — columns `[batch_idx_placeholder, class, cx, cy, w, h]` in normalized xywh. `batch_idx_placeholder` is filled by `collate_fn`.
- `path`: original image path (for debugging / per-image logging).
- `shapes`: original `(h0, w0)` + `((h_ratio, w_ratio), (pad_w, pad_h))` — needed by evaluation to un-letterbox predictions back to original-image coords for mAP computation.

**Letterbox** in `transforms.py`: resize longest side to 640 keeping aspect ratio, then pad with `114` gray to 640×640. Standard YOLOv5 letterbox math — reimplement in ~25 lines; see `yolov5/utils/augmentations.py::letterbox` as reference but **do not import it** (we own this side of the line).

**`collate_fn(batch)`**: stacks images, concatenates targets with the batch index prepended (so each row becomes `[i, class, cx, cy, w, h]`). This is what `ComputeLoss` expects.

**`Dataloader` wrapper class** (matches template): just constructs and exposes a `torch.utils.data.DataLoader` with `pin_memory=True` and `persistent_workers=True`.

### 4.4 `transforms.py`

- `letterbox(img, new_shape=640, color=(114,114,114))` — pure NumPy/OpenCV, returns `(img, ratio, (dw, dh))`.
- `to_tensor(img)` — HWC uint8 BGR → CHW float32 RGB `[0,1]`.
- Augmentation functions stubbed (`mosaic`, `hsv`, `random_hflip`) but gated behind the `AUG_*` flags in `constants.py`. All flags default `False` per user direction — only resize is active.

### 4.5 `yolov5s_model.py`

Three factories matching the template:

```python
def yolov5s_model(num_classes: int, weights: Path | None) -> tuple[nn.Module, Callable]:
    """Build YOLOv5s. Loads COCO-pretrained weights, replaces detect-head conv layers
    to output (num_classes + 5) * num_anchors channels. Returns (model, preprocess).
    `preprocess` is the letterbox+to_tensor pipeline so inference code can reuse it."""
    from yolov5.models.yolo import Model
    model = Model(cfg=MODEL_CONFIG, ch=3, nc=num_classes)
    if weights and weights.exists():
        ckpt = torch.load(weights, map_location="cpu")
        # filter out detect-head tensors (shape mismatch on nc=225 vs nc=80)
        state = {k: v for k, v in ckpt["model"].state_dict().items()
                 if k in model.state_dict() and model.state_dict()[k].shape == v.shape}
        model.load_state_dict(state, strict=False)
    model.nc = num_classes
    model.hyp = _hyp_dict()  # ComputeLoss reads box/cls/obj gain from here
    return model, preprocess

def model_optimizer(model) -> torch.optim.Optimizer:
    # Three param groups per YOLOv5 reference: weights w/ decay, biases, BN w/o decay
    # mlflow.log_param for OPTIMIZER, LEARNING_RATE, MOMENTUM, WEIGHT_DECAY

def model_scheduler(optimizer) -> torch.optim.lr_scheduler.LambdaLR:
    # Linear warmup for WARMUP_EPOCHS, then cosine decay from lr0 → lr0*LRF
```

The `_hyp_dict()` helper returns the `ComputeLoss` hyperparameters (`box`, `cls`, `obj`, `anchor_t`, etc.) — also surfaced in `constants.py` so they are MLflow-logged.

### 4.6 `loss.py`

Thin wrapper:

```python
from yolov5.utils.loss import ComputeLoss

class YoloLoss:
    def __init__(self, model): self.compute = ComputeLoss(model)
    def __call__(self, preds, targets) -> tuple[Tensor, dict]:
        total, parts = self.compute(preds, targets)
        return total, {"loss_box": parts[0].item(), "loss_obj": parts[1].item(),
                       "loss_cls": parts[2].item(), "loss_total": total.item()}
```

The dict form makes per-component MLflow logging trivial.

### 4.7 `evaluation.py`

```python
@torch.no_grad()
def evaluate(model, data_loader, device, conf_thres, iou_thres) -> dict:
    """Returns {'mAP50': float, 'mAP50-95': float, 'per_class_AP': {name: float}, ...}.
    Uses torchmetrics.detection.MeanAveragePrecision (which already implements COCO AP).
    Iterate loader, run model.eval() forward, apply NMS via yolov5.utils.general.non_max_suppression,
    un-letterbox boxes back to original-image coords using `shapes`, feed into the metric."""

def eval_log_mlflow(result: dict, prefix: str = "val") -> None:
    mlflow.log_metric(f"{prefix}/mAP50", result["mAP50"])
    mlflow.log_metric(f"{prefix}/mAP50-95", result["mAP50-95"])
    # log per-class AP as an MLflow table artifact (CSV)
```

`torchmetrics` is already in the dependency tree via `pytorch-lightning`; verify in `pyproject.toml` and add if missing.

### 4.8 `training_pipeline.py`

```python
class TrainingPipeline:
    def __init__(self, model, loss_fn, optimizer, scheduler,
                 dl_train, dl_val, dl_test, device, epochs, output_dir): ...

    def run_pipeline(self):
        best_map = -1.0
        for epoch in range(self.epochs):
            self._train_one_epoch(epoch)        # logs train/loss_{box,obj,cls,total} per epoch + lr
            val_result = evaluate(...)
            eval_log_mlflow(val_result, prefix="val")
            self.scheduler.step()
            if val_result["mAP50"] > best_map:
                best_map = val_result["mAP50"]
                self._save_checkpoint("best.pt")  # logged as MLflow artifact at end

        test_result = evaluate(self.model, self.dl_test, ...)
        eval_log_mlflow(test_result, prefix="test")
        mlflow.log_artifact(self.output_dir / "best.pt")
        mlflow.log_artifact(self.output_dir / "last.pt")
```

Per-batch logging is wired but throttled (every 50 steps) to avoid MLflow spam. Per-epoch metrics are always logged.

### 4.9 `run_training_pipeline.py`

```python
def training_run():
    set_seed(const.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlflow.log_params({...all constants...})

    model, preprocess = yolov5s_model(const.NUM_CLASSES, const.PRETRAINED_WEIGHTS)
    model.to(device)

    ds_train = CocoYoloDataset(const.ANNOTATIONS_TRAIN, const.DATA_ROOT.parent, image_size=const.IMAGE_SIZE, augment=True)
    ds_val   = CocoYoloDataset(const.ANNOTATIONS_VAL,   const.DATA_ROOT.parent, image_size=const.IMAGE_SIZE, augment=False)
    ds_test  = CocoYoloDataset(const.ANNOTATIONS_TEST,  const.DATA_ROOT.parent, image_size=const.IMAGE_SIZE, augment=False)
    # log dataset sizes + class counts to MLflow

    dl_train, dl_val, dl_test = (Dataloader(ds, ...).get_dataloader() for ds in (ds_train, ds_val, ds_test))

    optimizer = model_optimizer(model)
    scheduler = model_scheduler(optimizer)
    pipeline  = TrainingPipeline(model, YoloLoss(model), optimizer, scheduler,
                                 dl_train, dl_val, dl_test, device,
                                 const.EPOCH_COUNT, const.OUTPUT_DIR)
    pipeline.run_pipeline()

if __name__ == "__main__":
    load_dotenv(Path(__file__).parent / ".env")
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "yolov5s-wildlife225"))
    run_name = f"yolov5s-{datetime.now():%Y%m%d-%H%M%S}"
    with mlflow.start_run(run_name=run_name, tags={"model": "yolov5s", "dataset": "wildlife225", "seed": str(const.SEED)}):
        try:
            training_run()
        except Exception as e:
            mlflow.log_param("error", str(e)); raise
```

---

## 5. MLflow logging contract

| When | What | API |
|---|---|---|
| Run start | All `constants.py` hyperparams + dataset sizes + git SHA | `mlflow.log_params`, `mlflow.set_tags` |
| Every ~50 train steps | `train/loss_{box,obj,cls,total}`, `train/lr` | `mlflow.log_metric(step=global_step)` |
| End of each epoch | `train/epoch_loss_*`, `val/mAP50`, `val/mAP50-95`, `val/precision`, `val/recall` | `mlflow.log_metric(step=epoch)` |
| End of each epoch | Per-class AP table (CSV) | `mlflow.log_table` |
| End of run | `test/mAP50`, `test/mAP50-95`, per-class AP on test | `mlflow.log_metric`, `mlflow.log_table` |
| End of run | `best.pt`, `last.pt`, resolved `constants.py` snapshot, full `requirements.txt` | `mlflow.log_artifact` |

This contract is identical to what NanoDet / PicoDet pipelines will produce later — enables apples-to-apples runs view in MLflow.

---

## 6. Verification plan (end-to-end smoke test)

These steps validate the pipeline before a real multi-day training run.

1. **Dep install & import smoke** (validates A1 from §2):
   ```bash
   pip install yolov5 torchmetrics
   python -c "from yolov5.models.yolo import Model; from yolov5.utils.loss import ComputeLoss; print('ok')"
   ```

2. **Dataset smoke** — load a single batch and visualize:
   ```bash
   python -c "
   from scripts.training.yolov5s.dataset import CocoYoloDataset, Dataloader
   import scripts.training.yolov5s.constants as c
   ds = CocoYoloDataset(c.ANNOTATIONS_VAL, c.DATA_ROOT.parent, c.IMAGE_SIZE)
   print(len(ds)); img, tgt, path, shapes = ds[0]; print(img.shape, tgt.shape, path)
   "
   ```
   Expect `torch.Size([3, 640, 640])` and targets with `cx,cy,w,h ∈ [0,1]`.

3. **One-step forward + backward** — 1 epoch on 32 val images, `EPOCH_COUNT=1`, `BATCH_SIZE=4`, `NUM_WORKERS=0`. Confirms ComputeLoss accepts targets and gradients flow.

4. **MLflow connectivity** — point `.env` at the dev MLflow, run a 1-epoch toy run, confirm params/metrics/artifacts arrive.

5. **Evaluation correctness** — run `evaluate(...)` against val set using the COCO-pretrained model **before** any fine-tuning; mAP@0.5 will be ~0 (COCO classes ≠ wildlife classes) — this verifies the eval plumbing without expecting good numbers. Then run 5 epochs and confirm `val/mAP50` increases monotonically (or near-monotonically).

6. **Full run** — only after 1–5 pass: `EPOCH_COUNT=50`, full splits, submitted on the GPU box.

---

## 7. Files to change / create

**New files** (all under `scripts/training/yolov5s/`):
- `.env.example`
- `constants.py`
- `dataset.py`
- `transforms.py`
- `yolov5s_model.py`
- `loss.py`
- `evaluation.py`
- `training_pipeline.py`
- `run_training_pipeline.py`
- `__init__.py` (empty, makes the directory importable)

**Modified files:**
- `pyproject.toml` — add `yolov5 >= 7.0.13` and confirm `torchmetrics` is present (transitively via `pytorch-lightning` but pin explicitly).
- `.gitignore` — add `scripts/training/yolov5s/.env` and `scripts/training/yolov5s/model_exports/`.
- `Makefile` — add `yolov5s-train` target that runs `python -m scripts.training.yolov5s.run_training_pipeline` inside Docker on the MLflow network.
- `docs/plans/2026-06-02_yolov5s-training-pipeline.md` — copy of this plan (deliverable).

**Reference (do not modify) — for context only:**
- `data/real/annotations_{train,val,test}.json` — COCO inputs.
- `pyproject.toml` — already lists `torch`, `ultralytics`, `mlflow`, `python-dotenv` so most heavy deps are covered.
- Git history `HEAD~N:scripts/training/mlflow_yolov5_callback.py` — previous callback approach, kept as reference only; this plan replaces it.

---

## 8. Open questions / risks

- **`yolov5` PyPI version drift**: pin the version in `pyproject.toml`. If the API changes between versions, the wrapper modules (`loss.py`, `yolov5s_model.py`) localize the blast radius.
- **PyTorch 2.11 compatibility**: pyproject pins `torch >= 2.11`. The `yolov5` PyPI package historically targets PyTorch ≤ 2.x — confirm during the dep-install smoke test. If incompatible, downgrade torch in a dedicated Docker image for this pipeline (precedent: deleted plan `2026-05-19_yolov5-training-plan.md` proposed exactly that).
- **Class imbalance**: 50 of 225 classes have zero training annotations. They will silently never be predicted; mAP averages over all 225 will be depressed. The per-class AP table makes this visible — call out in the thesis methodology section.
- **Annotations from MegaDetector**: bboxes have `source: megadetector` and `conf` fields. They are pseudo-labels, not gold. Acceptable as a baseline but worth noting.
- **`.env` secrets hygiene**: repo root `.env` is currently **not** in `.gitignore`. Implementation must add the new `.env` path and verify it was never committed.
