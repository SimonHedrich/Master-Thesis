# YOLOv5 Training Plan — Wildlife 225-Class Detection

**Date:** 2026-05-19  
**Status:** Planning — not yet implemented  
**Role in thesis:** Phase 1 (Direct Fine-Tuning Baseline), YOLOv5n as additional student comparison point

---

## 1. Purpose & Role in the Thesis

This plan covers fine-tuning YOLOv5n on the 225-class wildlife dataset as the first end-to-end training run. It serves two purposes:

1. **Pipeline validation.** Before running expensive multi-seed experiments on all three students (YOLO11n, NanoDet, PicoDet), a single YOLOv5 run confirms that the full pipeline (dataset format → training → evaluation → MLflow logging) works correctly. YOLOv5 is the simplest YOLO codebase to debug.

2. **Additional baseline.** YOLOv5n (1.9M params, ~3.8MB) slots between the current student models in size. It provides a useful comparison point showing how an older nano architecture performs against YOLO11n on the same data, with no knowledge distillation.

This does **not** replace Phase 1 runs for YOLO11n, NanoDet, or PicoDet — it supplements them.

---

## 2. License Constraint

The YOLOv5 repo must be pinned to commit `5cdad89`. Later commits require an additional commercial license.

```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
git checkout 5cdad89
```

All subsequent paths assume this pinned checkout.

---

## 3. Model Variant

| Variant | Params | FLOPs | COCO mAP@0.5 | Role |
|---------|--------|-------|--------------|------|
| YOLOv5n | 1.9M | 4.5G | 45.7% | **Recommended — student-class comparison** |
| YOLOv5s | 7.2M | 16.5G | 56.8% | Alternative if a small teacher is needed |

**Recommendation: YOLOv5n.** Comparable parameter count to YOLO11n (2.6M) and NanoDet-Plus-m (1.8M), making results directly comparable without confounding architectural differences with size differences.

Pretrained weights: `yolov5n.pt` from the [commit's associated release](https://github.com/ultralytics/yolov5/releases) — COCO-pretrained, used as initialisation for transfer learning.

---

## 4. Environment

YOLOv5 at this commit was designed for PyTorch ≤ 2.0. The main `wildlife-training` Docker image pins `torch>=2.11.0` (CUDA 13.0), which may cause minor API incompatibilities (`torch.cuda.amp` API changes, `model.fuse()` deprecations). A **separate Docker image** is the cleanest solution, consistent with the existing pattern (NanoDet, PaddlePaddle each have their own images).

```dockerfile
# Dockerfile.yolov5
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

RUN pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
      --index-url https://download.pytorch.org/whl/cu118 && \
    pip install -r /opt/yolov5/requirements.txt

# Patches for PyTorch 2.0 compatibility if needed:
# - replace deprecated torch.cuda.amp.autocast with torch.autocast
# - replace model.fuse() calls if they error
```

Add a `make docker-yolov5-build` and `make docker-yolov5-shell` target to the Makefile, following the existing pattern.

---

## 5. Dataset Format

YOLOv5 uses the YOLO TXT format: one `.txt` file per image, each line is:
```
<class_id> <cx> <cy> <w> <h>
```
All values normalised to [0, 1]. A top-level YAML descriptor points to the splits:

```yaml
# data/training/wildlife225_yolov5.yaml
path: /home/debian/Master-Thesis/data/training
train: images/train
val:   images/val
test:  images/test

nc: 225
names:
  # filled in from resources/2026-03-19_student_model_labels.txt
  - squirrel family
  - eastern gray squirrel
  # ... (225 entries)
```

The bounding boxes come from MegaDetector pseudo-labels (already in the pipeline output as normalised `[cx, cy, w, h]` in the filter JSONL). A conversion script is needed to write the per-image `.txt` files.

---

## 6. Class Imbalance Strategy

The dataset has a severe long tail:
- 83 classes with ≥ 1,500 images (head — e.g., eastern gray squirrel: 31k images)
- 29 critical classes with < 100 images after filtering

Without mitigation, the head dominates gradients and tail-class mAP collapses.

### Recommended approach: Square-root resampling at the dataset level

Pre-balance the training set before writing images to disk: each class contributes `min(n_class, ceil(sqrt(n_class) * K))` images, where `K` is tuned so the total dataset stays manageable (~150–200k images). This avoids oversampling rare classes to the point of memorisation while still pulling them up from near-zero.

```python
# Pseudocode for resampling during dataset construction
MAX_PER_CLASS = 1500  # cap on head classes
for cls in classes:
    target = min(MAX_PER_CLASS, int(math.sqrt(len(images[cls])) * scale_factor))
    sampled = random.sample(images[cls], min(target, len(images[cls])))
```

YOLOv5 does not have a built-in per-class sampler for detection, so this must be done during data preparation, not at training time.

### Secondary: label smoothing

Set `label_smoothing: 0.1` in the hyperparameter YAML. With 225 classes, the model is prone to overconfidence on head classes. Label smoothing regularises the classification loss without touching the dataset.

### Do not: use `cls_pw` for class-level reweighting

YOLOv5's `cls_pw` parameter is the positive-weight for the *objectness* BCE loss, not a per-class weight vector. It cannot compensate for class frequency imbalance.

---

## 7. Optimizer

### Decision: **AdamW**

| | SGD (Nesterov) | AdamW |
|--|----------------|-------|
| Default in YOLOv5 | ✓ | — |
| Transfer learning convergence | Slower | Faster |
| Sensitivity to LR choice | High | Low |
| Long-tail behaviour | Unstable for rare classes with high LR | More stable, per-parameter LR |
| Final accuracy (long training) | Slightly better on large datasets from scratch | Comparable when fine-tuning |

**Why AdamW here:**
- We are fine-tuning from COCO pretrained weights, not training from scratch. Transfer learning strongly favours AdamW.
- With 225 classes and severe class imbalance, per-parameter adaptive learning rates help stabilise tail-class gradients.
- Consistent with the NanoDet (`AdamW, lr=0.001`) and modern YOLO configs, making cross-model comparisons cleaner.

Availability at commit `5cdad89`: the `--optimizer` flag was merged into YOLOv5 around v6.2. If the commit predates this, apply the following one-line patch to `train.py`:
```python
# In train.py, replace the optimizer block:
optimizer = torch.optim.AdamW(pg0, lr=hyp['lr0'], weight_decay=hyp['weight_decay'])
optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})
optimizer.add_param_group({'params': pg2})
```

**Parameters:**
```yaml
lr0: 0.001          # initial LR (AdamW — 10× lower than SGD default)
lrf: 0.01           # final LR ratio → final LR = lr0 × lrf = 1e-5
momentum: 0.937     # kept for SGD fallback; ignored by AdamW
weight_decay: 0.0005
```

---

## 8. Learning Rate Schedule

### Decision: **Cosine Annealing** (`--cos-lr`)

Options evaluated:

| Schedule | Behaviour | Verdict |
|----------|-----------|---------|
| Linear decay (default) | LR drops linearly to `lr0 × lrf` | Fine, but abrupt end |
| **Cosine annealing** | Smooth sinusoidal decay to `lr0 × lrf` | **Recommended** |
| OneCycleLR | Fast ramp up + steep decay | Good for short runs, harder to resume |
| Step decay | Discrete drops at fixed epochs | Less principled, requires tuning steps |

Cosine annealing is the current standard for YOLO fine-tuning, is available via `--cos-lr` in YOLOv5, and is consistent with the NanoDet config (`CosineAnnealingLR`). It eliminates the need to hand-tune decay step epochs.

**Warmup:** 3 epochs linear warmup from `lr0 × warmup_bias_lr / lr0` to `lr0`. YOLOv5 implements this by default.

```yaml
warmup_epochs: 3
warmup_momentum: 0.8
warmup_bias_lr: 0.1
```

---

## 9. Batch Size and Input Resolution

### Input resolution: **640 × 640**

| Resolution | Pros | Cons |
|------------|------|------|
| 416 × 416 | Faster training, lower VRAM | Misses small/distant animals |
| **640 × 640** | Better detection of small targets, standard | More VRAM, slower |
| 1280 × 1280 | Best small-object detection | ~3× VRAM, very slow |

Wildlife images frequently contain small animals at range (e.g., deer 50m away). 640 is the right tradeoff. The AX Visio binocular also captures high-resolution imagery where small-target detection matters.

### Batch size: **64** (nominal), accumulate to **128** if VRAM allows

On a single A40 (12GB VRAM):
- YOLOv5n at 640: ~4.5GB for bs=64 → fits comfortably
- YOLOv5s at 640: ~7GB for bs=64 → fits

YOLOv5 uses a nominal batch size (`nbs=64`) for gradient scaling. If actual batch fits in VRAM, use `--batch-size 64` directly; if not, halve it and set `--accumulate 2`.

### Multi-scale training: **enabled** (`--multi-scale`)

Randomly resizes images ±50% of target size each batch (480–800 px). This trains the model to detect animals at different scales, directly matching the field use case where animals appear at varying distances. Adds ~10–15% training time overhead; worth it.

---

## 10. Training Duration and Early Stopping

**Epochs:** 150  
**Early stopping:** patience = 50 epochs (stop if no improvement in mAP@0.5:0.95 on validation set for 50 consecutive epochs)

With ~150–200k training images and AdamW at lr0=0.001, convergence typically happens around epoch 80–120 for transfer learning. Epoch 150 gives headroom without over-training on tail classes.

```bash
--epochs 150 --patience 50
```

Best checkpoint (`best.pt`) is saved by `mAP@0.5:0.95` on the validation set — this metric is more discriminating than `mAP@0.5` and better reflects localization quality.

---

## 11. Augmentation

YOLOv5's default augmentation (`hyp.scratch-low.yaml`) is designed for training from scratch. For fine-tuning, some defaults should be reduced.

Recommended `hyp.finetune-wildlife.yaml`:

```yaml
# --- Augmentation ---
hsv_h: 0.015       # hue jitter — keep default (lighting conditions vary outdoors)
hsv_s: 0.7         # saturation jitter — keep default
hsv_v: 0.4         # brightness jitter — keep default (dawn/dusk images)
degrees: 0.0        # rotation — wildlife images are upright; rotation artifacts harm boxes
translate: 0.1      # translation — moderate
scale: 0.5          # scale — important for multi-distance detection
shear: 0.0          # shear — upright images only
perspective: 0.0    # perspective warp — introduces artifacts, disable
flipud: 0.0         # vertical flip — animals are gravity-bound; disable
fliplr: 0.5         # horizontal flip — valid for wildlife (animals face both ways)
mosaic: 1.0         # 4-image mosaic — keep; provides implicit multi-scale + context variety
mixup: 0.0          # MixUp — disable for baseline; can add in ablation
copy_paste: 0.0     # copy-paste — disable; requires instance masks

# --- Loss ---
box: 0.05           # box regression loss gain
cls: 0.5            # classification loss gain (scales internally by nc/80 ≈ 2.8×)
obj: 1.0            # objectness loss gain
label_smoothing: 0.1  # prevents overconfidence on head classes with 225-class vocab

# --- Optimizer ---
lr0: 0.001
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3
warmup_momentum: 0.8
warmup_bias_lr: 0.1
```

**Note on mosaic:** Mosaic composites four images into one, effectively giving the model mixed-class scenes with varied lighting. This is particularly valuable for tail classes, since it multiplies the effective number of contexts in which rare animals appear.

**Note on cls loss scaling:** YOLOv5 internally multiplies `cls` by `nc / 80`. For nc=225 this gives an effective weight of ~1.41. Monitor classification loss vs. box/obj loss in the first few epochs; if cls dominates, reduce `cls` from 0.5 to 0.3.

---

## 12. Training Command

```bash
python train.py \
  --weights yolov5n.pt \
  --cfg models/yolov5n.yaml \
  --data /home/debian/Master-Thesis/data/training/wildlife225_yolov5.yaml \
  --hyp /home/debian/Master-Thesis/scripts/training/configs/hyp.finetune-wildlife.yaml \
  --epochs 150 \
  --batch-size 64 \
  --imgsz 640 \
  --patience 50 \
  --cos-lr \
  --multi-scale \
  --label-smoothing 0.1 \
  --project /home/debian/Master-Thesis/output/yolov5_wildlife \
  --name yolov5n_wildlife225_v1 \
  --exist-ok \
  --workers 8 \
  --seed 42
```

Repeat with `--seed 1` and `--seed 7` for the three-seed statistical requirement (see Section 4.5 of the experimentation plan).

---

## 13. MLflow Logging

The main `wildlife-training` Docker image already includes `mlflow>=3.0.0`. For YOLOv5 (which uses its own W&B/CSV logger), add an MLflow callback wrapper:

```python
# scripts/training/mlflow_yolov5_callback.py
import mlflow

class MLflowCallback:
    """Thin adapter from YOLOv5's on_train_batch_end / on_val_end hooks to MLflow."""
    def __init__(self, run_name: str, tags: dict):
        mlflow.start_run(run_name=run_name, tags=tags)

    def on_fit_epoch_end(self, vals, epoch, best_fitness, fi):
        # vals = [train/box_loss, train/obj_loss, train/cls_loss,
        #         metrics/precision, metrics/recall, metrics/mAP50, metrics/mAP50-95,
        #         val/box_loss, val/obj_loss, val/cls_loss]
        keys = ["train/box_loss", "train/obj_loss", "train/cls_loss",
                "precision", "recall", "mAP@0.5", "mAP@0.5:0.95",
                "val/box_loss", "val/obj_loss", "val/cls_loss"]
        mlflow.log_metrics({k: v for k, v in zip(keys, vals)}, step=epoch)

    def on_train_end(self, last, best, epoch, results):
        mlflow.log_artifact(str(best))
        mlflow.end_run()
```

Alternatively, YOLOv5 natively supports W&B logging (`--logger wandb`) if the project uses that instead.

---

## 14. Evaluation After Training

After `best.pt` is saved, run the standard YOLOv5 evaluation:

```bash
python val.py \
  --weights output/yolov5_wildlife/yolov5n_wildlife225_v1/weights/best.pt \
  --data data/training/wildlife225_yolov5.yaml \
  --imgsz 640 \
  --batch-size 32 \
  --task test \
  --verbose \
  --save-json   # saves COCO-format predictions for per-class AP analysis
```

Key metrics to record for the experiment comparison matrix (Section 4.3 of the plan):
- `mAP@0.5`, `mAP@0.5:0.95`, macro-F1
- Per-class AP → bin into head / middle / tail tiers for long-tail analysis

---

## 15. Open Questions Before Implementation

| Question | Options | Recommended |
|----------|---------|-------------|
| Is `--optimizer AdamW` available at commit `5cdad89`? | (a) available natively, (b) requires one-line patch | Check and patch if needed |
| Should the dataset cap head classes at 1,500 or higher? | 1,500 (per Ultralytics guideline), 3,000, uncapped | 1,500 for first run (matches coverage report baseline) |
| Use synthetic images for critical-tier classes in this run? | Yes (add 50 synthetic/class), No (real only first) | Yes — helps diagnose whether synthetic data hurts or helps |
| Train YOLOv5s in parallel as a teacher? | Yes, same config with larger `--weights yolov5s.pt` | Defer until YOLOv5n baseline is validated |
| Freeze backbone for first N epochs? | 0 (fine-tune all), 10 (freeze backbone), 3 | 0 — dataset large enough to avoid forgetting |

---

## 16. Expected Timeline

| Step | Estimated time |
|------|----------------|
| Dataset format conversion (JSONL → YOLO TXT) | 2–4h |
| Docker image setup + compatibility patches | 1–2h |
| First training run (1 seed, 150 epochs, A40) | ~2–3h |
| Evaluation + MLflow logging | 30min |
| Three-seed replication | ~6–9h (parallel or sequential) |
