# Progress Notes – 24.04.2026

## Training Environment Setup & Model Smoke Test

**Context:** With the dataset pipeline still running (Wikimedia Commons download in progress), this note covers the training infrastructure needed for the next phase. It follows directly from the experimental design in [`2026-03-18_speciesnet-pipeline-and-experiment-design.md`](./2026-03-18_speciesnet-pipeline-and-experiment-design.md), specifically the proposed ladder: teacher fine-tuning → knowledge distillation → quantization.

---

## 1. Training Environment

### Host System

| Component | Version |
|-----------|---------|
| NVIDIA Driver | 595.45.04 |
| CUDA (host toolkit) | 13.2.0 |
| PyTorch | 2.11.0+cu130 |
| torchvision | 0.26.0 |
| ultralytics | 8.4.33 |
| Python | 3.13.5 |
| GPU | NVIDIA GeForce RTX 3060 (11.6 GB VRAM) |

### Docker Images

Four Docker images cover the full model portfolio. Each mounts the same repo at `/app` so scripts and configs are shared.

| Image | Dockerfile | Python | Purpose |
|-------|-----------|--------|---------|
| `wildlife-training` | `Dockerfile` | 3.13 | YOLO family (YOLOv8s, YOLO11/12/26n, RT-DETR-L) |
| `wildlife-speciesnet` | `Dockerfile.speciesnet` | 3.11 | SpeciesNet two-stage pipeline (teacher + baseline) |
| `wildlife-nanodet` | `Dockerfile.nanodet` | 3.11 | NanoDet-Plus-m training and NCNN export |
| `wildlife-paddle` | `Dockerfile.paddle` | 3.10 | PicoDet-S training and ONNX export (PaddlePaddle) |

**All images use `nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04`** except `wildlife-paddle` which uses `nvidia/cuda:12.0.0-cudnn8-runtime-ubuntu22.04` (PaddlePaddle wheels are compiled against CUDA 12.0).

The reasoning for using a CUDA 12.8 base despite the host having CUDA 13.2: no official `nvidia/cuda:13.x` images exist on Docker Hub yet. The NVIDIA container runtime injects `libcuda.so` from the host driver at container startup — the baked-in toolkit version only matters for build-time compilation. PyTorch's `cu130` wheels call into the injected host driver, so full CUDA 13.x capabilities are available inside the container.

**Python 3.11 / 3.10 for SpeciesNet / NanoDet / PaddlePaddle:** The `speciesnet` PyPI package requires Python `<3.13`; NanoDet uses legacy mmdet APIs removed in Python 3.12+. Python 3.13 (main image) is unsuitable for these frameworks. 3.11 is the sweet spot: modern enough for torch 2.11, old enough for all ecosystem packages.

**`--shm-size=8g`** is set on all Docker run commands to prevent PyTorch DataLoader shared-memory exhaustion with multiple workers.

**Key Makefile targets:**

| Target | Action |
|--------|--------|
| `make docker-build` | Build main image (YOLO/RT-DETR) |
| `make docker-shell` | Interactive GPU shell |
| `make smoke` | Run smoke test on host |
| `make smoke-docker` | Run smoke test in main Docker |
| `make docker-train` | Long-running teacher training (detached) |
| `make speciesnet-build` | Build `wildlife-speciesnet` image |
| `make speciesnet-smoke` | SpeciesNet pipeline smoke test |
| `make speciesnet-demo IMAGE=…` | Run pipeline on a single image |
| `make speciesnet-labels DIR=… OUT=…` | Batch soft-label generation for KD |
| `make nanodet-build` | Build `wildlife-nanodet` image |
| `make nanodet-smoke` | NanoDet forward-pass smoke test |
| `make nanodet-shell` | Interactive NanoDet training shell |
| `make paddle-build` | Build `wildlife-paddle` image |
| `make paddle-smoke` | PicoDet forward-pass smoke test |
| `make paddle-shell` | Interactive PicoDet training shell |
| `make uv-lock` | Pin main image dependencies to `uv.lock` |

---

## 2. Model Smoke Test Results

Script: `scripts/training/1-smoke_test_models.py`
Run: host, RTX 3060, CUDA 13.2, PyTorch 2.11.0+cu130, ultralytics 8.4.33

```
Model                       Status  Params(M) Latency(ms)  Device   Note
────────────────────────────────────────────────────────────────────────────────────────────
YOLOv8s (teacher)           PASS        11.17         7.5   cuda
RT-DETR-L (teacher)         PASS        32.97        41.4   cuda
YOLO11n (student)           PASS         2.62         7.2   cuda
YOLOv10n (student)          PASS         2.78         6.8   cuda
YOLO12n (student)           PASS         2.60        10.5   cuda
YOLO26n (student)           PASS         2.57         8.3   cuda
NanoDet-Plus-m              SKIP            —           —   —        make nanodet-build && make nanodet-smoke
PicoDet-S                   SKIP            —           —   —        make paddle-build && make paddle-smoke
SpeciesNet Pipeline         SKIP            —           —   —        make speciesnet-build && make speciesnet-smoke
yolov8s nc=225              PASS        11.22           —   cpu      Detect head nc=225 confirmed
rtdetr-l nc=225             PASS        33.19           —   cpu      RTDETRDecoder dec_score_head rebuilt for nc=225
yolo11n nc=225              PASS         2.66           —   cpu      Detect head nc=225 confirmed
yolo12n nc=225              PASS         2.64           —   cpu      Detect head nc=225 confirmed
yolo26n nc=225              PASS         2.61           —   cpu      Detect head nc=225 confirmed
────────────────────────────────────────────────────────────────────────────────────────────
PASS: 11   FAIL: 0   SKIP: 3
```

**Notes on the results:**

- **YOLO26n (2.57M params, 8.3ms)** is the newest nano model — September 2025 release. It achieves the best COCO mAP in the nano class (40.9%) with the fewest parameters (2.4M official count; 2.57M as measured here includes the detection head). The nc=225 head swap works identically to other YOLO variants.
- **RT-DETR-L latency (41.4ms)** is higher than YOLO variants on GPU. Expected — the Transformer decoder's cross-attention scales quadratically with spatial resolution. RT-DETR is teacher-only (not embedded), so raw GPU latency matters less than accuracy.
- **YOLO12n latency (10.5ms)** is slightly higher than YOLO11n (7.2ms) despite similar parameter counts. The A2C2f attention modules in YOLO12 add overhead on small single-image batches — less pronounced at the batch sizes used during actual training.
- **nc=225 head adaptation** confirmed for all 5 Ultralytics models. Parameter count increases ~0.04M vs. COCO nc=80, reflecting the larger `cv3` classification convolutions.
- **RT-DETR YAML init broken** in ultralytics 8.4.33 — `YOLO("rtdetr-l.yaml")` raises an unsupported-task error. The adaptation test loads from `.pt` instead and rebuilds `dec_score_head` in-place (documented in script).
- **3 expected SKIPs**: NanoDet, PicoDet, SpeciesNet each have their own Docker image and dedicated smoke test script. The SKIP messages include the exact `make` command to run them.

**Model ID discrepancy:** Research documents refer to "RT-DETR-R18" which does not exist in ultralytics 8.4.33. Available variants are `rtdetr-l` (ResNet-50, 33M params) and `rtdetr-x` (ResNet-101, 67M). `rtdetr-l` is used throughout.

---

## 3. SpeciesNet Two-Stage Pipeline (Teacher & Baseline)

Script: `scripts/training/0-teacher_speciesnet_pipeline.py`
Docker: `wildlife-speciesnet` (`Dockerfile.speciesnet`, Python 3.11)

### Architecture

| Stage | Model | Params | Size | Role |
|-------|-------|--------|------|------|
| Detection | MegaDetector v5a (YOLOv5x6) | 141.8M | ~270MB | Detects animal/person/vehicle boxes |
| Classification | SpeciesNet EfficientNetV2-M | 54M | ~87MB | Classifies crops → ~3,537 species labels |
| **Combined** | — | **~196M** | **~357MB** | — |

MegaDetector is loaded via `PytorchWildlife`; the SpeciesNet classifier is loaded via the `speciesnet` PyPI package (Python <3.13 requirement, hence the dedicated Docker image).

### Why SpeciesNet is the primary teacher

- Trained on 65M+ camera trap images — the largest publicly available wildlife training corpus
- 83% species-level accuracy, 98.6% empty image recall
- Apache 2.0 license — commercially usable for fine-tuning and distillation
- The two-stage (MegaDetector → crop → EfficientNetV2-M) is exactly the architecture that the AX Visio currently runs. A one-shot student model needs to exceed this to justify the redesign.

### Pipeline modes

```bash
# Verify pipeline loads (no real images needed)
make speciesnet-smoke

# Run on a single image (prints JSON with detections + top predictions)
make speciesnet-demo IMAGE=data/test/deer.jpg

# Batch mode: process full training directory → JSONL
make speciesnet-labels DIR=data/training OUT=output/teacher_labels.jsonl

# Soft-label mode: filter to 225 student classes, renormalise probabilities → JSONL
make speciesnet-labels DIR=data/training OUT=output/kd_soft_labels.jsonl
# (--soft-labels and --classes flags are set automatically by the Makefile target)
```

### Output format for KD

The `--soft-labels` mode writes one JSON object per image to the JSONL file:

```json
{
  "image": "data/training/deer/img001.jpg",
  "detections": [
    {
      "bbox_norm": [0.12, 0.08, 0.87, 0.94],
      "conf": 0.97,
      "soft_label": {
        "label_index": 42,
        "label_name": "Odocoileus virginianus",
        "probs_225": [0.001, ..., 0.850, ...]
      }
    }
  ],
  "predictions": [
    {"label": "Odocoileus virginianus", "score": 0.85},
    {"label": "Odocoileus hemionus",    "score": 0.10}
  ],
  "inference_ms": 145.2
}
```

The `probs_225` field is a 225-dimensional float vector renormalised over the student model's class set. This is the soft label used as the KD target distribution for Path A (response-based distillation).

### SpeciesNet Python version constraint

The `speciesnet` PyPI package requires Python `>=3.9,<3.13`. The main training image uses Python 3.13 (required for the latest ultralytics). These are incompatible in the same environment. Solution: dedicated `Dockerfile.speciesnet` with Python 3.11.

The SpeciesNet container is only needed for two operations:
1. **Soft label generation** — run once over the training dataset, save JSONL
2. **Baseline evaluation** — benchmark the two-stage pipeline on the test set

After JSONL files are generated, the student training and KD pipeline run entirely inside the main `wildlife-training` image.

---

## 4. NanoDet-Plus-m and PicoDet-S Environments

### NanoDet-Plus-m — `Dockerfile.nanodet`

NanoDet is not on PyPI and has legacy dependency requirements:
- `pytorch-lightning==1.9.5` (NanoDet uses `LightningModule` APIs removed in 2.x)
- Installed via `git clone https://github.com/RangiLyu/nanodet && pip install -e .`

Training config: `scripts/training/configs/nanodet-plus-m-wildlife225.yml`
- `num_classes: 225` in `head.name: NanoDetPlusHead`
- ShuffleNetV2 backbone with pre-trained ImageNet weights (separate from head)
- GhostPAN neck, 416×416 input, AdamW, 300 epochs

Key constraint: **NanoDet does not support the Ultralytics-style auto-rebuild.** A new config with `num_classes: 225` is required, and the head weights must be randomly re-initialised. The backbone (ShuffleNetV2) can be loaded from a COCO-pretrained checkpoint; only `pretrain_model_path` pointing to the backbone-only weights is needed (not the full NanoDet COCO checkpoint, which has an incompatible head).

NCNN export pipeline (for Raspberry Pi 5):
```bash
make nanodet-shell
# Inside container:
cd /opt/nanodet
python tools/export.py /app/scripts/training/configs/nanodet-plus-m-wildlife225.yml \
  --checkpoint /app/output/nanodet_wildlife225/model_best.ckpt \
  --out_path /app/output/nanodet_wildlife225_ncnn
```

### PicoDet-S — `Dockerfile.paddle`

PaddlePaddle + PaddleDetection environment:
- Base: `nvidia/cuda:12.0.0-cudnn8-runtime-ubuntu22.04` (PaddlePaddle compiled for CUDA 12.0)
- `paddlepaddle-gpu==2.6.2.post120` + `PaddleDetection` from GitHub

Training config: `scripts/training/configs/picodet-s-wildlife225.yml`
- `num_classes: 225` in `PicoHeadV2`
- ESNet backbone + CSP-PAN neck; 320×320 input, 300 epochs
- Reuses COCO backbone weights; head is randomly re-initialised

ONNX export pipeline (for SNPE conversion):
```bash
make paddle-shell
# Inside container:
cd /opt/PaddleDetection
python tools/export_model.py \
  -c /app/scripts/training/configs/picodet-s-wildlife225.yml \
  -o weights=/app/output/picodet_s_wildlife225/best_model
python -m paddle2onnx \
  --model_dir ./output/picodet_s_wildlife225_infer \
  --opset_version 11 --save_file /app/output/picodet_s_wildlife225.onnx
```

---

## 5. Class Adaptation — 225 Wildlife Classes

All student models need their detection heads replaced from nc=80 (COCO) to nc=225 (wildlife taxonomy). The exact approach differs by model family.

### YOLO Family (YOLOv8s, YOLO11n, YOLOv10n, YOLO12n)

The Detect head contains three components that depend on `nc`:
- `cv3`: a `ModuleList` of 3 `Sequential` modules (one per FPN scale), each ending in `Conv2d(ch, nc, 1)` — these are the classification convolutions
- `cv2`: bounding box regression branches — independent of `nc`, no change needed
- `nc` attribute on the Detect layer

**Recommended approach — Ultralytics train API (automatic):**
```python
from ultralytics import YOLO
model = YOLO("yolo11n.pt")
model.train(
    data="wildlife_225.yaml",   # nc: 225 declared in this file
    epochs=100,
    imgsz=640,
    device=0,
)
```
When `nc` in the dataset YAML differs from the loaded checkpoint, Ultralytics automatically rebuilds `cv3` with the new `nc` and randomly re-initializes those weights, while preserving the backbone and neck. This is the simplest and most reliable path.

**Manual head swap (for custom KD pipelines that bypass `model.train()`):**
```python
import torch.nn as nn
from ultralytics import YOLO

model = YOLO("yolo11n.pt").model
detect = model.model[-1]   # Detect layer
nc = 225

detect.nc = nc
for seq in detect.cv3:
    old = seq[-1]
    new = nn.Conv2d(old.in_channels, nc, 1)
    nn.init.normal_(new.weight, 0, 0.01)
    nn.init.zeros_(new.bias)
    seq[-1] = new
```
Only the final `1×1` conv in each cv3 branch is replaced. All other weights (backbone, neck, cv2 regression branches) are reused from the pretrained checkpoint.

### RT-DETR-L

RT-DETR's classification head is `RTDETRDecoder.dec_score_head`, a `ModuleList` of `nn.Linear(hidden_dim, nc)` layers (one per decoder layer, typically 6):

```python
import torch.nn as nn
from ultralytics import YOLO

model = YOLO("rtdetr-l.pt")
decoder = model.model.model[-1]   # RTDETRDecoder
nc = 225

decoder.dec_score_head = nn.ModuleList([
    nn.Linear(layer.in_features, nc)
    for layer in decoder.dec_score_head
])
decoder.num_classes = nc
```

Same train-API auto-rebuild applies as for YOLO variants. The encoder (hybrid CNN+Transformer) and all positional embeddings are preserved.

### NanoDet-Plus-m

NanoDet uses a config-YAML driven pipeline — there is no Ultralytics-style auto-rebuild:

1. Copy `config/nanodet-plus-m_416.yml` and change `num_classes: 80` → `num_classes: 225`
2. Update the dataset section with the wildlife data paths
3. Train from a pretrained backbone (not a COCO NanoDet checkpoint — head weights are incompatible):
   ```bash
   python tools/train.py config/nanodet-plus-m_wildlife225.yml \
     --config.model.backbone.pretrain_model_path shufflenetv2_x1.0.pth
   ```
4. Export to NCNN for Raspberry Pi deployment:
   ```bash
   python tools/export.py config/nanodet-plus-m_wildlife225.yml \
     --checkpoint workspace/nanodet_wildlife/model_best/model_best.ckpt \
     --out_path nanodet_wildlife
   ```

NanoDet requires a **separate installation** from the main training environment. It uses its own fork of mmdet/mmcv and conflicts with ultralytics. Options:
- Separate Python venv (simplest)
- Dedicated `Dockerfile.nanodet` based on `nvcr.io/nvidia/pytorch:24.xx-py3`

### PicoDet-S

PicoDet lives in the PaddlePaddle ecosystem and requires PaddleDetection:

1. Set `num_classes: 225` in the config YAML (`configs/picodet/picodet_s_320_coco_lcnet.yml`)
2. Train:
   ```bash
   python tools/train.py -c picodet_s_320_wildlife225.yml \
     --weight output/picodet_s_coco_lcnet/best_model    # reuse COCO backbone
   ```
3. Export to ONNX (required for SNPE conversion):
   ```bash
   python tools/export_model.py -c picodet_s_320_wildlife225.yml \
     -o weights=output/picodet_s_wildlife225/best_model \
        TestReader.inputs_def.im_shape=[1,3,320,320] \
        export.benchmark=True
   python -m paddle2onnx --model_dir=./picodet_s_infer --model_filename=model.pdmodel \
     --params_filename=model.pdiparams --opset_version=11 --save_file=picodet_s_wildlife225.onnx
   ```

PicoDet is best handled in a dedicated Docker image based on `paddlepaddle/paddle:2.6.2-gpu-cuda12.0-cudnn8.9-trt8.6`.

---

## 6. Open Questions

1. **SpeciesNet soft-label coverage:** The `probs_225` remapping discards probability mass from SpeciesNet labels not in the 225-class student set (e.g. birds, marine mammals, meta-classes like "mammalia"). For images where the in-set probability mass is low (e.g. a rare species with no student equivalent), the renormalised vector will be unreliable. Need a strategy for these cases — options: skip the image, use a genus-level fallback label, or assign uniform prior over the nearest taxonomic bucket.

2. **RT-DETR as teacher:** RT-DETR-L adds 41ms GPU latency per inference even with a full GPU, and its Transformer encoder is known to be problematic with SNPE's Hexagon DSP (unsupported attention ops). Its main advantage over YOLOv8s is mAP (47.9% vs 44.9%). Whether this delta justifies the complexity is an open question — the teacher only needs to run during KD training, not on embedded hardware, so raw GPU latency is less relevant than accuracy.

3. **NanoDet KD feasibility:** NanoDet's training pipeline doesn't natively support KD from a heterogeneous teacher (e.g. YOLOv8s). A custom distillation loss would need to be injected into NanoDet's training loop. Evaluate whether the expected mAP gain vs. YOLO11n justifies the implementation effort, especially given NanoDet's fixed pytorch-lightning 1.9.5 requirement.

4. **YOLO26n / YOLO12n attention on Hexagon DSP:** Both models use attention mechanisms (A2C2f modules) that may not map to supported Hexagon 685 ops. If attention falls back to the Kryo CPU, the latency advantage over YOLO11n disappears. Empirical verification needed once the SNPE toolchain is configured.

5. **`uv.lock` for reproducibility:** The `pyproject.toml` has no committed lock file — dependency resolution happens at Docker build time. Before any long training run, pin with `make uv-lock` and commit `uv.lock`.

---

## 7. Next Steps

1. **`make docker-build`** — build the main image and confirm CUDA 12.8 base + Python 3.13 + cu130 torch resolves via uv
2. **`make smoke-docker`** — verify GPU access inside the built container
3. **`make speciesnet-build && make speciesnet-smoke`** — verify the two-stage SpeciesNet pipeline loads in its Python 3.11 container
4. **Dataset YAML** — write `data/wildlife_225.yaml` with paths to the assembled dataset and all 225 class names (prerequisite for `model.train()`)
5. **`scripts/training/2-train_teacher.py`** — fine-tune YOLOv8s on wildlife data (primary KD teacher, Path A baseline)
6. **Soft-label generation** — once the dataset YAML is ready, run `make speciesnet-labels` to generate KD targets for the full training set
7. **SpeciesNet baseline evaluation** — benchmark the two-stage pipeline on the held-out test set to establish the accuracy ceiling the student must approach
