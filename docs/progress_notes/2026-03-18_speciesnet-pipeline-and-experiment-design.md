# Progress Notes – 18.03.2026

## Thinking Update – SpeciesNet Pipeline as Baseline & Experimental Design

---

### 1. Current Swarovski AX Visio Pipeline

Swarovski currently uses a **two-stage pipeline** on the AX Visio for mammal identification:

1. **Detection:** A fine-tuned **YOLOv5s** (the full MegaDetector was too large/slow for the device)
2. **Classification:** The **SpeciesNet EfficientNetV2-M** model classifies the cropped detections

This existing pipeline can serve as a **baseline benchmark** against which to evaluate the performance improvements (or trade-offs) of the approaches explored in this thesis.

---

### 2. Two-Stage vs. One-Shot Detection — A Research Gap

The paper *"To crop or not to crop"* (`research/To crop or not to crop.md`) demonstrates that a two-stage pipeline (species-agnostic detector + classifier on crops) yields ~25% better macro-average F1 than whole-image classification alone. However, the paper only compared:

- Whole-image classifiers (EfficientNetV2-M)
- Crop classifiers (EfficientNetV2-M on MegaDetector crops)
- Ensembles of both

**What was NOT evaluated:** Modern one-shot object detection models that perform detection and classification simultaneously (e.g., YOLO variants, NanoDet, PicoDet). This represents a **research gap** worth investigating.

#### Why a One-Shot Approach May Be Competitive

Several factors specific to the AX Visio use case suggest that the advantage of the two-stage pipeline may shrink or disappear:

1. **Reduced class space:** The SpeciesNet taxonomy covers ~3,500 animal species. For the AX Visio, the target is under 500 classes (common charismatic mammals only, excluding birds). The paper itself notes that birds and small animals likely drove much of the poor macro-avg F1 for the whole-image classifier — removing them from the class space should narrow the performance gap.

2. **Domain shift:** The paper evaluated on camera trap images (fixed-position, motion-triggered, often low-quality IR). The AX Visio produces user-aimed photographs of animals, which are typically higher quality, better framed, and more consistently lit — reducing the need for a detector to isolate small subjects from cluttered backgrounds.

3. **Latency advantage:** A single-stage model avoids the cascading latency of running a detector, cropping, and then running a classifier — critical for the 30ms inference budget on the QCS605.

#### Proposed Experimental Approach

1. Fine-tune a **full (heavy) one-shot detection model** (e.g., YOLOv12, RT-DETR) on the target wildlife species and compare its accuracy against the SpeciesNet two-stage pipeline
2. Then systematically reduce model size using **lightweight architectures** (YOLO-nano variants, NanoDet, PicoDet, EfficientDet-Lite) and measure how far the model can be compressed — via knowledge distillation and/or direct fine-tuning — before accuracy degrades unacceptably

---

### 3. Model Size Context — The Scale Challenge

Understanding the magnitude of the size difference between the SpeciesNet pipeline and the target student models is critical for setting realistic expectations, especially for knowledge distillation.

#### SpeciesNet Two-Stage Pipeline

| Component | Architecture | Parameters | Model Size (FP32) | COCO mAP* | Input Resolution |
|:---|:---|:---|:---|:---|:---|
| MegaDetector v5a | YOLOv5x6 | ~141.8M | ~270MB | 54.4% | 1280px |
| SpeciesNet Classifier | EfficientNetV2-M | ~54M | ~87MB | n/a (classifier) | 480×480 |
| **Combined pipeline** | — | **~196M** | **~357MB** | — | — |

*\*MegaDetector's COCO mAP refers to its base architecture (YOLOv5x6) evaluated on the standard COCO 80-class benchmark — MegaDetector itself was fine-tuned for only 3 meta-classes (animal/person/vehicle) and reports 99.2% precision at 97.3% recall on camera trap data. The SpeciesNet classifier (EfficientNetV2-M) is a pure classification model (85.1% ImageNet top-1 accuracy), so COCO detection mAP does not apply.*

#### Current AX Visio Detector

| Model | Parameters | COCO mAP | Input Resolution |
|:---|:---|:---|:---|
| YOLOv5s (fine-tuned) | 7.2M | 37.4% | 640px |

*This is the detector Swarovski currently deploys on the AX Visio — a useful reference point for what the target hardware can already run.*

#### Target Student Models (from `docs/object-detection-models-for-embedded-systems.md`)

| Model | Parameters | Model Size | COCO mAP | RPi 5 Latency (NCNN) | T4 GPU Latency |
|:---|:---|:---|:---|:---|:---|
| PicoDet-S | 0.99M | 2.1MB | 30.6% | 4.8ms (ARM A76) | — |
| NanoDet-Plus-m | 1.17M | 1.8MB | 30.4% | 19.77ms (ARM A76) | — |
| YOLO26n | 2.4M | 5MB | 40.9% | 67.69ms | 1.7ms (FP16) |
| YOLOv12-N | 2.6M | ~5.5MB | 40.6% | — | 1.64ms |
| YOLO11n | 2.7M | ~5.8MB | 39.4% | — | 1.5ms |
| EfficientDet-Lite0 | 3.2M | 4.4MB | 25.7% | 37ms (Pi 4) | — |

*YOLO26n is the latest Ultralytics release (Sep 2025). It achieves the highest COCO mAP among the nano-class models (40.9%) with the fewest parameters (2.4M). Its NMS-free end-to-end architecture simplifies embedded deployment. On the RPi 5 with NCNN it runs at ~67.7ms (14.8 FPS) — above the 30ms target but potentially improvable via INT8 quantization. License: AGPL-3.0 (same dual-license model as other recent Ultralytics YOLO versions — commercial use requires an Enterprise License).*

#### The Gap

**Size gap:**
- **Classifier alone vs. students:** The EfficientNetV2-M classifier has ~54M parameters — that is **20–55× more** than the student models (0.99M–2.7M).
- **Full pipeline vs. students:** The combined SpeciesNet pipeline (~196M parameters) is **72–198× larger** than the student models.
- **In terms of model size:** The full pipeline (~357MB FP32) vs. the smallest student (NanoDet at 1.8MB FP16) represents a **~200× size reduction**.

**COCO mAP gap (base architectures):**
- **Teacher-tier to student-tier:** The YOLOv5x6 base architecture (54.4%) vs. the student models (25.7–40.9%) shows a gap of **~14–29 percentage points**, depending on the student.
- **Current AX Visio detector as reference:** YOLOv5s achieves 37.4% COCO mAP with 7.2M params. Notably, **YOLO26n already exceeds this** (40.9% mAP) with only 2.4M params (3× fewer parameters) — suggesting the latest nano YOLO variants may match or surpass the detection capability Swarovski currently deploys.

**Important caveat:** COCO mAP measures general 80-class object detection. Wildlife species detection is a fundamentally different distribution with different challenges (long-tailed species, IR imagery, occlusion). These numbers provide a rough capability ranking, but the actual domain-specific performance gap will only be revealed by the KD experiments on wildlife data.

---

### 4. Dataset Status

Danielle provided approximately **66,000 images** labeled using the SpeciesNet pipeline. Initial assessment:

- **Probably not sufficient on its own** — detailed analysis needed to understand species distribution and label quality
- Labels are SpeciesNet-generated (not human-verified), so label noise is expected

**Next steps for data:**
1. Analyze the species distribution and general quality of the 66k dataset
2. Search for supplementary open datasets (iNaturalist, Snapshot Serengeti, iWildCam) to increase coverage
3. Identify or construct a clean test set for reliable performance evaluation
4. For species with insufficient training data: explore **synthetic image generation** (diffusion models) as augmentation — trained on mixed real+synthetic data, but evaluated strictly on real photographs only

---

### 5. Training Strategy — Knowledge Distillation Approach

#### Why Response-Based or Relation-Based KD

Given the situation — partially unlabeled data, or data labeled via the SpeciesNet pipeline (i.e., soft labels from a teacher model already available) — **response-based knowledge distillation** is a natural fit. The SpeciesNet pipeline itself would serve as the teacher:

1. Run **MegaDetector** to produce bounding box detections
2. Crop each detection and feed it through **SpeciesNet** for classification
3. Use the resulting soft probability distributions as supervision signal for the student model

**Relation-based KD** could additionally capture structural relationships between species (e.g., similar-looking animals producing similar feature representations), which may help the student generalize better across the long-tailed species distribution.

#### Why NOT Feature-Based KD

**Feature-based KD** (forcing the student to mimic the teacher's intermediate feature maps) is likely **not feasible** for this setup:

- The teacher is a **two-stage pipeline** (separate detector + classifier), while the student is a **single-stage detector** — the architectures are fundamentally different, making it unclear which feature maps to align
- The intermediate representations of an EfficientNetV2-M classifier operating on 480×480 crops are not directly comparable to the dense multi-scale feature pyramids of an anchor-free detector operating on full images
- The added implementation complexity does not justify the uncertain benefit, especially given that Channel-Wise KD (CWD) and similar techniques already exist to handle cross-architecture distillation more gracefully (see `docs/knowledge_distillation_research_overview.md`)

#### Alternative: Direct Fine-Tuning as Comparison Baseline

A core question of this thesis is whether KD actually helps over simply fine-tuning the student model directly on the labeled data. Both approaches should be compared:

- **Path A:** Student model trained via KD from the SpeciesNet teacher
- **Path B:** Student model fine-tuned directly on labeled wildlife data (no teacher)

The delta between these two paths — particularly across different student model sizes — is the key experimental result.

---

### 6. Further Optimization (Later Stage)

After establishing the distillation/fine-tuning results, additional compression techniques can be explored:

- **Quantization** (PTQ and QAT) — required for deployment on Hexagon 685 DSP via SNPE
- **Pruning** — structured or unstructured weight removal to further reduce model size

These are second-stage optimizations that build on top of the distilled/fine-tuned student model.
