# SOTA Model Survey Prompt

> This file contains the prompt to be given to an LLM (with web search capabilities) to conduct the comprehensive SOTA model survey for the thesis.

---

## Prompt

You are a computer vision research assistant. Your task is to produce a **comprehensive, structured survey of object detection models** relevant to a Master's thesis on real-time inference on embedded hardware. You have access to web search — use it extensively. Cross-reference multiple sources and verify your findings.

---

### Research Context

The thesis is titled **"Optimization and Deployment of Deep Learning Models for Real-Time Object Recognition on Resource-Constrained Embedded Hardware"**. It is conducted at inovex GmbH within the Data Management & Analytics department.

**Domain:** Real-time detection and classification of **non-bird mammal species** in camera trap / wildlife monitoring imagery.

**Target hardware:** Qualcomm QCS605 (Hexagon 685 DSP ~2.1 TOPS, Adreno 615 GPU, Kryo 300 CPU — an ARM-based 2018-era SoC). Development and benchmarking will be done on a **Raspberry Pi 5 (8GB)** as a proxy device.

**Primary technical approaches under investigation:**
1. **Quantization** (PTQ and QAT, INT8 / FP16 targets)
2. **Knowledge Distillation** (teacher → student compression)
3. **Transfer Learning** (pretrained → fine-tuned on wildlife domain)

**Key constraints:**
- Models must be runnable on ARM Cortex-A76-class CPUs and/or mobile GPUs (no CUDA/server-GPU-only architectures)
- Target: ≤30ms inference latency at ~640×640 input resolution on the proxy hardware (or clearly scalable to meet this)
- Memory budget: ≤500MB RAM for model + runtime
- Output format: bounding boxes (object detection), optionally combined with classification

---

### Survey Scope & Instructions

Conduct the survey in **four parts**. For each part, search the web for recent papers, benchmarks, GitHub repositories, and official documentation. Prefer sources from arXiv, Papers With Code, and official model repos (Ultralytics, Google, Meta, Microsoft, etc.).

---

#### Part 1: Real-Time Object Detection Architectures (the "model families")

Survey the major families of real-time object detection models — both historical foundations and current SOTA. For each model/family, cover:

- **Architecture type**: anchor-based, anchor-free, transformer-based, hybrid
- **Backbone family**: (e.g., CSPNet, EfficientNet, ResNet, ViT variants)
- **Year / version history**: Trace the evolution within the family (e.g., YOLO v1 → v11)
- **Typical input resolution**
- **Published benchmark accuracy** (COCO mAP, or domain-specific if available)
- **Published inference speed** (ms on a known device — note the device)
- **Model size** (parameters, MB)
- **License** (MIT, Apache 2.0, GPL, proprietary)
- **Framework & export formats** (PyTorch, TensorFlow, ONNX, TFLite, NCNN, CoreML, etc.)

**Minimum families to cover** (search for the latest version of each and trace their history):
- YOLO family: YOLOv1 through YOLOv11 / YOLO-NAS / YOLO-World / Gold-YOLO
- RT-DETR / RT-DETRv2 (Baidu)
- DAMO-YOLO (Alibaba DAMO Academy)
- EfficientDet family (Google)
- MobileNet-SSD variants (MobileNetV1/V2/V3 + SSD/SSDLite)
- NanoDet / NanoDetPlus
- FCOS
- CenterNet / CenterNet2
- SSD (original and variants)
- PicoDet (PaddlePaddle)
- SpeciesNet (iNaturalist / Google) — specifically relevant to this domain

**Also search for**: any other models that appear prominently in "real-time object detection on edge / embedded" literature published between 2022 and 2025.

---

#### Part 2: Embedded & Mobile Deployment

For the models identified in Part 1, research their suitability for embedded / edge deployment. Specifically:

- Which models have **official TFLite, ONNX, NCNN, or OpenVINO export pipelines**?
- Which have been **benchmarked on ARM hardware** (Raspberry Pi, Jetson Nano, mobile phones)?
  - Search for benchmarks on: Raspberry Pi 4/5, ARM Cortex-A class CPUs, Snapdragon 600-series devices
- Which models support **INT8 quantization** (PTQ via ONNX Runtime / TFLite, or QAT)?
- Which models support **FP16 half-precision** inference?
- Are there known accuracy/latency tradeoffs documented for quantized versions?

Search specifically for:
- "YOLOv8 Raspberry Pi benchmark"
- "real-time object detection ARM cortex benchmark 2023 2024"
- "edge AI object detection INT8 quantization benchmark"
- Papers With Code leaderboards for mobile/embedded detection

---

#### Part 3: Knowledge Distillation for Object Detection

Survey the state of knowledge distillation applied specifically to **object detection** (not just classification):

- What are the **main KD strategies** used for detection? (feature-level, response-level, relation-based, etc.)
- Which detector families have **published, reproducible KD pipelines**? List them with citations.
- What are the **typical compression ratios and accuracy retention** reported?
- Which models are commonly used as **teacher / student pairs** in the literature?
- Are there frameworks or libraries that provide off-the-shelf KD for detection (e.g., MMDetection, Ultralytics, PaddleDetection)?

Key papers to search for (and any that cite or follow them):
- "Distilling Object Detectors with Fine-Grained Feature Imitation" (Wang et al., 2019)
- "Knowledge Distillation for Object Detection" related work post-2020
- YOLO-specific distillation papers (search: "YOLOv8 knowledge distillation", "YOLO distillation")
- "CWD: Channel-wise Knowledge Distillation for Dense Prediction"
- "Localization Distillation for Object Detection"

---

#### Part 4: Wildlife / Camera Trap Domain

Survey models, datasets, and techniques specifically relevant to the **wildlife / camera trap detection domain**:

- Which object detection or classification models have been applied to wildlife camera trap imagery? What were the results?
- What are the **most-used open datasets** for wildlife species detection?
  - iNaturalist (competition datasets 2018–2021)
  - iWildCam (iNaturalist + WILDS benchmark)
  - Snapshot Serengeti
  - COCO (animals subset)
  - Any other relevant open datasets
- What domain-specific challenges are documented? (e.g., occlusion, small object size, night vision / IR imagery, class imbalance, rare species)
- **SpeciesNet**: Find its architecture, training data, published accuracy, and whether it is open-source and usable for transfer learning.
- **BioCLIP**: Find its architecture (vision-language, CLIP-based), training data (TreeOfLife-10M), and relevance to this domain.
- Are there published transfer learning / fine-tuning approaches from general detectors to wildlife?

---

### Output Format

Structure your output as follows:

#### 1. Executive Summary (1–2 paragraphs)
Which 3–5 models stand out as the strongest candidates for this thesis, and why?

#### 2. Model Comparison Matrix

A large Markdown table with one row per model (or model variant). Columns:

| Model | Family | Year | Architecture Type | Backbone | Params (M) | Size (MB) | COCO mAP | Inference (ms / device) | INT8 Support | TFLite/NCNN Export | KD Teacher Potential | KD Student Potential | License | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

Use "?" where data is not found. Use "✓" / "✗" for boolean columns.

#### 3. Knowledge Distillation Summary Table

| Paper / Framework | Year | Teacher | Student | Task | mAP Teacher | mAP Student | mAP Drop | Compression Ratio | Notes |
|---|---|---|---|---|---|---|---|---|---|

#### 4. Dataset Summary Table

| Dataset | Year | # Images | # Classes | Annotation Type | Domain | Open Access | Notes |
|---|---|---|---|---|---|---|---|---|

#### 5. Recommended Shortlist with Rationale

For each recommended model, explain in 3–5 bullet points:
- Why it is suitable for this thesis
- Its key weaknesses or risks
- What needs to be verified experimentally

#### 6. Open Questions & Gaps

List any areas where the literature is thin, contradictory, or where important benchmarks are missing for this specific use case.

---

### Quality Requirements

- **Cite every factual claim** with at minimum the paper title, authors, and year (full URL if possible).
- Distinguish between **official benchmarks** (from the paper/repo) and **third-party benchmarks**.
- If a model has multiple variants (nano, small, medium, large), include a row per variant in the matrix.
- Flag any models whose benchmarks were measured on hardware significantly different from ARM Cortex-A/embedded targets — GPU-only numbers are less useful.
- Prioritize **reproducibility**: prefer models with active GitHub repositories, clear training scripts, and recent maintenance (updated within the last 18 months as of March 2026).
