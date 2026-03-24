# A Review of Real-Time Deep Learning-Based Object Detection Models for Resource-Constrained Embedded Systems

**Awais Shah** — Department of Computer Science, Wapda Post Graduate College, Terbela, Pakistan

*NUML International Journal of Engineering and Computing (NIJEC), Vol. 4(2), 2025, pp. 13–30*
*Received: 1 Jan 2026 | Accepted: 4 Jan 2026*

**Keywords:** Object Detection, Embedded Systems, Edge AI, Lightweight Deep Learning Models, Resource-Constrained Devices

---

## Abstract

Computer vision relies heavily on object detection; from autonomous drones to industry monitoring it is everywhere. However, despite these advances, implementing cutting-edge object detection models on embedded systems proves difficult due to limitations in processing power, memory, and energy consumption. This paper presents a detailed examination of real-time object detection models designed for resource-constrained devices. We investigate widely used one-stage detectors like SSD (Single Shot MultiBox Detector) and YOLO (You Only Look Once), and discuss model compression methods like knowledge distillation, pruning, and quantization, which enable efficient deployment on embedded systems.

---

## 1. Introduction

Object Detection is the classifying and localization of objects in images and video streams [1]. The term neural network was first suggested in the 1940s [2] to mimic the human brain to solve the general problem of learning in a principled manner. In the 1980s and early 1990s, the introduction of back propagation [3] played a vital role in the widespread popularity of neural networks. However, in the early 2000s it became less popular due to the unavailability of large-scale annotated data, severe overfitting, scarcity of computational power, and negligible results. After addressing these limitations, neural networks were revived with more refined algorithms.

Traditional object detection relied on manually extracted features and classical machine learning techniques such as Viola-Jones [4], SIFT [5], and HOG combined with an SVM classifier [6]. They influenced the market for decades. However, while effective for constrained scenarios, they struggled in real-time scenarios due to the extensive need for manual feature engineering. The introduction of Deep Learning and CNNs (convolutional neural networks) [7] revolutionized object detection entirely. CNNs showed the capability to learn hierarchical image feature representations directly without manual feature crafting. The game changed with the appearance of AlexNet [8], achieving record-breaking results for image classification on the ImageNet benchmark.

> **Figure 1 — Object Detection taxonomy:** "Object Detection" branches into (1) **Traditional Methods** → Viola-Jones, HOG, SIFT; and (2) **Deep Learning Based Methods** → Regression/Proposal-based, Transformer-Based (DETR), Single-Stage Neural Networks.

In spite of these breakthroughs, a substantial "deployment gap" remains. Reliable object detection models are designed to run on costly GPUs and tend to be RAM- and power-hungry. However, these models need to work on low-end hardware like Raspberry Pi boards, NVIDIA Jetson Nano boards, and other ARM embedded systems, where battery power and time performance (typically ≥30 FPS) are the primary functional constraints. The trade-off between accuracy (measured by mAP) and efficiency (measured by FPS and memory) is the central challenge when designing for embedded systems. Two main solution directions have been proposed:

### 1.1 Lightweight Architectural Design

Designing efficient backbone architectures from scratch using approaches like depth-wise separable convolution [9], channel shuffling [10], inverted residuals [11], or neural architecture search (NAS), while reducing parameters and retaining detection accuracy.

### 1.2 Model Optimization

Post-training techniques that reduce computational and memory footprints without substantially reducing accuracy. These include:

- **Structured/unstructured pruning** [12] — eliminates redundant parameters
- **Quantization** [13] — reduces precision from FP32 to INT8 or lower
- **Knowledge distillation** [14] — transfers knowledge from large teacher networks to compact student models

This review presents a deployment-oriented systematic analysis of real-time deep learning-based object detection models for resource-constrained embedded systems. Unlike existing surveys that mainly emphasize detection accuracy or architectural design, this work jointly examines detection performance with practical deployment constraints: computational capacity, memory footprint, power consumption, thermal limitations, and hardware-specific optimization frameworks.

---

## 2. Literature Review

The early days of object detection were characterized by hand-designed feature extractors and shallow learning models. SIFT was developed to add robustness to scale and rotation transformations. HOG was established as a norm for pedestrian detection [15]. The Viola-Jones face detector, using Haar-like features [16] and an AdaBoost classifier [17], was one of the earliest real-time face detectors. These traditional approaches were ultimately limited by the semantic gap — the inability to bridge low-level pixel descriptors and high-level semantic understanding. It was not until CNNs demonstrated the ability to learn hierarchical feature representations automatically that object detection was truly revolutionized [18].

### 2.1 Deep Learning-Based Object Detection Architectures

Modern deep learning-based object detectors divide into two families: two-stage and single-stage detectors.

**a. Two-Stage Detectors** focus on high accuracy by splitting detection into region-proposal generation followed by classification and boundary refinement. R-CNN [19] initiated CNN-based region-proposal detection but processed each region separately, making it computationally intensive. Fast R-CNN [20] optimized this by sharing convolutional features across proposed regions. Faster R-CNN [21] introduced Region Proposal Networks (RPNs) for end-to-end processing, but the multi-stage pipeline still carries significant computational overhead.

> **Figure 2 — Two-Stage Detector pipeline:** Input image → Backbone CNN (Feature Extractor) → splits into RPN and ROI Pooling/Alignment → Classifier (Object/Class) and Regression (Bounding Box) → Output Detections.

**b. Single-Stage Detectors** treat object detection as a direct regression problem, bypassing the region-proposal step. YOLO [22] introduced this paradigm by predicting bounding boxes and class scores in a single pass. SSD [23] improved multi-scale detection using feature maps at multiple scales. These detectors are considerably faster than two-stage detectors but still resource-intensive for very constrained embedded systems.

> **Figure 3 — Single-Stage Detector pipeline:** Input image → Backbone CNN (Feature Extractor) → Classifier (Class Probability) / Regression (Bounding Coordinates) → Output Detections.

### 2.2 Lightweight Architectures for Embedded Systems

**a. MobileNet** [24] introduced depthwise separable convolutions — splitting standard convolution into a depthwise step (one filter per channel) and a pointwise 1×1 step — reducing computational complexity and parameter count by 8–9× while maintaining acceptable accuracy. MobileNetV2 [11] added inverted residual connections with linear bottlenecks. MobileNetV3 [25] incorporated NAS and Squeeze-and-Excitation attention.

**b. ShuffleNet** [26] introduced channel shuffle operations to enable efficient cross-group information exchange within group convolutions. ShuffleNetV2 [27] improved inference speed further through memory-access-cost-aware design heuristics and high parallelization.

**c. EfficientNet/EfficientDet** [28] proposed compound scaling of network depth, width, and resolution using a compound coefficient, yielding better accuracy/FLOP ratios than prior architectures (EfficientNet B0–B7). EfficientDet [29] adapts this for object detection with a weighted bidirectional feature pyramid network (BiFPN). EfficientNet-Lite removes operations unsupported on edge TPUs (e.g., Swish activation).

**d. NanoDet** [30] is a highly efficient anchor-free detector designed for mobile and edge devices, supporting >97 FPS on ARM CPUs. It uses ShuffleNetV2 or GhostNet as backbone and a light feature pyramid network. Model size is approximately 0.6–1.8 MB, making it one of the most efficient options for resource-constrained devices.

---

## 3. Model Compression Techniques for Edge Deployment

Model compression enables efficient execution of deep learning object detectors on edge devices with limited resources. The primary techniques are quantization, pruning, and knowledge distillation.

### 3.1 Quantization

Quantization reduces model size and computational cost by representing weights and activations in lower-bit precision. Standard deep learning models use 32-bit floating-point (FP32); quantization converts them to FP16 or INT8. INT8 quantization offers approximately 4× memory reduction and notably speeds up inference, especially on hardware with optimized integer arithmetic such as NVIDIA Jetson devices or Google Coral Edge TPUs.

Two primary approaches:

- **Post-Training Quantization (PTQ):** Applies quantization to an already-trained model without retraining. Simple to apply but may degrade accuracy for some models.
- **Quantization-Aware Training (QAT):** Simulates quantization effects during training so the model learns robustness to reduced numerical precision [13]. Typically results in better accuracy than PTQ, with loss often contained at 1–2% mAP, at the cost of more training time.

Combining INT8 quantization with inference frameworks such as TensorFlow Lite or TensorRT achieves up to 2–4× speedup over FP32 inference while nearly maintaining the original detection accuracy.

### 3.2 Pruning

Pruning reduces model complexity by eliminating less important parameters from trained networks [12]. Two main variants:

- **Unstructured pruning:** Removes individual weights based on magnitude or importance, creating sparse weight matrices. Can prune 70–90% of parameters but requires sparse computing libraries and offers limited benefits on standard embedded hardware.
- **Structured pruning:** Removes entire structural entities — neurons, channels, or layers — directly truncating the computational graph [31]. Results in dense, hardware-friendly models that accelerate inference on general hardware (e.g., Raspberry Pi, ARM processors) without special sparsity support. Structured pruning can achieve 40–60% FLOP reduction with less than 2% mAP loss when combined with fine-tuning [32].

The conventional pruning pipeline: (1) train network to convergence, (2) threshold parameters by a criterion (e.g., magnitude), (3) fine-tune to recover accuracy. Iterative alternation between pruning and fine-tuning generally achieves better accuracy/compression trade-offs than single-shot methods.

### 3.3 Knowledge Distillation

Knowledge Distillation (KD) [33] trains a smaller "student" model to mimic a larger "teacher" model. Rather than learning only from hard labels (e.g., "Dog" or "Cat"), the student learns from "soft targets" — the teacher's probability distributions over all classes (e.g., 85% dog, 10% wolf, 5% fox) — which encode inter-class relationships. For object detection, a student model like Tiny-YOLO can be trained to match not only the predictions but also internal feature maps and bounding box deltas of a full teacher model like YOLOv10. This enables the student to achieve significantly better performance than training from scratch, while remaining small enough for resource-constrained devices.

---

## 4. Deployment on Embedded Systems and Edge Devices

Embedded deployment imposes constraints that cloud platforms do not face: typical embedded systems (NVIDIA Jetson series, Raspberry Pi, smartphone SoCs) have 2–8 GB RAM and 5–15 W power budgets, compared to server-class GPUs.

### 4.1 Hardware Platforms and Optimization Frameworks

**a. NVIDIA Jetson Platform:** The Jetson series provides CUDA-compatible GPU acceleration and is the primary platform for edge AI requiring higher compute rates. The recommended inference optimization tool is TensorRT (NVIDIA), which uses layer fusion, auto-tuning, and mixed-precision inference (INT8/FP16). Performance varies by module:

- Jetson Nano: 20–30 FPS with lightweight models (YOLOv8-Nano, MobileNet-SSD) using TensorRT
- AGX Xavier NX: 60+ FPS with YOLOv5-Small

Power draw ranges from 5–15 W depending on the module and workload.

**b. Raspberry Pi Platform:** Raspberry Pi 4 and 5 use ARM Cortex-A processors without hardware neural-net accelerators, requiring highly optimized CPU inference engines such as TensorFlow Lite with XNNPACK delegation [34] or ONNX Runtime [35]. The Raspberry Pi 4 (quad-core Cortex-A72) achieves 5–10 FPS with INT8-quantized MobileNet-SSD under TensorFlow Lite [24]; the Raspberry Pi 5 (Cortex-A76) is 2–3× faster. Full-architecture models (ResNet50, VGG-16) are unsuitable due to compute and memory bandwidth constraints. Practical models are MobileNetV2, ShuffleNetV2, and NanoDet with INT8 quantization [11, 30].

### 4.2 Technical Deployment Challenges

- **Quantization trade-offs:** FP32→INT8 conversion reduces memory by ~4× with faster integer arithmetic [13], but typically degrades accuracy by 1–3% mAP depending on network architecture and quantization technique [36].
- **Thermal management and throttling:** Passive cooling (heat sinks) protects hardware from overheating, but sustained high-intensity inference triggers clock frequency scaling, causing variable latency and stuttering in real-time video processing.
- **Memory constraints:** Large backbone models with tens of millions of parameters can exceed embedded RAM (2–8 GB). Memory exhaustion causes inference failures and system instability, necessitating lightweight architectures (MobileNet, ShuffleNet, EfficientNet-Lite).
- **Power consumption and battery life:** A Jetson Nano running YOLOv5 at 30 FPS consumes ~10 W. Battery-constrained devices (drones, robots, IoT sensors) require adaptive strategies such as dynamic frame rate adjustment to remain within energy budgets.

---

## 5. Performance Evaluation and Comparative Analysis

**Table 1: Comparative Overview of Embedded Platforms for Real-Time Object Detection**

| Platform | Processor | Hardware | Models | Optimization | FPS Range | Power | Refs |
|----------|-----------|----------|--------|--------------|-----------|-------|------|
| GPU-Accelerated Edge | CUDA GPU | Jetson Nano, Xavier NX | YOLOv4-tiny, SSD-ResNet, MobileNet-SSD | TensorRT (FP16/INT8) | 5–60 | 5–15 W | [38] |
| CPU-Based Embedded | ARM CPU | Raspberry Pi 4/5 | SSD-MobileNet (TFLite), lightweight YOLO | TFLite (INT8), ARM NN, XNNPACK | 2–12 | 3–12 W | [39] |
| ASIC-Based Accelerators | Edge TPU | Google Coral Dev Board | SSD-MobileNetV2 (INT8), EfficientDet-Lite | Edge TPU Compiler (INT8) | 70–120 | <2.5 W | [40] |
| Mobile SoC | Heterogeneous SoC (CPU+NPU/DSP) | Modern Smartphones | YOLO-Lite, MobileNetV2/V3 | CoreML, NNAPI, SNPE | 20–45 | 2–5 W | [9] |

**Table 2: Comparison of Object Detection Benchmark Datasets**

| Dataset | Images | Categories | Avg Instances/Image | Primary Metric | Use for Embedded Systems |
|---------|--------|-----------|---------------------|----------------|--------------------------|
| MS COCO | 330K | 80 | 7.7 | mAP@0.5:0.95 | Primary benchmark; challenging scenes |
| PASCAL VOC 2007/2012 | 11.5K | 20 | 2.4 | mAP@0.5 | Lightweight model evaluation; fast benchmarking |
| ImageNet Detection | 450K | 200 | Variable | mAP@0.5 | Backbone pre-training; category diversity |
| Open Images V6 | 9M+ | 600 | Variable | mAP@0.5 | Large-scale pre-training; generalization testing |

**Table 3: Performance Characteristics of Lightweight Object Detection Models**

| Model | Backbone | Dataset | mAP | FPS | Size | Platform | Ref |
|-------|----------|---------|-----|-----|------|----------|-----|
| SSD-MobileNetV2 | MobileNetV2 | PASCAL VOC | ~72% | 20–30 | ~14 MB | Jetson Nano | [9] |
| YOLOv4-Tiny | CSPDarknet-Tiny | MS COCO | ~40% | 30–45 | ~23 MB | Jetson Nano | [41] |
| NanoDet | ShuffleNetV2 | MS COCO | ~33% | 60–90 | ~0.6–1.8 MB | ARM CPU | [30] |
| EfficientDet-Lite0 | EfficientNet-Lite | MS COCO | ~30% | 25–40 | ~4 MB | Edge TPU | [29] |
| YOLOv5-Nano | CSPDarknet-Nano | MS COCO | ~28% | 40–60 | ~1.9 MB | Jetson Xavier NX | [42] |

### 5.1 Platform-Level Performance Evaluation

As shown in Table 1, performance varies significantly by hardware. GPU-equipped devices (Jetson Nano, Xavier NX) leverage parallel computing for real-time inference with lightweight models and TensorRT, but at higher power draw. CPU-only platforms (Raspberry Pi) have limited inference speed. ASIC accelerators (Google Coral Edge TPU) demonstrate the best performance-per-watt ratio.

### 5.2 Model-Level Comparative Analysis

SSD-MobileNet architectures offer a good accuracy/efficiency balance and serve as a common baseline. Lightweight YOLO variants prioritize speed and small model size at the cost of detection performance, particularly for small objects. The EfficientDet-Lite series uses compound scaling and optimized feature fusion for improved accuracy at modest resource cost.

---

## 6. Future Trends in Embedded Object Detection

Research is moving from generic model compression toward **Hardware-Aware Neural Architecture Search (NAS)**, which automatically adapts network architecture to the computational profile of a target hardware platform (CPUs favor sequential computation, GPUs favor parallelism, NPUs favor fixed-point matrix arithmetic). Modern architectures like YOLOv10 and YOLOv11 are designed from the ground up with NPU constraints in mind.

### 6.1 Transformer-Based Detectors on the Edge

Vision Transformers (ViTs) are emerging as viable alternatives to CNNs at the edge:

- **Hybrid Models (e.g., MobileViT):** Combine CNNs' local-pattern focus with transformers' global context understanding.
- **Memory and Computational Complexity:** The primary hindrance is the quadratic complexity O(n²) with respect to the number of patches. Current research mitigates this with "linear attention" variants, reducing complexity to O(n).

### 6.2 Real-Time Adaptive Inference

Dynamic inference adaptively changes computational complexity based on runtime conditions (power budget, workload):

- **Adaptive Complexity Mechanisms:** Early-exit models or multi-backbone switching based on scene complexity [44]. For example, a security system might use a low-power MobileNet-SSD (2 W) during inactivity and switch to YOLOv8 (10 W) when a person is detected.

> **Figure 4 — Adaptive inference pipeline:** Input Video Stream → Low Power Trigger → (empty scene) MobileNet-SSD (2 W) ↔ (person detected) YOLOv8 (10 W) [Dynamic Switch, saves energy] → Output: Tracking & Alerting.

**Table 4: Expected Impact of Emerging Trends on Embedded Detectors**

| Trend | Expected Impact | Primary Hardware | Maturity |
|-------|----------------|------------------|----------|
| NPU Optimization | 5–10× speedup | Jetson Orin / RK3588 | High |
| Edge Transformers | Higher mAP in complex scenes | Jetson Xavier / Orin | Moderate |
| Adaptive Inference | ~40% power reduction | Raspberry Pi / ESP32-S3 | Emerging |

---

## 7. Conclusion

This review examined the development of single-stage object detection models and the challenges of migrating them from high-performance computing platforms to edge devices. The analysis highlights a paradigm shift toward hardware-friendly models capable of dynamically adjusting resource usage.

- **High-Performance Edge AI:** The NVIDIA Jetson line (Nano, Xavier, Orin) remains the best option due to TensorRT optimization and CUDA acceleration, achieving sub-30 ms latency.
- **Low-Cost / CPU-Constrained Deployment:** Raspberry Pi 4 and 5 are cost-effective when paired with lean models (MobileNet) or the Google Coral Edge TPU. Without dedicated acceleration, real-time performance (≥30 FPS) at high resolution remains out of reach.

## 8. Recommendations

- **Prioritize Hardware-Conscious NAS:** Optimize for latency, energy, and heat during the model design process.
- **Establish Standardized Benchmarking Protocols:** Measure not just mAP but also FPS, power consumption, and thermal characteristics.
- **Research Energy-Aware Training:** Design loss functions that account for energy usage in battery-powered systems.
- **Extend Multi-Task and Adaptive Frameworks:** Use dynamic inference to make optimal use of constrained edge capabilities.
