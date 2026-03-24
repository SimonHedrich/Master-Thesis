# Research & Experimentation Plan
## Knowledge Distillation for Wildlife Detection on Embedded Hardware

---

## Context

The AX Visio binocular (running on Qualcomm QCS605) needs a real-time wildlife species detector. The core academic question is: **does distilling a large teacher model into a lightweight student yield better results than directly fine-tuning the student on the target wildlife domain — especially given the domain shift from COCO-style classes to animal species?**

This plan designs a series of controlled experiments that answer this question empirically while producing a deployable INT8 model for the AX Visio as a concrete end product. The experiments are structured as an ablation ladder: each step adds one variable, making the contribution of each technique measurable.

---

## 1. Dataset

### 1.1 Primary Dataset: iNaturalist 2021 Competition

**Why iNaturalist:** It is open-licensed, large-scale (~2.7M images, 10,000 species), has clean taxonomy, and is already used in prior wildlife ML work (Van Horn thesis, BioCLIP). The competition train/val splits are standardized and reproducible.

**Class Filtering Pipeline:**
1. Start from SpeciesNet's class taxonomy as the canonical label set.
2. Filter to non-bird mammals only (matches AX Visio use case; excludes birds, marine life, plants).
3. Cross-reference against `resources/GBIF_image_counts.csv` — include only species with ≥100 GBIF images as a data availability signal.
4. Apply a minimum of ~50 iNaturalist training images per class to prevent degenerate training.
5. Expected result: ~120–200 mammal species.

**Why this threshold approach:** The long-tail literature (Van Horn 2017, "The Devil Is In The Tails") shows that below ~50 examples per class, supervised detectors rarely generalize. Having a data availability floor is better than including nearly-empty classes that inflate class count but harm metrics.

**Bounding Box Annotations:**
iNaturalist 2021 provides classification labels, not detection boxes. Two approaches, choose based on available time:
- **Preferred:** Use MegaDetector (YOLOv5x6) to generate pseudo bounding boxes for all training images. This is established practice (used by "To crop or not to crop", 2024) and produces high-quality boxes.
- **Fallback:** Use whole-image boxes (image boundary as bbox). This degrades localization metrics but preserves classification signal — acceptable for a first baseline.

**Geo-Filtering:** Applied **post-hoc at inference time only**, not baked into the dataset. The model learns all species; a regional allow-list filters predictions at deployment.

---

## 2. Model Selection

### 2.1 Teacher Models (too large for QCS605, used only during training)

| Model | Params | mAP (COCO) | Domain | Why |
|-------|--------|------------|--------|-----|
| YOLOv8s | 11.2M | 44.9 | General (COCO) | Mature KD ecosystem; CWD/FGD/LD all implemented for YOLOv8 |
| YOLOv12m or RT-DETRv2-R50 | ~26M | ~50+ | General (COCO) | Higher-capacity ceiling; tests if a bigger teacher helps |
| SpeciesNet | ~65M images trained | — | Wildlife | Already domain-aligned; the most interesting teacher |

**Why three teachers:** The key experimental variable is the teacher's domain alignment. A COCO-only teacher represents zero domain adaptation; YOLOv8s fine-tuned on wildlife represents a trained domain adapter; SpeciesNet represents a pre-existing expert. Comparing all three answers how much domain alignment in the teacher matters.

### 2.2 Student Models (target deployment candidates)

| Model | Params | Size | Framework | Why |
|-------|--------|------|-----------|-----|
| YOLOv11n | 2.6M | ~5.4MB | Ultralytics | Best accuracy/size in YOLO family; Ultralytics natively supports KD |
| NanoDet-Plus-m | 1.8M | ~3.6MB | NCNN | Designed for ARM; ShuffleNetV2 backbone; fastest on RPi |
| PicoDet-S | 1.1M | ~2.4MB | PaddleDetection | Smallest viable; ESNet + PAN neck |

**Why three students:** Each represents a different architectural paradigm and size point. If all three benefit from KD, the finding is robust. If only some do, it reveals which design choices are amenable to distillation.

**YOLOv5 note:** Excluded. Later commits require additional commercial licensing beyond commit `5cdad89`.

---

## 3. Training Pipeline

The experiments form a controlled ladder. Each condition differs by exactly one variable from its predecessor.

### Phase 0: Zero-Shot Baselines (no training on wildlife)

For each student:
- Run COCO-pretrained weights on wildlife test set → measures raw COCO→wildlife transfer.
- Run SpeciesNet on the same test set → upper-bound reference for domain-expert models.

**Purpose:** Establishes whether domain shift is actually a problem (likely: significant mAP drop expected), and gives a ceiling for what a perfect domain-expert achieves.

### Phase 1: Direct Fine-Tuning (Supervised Baseline — the comparison condition)

For each student:
- Fine-tune from COCO-pretrained weights on the wildlife training set.
- Use class-balanced sampling (square-root resampling) to mitigate long-tail bias.
- Train until convergence (~100–150 epochs); save best checkpoint by val mAP.
- Standard augmentations: mosaic, random horizontal flip, HSV jitter, scale jitter.

**Why class-balanced sampling:** Without it, head classes (e.g., common deer species) dominate gradient updates and macro-F1 suffers. Square-root resampling is a practical compromise between uniform (over-samples rare) and natural (ignores rare).

**This is the primary comparison baseline.** Every KD condition must beat this to justify the added complexity of distillation.

### Phase 2: Teacher Fine-Tuning

- Fine-tune **YOLOv8s** on the same wildlife training set → becomes the domain-adapted teacher.
- Evaluate teacher mAP → this is the distillation performance ceiling for feature-based methods.
- **SpeciesNet** is used as-is (no fine-tuning needed; it is already a domain expert).

**Why fine-tune the teacher separately before distillation:** Two-stage KD (fine-tune teacher first, then distill) consistently outperforms one-stage end-to-end KD because the teacher has already learned task-relevant features before being asked to transfer them.

### Phase 3: Knowledge Distillation Experiments

Each distillation variant uses a student initialized from COCO-pretrained weights (not yet fine-tuned on wildlife) and trains on the wildlife dataset with a distillation loss added.

#### 3a. Logit-Based KD (Hinton, Temperature Scaling)

- Soft cross-entropy between teacher and student class probability distributions.
- Temperature T ∈ {4, 8} — sweep to find optimal.
- Loss: `L = α·L_task + (1-α)·L_KD`, with α ∈ {0.5, 0.7}.
- Teacher: COCO-only YOLOv8s (not fine-tuned).

**Expected result:** Modest gain or neutral vs. direct FT. The semantic gap between COCO teacher outputs and wildlife student targets limits this approach — logit-based KD degrades when class vocabularies differ.

**Academic value:** Confirms the limitation of logit-based KD under domain shift; motivates feature-based approaches.

#### 3b. Feature-Based KD — Channel-Wise Distillation (CWD)

- Align intermediate feature map channel statistics between teacher and student.
- Use a projection layer to match channel dimensions where architectures differ.
- Apply to neck (FPN/PAN) feature maps — most semantically informative.
- Teacher: YOLOv8s fine-tuned on wildlife.

**Why CWD:** Avoids the semantic class mismatch problem entirely (features are architecture-agnostic). CWD normalizes across spatial dimensions so a large teacher FPN map can align with a smaller student FPN map without strict spatial correspondence. Established baseline in the object detection KD literature.

#### 3c. Feature-Based KD — Focal and Global Distillation (FGD)

- Adds a foreground-focused spatial attention mask to feature alignment: foreground regions (objects) are weighted more heavily than background.
- Addresses the key object detection KD challenge: most spatial locations are background, diluting the distillation signal.
- Teacher: YOLOv8s fine-tuned on wildlife.

**Why FGD on top of CWD:** Wildlife images often have small, partially occluded animals against cluttered backgrounds. Standard feature-level distillation averages over all spatial locations; FGD concentrates gradient on where the animals actually are.

#### 3d. Localization Distillation (LD)

- Teacher provides soft bounding box distributions (not hard boxes) → student learns from distributional targets.
- Replaces hard IoU loss with KL divergence between teacher and student box prediction distributions.
- Combined with 3c (FGD + LD = full pipeline).

**Why LD:** For wildlife detection, precise localization of small/partially visible animals is a key failure mode. LD transfers the teacher's uncertainty about box boundaries — more informative than binary hard labels.

#### 3e. Domain-Shift Experiment: Teacher Source Comparison

Run the best distillation method (from 3b–3d) with three different teachers:
1. YOLOv8s (COCO-only, not fine-tuned) — "cross-domain teacher"
2. YOLOv8s fine-tuned on wildlife — "domain-adapted teacher"
3. SpeciesNet — "domain-expert teacher"

This is the **core experiment** directly answering the thesis research question. Each teacher represents a different level of domain alignment.

### Phase 4: Quantization

Apply to the best-performing student configuration from Phase 3.

- **Step 1 — PTQ (Post-Training Quantization):** Convert to INT8 using a calibration set (~500 images). Measure mAP drop.
  - If mAP drop ≤ 2%: PTQ is sufficient → use for deployment.
  - If mAP drop > 2%: Proceed to QAT.
- **Step 2 — QAT (Quantization-Aware Training):** Fine-tune with fake quantization nodes inserted. Use Multi-level BatchNorm if layer statistics are unstable.
- **Step 3 — Format Conversion:** Export to ONNX → TFLite (for RPi 5 benchmarking) and ONNX → SNPE DLC (for QCS605 target).

---

## 4. Evaluation Framework

### 4.1 Primary Accuracy Metrics

- **mAP@0.5** — standard detection accuracy threshold (IoU=0.5)
- **mAP@0.5:0.95** — stricter; better reflects localization quality
- **Macro-averaged F1** — treats each species equally; critical for long-tail; matches AX Visio real-world need (users care about rare species, not just common deer)
- **Per-class AP** — diagnostic; identifies which species benefit/suffer from each method

### 4.2 Efficiency Metrics (measured on actual hardware)

- **Inference latency (ms) on Raspberry Pi 5 (8GB)** — actual wall-clock time, not FLOPs
- **Peak RAM usage (MB)** — constrained on embedded
- **Model file size (MB)** — constrained on AX Visio storage
- **FLOPs** — theoretical compute for comparison with published papers

**Benchmarking protocol:** Warm-up 20 runs, measure 100 runs, report median ± IQR. Batch size = 1 (embedded inference is sequential). Use NCNN for NanoDet, TFLite for YOLO/PicoDet.

### 4.3 Experiment Comparison Matrix

Repeat this table for each student model (YOLOv11n, NanoDet-Plus-m, PicoDet-S):

| Condition | mAP@.5 | mAP@.5:.95 | Macro-F1 | RPi5 ms | Size MB | ΔvsFT |
|-----------|--------|------------|----------|---------|---------|-------|
| Student: Zero-shot (COCO) | | | | | | ref |
| Student: Direct FT (baseline) | | | | | | 0 |
| Student: Logit KD (COCO teacher) | | | | | | |
| Student: CWD (domain teacher) | | | | | | |
| Student: FGD (domain teacher) | | | | | | |
| Student: FGD+LD (domain teacher) | | | | | | |
| Student: FGD+LD (COCO teacher) | | | | | | |
| Student: FGD+LD (SpeciesNet) | | | | | | |
| Teacher: FT reference | | | | | | |
| Student: Best + PTQ INT8 | | | | | | |
| Student: Best + QAT INT8 | | | | | | |

### 4.4 Long-Tail Analysis

Bin species into three frequency tiers based on training image count:
- **Head:** top 20% most frequent classes
- **Middle:** next 40%
- **Tail:** bottom 40% (rarest species)

For each condition, report mAP broken down by tier.

**Hypothesis:** KD from a domain-adapted teacher will disproportionately benefit tail-class species, because the teacher transfers learned representations for rare appearances. Direct fine-tuning on scarce tail data tends to overfit or underfit. This would be a meaningful academic finding.

### 4.5 Statistical Rigor

- Run **3 seeds** per experiment (different random inits + data shuffles).
- Report mean ± standard deviation.
- For key pairwise comparisons (KD best vs. direct FT), run a paired t-test or Wilcoxon signed-rank test across seeds.

---

## 5. Expected Results & Academic Contribution

### Predicted Outcomes

| Finding | Confidence | Reasoning |
|---------|-----------|-----------|
| Direct FT substantially beats zero-shot | High | Domain shift is real; iNaturalist species not in COCO |
| Logit KD ≈ direct FT (no gain) | Medium-high | Semantic gap between COCO classes and wildlife species limits soft targets |
| Feature-based KD (CWD/FGD) > direct FT by 2–5% mAP | Medium | Feature alignment avoids class vocabulary mismatch; well-supported in literature |
| Domain-adapted teacher > COCO-only teacher for KD | High | Teacher domain alignment matters — foundational result of the WAKD literature |
| SpeciesNet as teacher = best or near-best KD performance | Medium | Domain expert, but architectural gap may limit feature alignment |
| Tail classes benefit most from KD | Medium | Less training data → teacher supervision more impactful on rare species |
| PTQ causes >2% mAP loss; QAT recovers most of it | Medium | Typical for detection heads with heterogeneous activation ranges |

### Where the Plan Could Surprise You

- SpeciesNet may be architecturally incompatible with feature-based KD (different backbone family) → may only support logit-based KD, limiting its advantage.
- NanoDet-Plus may compress better than YOLOv11n under KD despite fewer parameters — its ShuffleNet architecture may be more quantization-friendly.
- If direct fine-tuning on iNaturalist is already very strong (dataset large enough), the KD benefit may be smaller than expected — which is itself a publishable null result with good methodology.

### Academic Contributions

1. **Systematic comparison of KD teacher domain alignment** under wildlife domain shift — answers a question not yet addressed empirically in the wildlife CV literature.
2. **Empirical evaluation of feature-based KD methods** (CWD, FGD, LD) on a wildlife detection task — applied to a non-standard domain.
3. **End-to-end pipeline** from KD to INT8 quantization to embedded benchmark — demonstrates real deployment feasibility, not just mAP improvements.
4. **Long-tail analysis** of KD benefit distribution across species frequency tiers.

### End Product for AX Visio

- Deployable YOLOv11n (or NanoDet-Plus-m, depending on RPi 5 benchmarks) INT8 model.
- Trained on iNaturalist ~150 mammal species.
- Optimized for Hexagon DSP via SNPE DLC format.
- Post-hoc geo-filter layer (configurable regional allow-list).
- Target: ≤30ms latency on QCS605 (capped at ~60% of RPi 5 measurement, per `hardware-proxy-selection.md`).

---

## 6. Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Data prep: filter, bbox generation (MegaDetector), dataset splits | ~2 weeks | Ready dataset, class list |
| Phase 0–1: Zero-shot + direct FT all students | ~2 weeks | Baseline numbers for all 3 students |
| Phase 2: Teacher fine-tuning | ~1 week | Domain-adapted YOLOv8s teacher |
| Phase 3a–3d: KD experiments | ~3 weeks | Full distillation ladder results |
| Phase 3e: Teacher source comparison | ~1 week | Core thesis result |
| Phase 4: Quantization + RPi 5 benchmarks | ~1.5 weeks | Deployment-ready models + latency table |
| Analysis, ablations, writing | ~3 weeks | Draft chapters |

---

## 7. Verification

- **Accuracy:** mAP on a held-out iNaturalist test set (not touched during training or hyperparameter tuning).
- **Latency:** Actual timed inference on a physical Raspberry Pi 5 with fixed batch size = 1.
- **Correctness of KD:** Training loss curves should show the distillation loss decreasing; student feature maps should become more correlated with teacher maps as training progresses (visualize with Grad-CAM).
- **Quantization:** Compare INT8 model predictions vs. FP32 model on a calibration set — should match >95% of the time on top-1 class.
- **End-to-end:** Run the final INT8 model on sample images from `resources/` (AX Visio binocular images) and verify visually that bounding boxes land on animals.
