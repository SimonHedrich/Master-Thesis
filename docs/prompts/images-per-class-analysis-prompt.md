# Prompt: How Many Images Per Class Are Needed for Training?

> **Usage:** Paste the text below (from "---" onward) into an LLM with strong ML knowledge (e.g. Claude Opus, GPT-4o, Gemini 1.5 Pro). The goal is a practical, project-specific analysis — not generic advice.

---

## Context: Master's Thesis — Wildlife Species Detection on Embedded Hardware

You are advising on a Master's thesis that trains and deploys wildlife animal detection models on embedded hardware. Here is the full context.

### Task

Train object-detection models to recognise **225 non-bird mammal species** (e.g. deer, wolves, primates, big cats, rodents) in photographs taken by binoculars or camera traps. The models must eventually run in real time on a **Qualcomm QCS605 SoC** (Hexagon 685 DSP, Adreno 615 GPU). A **Raspberry Pi 5 (8 GB)** is used as a proxy during development.

### Training Pipeline — Two Parallel Tracks

**Track A — Knowledge Distillation (primary):**
1. Fine-tune a large teacher model on the 225-class wildlife dataset.
2. Use that fine-tuned teacher to distil knowledge into a small student model.
3. Apply quantization-aware training.
4. Deploy student on QCS605.

**Track B — Direct Fine-Tuning (baseline):**
1. Fine-tune the small student model directly on the 225-class wildlife dataset.
2. Same QAT + deployment.

Comparison of Track A vs Track B is the core research question.

### Models

| Role | Model | Parameters | Notes |
|------|-------|-----------|-------|
| Teacher | SpeciesNet (EfficientNetV2-M backbone) | ~54 M | Google's wildlife classifier; already trained on global species |
| Teacher alternative | YOLOv8s | ~11 M | Object detector option for KD |
| Student | YOLOv11n / YOLOv12-N | ~2.6 M | Primary student candidates |
| Student | NanoDet-Plus-m | ~1.17 M | Lightweight option |
| Student | PicoDet-S | ~0.99 M | Most compact option |

The student must fit within the QCS605's memory budget. Domain shift is significant: teacher models were pre-trained on COCO or broad nature datasets; target domain is specifically **mammals in natural photography**.

### Dataset Sources

Images are collected from five open sources: iNaturalist, GBIF, Wikimedia Commons, Open Images, images_cv. All images pass a quality-filtering pipeline (metadata checks → MegaDetector → SpeciesNet classification → caption scoring).

---

## Current Dataset State

### Raw Coverage (before SpeciesNet quality filtering)

From `coverage_report.md` — images that passed initial heuristic filters (metadata, resolution, MegaDetector confidence ≥ 0.5):

| Tier | Image count range | Classes |
|------|-------------------|---------|
| Excellent | ≥ 1,500 | 83 |
| Good | 1,000 – 1,499 | 24 |
| Marginal | 500 – 999 | 30 |
| Low | 100 – 499 | 59 |
| Critical | < 100 | 29 |

Total raw passed images: **457,279** across 225 classes. Estimated usable after a 20% quality buffer: **365,830**. The Ultralytics recommended guideline is ≥ 1,500 images per class (≥ 1,200 after quality buffer).

### After SpeciesNet Quality Filtering

From `speciesnet_filter.md` — applying stricter classification-level filtering (SpeciesNet score ≥ 0.3, family-level species match required):

- Input images: **465,130**
- Passed: **158,667 (34.1%)**
- Failed: **306,463 (65.9%)**
- Primary failure reason: SpeciesNet returned no species match (35.3% of failures)

This means the **effective usable dataset is approximately 1/3 of raw counts**. A class that shows 1,500 raw images may only have ~500 after strict quality filtering. The 83 "Excellent" classes are likely reduced to far fewer usable images for many species.

Additionally:
- 80,538 images (17.3%) show multiple animals — these require bounding-box annotation decisions.
- Pass rates vary dramatically by species: some classes (e.g. Grevy's zebra 88.5%, giraffe 73.9%) are high; many others drop to 0% (eared seals, elephant seal, koala, brown-throated sloth had 0% in some datasets).

---

## Analysis Request

Please analyse the practical effects of different per-class image counts on fine-tuning and knowledge distillation for the models in this project. Structure your answer around the following count brackets:

- **10 images / class**
- **30 images / class**
- **50 images / class**
- **100 images / class**
- **200 images / class**
- **500 images / class**
- **1,000 images / class**
- **≥ 1,500 images / class**

For each bracket, address:

1. **Nano-scale students (0.99 M – 2.6 M params: PicoDet-S, NanoDet-Plus-m, YOLOv11n/12n):**
   - Expected detection quality and generalisation
   - Overfitting risk
   - Whether knowledge distillation from the teacher can compensate for low data volume at this bracket

2. **Mid-scale teacher (YOLOv8s at ~11 M params, SpeciesNet at ~54 M):**
   - Expected fine-tuning quality at this bracket
   - Can these larger models extract useful soft labels / feature maps even when fine-tuned on limited data?

3. **Class imbalance effects:**
   - This dataset is highly skewed: some classes have > 30,000 images, others < 30. What happens when a class at this bracket is trained alongside classes with 5,000 – 30,000 images?
   - Does the answer change between Track A (KD) and Track B (direct fine-tuning)?

4. **Impact on KD effectiveness:**
   - At each bracket, how confident and reliable are the teacher's soft labels?
   - Is there a minimum count below which teacher predictions become too noisy to be useful for distillation?

5. **Domain shift consideration:**
   - Wildlife photography has high intra-class variance (different lighting, occlusion, distance, camera angle). Does this change the minimum viable count compared to COCO-style benchmarks?

---

## Recommendation Request

After your analysis, please give a concrete recommendation addressing all of the following:

1. **Hard minimum threshold:** Below what per-class image count should a class be **excluded entirely** from training (for both tracks)? Justify based on the analysis above, not just literature rules of thumb.

2. **Soft target threshold:** What is the preferred minimum per-class count to include a class with acceptable expected quality? Separate your answer for:
   - Track A (teacher fine-tuning)
   - Track A (student trained via KD from that teacher)
   - Track B (direct student fine-tuning)

3. **Tiered strategy:** Given that 83 classes already have ≥ 1,500 raw images (but likely ~500 after strict quality filtering), and 88 classes have < 500 raw images, should the project train on all 225 classes or reduce the class set? If reducing, what count makes sense?

4. **Post-filter counts vs. raw counts:** The 20% quality buffer in the coverage report is optimistic — actual filtering removes ~66% of images. Should thresholds be set based on raw image counts or post-filter counts? What multiplier should be assumed when planning data collection?

5. **Augmentation:** Can standard augmentation strategies (mosaic, mixup, random crop, colour jitter, copy-paste) help close the gap for data-sparse classes? If so, by roughly how much does effective count increase, and does this affect your minimum threshold?

Be specific and practical. If relevant, cite expected mAP or recall ranges from transfer-learning literature for the count brackets you recommend.
