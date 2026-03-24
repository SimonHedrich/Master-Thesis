# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

This is a research repository for a Master's Thesis on **optimizing deep learning object detection models for real-time inference on embedded hardware**. It contains documentation, research notes, and utility scripts.

- **Domain:** Wildlife animal species detection (non-bird mammals)
- **Target hardware:** Qualcomm QCS605 (Hexagon 685 DSP, Adreno 615 GPU)
- **Proxy hardware:** Raspberry Pi 5 (8GB) — chosen for software stability during development
- **Institutional context:** inovex GmbH, Data Management & Analytics dept.

## Repository Structure

```
docs/         — Analysis documents and progress notes produced during the thesis
research/     — Papers (PDF + Markdown summaries) and literature notes
resources/    — Raw data files and example images from the AX Visio binocular
scripts/      — Utility scripts (data exploration, visualization, etc.)
```

**Key docs:**
- `docs/thesis-overview.md` — High-level research objectives and technical approaches
- `docs/hardware-proxy-selection.md` — Why RPi 5 was chosen over alternatives
- `docs/object-detection-models-for-embedded-systems.md` — Model architecture analysis
- `docs/knowledge_distillation_research_overview.md` — KD approaches and findings
- `docs/progress_notes/` — Chronological meeting and thinking notes

**Key research:**
- `research/cv-wildlife-classification-resources.md` — Curated reading list
- `research/A Review of Real-Time Deep Learning–Based Object Detection Models.md` — Primary survey paper on YOLO/SSD/NanoDet for edge deployment

## Thesis Research Context

### Core Research Question
Does distilling a large teacher model into a lightweight student model yield better results than directly fine-tuning the student on the target wildlife domain — especially given the domain shift from COCO-style classes to animal species?

### Technical Approach
1. **Teacher models** (too large for target hardware): YOLOv12, RT-DETR, SpeciesNet, DINOv3
2. **Student models** (deployable on QCS605): YOLO-nano variants, NanoDet, PicoDet, EfficientDet-Lite
3. **Pipeline:** Fine-tune teacher on wildlife species → distill into student → quantization-aware training → benchmark on RPi 5 proxy

### Dataset Strategy
- Primary: [iNaturalist Competition](https://www.kaggle.com/competitions/inaturalist-2021) (open dataset, preferred)
- Class universe: SpeciesNet taxonomy, filtered to non-bird mammals
- Species inclusion threshold: Based on GBIF image counts (`resources/GBIF_image_counts.csv`)
- Geo-filtering: Post-hoc output filter (not model-level input), applied after inference

### Important Constraints
- **YOLOv5 license:** Only commercially usable up to commit `5cdad89` — later commits require additional licensing
- Run own benchmarks rather than relying on published numbers
- Evaluate synthetic data only on real photographs (train on mix, test on real)

## Maintaining Documentation

Both `docs/README.md` and `research/README.md` serve as indices — keep them updated whenever files are added to those directories.
