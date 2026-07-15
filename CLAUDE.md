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
- `docs/2026-03-09_thesis-overview.md` — High-level research objectives and technical approaches
- `docs/2026-03-09_hardware-proxy-selection.md` — Why RPi 5 was chosen over alternatives
- `docs/2026-03-10_object-detection-models-for-embedded-systems.md` — Model architecture analysis
- `docs/2026-03-12_knowledge_distillation_research_overview.md` — KD approaches and findings
- `docs/progress_notes/` — Chronological meeting and thinking notes

**Key research:**
- `research/cv-wildlife-classification-resources.md` — Curated reading list
- `research/A Review of Real-Time Deep Learning–Based Object Detection Models.md` — Primary survey paper on YOLO/SSD/NanoDet for edge deployment

## Running Code

### Python scripts: always uv, always from the repo root
- Module-safe names: `uv run python -m scripts.<package>.<module>`
  (e.g. `uv run python -m scripts.training.yolov5s.run_training_pipeline`)
- Numbered pipeline scripts (`1-foo.py` — invalid module names, cannot use `-m`):
  `uv run python scripts/<dir>/<N>-<name>.py`
- Every runnable script's module docstring must state its exact run command in this form.

### Containers: one image, exec in, then uv
- `make build` builds the single `training` image; `make run` starts the container and
  execs a bash shell inside it; `make stop` stops/removes it.
- To run a script in the container: `make run` first, then the standard uv command inside.
- Default to the one shared container for everything. Only create a separate
  image/container when something genuinely cannot run in the default one
  (document the reason when you do).

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
- **Primary evaluation = the mixed (real + synthetic) test set.** The default headline metric is computed over the union of the real test images and the balanced 225×50 synthetic test set. Rationale: the consistent 50 synthetic images/class stabilise evaluation for classes with few or low-quality real photos (Band A), while remaining a negligible, consistent addition for well-resourced classes (Band D, up to 500 real test images). The model must **never** be judged on synthetic images alone. The **real-only breakout** is always reported alongside the mixed headline as the primary-evaluation figure and the anchor for any comparison to public (real-image) benchmarks. The real-vs-synthetic domain-shift delta is monitored as a watchdog: **if a clear discrepancy between the mixed and real (or synthetic and real) results emerges, the default evaluation axes will be revised.** See `docs/plans/2026-06-10_model-evaluation-strategy.md`.

## Maintaining Documentation

Both `docs/README.md` and `research/README.md` serve as indices — keep them updated whenever files are added to those directories.
