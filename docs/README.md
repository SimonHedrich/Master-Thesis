# Docs

> Keep this file up to date whenever new files are added to this directory.

This directory contains documentation, analysis, and notes produced during the Master's Thesis on optimizing deep learning object detection models for real-time inference on embedded hardware.

## Files

### Analysis & Reference

| File | Description |
|------|-------------|
| `thesis-overview.md` | High-level overview of the thesis subject, research objectives, development paths, and technical constraints — intended as a context prompt for LLMs. |
| `hardware-proxy-selection.md` | Analysis of SBC alternatives to the Qualcomm QCS605 for development and benchmarking. Recommends the Raspberry Pi 5 (8GB) as the primary proxy device. |
| `research-and-experimentation-plan.md` | Detailed research and experimentation plan covering dataset selection, model choices, training pipeline (zero-shot → fine-tuning → KD ladder → quantization), evaluation framework, and expected academic contributions. |
| `species-label-selection.md` | Final analysis and decisions for the student model's 225 output classes: research findings on class count ceilings for nano models, detailed consolidation reasoning by taxonomic group, complete label table, and summary statistics. |
| `species-label-selection-extended.md` | Extended 480-class label list: minimal pruning of the PO's 483 labels (removes nocturnal/tiny/range-limited species) plus 17 genus/family fallback entries for hierarchical inference. For use with larger student models or two-stage pipeline comparisons. |
| `species-label-research.md` | Deep research report on optimal taxonomic output design for lightweight mammal detection models, covering class capacity limits, label granularity strategy, North American coverage gaps, and hierarchical fallback architecture. |
| `supplementary-dataset-research-prompt.md` | LLM research prompt for identifying open wildlife image datasets to supplement the GBIF/SpeciesNet training data. Covers commercial license requirements, gap species analysis, and dataset combination strategy. |
| `supplementary-dataset-research.md` | Research report on commercially viable open wildlife datasets (LILA BC, GBIF, Open Images, COCO), gap coverage analysis, license risk assessment, and data combination strategy. |
| `dataset-supplementation-plan.md` | Concrete step-by-step plan for building the training dataset: LILA BC download, GBIF gap export, Open Images/COCO integration, label noise handling, synthetic data, and unified dataset assembly with estimated ~160k–300k final images. |

### Progress Notes

| File | Description |
|------|-------------|
| `progress_notes/2026-03-05_first-meeting.md` | Notes from the first supervisor meeting covering quantization, knowledge distillation, transfer learning, and methodology decisions. |
| `progress_notes/2026-03-11_dataset-stakeholder-meeting-and-model-architecture.md` | Meeting with Danielle (AX Visio Product Owner) on dataset preparation, YOLOv5 licensing, geo-filtering, and model architecture research (teacher/student candidates, research question). |
| `progress_notes/2026-03-18_speciesnet-pipeline-and-experiment-design.md` | Analysis of the current Swarovski pipeline (YOLOv5s + SpeciesNet), the two-stage vs. one-shot research gap, model size comparison (~200× gap), dataset status, and KD training strategy. |
| `progress_notes/2026-03-24_lila-bc-dataset-analysis.md` | Results from filtering 12.6M LILA BC camera trap images against the 225-class label set: 3.2M images matched, 115/225 classes covered, coverage gap analysis by region, and next steps. |
