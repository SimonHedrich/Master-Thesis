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
| `progress_notes/2026-03-24_dataset-gap-analysis-and-supplementation-strategy.md` | LILA BC filtering results (3.2M images, 115/225 classes covered), 110-class gap analysis by region, and supplementation strategy: iNaturalist Open Data S3, NACTI, Open Images/COCO, Wikimedia Commons, BioCLIP 2 pseudo-labeling. |
| `progress_notes/2026-03-30_wikimedia-category-crawling.md` | Motivation and implementation of the Wikimedia Commons category crawl (`scripts/crawl_wikimedia_categories.py`) and keyword-based filter (`scripts/filter_wikimedia_categories.py`) that produce `reports/wikimedia_categories_filtered/`. |
| `progress_notes/2026-04-24_synthetic-data-kd-experimental-design.md` | Experimental design proposal for the synthetic data + KD thesis contribution: three-tier class stratification (data-rich / parity / synth-only), budget analysis ($300 → 12–15 species × 150 images), synthetic generation strategy, KD-specific hypotheses under data scarcity, and evaluation methodology for small test sets. |
| `progress_notes/2026-04-24_training-setup-and-model-smoke-test.md` | Training environment setup (4 Docker images for YOLO/SpeciesNet/NanoDet/PaddlePaddle), smoke test results for all teacher and student models on RTX 3060, full SpeciesNet two-stage pipeline implementation (MegaDetector + EfficientNetV2-M, soft-label generation for KD), NanoDet-Plus-m and PicoDet-S training configs (nc=225), and class adaptation approach per model family. |
| `progress_notes/2026-04-25_synthetic-scenario-diversity.md` | Problem analysis for repetitive synthetic images (same posture/background per species), Gemini prompt length research (32K chars vs DALL-E 1K / SD 77 tokens), four solution options, and recommendation to pre-generate per-species ecological scenario banks via a cheap text LLM. |
| `automatic_image_qualitiy_filtering.md` | Design notes and rationale for the automated image quality filtering pipeline: staged funnel (metadata → heuristics → MegaDetector → Florence-2 VLM rescue → Florence-2 caption generation → LLM caption evaluation → SpeciesNet classification + 225-class filtering), implementation details for each script (1–7), key design decisions, and per-source strategy. |
| `synthetic-image-generation-model-research.md` | Research report on local text-to-image models for generating wildlife training images on an RTX 3060 12 GB: model profiles (FLUX.1, SD 3.5, SDXL family), benchmarked generation speeds, acceleration techniques (NF4, torch.compile, DeepCache, Nunchaku), and a tiered recommendation with sample diffusers code. |
| `gpu_training_options.md` | Analysis of NVIDIA A40 GPU options for the KD training campaign: VRAM requirements per model, comparison of 48 GB single-instance vs 2 × 12 GB split, and recommendation (two 12 GB instances) with rationale and practical scheduling notes. |

### Plans

Decision records and design documents written before implementation.

| File | Description |
|------|-------------|
| `plans/2026-04-29_pipeline-timeline-and-parallel-work.md` | Timeline and parallel-work coordination for the dataset quality pipeline. |
| `plans/2026-04-30_speciesnet-classification-strategy.md` | Detailed design decisions for pipeline steps 6–7: how to handle SpeciesNet misclassifications (hierarchical match levels), multi-animal images (retain for KD advantage), human co-presence (retain, track with flag), 225-class taxonomy mapping algorithm, and which detections to classify. Basis for `6-classify_speciesnet.py` and `7-filter_speciesnet.py`. |
| `plans/2026-05-04_dataset-construction-strategy.md` | Per-class data strategy: source trust levels (iNaturalist / Wikipedia / GBIF trusted; ImageCV / OpenImages require SpeciesNet filtering), four tiers by post-filter image count (< 100 / 100–499 / 500–1499 / ≥ 1500), synthetic image budget (≤ €500 / max ~10k images), manual review policy, and the Tier 3 real-vs-synthetic comparison design. |
| `plans/2026-05-04_dataset-construction-action-plan.md` | Companion to the construction strategy: evaluates the strategy against `images-per-class-analysis.md`, overlays track-specific thresholds (KD student 300 / teacher 500 / Track B 1000) on the four tiers, identifies the missing trusted-vs-unverified breakdown in `reports/speciesnet_filter.md`, and lists prioritised next steps and open decisions before dataset assembly. |
| `plans/2026-05-06_dataset-caps-and-synthetic-counts.md` | Final dataset split design: revised 4-band structure (A < 150 / B 150–249 / C 250–399 / D ≥ 400) with 200-image training budgets for Bands A–C, per-class training caps and test splits, synthetic image targets (13,700 images / €685), and real-vs-synthetic comparison experiment for 5 Band D classes. Supersedes the original 4-tier boundaries. |
| `plans/2026-05-06_unreviewed-sn-fail-classes.md` | Follow-on to the SN shrinkage investigation: audits all unreviewed Tier 2/3 classes where `family_mismatch_high_confidence` or `low_speciesnet_confidence` caused >60% image loss. Classifies cases by fail reason, estimates recoverable images, and recommends a two-track resolution (manual review for family_mismatch classes; threshold relaxation for near-threshold match_level classes). |
| `plans/2026-05-12_synthetic-image-generation-strategy.md` | Design document for generating the 12,600 synthetic training images (Band A: 200/class × 50 classes; Band B: 100/class × 26 classes). Covers the six dimensions of image variation (angle, distance, behavior, environment, lighting, occlusion), per-species ecological guild assignments, train/val split rationale (160/40 curated split for Band A; all-train for Band B), the full prompt template with worked examples (walrus, kinkajou), JSONL image-list schema, and pre-generation checklist (scientific name mapping, QC pipeline, bounding-box annotation via MegaDetector). |
