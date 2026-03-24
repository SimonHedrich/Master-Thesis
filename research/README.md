# Research

> Keep this file up to date whenever new files are added to this directory.

This directory contains papers, notes, and resources collected for the Master's Thesis on optimizing deep learning object detection models for real-time inference on embedded hardware.

## Files

### Papers (PDF + Markdown summaries)

| File | Description |
|------|-------------|
| `To crop or not to crop.pdf` / `.md` | Gadot et al. (2024), *IET Computer Vision* — Evaluates whole-image vs. detector-cropped classification on large camera trap datasets. Shows ~25% macro F1 improvement from incorporating object detection. |
| `Towards a Visipedia.pdf` / `.md` | PhD thesis by Grant Van Horn (Caltech, 2019) — Broad background on computer vision for species recognition, combining CV with expert communities. |
| `BIOCLIP.pdf` | BioCLIP (2023) — Vision foundation model for the Tree of Life using hierarchical taxonomic information encoded via CLIP. |
| `macroinvertebrate specimens.pdf` | PeerJ paper — Investigates how much training data is needed for accurate deep learning-based species identification; relevant for few-shot / low-data scenarios. |
| `Generalization for Rare Classes.pdf` | Beery et al., WACV 2020 — Explores using synthetic examples to improve generalization for rare classes in wildlife classification. |
| `A Review of Real-Time Deep Learning–Based Object Detection Models.md` | Shah (2025), *NIJEC* — Surveys real-time object detection models (YOLO, SSD, MobileNet, NanoDet) for resource-constrained embedded systems; covers model compression (quantization, pruning, knowledge distillation) and deployment on Jetson/Raspberry Pi/Coral hardware. |

### Notes & Resources

| File | Description |
|------|-------------|
| `cv-wildlife-classification-resources.md` | Curated reading list of computer vision resources for wildlife classification, including iNaturalist competitions, the iNat Geomodel, and links to all papers above. |
| `elicit_search_query.md` | Elicit search query used to find relevant literature on the thesis topic. |
