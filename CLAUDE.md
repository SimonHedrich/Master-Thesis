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
- `research/literature/README.md` — Zotero-managed bibliography (`references.bib`), LLM-generated survey summaries, and source PDFs + Markdown extractions (`literature/sources/`)

## Running Code

### Python scripts: always uv, always from the repo root
- Module-safe names: `uv run python -m scripts.<package>.<module>`
  (e.g. `uv run python -m scripts.training.yolov5s.run_training_pipeline`)
- Numbered pipeline scripts (`1-foo.py` — invalid module names, cannot use `-m`):
  `uv run python scripts/<dir>/<N>-<name>.py`
- Every runnable script's module docstring must state its exact run command in this form.
- **Exception A:** `scripts/literature/` (Zotero PDF sourcing/extraction pipeline) uses
  `uv run --script scripts/literature/<N>-<name>.py` (PEP 723 inline dependencies) instead.
  Plain `uv run` only resolves on Linux here (`pyproject.toml`'s `[tool.uv] environments`
  restriction, for the training stack), so it doesn't work on macOS outside the container —
  and this pipeline needs native host access to `~/Zotero`, which isn't mounted into the
  container. `uv run --script` sidesteps both without touching the shared project lockfile.
- **Exception B:** `scripts/benchmark/`'s device-side scripts (`2-bench_latency.py`,
  `3-bench_parity.py`, `pi/monitor.py`, `pi/device_profile.py`) and its host-side
  `1b-convert_and_verify.py` also use `uv run --script` (PEP 723). The device-side
  ones run on the Raspberry Pi 400, where `pyproject.toml`'s `pytorch-cu130` torch
  pin resolves to CUDA SBSA wheels that cannot run on an SBC — and where the repo
  isn't checked out at all, so they import no project code. `1b-` needs
  `onnxruntime`/`onnxslim`/`pnnx`, which would otherwise have to be added to the
  shared lockfile and synced into a venv that is often mid-training.
  See `scripts/benchmark/README.md`.

### Containers: one image, exec in, then uv
- `make build` builds the single `training` image; `make run` starts the container and
  execs a bash shell inside it; `make stop` stops/removes it.
- To run a script in the container: `make run` first, then the standard uv command inside.
- Default to the one shared container for everything. Only create a separate
  image/container when something genuinely cannot run in the default one
  (document the reason when you do).

## Version Control

**Commit directly to `main`, then push.** This repo deliberately does not use
feature branches — in a single-author thesis repo they cost more than they
return. This overrides the general assistant default of branching before
committing to a default branch: do not create a branch, and do not ask whether
to.

- Commit to `main` and `git push origin main` as one step. **This section is
  standing authorization for that push** — do not stop to confirm it.
- `git fetch` first if the remote may have moved. If a push is rejected,
  `git pull --rebase` then push again. Never force-push.
- Stage only the files belonging to the change at hand. This working tree
  routinely carries unrelated modified and untracked files — in-flight training
  run directories, scratch logs, local skill definitions, and sometimes live
  edits from a concurrent session — and none of those are yours to commit.
  Check `git status` before staging, and never `git add -A`.
- Because every commit is pushed immediately, treat it as public and permanent:
  no credentials, and no large regenerable artifacts (weights, exports,
  prediction caches — see `.gitignore`).
- **One exception:** a large, genuinely risky refactor may use a branch. The bar
  is high — work that could leave `main` broken, or that you expect to need
  review before it lands. Say out loud that you are branching and why. Anything
  routine — a fix, a doc, a new script, a results write-up — goes straight to
  `main`.

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
