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
scripts/      — Utility scripts, organized into subpackages (training, benchmark,
                literature, thesis, etc.), each with its own README where one exists
thesis/       — The manuscript (Overleaf-synced), thesis-phase toolchain docs, and
                writing-process notes
```

`docs/README.md`, `research/README.md`, and `thesis/README.md` are the maintained
indexes for their directories — check those rather than this file for current contents.

## Running Code

### Python scripts: always uv, always from the repo root
- Module-safe names: `uv run python -m scripts.<package>.<module>`
  (e.g. `uv run python -m scripts.training.yolov5s.run_training_pipeline`)
- Numbered pipeline scripts (`1-foo.py` — invalid module names, cannot use `-m`):
  `uv run python scripts/<dir>/<N>-<name>.py`
- Every runnable script's module docstring must state its exact run command in this form.
- **Exception A:** `scripts/literature/` (Zotero PDF sourcing/extraction pipeline) uses
  `uv run --script scripts/literature/<N>-<name>.py` (PEP 723 inline deps) instead — it
  needs native host access to `~/Zotero` and to run outside the Linux-only container.
- **Exception B:** `scripts/benchmark/`'s device-side scripts and host-side
  `1b-convert_and_verify.py` also use `uv run --script` (PEP 723) instead. See
  `scripts/benchmark/README.md` for which scripts and why.

### Thesis manuscript: Overleaf is the compiler
The manuscript (`thesis/manuscript/`) is compiled on Overleaf, not in the container.
Sync it with `make overleaf` (or `make overleaf-push|overleaf-pull|overleaf-status` to
force a direction), from the repo root and on the designated `thesis/overleaf-sync`
branch. `OVERLEAF_TOKEN` lives in `.env` and must stay the only copy. See
`scripts/thesis/README.md` for the full sync behavior.

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

The manuscript (`thesis/manuscript/chapters/`, especially Chapter 1 and Chapter 4) is
now the authoritative statement of the research question, technical approach, and
findings — read it, not this file, for the current framing. `thesis/README.md` has
the current draft status per chapter.

**Constraints that still apply to any code or writing** (not derivable from the
manuscript draft or code alone):
- **YOLOv5 license:** Only commercially usable up to commit `5cdad89` — later commits require additional licensing
- Run own benchmarks rather than relying on published numbers
- **Primary evaluation = the mixed (real + synthetic) test set.** The default headline metric is computed over the union of the real test images and the balanced 225×50 synthetic test set. Rationale: the consistent 50 synthetic images/class stabilise evaluation for classes with few or low-quality real photos (Band A), while remaining a negligible, consistent addition for well-resourced classes (Band D, up to 500 real test images). The model must **never** be judged on synthetic images alone. The **real-only breakout** is always reported alongside the mixed headline as the primary-evaluation figure and the anchor for any comparison to public (real-image) benchmarks. The real-vs-synthetic domain-shift delta is monitored as a watchdog: **if a clear discrepancy between the mixed and real (or synthetic and real) results emerges, the default evaluation axes will be revised.** See `docs/plans/2026-06-10_model-evaluation-strategy.md`.

## Maintaining Documentation

`docs/README.md`, `research/README.md`, and `thesis/README.md` all serve as indices — keep them updated whenever files are added to those directories.
