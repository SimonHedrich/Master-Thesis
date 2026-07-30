# Detector Architecture Selection for This Experiment

**Date:** 2026-07-21
**Status:** Decided — YOLO26n is the fixed detector for the generator comparison
**Depends on:** [`01_experiment-design.md`](01_experiment-design.md) §5,
[`06_evaluation-methodology.md`](06_evaluation-methodology.md),
`docs/2026-03-10_object-detection-models-for-embedded-systems.md`,
`docs/plans/2026-06-30_knowledge-distillation-and-teacher-finetuning-strategy.md`

---

## 1. The question

`01_experiment-design.md` §5 point 4 fixes "one detector, identical
schedule/seed policy for every cell" as a control, and had provisionally named
it "YOLOv5s (existing config)" without much justification. Axis C
(`06_evaluation-methodology.md`) trains this fixed detector on each
generator×prompt-regime cell's synthetic images and scores it on the real
9,742-image test set — described there as **"the thesis's actual goal."**
Which architecture that fixed detector should actually be was never argued for,
just defaulted to whatever pipeline existed first. This doc settles it.

The real tension: a detector **closer to the actual embedded deployment
target** is more directly relevant to the thesis's real question ("is this
synthetic data good enough for what we'll actually ship"), but a smaller model
could in principle be too capacity-starved to show a clear difference between
two decent datasets — while a **heavier** detector might resolve data-quality
differences more sharply, at the cost of being a worse proxy for the real
deployment target.

## 2. Candidates actually available

Two architectures already have complete, mutually comparable training
pipelines in this repo (shared dataset/optimizer/scheduler/MLflow contract,
per both packages' READMEs):

| Architecture | Params | COCO mAP | Pipeline | Tier |
|---|---|---|---|---|
| YOLOv5s | 7.2M | ~37.4% | `scripts/training/yolov5s/` | "small" |
| YOLO26n | 2.4M | 40.9% | `scripts/training/yolo26n/` | "nano" |

NanoDet-Plus-m (1.17–1.8M params) and PicoDet-S (0.99–1.1M params) — the
*true* long-term embedded targets per
`docs/2026-03-10_object-detection-models-for-embedded-systems.md` — have **no
pipeline in this repo at all**. Building one from scratch for this side-study
would confound new-pipeline engineering risk with the actual research question
(generator quality) and is out of scope here; the real choice is between the
two pipelines that already exist.

## 3. The simple-vs-heavy question is already answered elsewhere in this repo

`docs/plans/2026-06-30_knowledge-distillation-and-teacher-finetuning-strategy.md`
§1 (lines 122–159) already resolved an equivalent question for the main KD
study, independent of this sub-experiment, with citations:

- YOLOv5s (7.2M params) is **"approximately the size of what's already
  deployed"** — Swarovski's current AX Visio production pipeline already runs
  a fine-tuned YOLOv5s detector. That doc keeps YOLOv5s in the project only as
  **"the production-baseline reference row," explicitly "not as a KD
  student"** — i.e. it is not treated as an embedded-deployment candidate at
  all, just a stand-in for "how does the currently-shipping model perform."
- YOLO26n (2.4M params, 40.9% COCO mAP, anchor-free, NMS-free) is the
  recommended **actual embedded-deployment student** to start with, ahead of
  NanoDet-Plus-m/PicoDet-S (which need separate, non-Ultralytics toolchains),
  precisely because it shares enough Ultralytics/YOLOv5 tooling conventions to
  reuse the existing pipeline cheaply.

So the user's dichotomy — "simpler model closer to the embedded use case" vs.
"heavier model" — maps directly onto **YOLO26n vs. YOLOv5s**, and the project
has already made this call the same way for the main KD study. Using YOLOv5s
as this sub-experiment's headline detector would mean answering "is generator
X's data good enough for a model the size of what's already shipping" —
disconnected from the thesis's actual target. Using YOLO26n keeps this
side-study's conclusion directly load-bearing for the main thesis pipeline.

## 4. Is YOLO26n too small to show a data-quality difference?

Unlikely to matter at this experiment's data scale. Each cell trains on only
**100 synthetic images/class × 12 classes = 1,200 images**, fine-tuned from
COCO-pretrained weights. At that scale, neither a 2.4M- nor a 7.2M-parameter
model is anywhere near capacity-starved — both can comfortably fit 1,200
images. The limiting factor on real-test transfer is the **transferability of
the synthetic images' features**, not raw parameter count. The "heavier model
resolves quality differences better" intuition mainly applies when the
smaller model is capacity-starved relative to the task, which is not the
regime here (12 classes, COCO-pretrained init, small dataset).

## 5. Caveat: YOLO26n is the less battle-tested pipeline here

YOLO26n's pipeline is newer than YOLOv5s' in this repo — fewer completed full
runs, and it's still being extended with a KD training mode. YOLOv5s' own
recent hyp-scaling bug
(`docs/progress_notes/2026-07-16_yolov5s-underperformance-hyp-scaling-fix.md`,
a ~2.8x under-weighted classification loss from a missing nc-autoscaling step)
is a concrete reminder that any of these from-scratch training harnesses can
hide a silent, non-obvious bug. This is recorded as a documented risk, not a
blocker: this doc only settles the architecture choice; empirically
validating it (e.g. a pilot run on the incumbent cell, which already has all
1,200 images ready) before committing the full generator grid to it is a
natural next step, left out of scope for this decision-record task.

## 6. Training-time estimate

Figures below come directly from this repo's own production run logs, not
theoretical FLOP estimates.

**Full 225-class production runs (145,809 train images), observed:**

| Run | Batch | Train batches/epoch | Train s/epoch | Eval s/epoch (val) | Total |
|---|---|---|---|---|---|
| yolov5s | 16 | 9,113 | ~2,599s (0.285s/batch) | ~194s (12,560 img) | 50 epochs ≈ 39h ≈ 1.6 days |
| yolo26n | 64 | 2,278 | ~2,919s (1.28s/batch) | ~803s (12,608 img) | resumed run spanning ~1 week wall-clock |

This experiment's per-cell train set is **1,200 images — 0.8% of the
145,809-image production set** — across 12 classes instead of 225. Scaling the
observed per-batch rates down (with a small internal val holdout, see §7):

- **yolov5s:** ~60 train batches/epoch (1,200/16, minus the val holdout) ×
  0.285s ≈ 17s train + a few seconds eval ≈ **~25–40s/epoch**, likely
  overhead-bound (dataloader/Python loop) rather than compute-bound at this
  size.
- **yolo26n:** ~15 train batches/epoch (1,200/64) × 1.28s ≈ 19s train + eval
  ≈ **~35–45s/epoch**.

Even at a generous 150–200 epochs to convergence (early stopping will likely
end runs sooner on a ~1,000-image fine-tune), that's **roughly 1–2 hours per
single training run**, plus a one-time final eval pass over the full
9,742-image real test set (~3 min for yolov5s, ~10 min for yolo26n at the
observed per-image eval rates) — not days. With ~8 generator cells × ≥3 seeds
(per `06_evaluation-methodology.md`'s statistical-care recommendation) ≈ 24
runs, total compute is on the order of **1–2 GPU-days** on the project's A40 —
regardless of which of these two architectures is chosen.

**Conclusion: training time is not a differentiator between YOLOv5s and
YOLO26n for this specific sub-experiment.** The "yolo26n took a week"
recollection applies to the full 225-class/145,809-image production run; this
12-class/1,200-image side-study is roughly two orders of magnitude smaller
and cheap either way. The architecture choice should follow the relevance
argument in §3, not a time-budget argument.

## 7. Related recommendation: don't eval the full real test set every epoch

Carve a small internal validation split out of each cell's 1,200 synthetic
images (e.g. a stratified 80/20 per class) for per-epoch model
selection/early stopping, and reserve the full 9,742-image real test set
exclusively for the final Axis C report. Evaluating the full real test set
every epoch would add ~3–10 minutes per epoch (dwarfing the ~20–45s train
step) and dominate every run's wall-clock for no benefit; it would also mean
model selection peeks at the same test set used for the headline metric,
which is worth avoiding on methodological grounds independent of the time
cost.

## 8. Decision

**Use YOLO26n (`scripts/training/yolo26n/`) as the fixed detector for every
cell of this experiment.** Keep YOLOv5s available only as an optional
secondary/sensitivity check if time allows, consistent with its role
elsewhere in the thesis as the production-baseline reference, not an
embedded-deployment candidate.
