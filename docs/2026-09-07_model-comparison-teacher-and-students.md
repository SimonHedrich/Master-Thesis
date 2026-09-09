# Model Comparison: Teacher (SpeciesNet) vs. Students (YOLOv5s, YOLO26n)

Consolidates every eval report currently on disk for the core training campaign
(`TODO.md` §4) into one place. This is the §4.5 "Phase 4 comparison synthesis" the
[KD/teacher-finetuning strategy](plans/2026-06-30_knowledge-distillation-and-teacher-finetuning-strategy.md)
calls for — **partial**, not final: see [§5 Open gaps](#5-open-gaps--whats-missing) for
what's still outstanding before a thesis-final conclusion can be drawn.

All numbers below are from `eval_suite/run_evaluation.py`, scored against the same fixed
test set (63,802 real + 11,250 synthetic images), per
[the evaluation strategy](plans/2026-06-10_model-evaluation-strategy.md). **mixed** = the
project's headline metric (real + synthetic test images combined); **real** = the
real-only breakout, the anchor for any public-benchmark comparison. Both use the full
225-way species (`fine`) label space unless stated otherwise.

## 1. TL;DR

| Model | Role | Params | mixed mAP | real mAP | mixed mAP50 | real mAP50 |
|---|---|---:|---:|---:|---:|---:|
| MD+SN ensemble — pretrained | Teacher, zero-shot | ~196M | 0.487 | 0.445 | 0.487 | 0.445 |
| MD+SN ensemble — **fine-tuned** | Teacher, fine-tuned (§4.1) | ~196M | **0.549** | **0.536** | 0.549 | 0.536 |
| YOLOv5s — post-fix retrain (2026-08-13) | Student, direct-FT (§4.3) | 7.63M | 0.417 | 0.386 | 0.484 | 0.460 |
| YOLO26n — direct-FT (2026-07-15) | Student, direct-FT (§4.2 Phase 1) | 2.71M | **0.523** | **0.481** | 0.574 | 0.541 |
| YOLO26n — KD (2026-08-25) | Student, KD from fine-tuned teacher (§4.4) | 2.71M | 0.510 | 0.479 | 0.570 | 0.547 |

*(YOLOv5s pre-fix runs and the untrained-student zero-shot baseline are intentionally
excluded from this headline table — see §3.2 and §5.)*

**Bottom line for the thesis's core question:** at the single hyperparameter point tested
so far, KD does not beat direct fine-tuning on the headline number, and — contrary to this
doc's original read — the band breakdown does **not** yet give a clean answer on the
long-tail question either, because a pipeline bug means none of the models below were ever
trained on the 50 Band-A classes at all (§3.4, `TODO.md` §4.7). Every Band-A number in this
document is zero-shot/held-out behavior, not "weak but present" supervision. The long-tail
KD hypothesis remains untested until that's fixed.

## 2. Key findings

1. **YOLO26n is both smaller and much better than YOLOv5s.** 2.71M params vs. 7.63M
   (2.8× smaller) while scoring +0.106 mixed mAP / +0.095 real mAP over YOLOv5s's best
   evaluated (post-fix) run. For a project targeting embedded inference, this is a strong
   signal in favor of YOLO26n as the deployment candidate — smaller model, better accuracy.

2. **~~KD wins 10× on Band A~~ — CORRECTED 2026-09-08: neither model is actually
   trained on Band A at all (see §3.4).** The paragraph below is what a first read of
   the band table suggests; §3.4 explains why that reading is wrong.

   | Band (real-image pool per class) | YOLO26n direct-FT (mixed mAP) | YOLO26n KD (mixed mAP) |
   |---|---:|---:|
   | A — <150 imgs, 50 classes, mostly synthetic-trained | 0.019 | **0.191** (10×) |
   | B — 150–249 imgs, 26 classes | 0.672 | 0.658 |
   | C — 250–399 imgs, 26 classes | 0.726 | 0.695 |
   | D — ≥400 imgs, 122 classes | 0.788 | 0.774 |

   It's tempting to read this as "KD trades a little B/C/D accuracy for a 10× Band-A
   win — direct evidence for the strategy doc's long-tail hypothesis." **That reading is
   wrong.** Investigation (§3.4, `TODO.md` §4.7) found that the training pipeline never
   loads *any* Band-A images (real or synthetic) for either model — both scores above are
   fully held-out/zero-shot behavior, not "KD helps a weakly-represented class." The real
   finding is narrower but still interesting: KD's soft labels leak just enough incidental
   signal about classes the student never saw to produce a non-trivial AP on some of them,
   where direct-FT's hard labels barely do and YOLOv5s does not at all. That's a
   knowledge-transfer curiosity, not evidence about long-tail performance — the actual
   long-tail question is still untested pending the §4.7 fix.

3. **Every model's Band A number is a symptom of one shared pipeline bug, not five
   separate model behaviors.** `TODO.md` §4.7 (full writeup in §3.4 below): the training
   code for all three pipelines (`yolov5s`, `yolo26n`, `teacher_finetune`) hardcodes
   `data/real/annotations_{train,val}.json`, which contains **zero** images for the 50
   Band-A classes — verified directly (0/145,728 train images, 0/12,543 val images). The
   ~8,000 Band-A synthetic training images that were actually generated for this purpose
   sit unused in `data/synthetic/annotations_train.json`. So *every* checkpoint evaluated
   in this document has had **zero training exposure** to Band A, and the differences below
   are all downstream of that one gap, not independent findings about each architecture:

   - **YOLOv5s** collapses to an exact `0.000` mAP on all 50 Band-A classes (§3.2) — the
     most complete failure, consistent with a classifier head that's had ~146k pure-negative
     gradient updates per Band-A class and nothing else.
   - **YOLO26n direct-FT** also never trains on Band A, but its head doesn't collapse as
     completely — 46/50 classes retain a tiny nonzero AP (§3.3), likely incidental
     generalization from visually related trained classes rather than real learning.
   - **YOLO26n KD** retains more residual Band-A signal than direct-FT (§3.3) — plausibly
     because the frozen teacher's soft-label distribution puts some incidental probability
     mass on Band-A classes when scoring *other* training images, giving the KD student a
     weak indirect signal that hard 0/1 labels never provide. This is a real and interesting
     knowledge-transfer effect, but it is **not** evidence for the dataset design's long-tail
     hypothesis (which was about classes with *some* real data, not *zero* training exposure)
     — that hypothesis is still untested.
   - **The SpeciesNet teacher fine-tune actively regresses on Band A** (0.397 → 0.013,
     mixed) rather than merely failing to improve — because fine-tuning starts from a
     pretrained model that had *some* generic Band-A recognition capability, and 200 epochs
     of Band-A-absent, pure-negative supervision actively unlearns it. This is the sharpest
     illustration of the bug's cost: the fine-tune trades away real capability the
     off-the-shelf model had, in exchange for large gains elsewhere (Band D: 0.603 → 0.823).

4. **Localization is a solved problem here; classification is where all the models
   struggle** — on the bands they actually trained on. The class-agnostic "detect" analog
   (bounding box only, no species label) is ≥0.77 mAP for every model, including
   COCO-pretrained-architecture students on this domain (YOLOv5s 0.77–0.79, YOLO26n
   0.79–0.81, MD+SN teacher 0.95–0.97). The gap to the fine-grained 225-way headline number
   (0.42–0.55) is almost entirely a species-classification problem, not a detection one —
   see the granularity decomposition in §3 for each model.

## 3. Detailed results

### 3.1 Teacher — SpeciesNet classifier head (in the MD+SN ensemble)

| | pretrained (zero-shot) | fine-tuned |
|---|---:|---:|
| mixed mAP / real mAP | 0.487 / 0.445 | 0.549 / 0.536 |
| mixed mAP50 / real mAP50 | 0.487 / 0.445 | 0.549 / 0.536 |
| detect-only mAP (mixed / real) | 0.952 / 0.947 | 0.969 / 0.964 |
| coarse (look-alikes merged) mAP | 0.514 | 0.572 |
| Band A mAP (mixed / real) | 0.397 / 0.324 | 0.013 / 0.008 |
| Band B mAP (mixed / real) | 0.454 / 0.408 | 0.630 / 0.574 |
| Band C mAP (mixed / real) | 0.473 / 0.515 | 0.634 / 0.605 |
| Band D mAP (mixed / real) | 0.603 / 0.551 | 0.823 / 0.801 |

Source: `scripts/training/megadet_speciesnet_ensemble/model_exports/{pretrained,finetuned-teacher-finetune-20260806-131233}/eval/evaluation_report.md`

### 3.2 Student — YOLOv5s (direct fine-tune)

Three evaluated runs exist; the first two **predate** the anchor/loss-autoscaling fix
(`docs/progress_notes/2026-07-16_yolov5s-underperformance-hyp-scaling-fix.md`) and are
listed only for the historical record — they are not representative of the current
pipeline and should not be used in any final comparison.

| | 2026-06-02 (stale) | 2026-06-29 (stale) | 2026-07-14 (stale) | **2026-08-13 (current, post-fix)** |
|---|---:|---:|---:|---:|
| mixed mAP / real mAP | 0.254 / 0.228 | 0.371 / 0.347 | 0.360 / 0.336 | **0.417 / 0.386** |
| mixed mAP50 / real mAP50 | 0.295 / 0.272 | 0.422 / 0.404 | 0.405 / 0.387 | **0.484 / 0.460** |
| detect-only mAP (mixed / real) | 0.651 / 0.611 | 0.787 / 0.764 | 0.788 / 0.765 | 0.773 / 0.749 |
| coarse mAP | 0.279 | 0.403 | 0.393 | 0.437 |
| Band A mAP (mixed / real) | — | — | — | **0.000 / 0.000** |
| Band B mAP (mixed / real) | — | — | — | 0.364 / 0.290 |
| Band C mAP (mixed / real) | — | — | — | 0.540 / 0.385 |
| Band D mAP (mixed / real) | — | — | — | 0.693 / 0.650 |

The fix clearly helped (+0.057 mixed mAP over the last stale run), but the Band A total
failure (finding 3 above, root-caused in §3.4) means this checkpoint should not be treated
as representing YOLOv5s's true ceiling — its Band-A number will need re-measuring once
§4.7's training-data gap is fixed and the model is retrained.

Source: `scripts/training/yolov5s/model_exports/{yolov5s-20260602-233434,yolov5s-20260629-235646,yolov5s-20260714-010652}/eval_best/evaluation_report.md`,
`scripts/training/yolov5s/model_exports/yolov5s-20260813-162031/eval_best/evaluation_report.md` (run 2026-09-07 for this comparison).

### 3.3 Student — YOLO26n (direct-FT vs. KD)

| | direct-FT (2026-07-15) | KD from fine-tuned teacher (2026-08-25) |
|---|---:|---:|
| mixed mAP / real mAP | **0.523** / **0.481** | 0.510 / 0.479 |
| mixed mAP50 / real mAP50 | 0.574 / 0.541 | 0.570 / 0.547 |
| detect-only mAP (mixed / real) | 0.812 / 0.778 | 0.788 / 0.763 |
| coarse mAP | 0.536 | 0.527 |
| Band A mAP (mixed / real) | 0.019 / 0.011 | **0.191** / **0.143** |
| Band B mAP (mixed / real) | 0.672 / 0.563 | 0.658 / 0.585 |
| Band C mAP (mixed / real) | 0.726 / 0.601 | 0.695 / 0.604 |
| Band D mAP (mixed / real) | 0.788 / 0.741 | 0.774 / 0.734 |

Training config: KD run used the fine-tuned MD+SN ensemble (§3.1) as the frozen teacher,
`T=4 / α=0.5` (the strategy doc's default point; the full `{4,8}×{0.5,0.7}` grid has not
been run — §5). Early-stopped at epoch 182 (best checkpoint epoch 161).

Source: `scripts/training/yolo26n/model_exports/{yolo26n-20260715-010031/eval_best,yolo26n-kd-20260825-164250/evaluation}/evaluation_report.md`

### 3.4 Root-cause investigation: why every model's Band A number is broken

**Question:** why does YOLOv5s score an exact `0.000` mAP on all 50 Band-A classes, and why
does YOLO26n do slightly, but not meaningfully, better?

**Finding: neither model is ever trained on a single Band-A image.** Confirmed by reading
the code, not by inference:

- `scripts/training/yolov5s/constants.py:13-14` and `scripts/training/yolo26n/constants.py:20-21`
  hardcode `ANNOTATIONS_TRAIN`/`ANNOTATIONS_VAL` to `data/real/annotations_{train,val}.json`.
  `scripts/training/teacher_finetune/constants.py:37` does the same for `ANNOTATIONS_TRAIN`.
- `run_training_pipeline.py` for all three models constructs `CocoYoloDataset`/training data
  exclusively from those two constants — there is no CLI flag, code path, or config option
  anywhere in any of the three pipelines that reads `data/synthetic/annotations_train.json`
  or `annotations_val.json`. Synthetic data is wired in only as a **test**-time domain
  (`eval_suite --synth-ann`), never for training or validation.
- Directly checked `data/real/annotations_train.json`: **0 of 145,728** images have
  `band == "A"` (bands present: D=137,478, C=4,418, B=2,210, negative=1,622). Same for
  `annotations_val.json`: **0 of 12,543**.
- The ~8,000 Band-A synthetic training images (plus per-class synthetic val images) that
  were actually generated for this purpose live only in `data/synthetic/annotations_train.json`
  (8,000 Band-A + 2,080 Band-B images) — a file built by a completely separate pipeline
  (`scripts/synthetic/6-export_coco.py`) that nothing downstream ever merges into the real
  annotation files (`scripts/dataset_quality/12-assign_dataset_splits.py`, which builds
  `data/real/annotations_*.json`, has no synthetic-handling code at all).

**This is a confirmed gap against the documented design, not a known/accepted limitation.**
`docs/plans/2026-05-19_synthetic-test-set.md` ("Band A note") states explicitly: *"the 40
synthetic val images used during training... are part of the training pipeline (used only
for early stopping)"* — i.e. Band-A synthetic train/val images were always meant to reach
the model. They never do, in any of the three training pipelines that currently exist.

**Why YOLOv5s hits an exact `0.000` and YOLO26n doesn't, given identically zero
supervision:** both models' eval code is shared (`scripts/training/yolov5s/eval_suite/{scoring,report,grouping}.py`,
imported unmodified by `yolo26n/eval_suite/run_evaluation.py`), so this isn't a scoring
difference — it's genuine model-output behavior. The most likely explanation is
architectural: YOLOv5s's classic per-anchor independent-sigmoid BCE classification head,
driven by ~146k consecutive pure-negative gradient updates per Band-A channel with no
positive example ever, converges its output for those classes below the `0.001` eval
confidence threshold on literally all 8,288 Band-A test images. YOLO26n's newer anchor-free
head (task-aligned label assignment + a DFL/varifocal-style classification loss) evidently
doesn't collapse as completely, leaking enough residual signal — likely generalization
spillover from visually related trained classes — to clear that threshold on a handful of
images for most Band-A classes. This part is an informed hypothesis, not something directly
instrumented; the zero-training-data root cause above is the verified fact.

**Fix and impact:** `data/synthetic/annotations_{train,val}.json` needs to be merged into
the dataloader construction for all three pipelines (at minimum for Band A; Band B's real
pool is also thin-by-design and may warrant the same treatment). **Status (2026-09-09): the
code fix has landed and is verified** (`CocoYoloDataset`/`SpeciesNetCropDataset` now merge
multiple annotation sources; all three pipelines' train/val construction updated) — see
`TODO.md` §4.7 for the full implementation note. **None of the checkpoints in this document
have been retrained on the fix yet**, so every Band-A result in §1 and §3.1–3.3 still
reflects the old, zero-shot/held-out behavior described above — including the teacher
fine-tune's Band A regression (finding 3) and any future KD-vs-direct-FT long-tail claim.
Retraining (a multi-day GPU campaign, per past KD/YOLOv5s run times) is the remaining step
before this table can be refreshed.

## 4. Band definitions (for context)

Per [`plans/2026-05-04_dataset-construction-strategy.md`](plans/2026-05-04_dataset-construction-strategy.md),
classes are split into four bands by real-image pool size, which also determines how much
of each class's *training* data is real vs. synthetic:

| Band | Real-image pool | # classes | Training composition |
|---|---|---:|---|
| A | < 150 | 50 | Mostly/entirely synthetic (too few real images) |
| B | 150–249 | 26 | Mixed real + synthetic |
| C | 250–399 | 26 | Mostly real |
| D | ≥ 400 | 122 | Real only, up to 500 imgs/class |

Band A's real *test* images are still real (never synthetic-only) — the *test* set is fine.
The problem (finding 3, §3.4) is entirely on the training side: none of Band A's synthetic
*training* images ever reach any of the three training pipelines, so Band A is currently
tested but never trained.

## 5. Open gaps — what's missing

This comparison is **not** the final §4.5 synthesis yet. Still outstanding, per `TODO.md` §4:

- **Band-A (and possibly Band-B) training-data gap (§4.7 — see §3.4). Code fixed
  2026-09-09; retraining still pending.** Every model evaluated here had zero training
  exposure to Band A, which taints every Band-A number and the teacher fine-tune's Band-A
  regression. The dataloader fix (merging `data/synthetic/annotations_{train,val}.json`)
  is in and verified; actually retraining YOLOv5s, YOLO26n (direct-FT + KD), and the
  SpeciesNet classifier on it is the prerequisite still outstanding before any real
  long-tail/thin-class conclusion can be drawn.
- **Student zero-shot baseline (Phase 0).** Raw-COCO-weights YOLO26n eval crashed on a
  `data/blanks/` gap (2026-08-13); the gap was fixed 2026-08-25 but the eval was never
  retried. Without it, there's no "how bad is COCO→wildlife domain shift" floor to anchor
  the direct-FT and KD gains against.
- **KD hyperparameter grid.** Only the default `T=4/α=0.5` point has been run. The
  strategy doc's full `{4,8}×{0.5,0.7}` grid remains deferred; worth revisiting once §4.7 is
  fixed, since the current Band-A comparison is not a valid read of KD's long-tail benefit.
- **Multi-animal subset cut.** The strategy doc's Phase 4 explicitly calls for a KD-vs-
  direct-FT comparison restricted to `multi_animal=true` images — the concrete test of the
  KD-from-multi-box-supervision advantage claim. Not yet done for any model pair.
- **MegaDetector fine-tuning.** Marked optional/secondary in the strategy doc (already
  99.2%/97.3% precision/recall out of the box); not attempted, and this comparison treats
  MD as fixed throughout.
- **Human qualitative rating axis** (§3.4 of the synthetic-data comparison track) is
  unrelated to this table but is the other open input the strategy doc eventually wants
  alongside downstream mAP.
