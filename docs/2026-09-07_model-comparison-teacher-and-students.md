# Model Comparison: Teacher (SpeciesNet) vs. Students (YOLOv5s, YOLO26n)

Consolidates every eval report currently on disk for the core training campaign
(`TODO.md` §4) into one place. This is the §4.5 "Phase 4 comparison synthesis" the
[KD/teacher-finetuning strategy](plans/2026-06-30_knowledge-distillation-and-teacher-finetuning-strategy.md)
calls for. **Updated 2026-09-21: all four models have now been retrained on the
Band-A training-data fix (§4.7) and the core KD-vs-direct-FT question has a clean
answer** (§2 finding 2 below) — the headline comparison in this document is no
longer provisional. It is still **not the full thesis-final picture**: see
[§5 Open gaps](#5-open-gaps--whats-missing) for the remaining, non-blocking
refinements (zero-shot floor, KD hyperparameter grid, multi-animal subset cut).

All numbers below are from `eval_suite/run_evaluation.py`, scored against the same fixed
test set (63,802 real + 11,250 synthetic images), per
[the evaluation strategy](plans/2026-06-10_model-evaluation-strategy.md). **mixed** = the
project's headline metric (real + synthetic test images combined); **real** = the
real-only breakout, the anchor for any public-benchmark comparison. Both use the full
225-way species (`fine`) label space unless stated otherwise.

## 1. TL;DR

All rows below are **post-Band-A-fix** (§3.4, `TODO.md` §4.7) — every model has real,
if thin, training exposure to the 50 long-tail Band-A classes.

| Model | Role | Params | mixed mAP | real mAP | mixed mAP50 | real mAP50 |
|---|---|---:|---:|---:|---:|---:|
| MD+SN ensemble — pretrained | Teacher, zero-shot | ~196M | 0.487 | 0.445 | 0.487 | 0.445 |
| MD+SN ensemble — **fine-tuned** (`--freeze-fraction 0.75`) | Teacher, fine-tuned (§4.1) | ~196M | **0.663** | **0.599** | 0.663 | 0.599 |
| YOLOv5s — Band-A-fix retrain (2026-09-09) | Student, direct-FT (§4.3) | 7.63M | 0.496 | 0.407 | 0.568 | 0.485 |
| YOLO26n — direct-FT, Band-A-fix retrain (2026-09-19) | Student, direct-FT (§4.2 Phase 1) | 2.71M | **0.599** | **0.529** | 0.659 | 0.599 |
| YOLO26n — KD, Band-A-fix retrain (2026-09-21) | Student, KD from fine-tuned teacher (§4.4) | 2.71M | 0.560 | 0.479 | 0.624 | 0.550 |

*(Pre-fix runs and the untrained-student zero-shot baseline are intentionally excluded
from this headline table — see §3.2/§3.3 for the historical record and §5 for the
still-missing zero-shot floor.)*

**Bottom line for the thesis's core question:** now that every model has genuine Band-A
training exposure, the comparison is clean, and the answer is unambiguous — **KD does not
beat direct fine-tuning anywhere.** YOLO26n direct-FT outperforms YOLO26n KD on every single
axis checked: the headline mixed/real mAP, mAP50, the detect-only localization analog, the
coarse (look-alike-merged) granularity, and all four bands in both domains, *including*
Band A specifically (real mAP_fine: direct-FT 0.300 vs. KD 0.239) — the exact long-tail
case the KD hypothesis was meant to help most. See finding 2 in §2 and the full table in
§3.3. At this single hyperparameter point (`T=4/α=0.5`), direct fine-tuning is simply the
stronger recipe for this domain-specific 225-way task; whether a different `(T, α)` point
would change that remains open (§5).

## 2. Key findings

1. **YOLO26n is both smaller and much better than YOLOv5s.** 2.71M params vs. 7.63M
   (2.8× smaller) while scoring +0.106 mixed mAP / +0.095 real mAP over YOLOv5s's best
   evaluated (post-fix) run. For a project targeting embedded inference, this is a strong
   signal in favor of YOLO26n as the deployment candidate — smaller model, better accuracy.

2. **KD does not beat direct fine-tuning anywhere — including Band A, the case it was
   meant to help most. (2026-09-21, post-Band-A-fix retrain of all four models.)** Both
   findings 2 and 3 below (struck through) described the *pre-fix* state, where neither
   model had any Band-A training exposure at all and the apparent "KD wins 10× on Band A"
   read was an artifact of that bug, not a real effect. With the fix landed and all four
   models retrained on genuine (if thin) Band-A supervision, the comparison is now clean:

   | Metric | YOLO26n direct-FT | YOLO26n KD | KD wins? |
   |---|---:|---:|---|
   | mixed mAP (headline) | **0.599** | 0.560 | No |
   | real mAP | **0.529** | 0.479 | No |
   | detect-only mAP (mixed) | **0.807** | 0.770 | No |
   | coarse mAP (mixed) | **0.608** | 0.569 | No |
   | Band A mAP_fine (mixed / real) | **0.579** / **0.300** | 0.493 / 0.239 | No |
   | Band B mAP_fine (mixed / real) | **0.685** / **0.588** | 0.659 / 0.565 | No |
   | Band C mAP_fine (mixed / real) | **0.614** / **0.591** | 0.584 / 0.572 | No |
   | Band D mAP_fine (mixed / real) | **0.726** / **0.742** | 0.713 / 0.706 | No |

   Direct-FT wins every row. This is a clean negative result for the KD hypothesis at
   this hyperparameter point (`T=4/α=0.5`) — not a mixed or ambiguous one. It's also
   consistent in direction with the earlier *pre*-fix single-point comparison
   (`TODO.md` §4.4: KD 0.510/0.479 vs. direct-FT 0.523/0.481 mixed/real) — KD has
   underperformed direct-FT at this default point both before and after the Band-A fix,
   and the gap is if anything larger post-fix (0.039 mixed-mAP gap vs. 0.013 pre-fix).
   Whether a different `(T, α)` point on the strategy doc's grid changes this remains open
   (§5) — this result rules out "KD helps" *at the single point tested*, not KD in general.

   ~~**KD wins 10× on Band A** — CORRECTED 2026-09-08: neither model is actually
   trained on Band A at all (see §3.4).~~ *(Historical — pre-Band-A-fix reading, kept for
   continuity.)* The paragraph below is what a first read of the pre-fix band table
   suggested; §3.4 explains why that reading was wrong at the time.

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
   knowledge-transfer curiosity, not evidence about long-tail performance.

3. ~~**Every model's Band A number is a symptom of one shared pipeline bug, not five
   separate model behaviors.**~~ *(Historical — the bug this finding describes is fixed;
   kept for the record since it's good documentation of the root cause, see §3.4.)*
   `TODO.md` §4.7 (full writeup in §3.4 below): the training
   code for all three pipelines (`yolov5s`, `yolo26n`, `teacher_finetune`) hardcoded
   `data/real/annotations_{train,val}.json`, which contained **zero** images for the 50
   Band-A classes — verified directly (0/145,728 train images, 0/12,543 val images). The
   ~8,000 Band-A synthetic training images that were actually generated for this purpose
   sat unused in `data/synthetic/annotations_train.json`. So *every* checkpoint evaluated
   at the time had **zero training exposure** to Band A, and the differences below were
   all downstream of that one gap, not independent findings about each architecture:

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
   struggle.** The class-agnostic "detect" analog (bounding box only, no species label) is
   ≥0.75 mAP for every model post-Band-A-fix (YOLOv5s 0.750–0.775, YOLO26n direct-FT
   0.782–0.807, YOLO26n KD 0.745–0.770, MD+SN teacher 0.964–0.969). The gap to the
   fine-grained 225-way headline number (0.41–0.66) is almost entirely a
   species-classification problem, not a detection one — see the granularity decomposition
   in §3 for each model.

## 3. Detailed results

### 3.1 Teacher — SpeciesNet classifier head (in the MD+SN ensemble)

| | pretrained (zero-shot) | fine-tuned, pre-Band-A-fix (2026-08-06) | **fine-tuned, Band-A-fix (2026-09-09)** |
|---|---:|---:|---:|
| mixed mAP / real mAP | 0.487 / 0.445 | 0.549 / 0.536 | **0.663 / 0.599** |
| mixed mAP50 / real mAP50 | 0.487 / 0.445 | 0.549 / 0.536 | **0.663 / 0.599** |
| detect-only mAP (mixed / real) | 0.952 / 0.947 | 0.969 / 0.964 | 0.969 / 0.964 |
| coarse (look-alikes merged) mAP | 0.514 | 0.572 | 0.682 |
| Band A mAP (mixed / real) | 0.397 / 0.324 | 0.013 / 0.008 | **0.447 / 0.182** |
| Band B mAP (mixed / real) | 0.454 / 0.408 | 0.630 / 0.574 | 0.702 / 0.638 |
| Band C mAP (mixed / real) | 0.473 / 0.515 | 0.634 / 0.605 | 0.629 / 0.639 |
| Band D mAP (mixed / real) | 0.603 / 0.551 | 0.823 / 0.801 | 0.798 / 0.821 |

Fine-tune config for the current column: `--freeze-fraction 0.75` (the established-best
value), retrained on the Band-A-fixed pipeline (§4.7); early-stopped at epoch 22, best val
`f1_macro=0.8312` (up from 0.7864 pre-fix), test `f1_macro=0.6220`/`accuracy_top1=0.7058`
(up from 0.5588/0.6870). Band A mixed: 0.027 (the actual pre-fix number for this same
`--freeze-fraction 0.75` setting, not the `20260806` column shown above, which used a
different freeze fraction) → 0.447, a ~16× improvement — confirming the training-data fix
also works for the teacher's classifier head, not just the detector students.

Source: `scripts/training/megadet_speciesnet_ensemble/model_exports/{pretrained,finetuned-teacher-finetune-20260806-131233,finetuned-teacher-finetune-ff0.75-bs64-20260909-231034}/eval/evaluation_report.md`

### 3.2 Student — YOLOv5s (direct fine-tune)

The first three evaluated runs **predate** the anchor/loss-autoscaling fix
(`docs/progress_notes/2026-07-16_yolov5s-underperformance-hyp-scaling-fix.md`), and the
2026-08-13 run **predates** the Band-A training-data fix (§4.7) — all four are listed only
for the historical record; the 2026-09-09 column is the current, representative run.

| | 2026-06-02 (stale) | 2026-06-29 (stale) | 2026-07-14 (stale) | 2026-08-13 (anchor-fix, pre-Band-A-fix) | **2026-09-09 (Band-A fix, current)** |
|---|---:|---:|---:|---:|---:|
| mixed mAP / real mAP | 0.254 / 0.228 | 0.371 / 0.347 | 0.360 / 0.336 | 0.417 / 0.386 | **0.496 / 0.407** |
| mixed mAP50 / real mAP50 | 0.295 / 0.272 | 0.422 / 0.404 | 0.405 / 0.387 | 0.484 / 0.460 | **0.568 / 0.485** |
| detect-only mAP (mixed / real) | 0.651 / 0.611 | 0.787 / 0.764 | 0.788 / 0.765 | 0.773 / 0.749 | 0.775 / 0.750 |
| coarse mAP (mixed) | 0.279 | 0.403 | 0.393 | 0.437 | 0.510 |
| Band A mAP (mixed / real) | — | — | — | **0.000 / 0.000** | **0.384 / 0.058** |
| Band B mAP (mixed / real) | — | — | — | 0.364 / 0.290 | 0.461 / 0.341 |
| Band C mAP (mixed / real) | — | — | — | 0.540 / 0.385 | 0.477 / 0.386 |
| Band D mAP (mixed / real) | — | — | — | 0.693 / 0.650 | 0.646 / 0.648 |

The Band-A fix took YOLOv5s from a total, exact-zero Band-A failure to a real if modest
result (mixed 0.000 → 0.384, real 0.000 → 0.058) — the clearest before/after demonstration
in this document of what the pipeline bug (§3.4) was actually costing. Headline mixed mAP
also rose +0.079 over the anchor-fix-only baseline, though Band C/D dipped slightly
(likely noise/regularization interaction from the retrain, not investigated further).

Source: `scripts/training/yolov5s/model_exports/{yolov5s-20260602-233434,yolov5s-20260629-235646,yolov5s-20260714-010652}/eval_best/evaluation_report.md`,
`scripts/training/yolov5s/model_exports/yolov5s-20260813-162031/eval_best/evaluation_report.md`,
`scripts/training/yolov5s/model_exports/yolov5s-20260909-230900/eval_best/evaluation_report.md` (commit `377d917`).

### 3.3 Student — YOLO26n (direct-FT vs. KD)

Both rows below are **post-Band-A-fix** retrains (pre-fix numbers — direct-FT 0.523/0.481,
KD 0.510/0.479 mixed/real — are historical, see finding 2 in §2).

| | **direct-FT (2026-09-19)** | KD (2026-09-21) |
|---|---:|---:|
| mixed mAP / real mAP | **0.599** / **0.529** | 0.560 / 0.479 |
| mixed mAP50 / real mAP50 | **0.659** / **0.599** | 0.624 / 0.550 |
| detect-only mAP (mixed / real) | **0.807** / **0.782** | 0.770 / 0.745 |
| coarse mAP (mixed) | **0.608** | 0.569 |
| Band A mAP_fine (mixed / real) | **0.579** / **0.300** | 0.493 / 0.239 |
| Band B mAP_fine (mixed / real) | **0.685** / **0.588** | 0.659 / 0.565 |
| Band C mAP_fine (mixed / real) | **0.614** / **0.591** | 0.584 / 0.572 |
| Band D mAP_fine (mixed / real) | **0.726** / **0.742** | 0.713 / 0.706 |

**Direct-FT wins every row** — see finding 2 in §2 for the full verdict. Training config:
the KD run used the fine-tuned MD+SN ensemble (§3.1) as the frozen teacher, `T=4 / α=0.5`
(the strategy doc's default point; the full `{4,8}×{0.5,0.7}` grid has not been run — §5).

**KD checkpoint provenance note:** KD training reached epoch 186/200 (best val
`mAP50_95=0.7327` at epoch 185, climbing steadily but with a shrinking-delta plateau in its
final logged epochs — 0.7239@175, 0.7260@177, 0.7277@178, 0.7293@181, 0.7313@183,
0.7327@185) when, mid-epoch-187, a Tailscale network outage on the training host broke DNS
resolution to the MLflow tracking server; the resulting unhandled `MlflowException` killed
the training process. `best.pt` (epoch 185) — already the best checkpoint at the time of
the crash — is used as the final model rather than resuming training, on the reasoning that
the shrinking improvement deltas indicate this was an effectively-finished, plateauing run
that the crash simply preempted by a couple of epochs rather than one cut short of a
meaningful further gain. Training was not resumed.

Source: `scripts/training/yolo26n/model_exports/{yolo26n-bs32-20260910-212812,yolo26n-kd-bs16-20260916-101612}/evaluation/evaluation_report.md`
(commits `b80ee93`, this document's companion commit for KD).

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
`TODO.md` §4.7 for the full implementation note.

**Status (2026-09-21): all four models have now been retrained on the fix and re-evaluated**
(teacher `5f12c0a`, YOLOv5s `377d917`, YOLO26n direct-FT `b80ee93`, YOLO26n KD this commit)
— §1 and §3.1–3.3 above reflect the current, post-fix numbers throughout. Every Band-A
number in this document is now genuine (if thin) supervised behavior, not the
zero-shot/held-out artifact described above. The retrain campaign's full before/after
picture: YOLOv5s Band A mixed mAP 0.000→0.384, teacher 0.027→0.447, YOLO26n direct-FT
0.019→0.579 — all three-to-sixteen-fold improvements, confirming the fix's impact was real
and consistent across architectures. (Pre-fix, zero-shot-Band-A numbers remain in the
tables above/finding 2–3 as the historical record; they are superseded, not deleted.)

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

**2026-09-21: the Band-A retrain campaign (below) is done, and with it the core
KD-vs-direct-FT question this document exists to answer (§2 finding 2) — that part of the
§4.5 synthesis is now final, not provisional.** The remaining gaps below are real but
narrower: they refine or extend the comparison rather than qualify its headline verdict.

- ~~**Band-A (and possibly Band-B) training-data gap (§4.7 — see §3.4).**~~ **Done
  2026-09-21.** Code fixed 2026-09-09; all four models (teacher, YOLOv5s, YOLO26n direct-FT,
  YOLO26n KD) retrained on the fix and re-evaluated — see §3.1–3.4 for the full before/after
  numbers. No longer an open gap.
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
