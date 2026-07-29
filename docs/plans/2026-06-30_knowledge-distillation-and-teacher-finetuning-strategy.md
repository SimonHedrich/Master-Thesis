# Knowledge Distillation & Teacher Fine-Tuning Strategy

**Date:** 2026-06-30
**Status:** Plan — implementation not yet started
**Depends on / supersedes in part:**
`docs/2026-03-12_knowledge_distillation_research_overview.md`,
`docs/2026-03-12_research-and-experimentation-plan.md`,
`docs/progress_notes/2026-03-18_speciesnet-pipeline-and-experiment-design.md`,
`docs/plans/2026-04-30_speciesnet-classification-strategy.md`,
`docs/plans/2026-06-10_model-evaluation-strategy.md`,
`docs/2026-06-03_yolov5s-training-pipeline.md`,
`docs/plans/2026-06-02_yolov5s-training-pipeline.md`

---

## 0. Context

The custom YOLOv5s fine-tuning pipeline (`scripts/training/yolov5s/`) is now functional and
trains successfully on the full 225-class wildlife dataset. This document is the next step:
deciding how to (a) bring in additional, smaller student model architectures and (b)
implement a knowledge-distillation (KD) training strategy, using the
**MegaDetector → SpeciesNet (MD+SN)** pipeline as the teacher — per the user's explicit
direction, because MD+SN is the strongest available species-ID model for this domain and is
also the architecture family Swarovski's current AX Visio production pipeline is built on
(see §1.3 below).

Two earlier documents sketched a KD strategy in March 2026
(`docs/2026-03-12_knowledge_distillation_research_overview.md`, a literature survey, and
`docs/2026-03-12_research-and-experimentation-plan.md`, a concrete experiment design). Both
were written *before* the dataset and training pipeline existed, and both assumed a
**separately fine-tuned YOLOv8s** as the domain-adapted teacher (with SpeciesNet only as an
optional "ceiling reference" teacher), plus an iNaturalist-direct dataset construction that
the project has since superseded with the real, much more elaborate pipeline (225 classes,
training bands A–D, contamination review, look-alike groups, synthetic data, the
multi-axis evaluation framework).

**What this document supersedes:** the specific teacher choice (YOLOv8s fine-tuned on
wildlife → replaced by MD+SN fine-tuned on wildlife) and the assumption of a from-scratch
iNaturalist dataset build (→ replaced by the dataset already built under `data/real/` and
`data/synthetic/`).

**What this document carries forward unchanged** from the March research, because it is
still correct and independently corroborated by the project's own pipeline-design notes:
the capacity-gap risk analysis, the conclusion that feature-based KD does not transfer
across architecturally mismatched teacher/student pairs, and the staged
zero-shot → direct-fine-tune → teacher-fine-tune → KD → comparison experimental ladder.

This document does **not** cover quantization / QAT / SNPE export — that remains the
domain of `docs/2026-03-10_object-detection-models-for-embedded-systems.md` and should be
revisited as its own plan once a winning training strategy and student architecture are
chosen.

---

## 1. Direct answers to the three open questions

### 1.1 Does fine-tuning MD+SN on the dataset make sense, given many labels come from these models?

**Yes — and it is in fact necessary if MD+SN is to be used as a KD teacher at all** (see
§1.3). But the circularity concern is legitimate and needs a specific, scoped mitigation
rather than a blanket dismissal.

What the dataset's label provenance actually is, per the dataset-construction pipeline
(`scripts/dataset_quality/`, `docs/plans/2026-05-25_dataset-split-real-image-selection.md`,
`docs/plans/2026-04-30_speciesnet-classification-strategy.md`):

- **Species labels originate from directory metadata assigned at ingestion** — i.e.
  iNaturalist/GBIF community taxonomic identification — **not from SpeciesNet predictions.**
  SpeciesNet enters the pipeline only at steps 6–7 (`6-classify_speciesnet.py`,
  `7-filter_speciesnet.py`) as a **label-verification filter**: it checks whether its own
  prediction agrees with the pre-existing directory label at a given taxonomic level
  (species/genus/family/order/class), and rejects images where it disagrees strongly. It
  does not assign the positive label in the first place.
- Per `docs/plans/2026-05-25_dataset-split-real-image-selection.md` §1 ("Completed"),
  SpeciesNet-based filtering was applied **specifically to the less-trusted sources
  (OpenImages, ImagesCV)** — not blanket-applied to iNaturalist/GBIF, which make up the bulk
  of the dataset and carry their own independent provenance.
- **17,737 images went through manual human review (13,827 approved)** — a verification
  channel fully independent of SpeciesNet.
- **Bounding boxes** come from MegaDetector, but went through a separate, human-reviewed
  contamination pipeline (`14-flag_multi_animal_contamination.py` through
  `15-apply_contamination_decisions.py`, plus the FiftyOne review workflow) rather than being
  trusted blindly from raw MD output.

So: fine-tuning SpeciesNet's classification head on this dataset teaches it a genuinely new
mapping — photographs (camera-trap and user-aimed/binocular-style alike) → the project's
specific 225-class taxonomy — using labels it largely did not generate. This is ordinary
transfer learning, not a tautology, for the bulk of the data.

**Where the circularity risk is real:** for the SpeciesNet-gated subset (OpenImages,
ImagesCV, plus any image that survived purely because it cleared SpeciesNet's own
confidence/match-level threshold), evaluating the fine-tuned teacher's accuracy *on that
exact subset* would partly measure agreement with itself, because the subset was selected
for agreeing with the pre-fine-tuning model.

**Mitigation — disclosure, not a different test set.** The temptation is to evaluate the
fine-tuned teacher on a restricted, circularity-safe slice of the test set (e.g.
iNaturalist/GBIF and manually-reviewed images only, excluding the SpeciesNet-gated
OpenImages/ImagesCV portion). **This plan deliberately does not do that.** Every model in the
comparison matrix — zero-shot, direct-FT students, the teacher, KD students — must be scored
on **one fixed test set**, identical across all of them, or none of the headline numbers in
the thesis are comparable to each other. That single-fixed-instrument principle is the whole
point of `docs/plans/2026-06-10_model-evaluation-strategy.md`'s evaluation framework, and
carving out a bespoke test slice for just the teacher's before/after claim would quietly
violate it.

Instead, the circularity risk is **disclosed, not engineered around**: wherever the thesis
reports the teacher's fine-tuned-vs-off-the-shelf species-accuracy delta, it must explicitly
state that part of the test set's OpenImages/ImagesCV portion was originally filtered using
the *pre-fine-tuning* SpeciesNet, so that portion's contribution to the reported improvement
is optimistic by an unknown (likely small, given OpenImages/ImagesCV's modest share of the
full dataset) amount. Report the per-source composition of the test set alongside the number
(reusing the existing per-class/per-source breakdowns already produced by the dataset
pipeline, e.g. `reports/dataset_split_summary.json`) so the reader can judge the scale of the
effect themselves. This keeps every evaluation in the thesis comparable on the same
instrument, at the cost of an explicit, documented caveat on one specific number rather than
a silent methodological patch.

### 1.2 Is YOLOv5s an appropriate KD student?

**No.** Two independent reasons converge on this:

1. **It is approximately the size of what's already deployed.** Per
   `docs/progress_notes/2026-03-18_speciesnet-pipeline-and-experiment-design.md` §1,
   Swarovski's current AX Visio pipeline already runs a **fine-tuned YOLOv5s detector**
   (the full MegaDetector was too large/slow for the device) followed by the SpeciesNet
   classifier on crops. YOLOv5s (7.2M params, ~37.4% COCO mAP) is therefore not a meaningful
   KD "compression target" — there is no capacity reduction to study, and no product reason
   to distill into a model the same size as what is already shipping.
2. **It is 3–7× larger than the documented target student candidates.** Per
   `docs/2026-03-10_object-detection-models-for-embedded-systems.md` and
   `docs/progress_notes/2026-03-18`, the actual embedded-deployment candidates are:

   | Model | Params | COCO mAP | Notes |
   |---|---|---|---|
   | NanoDet-Plus-m | 1.17–1.8M | 30.4% | ShuffleNetV2 backbone, ARM-friendly, ~20ms on RPi5 |
   | PicoDet-S | 0.99–1.1M | 30.6% | Smallest; ESNet+PAN; PaddleDetection-native toolchain |
   | YOLO11n | 2.6–2.7M | 39.4% | Ultralytics ecosystem |
   | YOLO26n | 2.4M | 40.9% | Best mAP/param ratio in the nano tier; NMS-free; AGPL-3.0 |

**What YOLOv5s should keep being used for:** it is already the project's Phase-1
direct-fine-tuning baseline, and — because it mirrors Swarovski's deployed detector — it
doubles as a stand-in for "how does the currently-shipping model perform under our harder,
fine-grained 225-class evaluation suite." Keep it in the comparison matrix as the
**production-baseline reference row**, not as a KD student.

**Recommendation for the actual KD student.** Start engineering with **YOLO11n or YOLO26n**
(Ultralytics ecosystem) rather than jumping straight to NanoDet-Plus-m / PicoDet-S, which are
the true longer-term embedded targets. Reasoning: the existing pipeline's model factory
(`yolov5s_model.py`) already depends on Ultralytics/YOLOv5 tooling conventions (loading a
YOLO YAML config, `ComputeLoss` from the `yolov5` package). YOLO11n/26n share enough of that
tooling that the model-factory + loss-wrapper adaptation is the smallest possible engineering
lift, letting the genuinely novel and risky part of this work — the KD loss and teacher-output
caching infrastructure (§3, §5) — be validated against a low-friction architecture first.
NanoDet-Plus-m (NCNN/PyTorch hybrid) and PicoDet-S (PaddleDetection-native) require separate
model-loading/export toolchains and should follow once the KD machinery is proven on
YOLO11n/26n. Note the AGPL-3.0 license on YOLO11n/26n as a **deployment** caveat for the
thesis text (Swarovski would need an Enterprise License to ship it) — it does not affect the
validity of the *research* comparison.

### 1.3 Doesn't fine-tuning the teacher (option 1) duplicate "the teacher trains in parallel" (option 2)?

This is a real and common point of confusion, and the answer resolves it directly: **in the
documented, recommended KD setup, the teacher does not train in parallel with the student.**
Offline, two-stage KD is the standard here. Per
`docs/2026-03-12_research-and-experimentation-plan.md` (Phase 2 rationale): *"Two-stage KD
(fine-tune teacher first, then distill) consistently outperforms one-stage end-to-end KD
because the teacher has already learned task-relevant features before being asked to
transfer them."*

Concretely:

1. **Stage A — "option 1":** Fine-tune MegaDetector + SpeciesNet on the dataset. Freeze the
   result. This is Phase 2 in the experimental ladder (§4).
2. **Stage B — "option 2":** Train the student (COCO-pretrained init) using a loss that
   combines (a) the normal hard-label task loss and (b) a distillation loss against the
   *frozen* Stage-A teacher's outputs — soft class probabilities from SpeciesNet, plus
   auxiliary localization signal from MegaDetector. This is Phase 3.

**Option 1 is not an alternative to option 2 — it is option 2's prerequisite phase.** There
is no redundancy to resolve. The "teacher trains in parallel" framing the user recalls
describes a different, less common KD variant — *online* or *mutual* distillation (e.g.
Deep Mutual Learning), where teacher and student co-train simultaneously because no strong
pretrained teacher exists yet. That is not the situation here: a strong, already-pretrained
MD+SN exists, and every project document that discusses this converges on the offline,
two-stage approach. Online KD is noted here only to close the loop on the terminology — it
is **not** part of this plan's scope.

A practical corollary follows directly: because the teacher is frozen after Stage A, its
outputs can be **precomputed once and cached** rather than re-run live inside the student's
training loop. This matters because MD+SN is ~196M params / ~357MB combined
(`docs/progress_notes/2026-03-18` §3), while the project's GPU budget
(`docs/2026-04-29_gpu_training_options.md`: two 12GB A40 instances) is sized for student-scale
models, not for running a ~200M-parameter ensemble live alongside each student training step.
See §5.

---

## 2. Why feature-based KD is excluded

Both `docs/2026-03-12_knowledge_distillation_research_overview.md` (§2, §6) and
`docs/progress_notes/2026-03-18_speciesnet-pipeline-and-experiment-design.md` (§5)
independently reach the same conclusion: feature-based KD (CWD, FGD, LD — aligning
intermediate feature-pyramid activations between teacher and student) requires the two
networks to have **comparable, spatially corresponding feature maps**. That assumption holds
for same-task, same-lineage pairs (e.g. YOLOv8s → YOLOv8n) but breaks down here:

- The **teacher is a two-stage pipeline**: a class-agnostic detector (MegaDetector) followed
  by a separate classifier (SpeciesNet's EfficientNetV2-M) operating on 480×480 *crops*.
- The **student is a single-stage detector** operating end-to-end on full images at 640px,
  with a dense multi-scale FPN/PAN feature pyramid that has no architectural counterpart in
  SpeciesNet's crop-classifier backbone.

There is no principled way to pick "which student feature map corresponds to which teacher
feature map" across this gap, and the added implementation complexity (adapter/projection
layers between fundamentally mismatched feature spaces) isn't justified by an uncertain
benefit. **Response-based** and **relation-based** KD are used instead (§3) — both operate on
the teacher's final outputs (class probabilities, detections) rather than intermediate
activations, so they sidestep the architecture-mismatch problem entirely.

---

## 3. Recommended KD design

**Classification branch — response-based (Hinton) KD.**
SpeciesNet's softmax output is projected onto the project's 225-class space using the
**already-implemented** mapping in `scripts/dataset_quality/7-filter_speciesnet.py`
(`load_classes_225()`, `compute_probs_225()` — species/genus/family lookup tables). Reuse
this directly; an earlier design doc
(`docs/plans/2026-04-30_speciesnet-classification-strategy.md` §"Reuse existing code")
proposed this logic live in a separate `scripts/training/0-teacher_speciesnet_pipeline.py`
script, but that script was never built — the real, working implementation is inside script 7,
so build on that rather than re-implementing the projection.

Loss on the classification branch:

```
L_cls = (1 - α) · CE(student_logits, hard_label)
      + α · T² · KL( softmax(student_logits / T) , probs_225_from_teacher )
```

with temperature `T` and weight `α` as sweep hyperparameters (same convention the March
plan proposed: `T ∈ {4, 8}`, `α ∈ {0.5, 0.7}` is a reasonable starting grid). Do **not** apply
logit-style KD to the box-regression branch — there is no shared box-distribution
representation between MegaDetector's NMS'd output and a different-anchor-scheme student.

**Localization branch — MegaDetector pseudo-labels, not feature mimicry.**
Two concrete, low-risk mechanisms, both already foreshadowed in
`docs/plans/2026-04-30_speciesnet-classification-strategy.md` §2a–2b:

1. **Secondary-detection pseudo-GT.** In multi-animal images, MegaDetector's
   lower-confidence animal detections beyond the primary one (already tracked via the
   `multi_animal` / `n_animal_detections` fields from the dataset-quality pipeline) become
   additional pseudo-GT boxes, paired with their own SpeciesNet soft label. This is the one
   piece of *extra* supervision that hard-label training structurally cannot use — an image
   with one labeled subject plus two background animals gives KD three supervision signals
   where direct fine-tuning gives one. This is the most concrete practical KD advantage in
   this setup, and is worth an explicit ablation (single-animal vs. multi-animal training
   subsets) as a thesis result, exactly as proposed in the source doc.
2. **Person-class downweighting**, carried over unchanged from the existing dataset/training
   convention (0.3× loss weight on the person class) to keep the KD and hard-label loss
   branches consistent on the deployment-motivated animal-priority behavior.

**Capacity-gap risk.** The parameter ratio between MD+SN (~196M combined) and a nano student
(~1–2.7M) is roughly **70–170×** — far past the ~2.5–3.7× gaps the capacity-gap literature
treats as empirically safe
(`docs/2026-03-12_knowledge_distillation_research_overview.md` §1). Response-based KD is less
exposed to this failure mode than feature-based KD (it doesn't require matching
representational capacity in the feature space), but negative transfer on the classification
branch (KD underperforming direct fine-tuning) is a real, plausible experimental outcome here,
not just a theoretical edge case — treat it as a first-class possible result (§6), not a bug
to debug away. If it does appear consistently, the documented fallback is a
**teacher-assistant bridge**: fine-tune an intermediate-capacity model (e.g. YOLO11m) on the
wildlife data and distill MD+SN → bridge → nano student in two hops. This is noted here as a
contingency to reach for if needed, not part of the default Phase 3 scope.

---

## 4. Experimental ladder

Adapted from the March research-and-experimentation-plan's phase structure, re-grounded in
the actual current dataset and evaluation framework
(`docs/plans/2026-06-10_model-evaluation-strategy.md`).

**Phase 0 — Zero-shot baselines.**
Run the COCO-pretrained student (no wildlife training) and the off-the-shelf (not yet
fine-tuned) MD+SN pipeline on the eval suite's test domains. Establishes how severe the
COCO→wildlife domain shift actually is and gives an upper-bound reference.

**Phase 1 — Direct fine-tuning baseline.**
Already done for YOLOv5s (kept as the production-baseline reference row, §1.2). Extend the
same pipeline pattern to the chosen student (YOLO11n/26n first, §1.2) trained directly on
`data/real/` (+ synthetic per band) with no teacher involved. This is the baseline every KD
condition must beat to justify its added complexity.

**Phase 2 — Teacher fine-tuning.**
Fine-tune the **SpeciesNet classifier head** on the dataset's MegaDetector crops, output
mapped to the 225-class space (primary focus — see below for why). Evaluate species
accuracy/F1 on the **same fixed test set used by every other phase and every other model** —
no special slice (see the disclosure-based mitigation in §1.1) — and report the per-source
test-set composition alongside the number so the circularity caveat is visible to the reader
rather than hidden by the evaluation methodology. **MegaDetector fine-tuning
is optional/secondary**: it already reports 99.2% precision / 97.3% recall on camera-trap
data (`docs/progress_notes/2026-03-18` §3) and is class-agnostic (animal/person/vehicle only,
no species-specific adaptation needed) — there is limited expected headroom from fine-tuning
it further. Revisit this default only if Phase 4 diagnostics show MD's localization, not
SpeciesNet's classification, is the bottleneck on this specific imagery domain.

**Phase 3 — Knowledge distillation.**
Train the student (COCO-pretrained init, *not* the Phase-1 fine-tuned weights, to isolate the
KD signal) using the loss design in §3, with the frozen Phase-2 teacher as the soft-label
source.

**Phase 4 — Comparison.**
Score zero-shot, direct-FT, and KD checkpoints through the **existing**
`scripts/training/yolov5s/eval_suite/run_evaluation.py` — no new evaluation code path, only
new checkpoints fed into the established tool. Report:
- The headline mixed cell (`G=fine, D=mixed, B=all`) plus the real-only breakout, per
  `docs/plans/2026-06-10_model-evaluation-strategy.md` §3.
- The band × granularity grid (does KD help disproportionately on thin Band A/B classes,
  per the long-tail hypothesis already documented in the March plan).
- A dedicated **multi-animal-subset cut**: KD vs. direct-FT mAP restricted to images flagged
  `multi_animal=true` vs. single-animal images — the concrete test of the §3 KD advantage
  claim.

**Explicitly out of scope:** quantization, QAT, and SNPE/DSP export. These remain governed by
`docs/2026-03-10_object-detection-models-for-embedded-systems.md` and should be the subject of
a separate plan once a winning student + training-strategy combination is selected from
Phase 4's results.

---

## 5. Engineering plan

| Component | Action | Notes |
|---|---|---|
| `scripts/training/yolo11n/` (or chosen student name) | **New** — model factory + loss wrapper only | Mirrors `scripts/training/yolov5s/`'s module map. `dataset.py`, `training_pipeline.py`, `evaluation.py`, `eval_suite/` are explicitly designed to be reused unchanged — `scripts/training/yolov5s/README.md`: *"owned by this package so it can be reused for NanoDet / PicoDet runs."* |
| `scripts/training/teacher_finetune/` | **New** | Fine-tunes the SpeciesNet classifier head on the dataset's crops (224/480px), output via the existing `classes_225.csv` projection. Reuses MegaDetector bboxes already stored in `filter_results.jsonl` — does not re-run MegaDetector. |
| Teacher-output cache | **New step** | Re-run the `6-classify_speciesnet.py` batching/threading pattern using the **fine-tuned** Stage-A teacher checkpoint, producing a `speciesnet_results.jsonl`-equivalent cache of soft labels per training image, computed once. The student's dataset loader reads this cache as an extra label source — the ~196M-param teacher never runs inside the student's training loop (resolves the GPU-budget concern in §1.3). |
| KD loss module | **New**, extends `loss.py`'s wrapper pattern | Adds the soft-CE/KL term (§3) to the existing `YoloLoss` output; logs `loss_kd` as an extra MLflow scalar alongside the existing `loss_box/obj/cls`. |
| `constants.py`-equivalent for new students | **Modify** | Add `KD_TEMPERATURE`, `KD_ALPHA` following the existing `as_dict()`-logged convention. |

---

## 6. Verification

- **Phase 1 sanity gate.** Each new student's direct-FT run must reach a non-degenerate
  mAP50_95 on val (same order of magnitude as YOLOv5s' figures) before any KD run is
  attempted — mirrors how the YOLOv5s pipeline itself was smoke-tested
  (`scripts/training/yolov5s/smoke_test_augmentation.py`).
- **Phase 2 success criterion.** SpeciesNet's species accuracy/F1 on the 225-class
  projection improves, *measured on the same fixed test set used throughout the project* —
  no special slice (§1.1) — relative to the off-the-shelf checkpoint. Report the test set's
  per-source composition alongside the number and state the circularity caveat explicitly in
  the write-up rather than excluding any portion of the test set to engineer around it.
- **Phase 3/4 comparison.** Run through `eval_suite/run_evaluation.py` exactly as for any
  other trained checkpoint — no bespoke evaluation logic for KD models.
- **Negative-transfer reporting.** If any KD condition's headline mAP comes in below its
  direct-FT counterpart, report it explicitly rather than discarding the run — per the March
  research framing, a clean negative result (logit-based KD failing under domain shift,
  capacity gap exceeding safe limits) is a valid, publishable thesis finding in its own right.
- **No duplication check.** Confirm this document does not re-litigate quantization/QAT
  (owned by `docs/2026-03-10_object-detection-models-for-embedded-systems.md`) or the
  evaluation-axis design (owned by `docs/plans/2026-06-10_model-evaluation-strategy.md`) —
  it only references and reuses them.
