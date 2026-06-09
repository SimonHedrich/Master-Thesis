# Data Augmentation Strategy for the Distillation-vs-Direct Comparison

**Date:** 2026-06-07 (revised 2026-06-09)
**Status:** Plan only — no implementation yet
**Relates to:** `docs/progress_notes/2026-06-03_yolov5s-training-pipeline.md`, `docs/plans/2026-06-02_yolov5s-training-pipeline.md`, `docs/plans/2026-06-08_fix-incomplete-gt-annotations.md`, `docs/thesis-overview.md`

---

## 1. Problem Statement & Goal

The baseline training pipeline has all augmentation disabled (`AUG_MOSAIC=False`, `AUG_HSV=False`, `AUG_HFLIP=False`). This was intentional for a clean first run. The question this plan answers is **what augmentation to enable, and how to structure it so it serves the thesis's core research question** rather than an adjacent one.

The thesis's core question is **whether distilling a teacher into the student beats directly fine-tuning the student** on the wildlife domain. Augmentation is not a side concern here — it is a confounder that must be controlled, and it is also an *alternative* lever for improving the student that distillation has to be measured against. So the augmentation design is built directly around the comparison (Section 3).

### What augmentation is and is not for, in this thesis

- **It is for in-domain robustness.** Training data comes from several photo sources (GBIF, iNaturalist, OpenImages, Wikimedia, plus `images_cv`) and is supplemented with synthetically generated images for rare classes. The model is **evaluated on the held-out test split drawn from the same sources** — *not* on live AX Visio binocular footage. Augmentation should therefore make the model robust to the variability *within that training distribution* (lighting, scale, pose, partial occlusion, left/right symmetry), which the in-domain test split rewards.
- **It is not for bridging a sensor domain shift.** There is no AX-Visio-specific evaluation in scope, so there is no reason to add transforms that simulate binocular optics (JPEG/optical artifacts, specific framing). *(This corrects the original draft, which incorrectly motivated augmentation by an iNaturalist→AX-Visio domain shift.)*
- **Rare-class note.** The low-count classes (Band A in the split report — 51 classes) are trained on **synthetic images**, not zero data, so augmentation applies to them normally. The one caveat is that these classes train on synthetic and test on real, so augmentation strength is effectively a synthetic→real bridging knob for them; their test mAP should be tracked separately (Section 5) to see whether the shared augmentation helps or amplifies synthetic-texture artifacts.

---

## 2. Two Axes That Govern Every Augmentation Choice

Augmentations differ along two independent axes, and conflating them is what made the original plan ambiguous.

**Axis 1 — Distillation-compatibility: single-image vs. multi-image compositing.**
In distillation the teacher must produce soft targets for *the exact pixels the student sees*.
- **Single-image** transforms (flip, scale, translate, photometric, blur) are fine: run the teacher on the same transformed image and its targets stay valid.
- **Multi-image compositing** transforms (mosaic, mixup, cutmix, copy-paste) stitch/blend several source images. There is no well-defined teacher view of the composite, so these **cannot be used in a distillation run**. This — not "the whole YOLO recipe" — is the precise set that is off-limits to the teacher+student setup.

**Axis 2 — Architecture-specificity.**
- **Generic**: mechanically independent of the detector's label assignment/loss (flip, photometric, scale/translate).
- **Architecture-coupled**: tied to a model's anchor matching / loss / training protocol — mosaic (YOLO anchor coverage), SSD-style IoU-constrained crop (anchor matching), multi-scale training (stride/FPN). Applying one model's coupled augmentation to another either hurts it or gives an unfair edge.

The basic set (Section 4.1) is deliberately chosen from the **single-image + generic** quadrant so it is valid for both distillation and direct training and does not favor any architecture.

---

## 3. The Three Training Setups

The student architecture is held **fixed** across all three (e.g. YOLOv5s). Only one factor changes between each adjacent pair — a single-factor design.

| Setup | Loss | Augmentation | Role |
|---|---|---|---|
| **A** | teacher→student distillation | **basic set** (single-image, Section 4.1) | distillation condition |
| **B** | direct detection | **basic set** — *identical to A* | direct-fine-tune baseline |
| **C** | direct detection | **basic set + everything easily available** (compositing, Section 4.2) | the student's honest augmentation ceiling |

This is a **nested** design: B = A with the loss swapped; C = B with the compositing layer added on top. What each comparison isolates:

- **A vs. B** — *the distillation benefit.* Augmentation is byte-for-byte identical (Section 5), so any difference is attributable to the teacher. **This is the thesis headline.**
- **B vs. C** — *the compositing/architecture-specific augmentation benefit alone.* Same loss, the only difference is the compositing layer.
- **A vs. C** — *the practitioner's question:* does standing up a full teacher+distillation pipeline beat simply turning on the student's native aggressive augmentation? Note this pair differs by **two** factors (loss *and* compositing), so it is a head-to-head of techniques, **not** a clean single-factor isolation — the clean isolations are A-vs-B and B-vs-C. If C ≥ A, that is a strong result against distillation's added complexity.

Why there is no fourth "teacher+student with compositing" cell: compositing breaks the teacher view (Axis 1), so it is genuinely undefined, not a missing experiment.

---

## 4. Augmentation Catalogue

### 4.1 Basic set — setups A & B

Chosen as the compromise between "everything possible" and "easy to implement with little bug surface." The dividing line is **bbox-coordinate risk**: flip and photometric carry essentially none; aggressive crop/shear/perspective carry the most. We take the high-value, low-risk transforms and stop there.

| Transform | Value for wildlife | Bbox risk | Decision |
|---|---|---|---|
| Horizontal flip (p=0.5) | Animals face either way; no semantic L/R | `x→1-x`, ~zero | **Include** |
| HSV jitter (H 0.015, S 0.7, V 0.4) | Outdoor lighting/shadow/habitat color | none (no geometry) | **Include** |
| Random scale (±0.5) + translate (0.1) | Scale invariance: distant vs. close animal | moderate | **Include — via reused tested code** |
| Rotation / shear / perspective | Minor tilt benefit | high (bbox distortion) | **Off** — low value, high bug surface |
| Vertical flip | Animals are upright | n/a | **Off** |
| Blur / CutOut / grayscale | Mild robustness | low | **Off** — keep set minimal (Section 9) |

**Bug-aversion stance:** the scale/translate transform is the only geometrically non-trivial member, and it is exactly where hand-rolled bbox math goes wrong. It must therefore **reuse the tested implementation from the `yolov5` package** (`yolov5.utils.augmentations.random_perspective` with `degrees=shear=perspective=0`, i.e. scale+translate only), not a bespoke crop-and-clip routine. Flip and HSV are trivial enough to keep in our own `transforms.py` if preferred, but reusing `augment_hsv` is equally fine.

### 4.2 Setup C — "everything easily available"

Setup C enables the student's native, already-implemented augmentation on top of the basic set. For YOLOv5s, "easily available" means the standard YOLOv5 hyperparameter recipe, driven through the library's own augmentation code — **not** reimplemented by us:

- **Mosaic** (p≈1.0) — 4-image stitch; the primary architecture-specific augmentation.
- **MixUp** (p≈0.1) — image+label blend (YOLOv5/YOLOX default companion to mosaic).
- **Copy-Paste** — present in the recipe but a **no-op on our data**: it requires instance masks, and our GT is box-only MegaDetector output. Document it as enabled-but-inert rather than silently expecting an effect.
- **Close-mosaic tail** (~last 10 epochs) — disable mosaic for the final epochs so the model adapts to real image statistics. This is part of getting an *honest* mosaic ceiling; running mosaic to the last epoch would understate setup C.

Because these run through the library's debugged pipeline, setup C adds little implementation risk despite being the "aggressive" arm.

### 4.3 Reference — full catalogue (for a possible future multi-architecture bake-off)

Not in scope for the current single-student experiment, but recorded so the architecture-specific augmentations are not re-discovered later. If NanoDet / PicoDet / EfficientDet-Lite ever enter the comparison, each must be granted **its own** native recipe rather than having YOLO's imposed on it.

| Augmentation | Axis 1 | Axis 2 (coupling) | Current experiment |
|---|---|---|---|
| HFlip, HSV | single-image | generic | Basic (A, B, C) |
| Scale + translate | single-image | generic | Basic (A, B, C) |
| Rotation / shear / perspective | single-image | YOLO recipe | Off (low value, bug-prone) |
| Blur / CutOut / grayscale | single-image | generic | Off (open question) |
| Mosaic / Mosaic-9 | **compositing** | YOLO anchor coverage | Setup C only |
| MixUp | **compositing** | YOLO/YOLOX recipe | Setup C only |
| CutMix | **compositing** | recipe | Off |
| Copy-Paste | **compositing** | needs masks | Setup C (inert on box data) |
| SSD zoom-out "expand" + IoU-constrained crop | region | SSD/EfficientDet anchor matching | Future multi-arch only |
| AutoAugment-for-detection + large-scale jitter | single-image ops | EfficientDet recipe | Future multi-arch only |
| Multi-scale training | protocol | anchor/FPN stride | Off (fixed embedded input) |
| Close-mosaic last N epochs | protocol | travels with mosaic | Setup C (with mosaic) |

---

## 5. Reproducibility — Keeping A and B Identical

The A-vs-B comparison is only valid if the data pipeline is **byte-for-byte identical** between the two runs; the *only* permitted difference is the loss function. Requirements:

1. **Single shared code path.** Both the distillation training entry point (A) and the direct entry point (B) consume the **same** `CocoYoloDataset` + `transforms.py` module and the **same** `AUG_*` constants. No copy of the augmentation logic per entry point.
2. **Same RNG discipline.** Identical `SEED`, identical `worker_init_fn`/`generator` seeding, identical per-epoch shuffling. Augmentation randomness must be seeded from the same source so epoch *n* presents the same augmented images in both runs.
3. **One augmented image, two consumers (setup A).** In distillation, the dataloader emits a single augmented (single-image) sample; the teacher and the student both forward-pass *that same image*. The teacher is never shown a differently-augmented or un-augmented copy.
4. **Compositing flags provably off in A.** `AUG_MOSAIC/MIXUP/COPY_PASTE` must be unreachable in the distillation path (assert at startup), so a config slip cannot silently invalidate the headline comparison.
5. **Log the resolved augmentation config to MLflow** for every run (the existing `as_dict()` covers this) so A and B can be diffed and shown equal.

Additionally, **log Band-A (synthetic-trained) test mAP as its own metric** alongside aggregate test mAP, so the augmentation's effect on the synthetic→real rare-class tail is visible rather than buried.

Val and test dataloaders always receive `augment=False`; only letterbox-resize is applied.

---

## 6. Proposed Augmentation Constants

Shared default values encode **setups A & B** (basic set on, compositing off). Setup C overrides the compositing block; the basic block is unchanged between all three.

```python
# ─── Basic shared augmentation — setups A & B (single-image, distillation-safe) ──
# Applied identically in the shared dataloader; A and B differ ONLY in the loss.
# The scale/translate transform reuses yolov5.utils.augmentations.random_perspective
# (tested) rather than hand-rolled bbox math, to minimise bug surface.

AUG_HFLIP        = True    # horizontal flip, p=0.5  (bbox x -> 1-x; ~zero risk)
AUG_HSV          = True    # photometric jitter; no bbox math at all
AUG_HSV_H        = 0.015   # hue gain        (YOLOv5 reference default)
AUG_HSV_S        = 0.7     # saturation gain
AUG_HSV_V        = 0.4     # value gain
AUG_SCALE        = 0.5     # random scale gain (±0.5), via random_perspective
AUG_TRANSLATE    = 0.1     # random translation fraction, via random_perspective
AUG_DEGREES      = 0.0     # rotation OFF       — low value for upright animals
AUG_SHEAR        = 0.0     # shear OFF          — bbox-distortion risk
AUG_PERSPECTIVE  = 0.0     # perspective OFF    — bug-prone, low value
AUG_FLIPUD       = 0.0     # vertical flip OFF  — animals are upright

# ─── Compositing augmentation — setup C ONLY (multi-image; breaks distillation) ──
# No valid teacher view of a composite, so these are NEVER enabled in setup A.
# Shared default = OFF (setups A & B). Comments show the setup-C override values.

AUG_MOSAIC       = 0.0     # setup C: 1.0   (4-image stitch)
AUG_MIXUP        = 0.0     # setup C: 0.1   (image+label blend)
AUG_COPY_PASTE   = 0.0     # setup C: 0.0   — needs masks; NO-OP on box-only GT
AUG_CLOSE_MOSAIC = 0       # setup C: 10    — epochs of mosaic-off tail before end
```

---

## 7. Implementation Sequence

When implementing (not now):

1. **Sequence after the GT-annotation fix** (`2026-06-08_fix-incomplete-gt-annotations.md`). That fix turns the data from 1-box-per-image into genuine multi-instance, which changes how scale/translate visibility and (in setup C) mosaic behave. Implementing and smoke-testing augmentation against the *current* single-box JSONs would bake in wrong assumptions.
2. Add the constants in Section 6 to the model `constants.py`.
3. Wire the **basic set** into the shared `CocoYoloDataset.__getitem__`, gated on the `AUG_*` flags, reusing `yolov5.utils.augmentations` (`augment_hsv`, `random_perspective` with rotation/shear/perspective = 0) and a trivial horizontal flip. Keep this the single code path used by both the distillation (A) and direct (B) entry points.
4. Add the startup assertion that compositing flags are off whenever the distillation path is active (Section 5.4).
5. For **setup C**, drive mosaic/mixup/close-mosaic through the library's existing augmentation pipeline rather than reimplementing them.
6. **Smoke test** (1 epoch, val split) against the *fixed multi-box* JSONs: confirm augmented batches render correctly, all targets stay in `[0, 1]`, and multi-instance boxes survive the basic transforms.
7. MLflow experiments: keep `yolov5s-wildlife225` (no-aug baseline) as reference; create runs for setups A, B, C with the resolved augmentation config logged for each.

---

## 8. Open Questions

- **Scale range:** `±0.5` (scale 0.5–1.5) is the YOLOv5 default. A narrower range avoids extreme downscaling that makes distant animals vanish; may need tuning once the multi-box data is in place.
- **Blur / CutOut:** deliberately excluded to keep the basic set minimal and bug-free. Cheap to add later as a single-image extension to setups A/B if rare-class recall is weak — but only if added to **both** A and B simultaneously.
- **Synthetic rare classes:** Band-A classes train on synthetic and test on real. If their separately-logged test mAP regresses under augmentation, the basic-set strength (especially scale) may need to be reduced for synthetic samples specifically to avoid amplifying texture artifacts.
- **MixUp probability in C:** `0.1` is the YOLOv5 default; whether a higher value helps is a setup-C-internal tuning question that does not affect the A/B comparison.
