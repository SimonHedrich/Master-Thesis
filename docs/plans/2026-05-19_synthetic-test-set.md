# Synthetic Test Set — 225 × 50 Images

**Date:** 2026-05-19  
**Status:** Planned

---

## Background

~12,600 synthetic training images were generated using the company Gemini API. The batch API (half price) was not used. To compensate, ~11,250–12,600 more images will be self-funded. Rather than generating more training images, these are spent on a **standardized synthetic test set** covering all 225 classes.

---

## Overview

```
225 classes × 50 images = 11,250 images  (≈ budget target)
Optional: 225 × 56 = 12,600 images exactly (add one more behavior variant)
```

---

## Why a Synthetic Test Set

| Problem | What the synthetic test set solves |
|---------|------------------------------------|
| Real test set varies wildly: Band A has only 6–149 real images/class | Fixed 50 images/class regardless of real availability |
| Band D (122 classes) has large real training sets but no controlled evaluation | Every class gets a standardized probe |
| Cross-band accuracy comparison is confounded by test set size differences | Equal 50-image test weight per class |
| Condition-specific failure analysis is impossible with uncontrolled real images | Known angle/behavior/lighting per image enables structured ablations |

---

## Relationship to Real Test Set

CLAUDE.md constraint: "Evaluate synthetic data only on real photographs (train on mix, test on real)."

The real test set remains the **primary evaluation** reported in the thesis. The synthetic test set is a **secondary analysis instrument** used for:
- Ablations and condition-level failure analysis
- Standardized per-class accuracy when real test sizes are too unequal
- Real-vs-synthetic gap measurement (domain shift quantification)

---

## Image Design: 50 per Class

Use the same "prototypical" distribution as the Band A val set (defined in `docs/plans/2026-05-12_synthetic-image-generation-strategy.md` Section 4.1):

| Count | Angle | Distance | Behavior |
|-------|-------|----------|----------|
| 10 | `eye_level` | `medium` | `standing_alert` |
| 10 | `eye_level` | `medium` | `walking` |
| 10 | `three_quarter_front` | `medium` | `eating_foraging` |
| 10 | `eye_level` | `medium` | `resting` |
| 10 | `three_quarter_front` | `medium` | `looking_at_camera` |

All 50: overcast or golden-hour lighting, no occlusion, primary habitat.

**Style:** binocular (no bokeh) — consistent with training images, no style-induced domain gap.

### Optional 6th behavior (to reach 12,600 total)

Add 6 images per class (`drinking` at water source, `eye_level/medium`):
- 225 × 56 = 12,600 exactly
- Adds one more behavioral variety at zero extra planning cost

---

## Band-Specific Notes

| Band | Real test set | What synthetic test adds |
|------|--------------|--------------------------|
| A (50 classes) | 6–149 real images (all used for test) | 50 controlled images; standardizes the tiny Band A per-class test |
| B (26 classes) | 50–149 real surplus | 50 controlled synthetic supplement |
| C (26 classes) | 50–199 real surplus | 50 controlled synthetic supplement |
| D (122 classes) | Large real test set | First controlled synthetic evaluation for these classes |

**Band A note:** The 40 synthetic val images used during training have a similar design but are part of the training pipeline (used only for early stopping). The 50 synthetic test images are held out entirely — never seen during training.

---

## File Layout

```
data/synthetic/
├── test_index.jsonl          ← metadata for all 11,250–12,600 test images
├── test_prompts/
│   ├── walrus/
│   │   ├── 001.txt
│   │   └── ...050.txt (or 056.txt)
│   └── ...225 class dirs
└── images/
    └── test/
        └── {class}/
```

**test_index.jsonl** record format (same schema as `index.jsonl`, with `split: "test"`):

```json
{
  "filename":    "t_walrus_001.jpg",
  "class":       "walrus",
  "scientific":  "Odobenus rosmarus",
  "band":        "A",
  "split":       "test",
  "shot_type":   "eye_level",
  "distance":    "medium",
  "behavior":    "standing_alert",
  "lighting":    "overcast",
  "occlusion":   "none",
  "prompt_file": "test_prompts/walrus/001.txt",
  "bokeh":       false,
  "status":      "pending"
}
```

Filename prefix `t_` distinguishes test images from training images (`a_`, `b_`).

---

## Implementation Plan

### Script: `scripts/synthetic/1-generate_test_image_list.py`

Mirrors `1-generate_image_list.py` but:
- Iterates over all 225 classes (not just 76)
- Generates 50 (or 56) images per class using the val-set shot schedule above
- Writes prompts to `data/synthetic/test_prompts/{class_slug}/{nnn:03d}.txt`
- Appends records to `data/synthetic/test_index.jsonl`

### Generation

Reuse `scripts/synthetic/2-generate_images.py` pointed at `test_index.jsonl` instead of `index.jsonl`. No changes needed to the generation script itself.

### Quality control

Same MegaDetector + SpeciesNet pipeline as training images. Test images with no detection are flagged and regenerated. The 10% failure allowance applies.

