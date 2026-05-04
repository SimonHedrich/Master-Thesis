# Dataset Construction Strategy

**Date:** 2026-05-04
**Status:** Draft — pending class distribution analysis

---

## 1. Data Source Trust Levels

Sources are split by label reliability before any tier-based processing is applied.

| Source | Trust Level | SpeciesNet Filtering |
|---|---|---|
| iNaturalist | Trusted | Not required |
| Wikipedia | Trusted | Not required |
| GBIF | Trusted | Not required |
| ImageCV | Unverified | Required |
| OpenImages | Unverified | Required |

Trusted sources are assumed to have the correct species in the designated category folder. Unverified sources require SpeciesNet confidence-threshold filtering to confirm label integrity before training use.

---

## 2. Training Tiers by Available Image Count

Classes are assigned to one of four tiers based on the number of images available **after** quality filtering. Tier boundaries are set at 100 / 500 / 1,500 images but are open to adjustment once the full class distribution from the SpeciesNet filter output is known.

---

### Tier 1 — Critical Scarcity: < 100 images

**Filtering:**
SpeciesNet filtering is skipped for this tier to avoid further reducing an already critical image count. Images where MegaDetector detected an animal are **manually reviewed** to verify label correctness.

**Training data:** A small number of synthetically generated images — up to a few hundred per class (exact count to be determined; not 1,500, as that would cost ~€75/class and exceed budget).

**Evaluation data:** The real images are withheld entirely from training and used exclusively for evaluation.

**Goal:** Assess whether synthetic-only training can produce a usable detector for data-scarce species, with real images providing a clean, uncontaminated evaluation set.

---

### Tier 2 — Low Data: 100–499 images

**Filtering:**
Same as Tier 1: SpeciesNet filtering is skipped; images with a MegaDetector detection are **manually reviewed**.

**Training data:** Real images supplemented with a small number of synthetic images (up to a few hundred per class) to improve variety. The total training count remains well below 1,500.

**Evaluation data:** Held-out portion of the real images.

---

### Tier 3 — Sufficient Data: 500–1,499 images

**Filtering:**
- Unverified sources (ImageCV, OpenImages): SpeciesNet filtering applied as normal.
- Trusted sources (iNaturalist, Wikipedia, GBIF): if an image **fails** SpeciesNet filtering, it is flagged for **manual review** rather than automatically discarded — the label may still be correct despite a low confidence score.

**Training data:** Primarily real images. Synthetic supplementation is generally not expected to be necessary.

**Evaluation data:** Held-out portion of real images.

**Comparison goal:** This tier is also used to directly compare training on real images versus training on synthetic images. To ensure a fair comparison, the training set for this tier may be **capped to the same image count used in Tiers 1 and 2**, keeping the experimental conditions consistent across tiers.

---

### Tier 4 — Abundant Data: ≥ 1,500 images

**Filtering:** Standard pipeline; SpeciesNet filtering applied for unverified sources.

**Training data:** Capped at 1,500 real images (randomly sampled). No synthetic generation needed.

**Evaluation data:** Remaining real images beyond the 1,500 cap.

**Note:** Tier 4 classes are predominantly common, well-represented animals (e.g., deer, wild boar, fox). Robust performance on these species is important for real-world customer use cases, where the model must reliably recognise high-frequency wildlife.

---

## 3. Cost Estimate

| Parameter | Value |
|---|---|
| Cost per synthetic image | €0.05 |
| Maximum budget | €500 |
| Maximum synthetic images | **10,000** |

With a cap of a few hundred synthetic images per class, the budget of 10,000 images can cover approximately 20–50 classes in Tiers 1 and 2 (depending on the exact per-class count chosen). The precise allocation will be determined once the class distribution from `scripts/dataset_quality/7-filter_speciesnet.py` is available.

---

## 4. Open Questions

- Exact per-class synthetic image count for Tiers 1 and 2 (budget-driven decision).
- Exact cap for the Tier 3 real-vs-synthetic comparison (should match Tier 1/2 totals).
- Final tier boundaries: currently 100 / 500 / 1,500; to be revisited after full class distribution analysis.
