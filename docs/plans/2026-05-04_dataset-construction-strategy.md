# Dataset Construction Strategy

**Date:** 2026-05-04 → 2026-05-06
**Status:** Final — ready for implementation

---

## Strategy Evolution

The initial draft (2026-05-04) proposed four Tiers with boundaries at 100/500/1,500 images, leaving synthetic image counts and the Tier 3 comparison cap as open questions pending class distribution analysis. After running the full SpeciesNet filter pipeline and completing the manual image review (17,737 images reviewed, 13,827 approved), the actual class distribution shifted the effective pool sizes enough that the tier boundaries were revised to 150/250/400 and the tiers were renamed Bands A–D to avoid confusion with the old thresholds. All previously open questions were resolved at that point, and synthetic counts were fixed at 200/class for Band A and 100/class for Band B.

---

## 1. Data Source Trust Levels

Sources are split by label reliability before any band-based processing is applied.

| Source | Trust Level | SpeciesNet Filtering |
|---|---|---|
| iNaturalist | Trusted | Not required |
| Wikipedia | Trusted | Not required |
| GBIF | Trusted | Not required |
| ImageCV | Unverified | Required |
| OpenImages | Unverified | Required |

Trusted sources are assumed to have the correct species in the designated category folder. Unverified sources require SpeciesNet confidence-threshold filtering to confirm label integrity before training use. For trusted sources where an image fails SpeciesNet filtering, the image is flagged for manual review rather than automatically discarded — the label may still be correct despite a low confidence score.

---

## 2. Training Bands by Available Image Count

Training uses a 4-band system built around a **200-image training budget** for Bands A–C, making those three conditions directly comparable (synthetic-only, mixed, all-real).

The primary comparison is therefore **cross-band**: Band A classes (trained on 200 synthetic images) are compared against Band C classes (trained on 200 real images), with Band B (mixed) sitting between them. These are different species that fell into different bands due to data availability — not a controlled within-class experiment — but the equal 200-image budget makes the band-level comparison meaningful.

**Effective pool** = `effective_trusted + unverified_sn_pass` from `reports/class_distribution_reviewed.csv`.

| Band | Pool range | Training images | Test images | Synthetic? |
|------|-----------|-----------------|-------------|------------|
| A | < 150 | 200 synthetic | all real (0–149) | Yes — 200/class |
| B | 150–249 | 100 real + 100 synth = 200 | 50–149 real | Yes — 100/class |
| C | 250–399 | 200 real | 50–199 real | No |
| D | ≥ 400 | min(pool − test, 1,500) | max(⌊pool × 0.2⌋, 50), capped at 500 | No (except optional apples-to-apples extension) |

Key constraint: **never discard usable real images to enforce a training cap** — surplus images beyond the training budget go to the test set instead.

**Exclusions:** `human` (0 real images) and `unmatched` (7,715 images, no real species category) are excluded from all bands and experiments.

---

### Band A — pool < 150 (50 classes)

All 37 former Tier-1 classes (pool < 100) plus 13 former Tier-2 classes with pool 100–149:

| Class | Pool |
|-------|------|
| walrus | 101 |
| old world porcupine family | 102 |
| raccoon dog | 105 |
| callicebus genus | 112 |
| wild cat | 112 |
| black-backed jackal | 112 |
| ringtail | 123 |
| kinkajou | 125 |
| genet genus | 132 |
| leopardus species | 132 |
| water deer | 133 |
| eurasian badger | 141 |
| nine-banded armadillo | 145 |

Training: 200 synthetic images per class.
Test pool: all available real images (review_approved / effective_trusted).

Average real test images across Band A ≈ 67. The old-Tier-1 subgroup averages ~49.
**Flag in results:** ~17 classes have fewer than 30 real test images — these evaluations are test-limited by definition, not a model failure.

---

### Band B — pool 150–249 (26 classes)

| Class | Pool |
|-------|------|
| canada lynx | 151 |
| spectacled bear | 153 |
| caracal | 159 |
| eurasian lynx | 161 |
| black wildebeest | 162 |
| giant panda | 169 |
| serval | 170 |
| patas monkey | 175 |
| american mink | 176 |
| gerenuk | 177 |
| dhole | 190 |
| bat-eared fox | 190 |
| baird's tapir | 193 |
| grevy's zebra | 198 |
| asian elephant | 207 |
| kirk's dik-dik | 211 |
| american badger | 215 |
| chimpanzee | 216 |
| african wild dog | 229 |
| glaucomys species | 240 |
| common wombat | 241 |
| european bison | 243 |
| lowland tapir | 243 |
| tayra | 243 |
| eurasian otter | 247 |
| springbok | 247 |

Training: 100 real images (random sample) + 100 synthetic = 200 total.
Test: the surplus 50–149 real images after training sample is reserved.

---

### Band C — pool 250–399 (26 classes)

| Class | Pool |
|-------|------|
| muntjac genus | 250 |
| roan antelope | 250 |
| giant otter | 251 |
| giant anteater | 258 |
| bornean orangutan | 277 |
| red panda | 283 |
| kob | 286 |
| dromedary camel | 296 |
| domestic goat | 297 |
| bushbuck | 307 |
| northern chamois | 315 |
| cercopithecus species | 337 |
| puma | 338 |
| quokka | 341 |
| grant's gazelle | 341 |
| red kangaroo | 342 |
| sambar | 342 |
| grey wolf | 348 |
| ring-tailed lemur | 358 |
| chital | 360 |
| snow leopard | 363 |
| meerkat | 364 |
| common eland | 385 |
| hartebeest | 386 |
| domestic donkey | 398 |
| dingo | 399 |

Training: 200 real images (random sample).
Test: the surplus 50–199 real images after training sample is reserved.
No synthetic generation needed.

---

### Band D — pool ≥ 400 (122 species classes)

Sub-groups by origin:
- **16 former Tier-2 classes** (pool 400–499): japanese macaque, common duiker, south american coati, hippopotamus, alpine ibex, sable antelope, thomson's gazelle, grey fox, blackbuck, jaguar, waterbuck, mountain zebra, swamp wallaby, giraffe, blesbok (439), sika deer (471)
- **26 former Tier-3 classes** (pool 514–1,484): includes nyala (554), golden jackal (551), gemsbok (563) — moved up from Band C/B after manual review (2026-05-07)
- **81 former Tier-4 classes** (pool 1,518–31,159)

Training: min(pool − test, 1,500) real images.
Test: max(⌊pool × 0.2⌋, 50), capped at 500 images.

Representative examples:

| Pool | Train | Test | Example class |
|------|-------|------|---------------|
| 420 | 336 | 84 | japanese macaque |
| 480 | 384 | 96 | giraffe |
| 730 | 584 | 146 | cheetah |
| 1,484 | 1,187 | 297 | pikas |
| 1,500 | 1,200 | 300 | brown-throated sloth |
| 5,500 | 1,500 | 500 | elk |
| 31,159 | 1,500 | 500 | squirrel family |

No synthetic generation needed (except for the optional apples-to-apples extension — see Section 5).

Band D represents the most common, well-represented wildlife species. Robust performance on these classes is important for real-world customer use cases where the model must reliably recognise high-frequency animals.

---

## 3. Training Cap and Test Split per Band

| Band | Train cap | Source | Test allocation | Min test |
|------|-----------|--------|----------------|---------|
| A | 200 | synthetic | all real available | 0 (test-limited) |
| B | 100 real + 100 synth | mixed | pool − 100 real training ≈ 50–149 | 50 |
| C | 200 real | real | pool − 200 ≈ 50–199 | 50 |
| D | min(pool − test, 1,500) | real | max(⌊pool × 0.2⌋, 50), ≤ 500 | 50 |

For Bands B and C, the random training sample is drawn first; all remaining real images go to the test set. This ensures no usable real image is discarded.

---

## 4. Synthetic Image Targets and Budget

| Item | Classes | Synth images | Cost (@€0.05/img) |
|------|---------|-------------|-------------------|
| Band A training (200/class) | 50 | 10,000 | €500 |
| Band B supplement (100/class) | 26 | 2,600 | €130 |
| **Total** | | **12,600** | **€630** |

Fits within the €800 hard cap with **€170 buffer** for re-generation or prompt iteration.

Copy-paste augmentation (2–3× variance multiplier) applies on top of generated images. 200 synthetic images in Band A ≈ 400–600 effective image-equivalents in training diversity.

---

## 5. Optional Extension: Apples-to-Apples Comparison (time-permitting)

**This experiment is not part of the fixed plan.** The primary synthetic-vs-real comparison is cross-band (Band A vs. Band C — see Section 2). If time allows, an additional apples-to-apples experiment can be run to isolate the synthetic data effect on the same species: five Band D classes with large surplus pools are selected and each is trained under two conditions on identical test sets:

1. **Synthetic-only (200)** — generated images, no real training data
2. **Real-only (200)** — random subsample from training pool

(The full-real Band D training run serves as the natural baseline and requires no extra training.)

Selected classes:

| Class | Pool | Test set | Comparison train |
|-------|------|---------|-----------------|
| cheetah | 863 | 173 | 200 synth / 200 real |
| tiger | 798 | 160 | 200 synth / 200 real |
| gorilla | 609 | 122 | 200 synth / 200 real |
| sea otter | 1,404 | 281 | 200 synth / 200 real |
| mountain goat | 793 | 159 | 200 synth / 200 real |

All five have effective pool > 500, giving stable test sets of 100–281 images. If conducted, synthetic cost: 5 × 200 = 1,000 images = €50 (addable to the budget if time permits).

---

## 6. Test Set Quality Analysis

| Band | Min test images | Avg test images | Notes |
|------|----------------|-----------------|-------|
| A | 0 (human excluded) | ~67 | ~17 classes < 50 real; flag as thin eval in results |
| B | 50 | ~100 | Acceptable for all 26 classes |
| C | 50 | ~125 | Good for all 26 classes |
| D (pool 400–499) | 80–100 | ~89 | Borderline but acceptable |
| D (pool 500–1,500) | 100–300 | ~180 | Good |
| D (pool > 1,500) | 300–500 | ~460 | Robust |

Target of ≥ 50 test images per class is met for all Band B, C, and D classes. Band A classes are inherently test-limited for species with few real images — this is a data scarcity finding, not an experimental design choice, and should be reported as such.

---

## 7. Exclusions

| Class | Reason | Action |
|-------|--------|--------|
| `human` | 0 real images; not a target species | Excluded from all bands and experiments |
| `unmatched` | 7,715 images with no real species category; label is a catch-all artifact | Excluded from all bands and experiments |

---

## 8. Experimental Conditions Summary

| Condition | Training images | Source | Bands involved |
|-----------|----------------|--------|----------------|
| Synthetic-only | 200 | All synthetic | Band A |
| Half-real + half-synth | 100 real + 100 synth | Mixed | Band B |
| Real-only 200 | 200 real | All real | Band C |
| Real 400–1,500 | pool-dependent | All real | Band D |
| Synthetic vs. real @ 200 *(optional)* | 200 each condition | Separate | 5 selected Band D classes |

The 200-image budget is consistent across Bands A–C, enabling the primary cross-band comparison: Band A (synthetic-only) vs. Band B (mixed) vs. Band C (real-only) at equal training size across different species groups.

---

## 9. Open Items Resolved

- **Q1 (synthetic counts):** Resolved. Band A = 200/class, Band B = 100/class.
- **Q2 (Tier 3 cap):** Resolved. The comparable 200-image condition is handled by Band C. Full Band D training uses all images up to 1,500.
- **Q3 (trusted-source review priority):** Resolved via the review server (Step 3 complete; 17,737 reviewed, 13,827 approved, 3,910 declined).
- **Q4 (OpenImages bbox):** Excluded — 7,499 records (~1.6% of total), not worth re-patching.

**Remaining pipeline task (Step 5):** Copy-paste augmentation implementation — online, using MegaDetector bounding boxes from `filter_results.jsonl`.
