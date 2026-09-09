# Model Evaluation Report

**Checkpoint:** `scripts/training/yolov5s/model_exports/yolov5s-20260813-162031/best.pt`  
**max_det:** 100 · **synthetic:** True


## Tier 1 — Headline

Default cell `G=fine · D=mixed · B=all` (cross-model ranking number), with the `D=real` breakout (public-comparison anchor) alongside.

| Metric | mixed (headline) | real (breakout) |
|--------|------------------|-----------------|
| map | 0.417 | 0.386 |
| map_50 | 0.484 | 0.460 |
| map_75 | 0.450 | 0.419 |
| map_small | — | — |
| map_medium | 0.305 | 0.308 |
| map_large | 0.448 | 0.415 |
| mar_1 | 0.422 | 0.382 |
| mar_10 | 0.517 | 0.491 |
| mar_100 | 0.521 | 0.496 |
| mar_small | — | — |
| mar_medium | 0.421 | 0.425 |
| mar_large | 0.547 | 0.520 |

**Public-comparison analog** — class-agnostic `mAP_detect`: mixed 0.773 (mAP50 0.922), real 0.749 (mAP50 0.912).


**Statistical hygiene** — headline mixed mAP 0.417 → 0.434 excluding the 9 test-limited (<30 real img) classes. Count-weighted (micro) mixed mAP: 0.448.

## Tier 2.1 — Granularity gap decomposition (mixed, all classes)

| Level | mAP | mAP50 |
|-------|-----|-------|
| detect (localisation only) | 0.773 | 0.922 |
| coarse (look-alikes merged) | 0.437 | 0.512 |
| fine (full 225-way) | 0.417 | 0.484 |

Δ_coarse (cross-group cost) = 0.336 · Δ_fine (look-alike cost) = 0.020 (mAP).

## Tier 2.2 — Band × granularity grid


**Domain: mixed**

| Band | n_img | mAP_fine | mAP50_fine | mAP_coarse | mAP50_coarse |
|------|-------|----------|------------|------------|--------------|
| A | 8294 | 0.000 | 0.000 | 0.086 | 0.093 |
| B | 9304 | 0.364 | 0.424 | 0.444 | 0.521 |
| C | 14253 | 0.540 | 0.620 | 0.550 | 0.638 |
| D | 42775 | 0.693 | 0.800 | 0.701 | 0.815 |

**Domain: real**

| Band | n_img | mAP_fine | mAP50_fine | mAP_coarse | mAP50_coarse |
|------|-------|----------|------------|------------|--------------|
| A | 5794 | 0.000 | 0.000 | 0.039 | 0.046 |
| B | 8004 | 0.290 | 0.359 | 0.374 | 0.460 |
| C | 12953 | 0.385 | 0.486 | 0.419 | 0.528 |
| D | 36625 | 0.650 | 0.772 | 0.660 | 0.789 |

## Tier 2.3 — Domain shift (real − synthetic), fine granularity

Mean paired Δ (fine): -0.222 · coarse: -0.228

| Band | mean Δ (real − synth), fine |
|------|------------------------------|
| A | 0.000 |
| B | -0.351 |
| C | -0.475 |
| D | -0.232 |

> Watchdog (strategy §3.1): a large/systematic real−synth gap is the signal to revise the `mixed` default.

## Tier 2.3b — Within look-alike group confusion

Overall within-group fine-confusion rate: 0.207

| Look-alike group | matched | confused | rate |
|------------------|---------|----------|------|
| elephant | 2799 | 1451 | 0.518 |
| equine_unstriped | 2207 | 661 | 0.300 |
| tragelaphus | 1910 | 308 | 0.161 |
| canis | 1878 | 758 | 0.404 |
| sciurus | 1792 | 241 | 0.134 |
| zebra | 1665 | 160 | 0.096 |
| ovis | 1644 | 76 | 0.046 |
| gazelle | 1524 | 663 | 0.435 |
| cervus | 1400 | 94 | 0.067 |
| marmota | 1158 | 58 | 0.050 |
| odocoileus | 1004 | 128 | 0.127 |
| panthera_rosette | 984 | 244 | 0.248 |
| ursus | 961 | 201 | 0.209 |
| lepus | 958 | 106 | 0.111 |
| bison | 868 | 155 | 0.179 |
| connochaetes | 842 | 108 | 0.128 |
| sylvilagus | 817 | 124 | 0.152 |
| bos | 805 | 117 | 0.145 |
| macaca | 715 | 63 | 0.088 |
| macropus | 620 | 2 | 0.003 |
| sus | 591 | 68 | 0.115 |
| lynx_caracal_cluster | 562 | 103 | 0.183 |
| nasua | 557 | 65 | 0.117 |
| felis | 524 | 100 | 0.191 |
| capra | 374 | 11 | 0.029 |
| hippotragus | 361 | 13 | 0.036 |
| tapirus | 360 | 137 | 0.381 |
| kobus | 301 | 0 | 0.000 |
| hyaena | 281 | 86 | 0.306 |
| leopardus | 0 | 0 | — |

---
*Per-class (225-row) table, per-band COCO-12 vectors and the confusion pairs are emitted as CSV/JSON artifacts alongside this file.*
