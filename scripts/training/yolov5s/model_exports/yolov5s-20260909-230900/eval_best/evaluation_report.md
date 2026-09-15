# Model Evaluation Report

**Checkpoint:** `scripts/training/yolov5s/model_exports/yolov5s-20260909-230900/best.pt`  
**max_det:** 100 · **synthetic:** True


## Tier 1 — Headline

Default cell `G=fine · D=mixed · B=all` (cross-model ranking number), with the `D=real` breakout (public-comparison anchor) alongside.

| Metric | mixed (headline) | real (breakout) |
|--------|------------------|-----------------|
| map | 0.496 | 0.407 |
| map_50 | 0.568 | 0.485 |
| map_75 | 0.532 | 0.443 |
| map_small | — | — |
| map_medium | 0.310 | 0.313 |
| map_large | 0.533 | 0.440 |
| mar_1 | 0.501 | 0.399 |
| mar_10 | 0.599 | 0.511 |
| mar_100 | 0.603 | 0.516 |
| mar_small | — | — |
| mar_medium | 0.428 | 0.432 |
| mar_large | 0.634 | 0.542 |

**Public-comparison analog** — class-agnostic `mAP_detect`: mixed 0.775 (mAP50 0.922), real 0.750 (mAP50 0.911).


**Statistical hygiene** — headline mixed mAP 0.496 → 0.488 excluding the 9 test-limited (<30 real img) classes. Count-weighted (micro) mixed mAP: 0.475.

## Tier 2.1 — Granularity gap decomposition (mixed, all classes)

| Level | mAP | mAP50 |
|-------|-----|-------|
| detect (localisation only) | 0.775 | 0.922 |
| coarse (look-alikes merged) | 0.510 | 0.590 |
| fine (full 225-way) | 0.496 | 0.568 |

Δ_coarse (cross-group cost) = 0.265 · Δ_fine (look-alike cost) = 0.014 (mAP).

## Tier 2.2 — Band × granularity grid


**Domain: mixed**

| Band | n_img | mAP_fine | mAP50_fine | mAP_coarse | mAP50_coarse |
|------|-------|----------|------------|------------|--------------|
| A | 8294 | 0.384 | 0.402 | 0.405 | 0.428 |
| B | 9304 | 0.461 | 0.527 | 0.520 | 0.602 |
| C | 14253 | 0.477 | 0.552 | 0.498 | 0.582 |
| D | 42775 | 0.646 | 0.751 | 0.657 | 0.769 |

**Domain: real**

| Band | n_img | mAP_fine | mAP50_fine | mAP_coarse | mAP50_coarse |
|------|-------|----------|------------|------------|--------------|
| A | 5794 | 0.058 | 0.067 | 0.086 | 0.101 |
| B | 8004 | 0.341 | 0.420 | 0.418 | 0.515 |
| C | 12953 | 0.386 | 0.483 | 0.423 | 0.530 |
| D | 36625 | 0.648 | 0.770 | 0.659 | 0.788 |

## Tier 2.3 — Domain shift (real − synthetic), fine granularity

Mean paired Δ (fine): -0.357 · coarse: -0.359

| Band | mean Δ (real − synth), fine |
|------|------------------------------|
| A | -0.815 |
| B | -0.614 |
| C | -0.268 |
| D | -0.135 |

> Watchdog (strategy §3.1): a large/systematic real−synth gap is the signal to revise the `mixed` default.

## Tier 2.3b — Within look-alike group confusion

Overall within-group fine-confusion rate: 0.190

| Look-alike group | matched | confused | rate |
|------------------|---------|----------|------|
| elephant | 2863 | 1453 | 0.508 |
| equine_unstriped | 2248 | 574 | 0.255 |
| tragelaphus | 1953 | 283 | 0.145 |
| canis | 1846 | 723 | 0.392 |
| sciurus | 1792 | 259 | 0.145 |
| zebra | 1676 | 145 | 0.087 |
| ovis | 1650 | 46 | 0.028 |
| gazelle | 1479 | 545 | 0.368 |
| cervus | 1384 | 101 | 0.073 |
| marmota | 1155 | 54 | 0.047 |
| odocoileus | 988 | 117 | 0.118 |
| lepus | 969 | 106 | 0.109 |
| ursus | 939 | 166 | 0.177 |
| bison | 911 | 184 | 0.202 |
| panthera_rosette | 885 | 271 | 0.306 |
| bos | 844 | 94 | 0.111 |
| connochaetes | 842 | 61 | 0.072 |
| sylvilagus | 830 | 116 | 0.140 |
| macaca | 690 | 84 | 0.122 |
| macropus | 669 | 43 | 0.064 |
| lynx_caracal_cluster | 591 | 96 | 0.162 |
| nasua | 554 | 47 | 0.085 |
| sus | 550 | 38 | 0.069 |
| felis | 538 | 89 | 0.165 |
| hyaena | 374 | 44 | 0.118 |
| capra | 363 | 9 | 0.025 |
| tapirus | 360 | 53 | 0.147 |
| hippotragus | 345 | 15 | 0.043 |
| kobus | 289 | 0 | 0.000 |
| leopardus | 94 | 2 | 0.021 |

---
*Per-class (225-row) table, per-band COCO-12 vectors and the confusion pairs are emitted as CSV/JSON artifacts alongside this file.*
