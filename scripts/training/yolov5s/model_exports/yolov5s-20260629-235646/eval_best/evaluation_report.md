# Model Evaluation Report

**Checkpoint:** `scripts/training/yolov5s/model_exports/yolov5s-20260629-235646/best.pt`  
**max_det:** 100 · **synthetic:** True


## Tier 1 — Headline

Default cell `G=fine · D=mixed · B=all` (cross-model ranking number), with the `D=real` breakout (public-comparison anchor) alongside.

| Metric | mixed (headline) | real (breakout) |
|--------|------------------|-----------------|
| map | 0.371 | 0.347 |
| map_50 | 0.422 | 0.404 |
| map_75 | 0.396 | 0.374 |
| map_small | — | — |
| map_medium | 0.273 | 0.276 |
| map_large | 0.398 | 0.372 |
| mar_1 | 0.380 | 0.348 |
| mar_10 | 0.466 | 0.446 |
| mar_100 | 0.470 | 0.450 |
| mar_small | — | — |
| mar_medium | 0.380 | 0.383 |
| mar_large | 0.493 | 0.472 |

**Public-comparison analog** — class-agnostic `mAP_detect`: mixed 0.787 (mAP50 0.924), real 0.764 (mAP50 0.915).


**Statistical hygiene** — headline mixed mAP 0.371 → 0.386 excluding the 9 test-limited (<30 real img) classes. Count-weighted (micro) mixed mAP: 0.394.

## Tier 2.1 — Granularity gap decomposition (mixed, all classes)

| Level | mAP | mAP50 |
|-------|-----|-------|
| detect (localisation only) | 0.787 | 0.924 |
| coarse (look-alikes merged) | 0.403 | 0.462 |
| fine (full 225-way) | 0.371 | 0.422 |

Δ_coarse (cross-group cost) = 0.384 · Δ_fine (look-alike cost) = 0.032 (mAP).

## Tier 2.2 — Band × granularity grid


**Domain: mixed**

| Band | n_img | mAP_fine | mAP50_fine | mAP_coarse | mAP50_coarse |
|------|-------|----------|------------|------------|--------------|
| A | 8294 | 0.000 | 0.000 | 0.088 | 0.093 |
| B | 9304 | 0.090 | 0.102 | 0.227 | 0.263 |
| C | 14253 | 0.459 | 0.513 | 0.481 | 0.546 |
| D | 42775 | 0.672 | 0.762 | 0.691 | 0.788 |

**Domain: real**

| Band | n_img | mAP_fine | mAP50_fine | mAP_coarse | mAP50_coarse |
|------|-------|----------|------------|------------|--------------|
| A | 5794 | 0.000 | 0.000 | 0.048 | 0.055 |
| B | 8004 | 0.065 | 0.078 | 0.185 | 0.224 |
| C | 12953 | 0.285 | 0.354 | 0.335 | 0.415 |
| D | 36625 | 0.644 | 0.749 | 0.658 | 0.771 |

## Tier 2.3 — Domain shift (real − synthetic), fine granularity

Mean paired Δ (fine): -0.189 · coarse: -0.207

| Band | mean Δ (real − synth), fine |
|------|------------------------------|
| A | 0.000 |
| B | -0.113 |
| C | -0.396 |
| D | -0.238 |

> Watchdog (strategy §3.1): a large/systematic real−synth gap is the signal to revise the `mixed` default.

## Tier 2.3b — Within look-alike group confusion

Overall within-group fine-confusion rate: 0.276

| Look-alike group | matched | confused | rate |
|------------------|---------|----------|------|
| elephant | 2851 | 1835 | 0.644 |
| equine_unstriped | 2141 | 845 | 0.395 |
| sciurus | 1789 | 304 | 0.170 |
| canis | 1754 | 959 | 0.547 |
| zebra | 1668 | 351 | 0.210 |
| ovis | 1640 | 77 | 0.047 |
| tragelaphus | 1538 | 372 | 0.242 |
| gazelle | 1511 | 972 | 0.643 |
| cervus | 1355 | 140 | 0.103 |
| marmota | 1166 | 92 | 0.079 |
| ursus | 982 | 206 | 0.210 |
| panthera_rosette | 961 | 271 | 0.282 |
| lepus | 953 | 114 | 0.120 |
| bison | 888 | 225 | 0.253 |
| odocoileus | 880 | 129 | 0.147 |
| sylvilagus | 825 | 114 | 0.138 |
| bos | 808 | 115 | 0.142 |
| connochaetes | 807 | 137 | 0.170 |
| macaca | 701 | 94 | 0.134 |
| macropus | 607 | 1 | 0.002 |
| sus | 562 | 68 | 0.121 |
| nasua | 552 | 77 | 0.139 |
| felis | 518 | 96 | 0.185 |
| lynx_caracal_cluster | 516 | 191 | 0.370 |
| hippotragus | 328 | 36 | 0.110 |
| hyaena | 320 | 130 | 0.406 |
| kobus | 247 | 1 | 0.004 |
| tapirus | 222 | 108 | 0.486 |
| capra | 168 | 12 | 0.071 |
| leopardus | 0 | 0 | — |

---
*Per-class (225-row) table, per-band COCO-12 vectors and the confusion pairs are emitted as CSV/JSON artifacts alongside this file.*
