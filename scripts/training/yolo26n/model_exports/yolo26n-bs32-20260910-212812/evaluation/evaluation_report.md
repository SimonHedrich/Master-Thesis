# Model Evaluation Report

**Checkpoint:** `/app/scripts/training/yolo26n/model_exports/yolo26n-bs32-20260910-212812/best.pt`  
**max_det:** 100 · **synthetic:** True


## Tier 1 — Headline

Default cell `G=fine · D=mixed · B=all` (cross-model ranking number), with the `D=real` breakout (public-comparison anchor) alongside.

| Metric | mixed (headline) | real (breakout) |
|--------|------------------|-----------------|
| map | 0.599 | 0.529 |
| map_50 | 0.659 | 0.599 |
| map_75 | 0.627 | 0.561 |
| map_small | — | — |
| map_medium | 0.379 | 0.378 |
| map_large | 0.641 | 0.573 |
| mar_1 | 0.696 | 0.626 |
| mar_10 | 0.836 | 0.788 |
| mar_100 | 0.845 | 0.798 |
| mar_small | — | — |
| mar_medium | 0.671 | 0.673 |
| mar_large | 0.872 | 0.825 |

**Public-comparison analog** — class-agnostic `mAP_detect`: mixed 0.807 (mAP50 0.918), real 0.782 (mAP50 0.907).


**Statistical hygiene** — headline mixed mAP 0.599 → 0.594 excluding the 9 test-limited (<30 real img) classes. Count-weighted (micro) mixed mAP: 0.597.

## Tier 2.1 — Granularity gap decomposition (mixed, all classes)

| Level | mAP | mAP50 |
|-------|-----|-------|
| detect (localisation only) | 0.807 | 0.918 |
| coarse (look-alikes merged) | 0.608 | 0.674 |
| fine (full 225-way) | 0.599 | 0.659 |

Δ_coarse (cross-group cost) = 0.199 · Δ_fine (look-alike cost) = 0.009 (mAP).

## Tier 2.2 — Band × granularity grid


**Domain: mixed**

| Band | n_img | mAP_fine | mAP50_fine | mAP_coarse | mAP50_coarse |
|------|-------|----------|------------|------------|--------------|
| A | 8294 | 0.579 | 0.602 | 0.576 | 0.601 |
| B | 9304 | 0.685 | 0.763 | 0.701 | 0.787 |
| C | 14253 | 0.614 | 0.687 | 0.597 | 0.675 |
| D | 42775 | 0.726 | 0.806 | 0.734 | 0.819 |

**Domain: real**

| Band | n_img | mAP_fine | mAP50_fine | mAP_coarse | mAP50_coarse |
|------|-------|----------|------------|------------|--------------|
| A | 5794 | 0.300 | 0.332 | 0.285 | 0.319 |
| B | 8004 | 0.588 | 0.687 | 0.617 | 0.724 |
| C | 12953 | 0.591 | 0.696 | 0.598 | 0.707 |
| D | 36625 | 0.742 | 0.838 | 0.746 | 0.847 |

## Tier 2.3 — Domain shift (real − synthetic), fine granularity

Mean paired Δ (fine): -0.362 · coarse: -0.358

| Band | mean Δ (real − synth), fine |
|------|------------------------------|
| A | -0.729 |
| B | -0.462 |
| C | -0.318 |
| D | -0.204 |

> Watchdog (strategy §3.1): a large/systematic real−synth gap is the signal to revise the `mixed` default.

## Tier 2.3b — Within look-alike group confusion

Overall within-group fine-confusion rate: 0.193

| Look-alike group | matched | confused | rate |
|------------------|---------|----------|------|
| elephant | 3057 | 1554 | 0.508 |
| equine_unstriped | 2250 | 611 | 0.272 |
| tragelaphus | 2050 | 444 | 0.217 |
| canis | 1936 | 666 | 0.344 |
| sciurus | 1775 | 213 | 0.120 |
| zebra | 1687 | 161 | 0.095 |
| ovis | 1678 | 53 | 0.032 |
| gazelle | 1627 | 645 | 0.396 |
| cervus | 1458 | 113 | 0.078 |
| marmota | 1176 | 67 | 0.057 |
| odocoileus | 997 | 106 | 0.106 |
| ursus | 988 | 182 | 0.184 |
| lepus | 968 | 94 | 0.097 |
| bison | 947 | 181 | 0.191 |
| panthera_rosette | 886 | 216 | 0.244 |
| sylvilagus | 874 | 112 | 0.128 |
| connochaetes | 862 | 71 | 0.082 |
| bos | 832 | 83 | 0.100 |
| macaca | 704 | 76 | 0.108 |
| macropus | 687 | 52 | 0.076 |
| lynx_caracal_cluster | 613 | 98 | 0.160 |
| sus | 578 | 65 | 0.112 |
| nasua | 562 | 59 | 0.105 |
| felis | 553 | 99 | 0.179 |
| tapirus | 416 | 58 | 0.139 |
| capra | 376 | 4 | 0.011 |
| hyaena | 370 | 26 | 0.070 |
| hippotragus | 345 | 11 | 0.032 |
| kobus | 313 | 2 | 0.006 |
| leopardus | 107 | 4 | 0.037 |

---
*Per-class (225-row) table, per-band COCO-12 vectors and the confusion pairs are emitted as CSV/JSON artifacts alongside this file.*
