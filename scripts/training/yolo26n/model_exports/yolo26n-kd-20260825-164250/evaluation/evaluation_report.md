# Model Evaluation Report

**Checkpoint:** `/app/scripts/training/yolo26n/model_exports/yolo26n-kd-20260825-164250/best.pt`  
**max_det:** 100 · **synthetic:** True


## Tier 1 — Headline

Default cell `G=fine · D=mixed · B=all` (cross-model ranking number), with the `D=real` breakout (public-comparison anchor) alongside.

| Metric | mixed (headline) | real (breakout) |
|--------|------------------|-----------------|
| map | 0.510 | 0.479 |
| map_50 | 0.570 | 0.547 |
| map_75 | 0.538 | 0.509 |
| map_small | — | — |
| map_medium | 0.362 | 0.365 |
| map_large | 0.549 | 0.517 |
| mar_1 | 0.673 | 0.617 |
| mar_10 | 0.806 | 0.771 |
| mar_100 | 0.810 | 0.775 |
| mar_small | — | — |
| mar_medium | 0.675 | 0.682 |
| mar_large | 0.832 | 0.796 |

**Public-comparison analog** — class-agnostic `mAP_detect`: mixed 0.788 (mAP50 0.902), real 0.763 (mAP50 0.890).


**Statistical hygiene** — headline mixed mAP 0.510 → 0.530 excluding the 9 test-limited (<30 real img) classes. Count-weighted (micro) mixed mAP: 0.564.

## Tier 2.1 — Granularity gap decomposition (mixed, all classes)

| Level | mAP | mAP50 |
|-------|-----|-------|
| detect (localisation only) | 0.788 | 0.902 |
| coarse (look-alikes merged) | 0.527 | 0.592 |
| fine (full 225-way) | 0.510 | 0.570 |

Δ_coarse (cross-group cost) = 0.262 · Δ_fine (look-alike cost) = 0.016 (mAP).

## Tier 2.2 — Band × granularity grid


**Domain: mixed**

| Band | n_img | mAP_fine | mAP50_fine | mAP_coarse | mAP50_coarse |
|------|-------|----------|------------|------------|--------------|
| A | 8294 | 0.191 | 0.204 | 0.234 | 0.250 |
| B | 9304 | 0.658 | 0.736 | 0.664 | 0.749 |
| C | 14253 | 0.695 | 0.772 | 0.684 | 0.766 |
| D | 42775 | 0.774 | 0.858 | 0.778 | 0.867 |

**Domain: real**

| Band | n_img | mAP_fine | mAP50_fine | mAP_coarse | mAP50_coarse |
|------|-------|----------|------------|------------|--------------|
| A | 5794 | 0.143 | 0.159 | 0.160 | 0.181 |
| B | 8004 | 0.585 | 0.681 | 0.593 | 0.695 |
| C | 12953 | 0.604 | 0.713 | 0.602 | 0.713 |
| D | 36625 | 0.734 | 0.836 | 0.738 | 0.844 |

## Tier 2.3 — Domain shift (real − synthetic), fine granularity

Mean paired Δ (fine): -0.238 · coarse: -0.243

| Band | mean Δ (real − synth), fine |
|------|------------------------------|
| A | -0.142 |
| B | -0.386 |
| C | -0.390 |
| D | -0.214 |

> Watchdog (strategy §3.1): a large/systematic real−synth gap is the signal to revise the `mixed` default.

## Tier 2.3b — Within look-alike group confusion

Overall within-group fine-confusion rate: 0.208

| Look-alike group | matched | confused | rate |
|------------------|---------|----------|------|
| elephant | 2947 | 1609 | 0.546 |
| equine_unstriped | 2274 | 688 | 0.303 |
| canis | 1988 | 739 | 0.372 |
| tragelaphus | 1974 | 359 | 0.182 |
| sciurus | 1783 | 200 | 0.112 |
| ovis | 1668 | 75 | 0.045 |
| zebra | 1654 | 182 | 0.110 |
| gazelle | 1579 | 775 | 0.491 |
| cervus | 1458 | 100 | 0.069 |
| marmota | 1182 | 52 | 0.044 |
| panthera_rosette | 1013 | 175 | 0.173 |
| ursus | 1010 | 206 | 0.204 |
| odocoileus | 1003 | 93 | 0.093 |
| lepus | 952 | 91 | 0.096 |
| bison | 921 | 167 | 0.181 |
| sylvilagus | 858 | 92 | 0.107 |
| connochaetes | 841 | 121 | 0.144 |
| bos | 822 | 133 | 0.162 |
| macaca | 743 | 55 | 0.074 |
| macropus | 634 | 3 | 0.005 |
| sus | 622 | 73 | 0.117 |
| lynx_caracal_cluster | 613 | 97 | 0.158 |
| nasua | 565 | 61 | 0.108 |
| felis | 526 | 95 | 0.181 |
| capra | 408 | 14 | 0.034 |
| tapirus | 400 | 119 | 0.297 |
| hyaena | 356 | 162 | 0.455 |
| hippotragus | 347 | 15 | 0.043 |
| kobus | 326 | 0 | 0.000 |
| leopardus | 0 | 0 | — |

---
*Per-class (225-row) table, per-band COCO-12 vectors and the confusion pairs are emitted as CSV/JSON artifacts alongside this file.*
