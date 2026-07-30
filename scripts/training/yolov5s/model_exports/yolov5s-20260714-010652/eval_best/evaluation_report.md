# Model Evaluation Report

**Checkpoint:** `scripts/training/yolov5s/model_exports/yolov5s-20260714-010652/best.pt`  
**max_det:** 100 · **synthetic:** True


## Tier 1 — Headline

Default cell `G=fine · D=mixed · B=all` (cross-model ranking number), with the `D=real` breakout (public-comparison anchor) alongside.

| Metric | mixed (headline) | real (breakout) |
|--------|------------------|-----------------|
| map | 0.360 | 0.336 |
| map_50 | 0.405 | 0.387 |
| map_75 | 0.382 | 0.360 |
| map_small | — | — |
| map_medium | 0.273 | 0.275 |
| map_large | 0.386 | 0.360 |
| mar_1 | 0.376 | 0.345 |
| mar_10 | 0.460 | 0.440 |
| mar_100 | 0.463 | 0.444 |
| mar_small | — | — |
| mar_medium | 0.385 | 0.389 |
| mar_large | 0.484 | 0.463 |

**Public-comparison analog** — class-agnostic `mAP_detect`: mixed 0.788 (mAP50 0.917), real 0.765 (mAP50 0.908).


**Statistical hygiene** — headline mixed mAP 0.360 → 0.375 excluding the 9 test-limited (<30 real img) classes. Count-weighted (micro) mixed mAP: 0.381.

## Tier 2.1 — Granularity gap decomposition (mixed, all classes)

| Level | mAP | mAP50 |
|-------|-----|-------|
| detect (localisation only) | 0.788 | 0.917 |
| coarse (look-alikes merged) | 0.393 | 0.446 |
| fine (full 225-way) | 0.360 | 0.405 |

Δ_coarse (cross-group cost) = 0.395 · Δ_fine (look-alike cost) = 0.033 (mAP).

## Tier 2.2 — Band × granularity grid


**Domain: mixed**

| Band | n_img | mAP_fine | mAP50_fine | mAP_coarse | mAP50_coarse |
|------|-------|----------|------------|------------|--------------|
| A | 8294 | 0.000 | 0.000 | 0.075 | 0.080 |
| B | 9304 | 0.041 | 0.046 | 0.182 | 0.210 |
| C | 14253 | 0.425 | 0.471 | 0.450 | 0.506 |
| D | 42775 | 0.672 | 0.755 | 0.692 | 0.782 |

**Domain: real**

| Band | n_img | mAP_fine | mAP50_fine | mAP_coarse | mAP50_coarse |
|------|-------|----------|------------|------------|--------------|
| A | 5794 | 0.000 | 0.000 | 0.041 | 0.047 |
| B | 8004 | 0.027 | 0.032 | 0.151 | 0.182 |
| C | 12953 | 0.256 | 0.316 | 0.306 | 0.377 |
| D | 36625 | 0.645 | 0.744 | 0.661 | 0.767 |

## Tier 2.3 — Domain shift (real − synthetic), fine granularity

Mean paired Δ (fine): -0.186 · coarse: -0.207

| Band | mean Δ (real − synth), fine |
|------|------------------------------|
| A | 0.000 |
| B | -0.045 |
| C | -0.396 |
| D | -0.248 |

> Watchdog (strategy §3.1): a large/systematic real−synth gap is the signal to revise the `mixed` default.

## Tier 2.3b — Within look-alike group confusion

Overall within-group fine-confusion rate: 0.281

| Look-alike group | matched | confused | rate |
|------------------|---------|----------|------|
| elephant | 2810 | 1795 | 0.639 |
| equine_unstriped | 1930 | 801 | 0.415 |
| sciurus | 1765 | 295 | 0.167 |
| canis | 1729 | 954 | 0.552 |
| zebra | 1647 | 341 | 0.207 |
| ovis | 1632 | 81 | 0.050 |
| tragelaphus | 1456 | 354 | 0.243 |
| gazelle | 1431 | 1053 | 0.736 |
| cervus | 1316 | 129 | 0.098 |
| marmota | 1151 | 86 | 0.075 |
| ursus | 976 | 203 | 0.208 |
| panthera_rosette | 951 | 328 | 0.345 |
| lepus | 949 | 121 | 0.128 |
| odocoileus | 949 | 124 | 0.131 |
| bison | 871 | 208 | 0.239 |
| sylvilagus | 824 | 137 | 0.166 |
| bos | 820 | 138 | 0.168 |
| connochaetes | 784 | 146 | 0.186 |
| macaca | 687 | 114 | 0.166 |
| macropus | 598 | 1 | 0.002 |
| sus | 559 | 70 | 0.125 |
| nasua | 547 | 72 | 0.132 |
| felis | 522 | 98 | 0.188 |
| lynx_caracal_cluster | 505 | 191 | 0.378 |
| hippotragus | 313 | 29 | 0.093 |
| hyaena | 304 | 111 | 0.365 |
| kobus | 210 | 1 | 0.005 |
| capra | 195 | 19 | 0.097 |
| tapirus | 15 | 7 | 0.467 |
| leopardus | 0 | 0 | — |

---
*Per-class (225-row) table, per-band COCO-12 vectors and the confusion pairs are emitted as CSV/JSON artifacts alongside this file.*
