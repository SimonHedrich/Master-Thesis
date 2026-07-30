# Model Evaluation Report

**Checkpoint:** `scripts/training/yolo26n/model_exports/yolo26n-20260715-010031/best.pt`  
**max_det:** 100 · **synthetic:** True


## Tier 1 — Headline

Default cell `G=fine · D=mixed · B=all` (cross-model ranking number), with the `D=real` breakout (public-comparison anchor) alongside.

| Metric | mixed (headline) | real (breakout) |
|--------|------------------|-----------------|
| map | 0.523 | 0.481 |
| map_50 | 0.574 | 0.541 |
| map_75 | 0.549 | 0.511 |
| map_small | — | — |
| map_medium | 0.326 | 0.329 |
| map_large | 0.556 | 0.516 |
| mar_1 | 0.709 | 0.650 |
| mar_10 | 0.756 | 0.707 |
| mar_100 | 0.757 | 0.709 |
| mar_small | — | — |
| mar_medium | 0.650 | 0.657 |
| mar_large | 0.770 | 0.722 |

**Public-comparison analog** — class-agnostic `mAP_detect`: mixed 0.812 (mAP50 0.907), real 0.778 (mAP50 0.889).


**Statistical hygiene** — headline mixed mAP 0.523 → 0.545 excluding the 9 test-limited (<30 real img) classes. Count-weighted (micro) mixed mAP: 0.586.

## Tier 2.1 — Granularity gap decomposition (mixed, all classes)

| Level | mAP | mAP50 |
|-------|-----|-------|
| detect (localisation only) | 0.812 | 0.907 |
| coarse (look-alikes merged) | 0.536 | 0.592 |
| fine (full 225-way) | 0.523 | 0.574 |

Δ_coarse (cross-group cost) = 0.276 · Δ_fine (look-alike cost) = 0.013 (mAP).

## Tier 2.2 — Band × granularity grid


**Domain: mixed**

| Band | n_img | mAP_fine | mAP50_fine | mAP_coarse | mAP50_coarse |
|------|-------|----------|------------|------------|--------------|
| A | 8308 | 0.019 | 0.022 | 0.118 | 0.124 |
| B | 9325 | 0.672 | 0.742 | 0.685 | 0.758 |
| C | 14265 | 0.726 | 0.796 | 0.716 | 0.788 |
| D | 42791 | 0.788 | 0.858 | 0.793 | 0.867 |

**Domain: real**

| Band | n_img | mAP_fine | mAP50_fine | mAP_coarse | mAP50_coarse |
|------|-------|----------|------------|------------|--------------|
| A | 5808 | 0.011 | 0.014 | 0.068 | 0.076 |
| B | 8025 | 0.563 | 0.651 | 0.578 | 0.670 |
| C | 12965 | 0.601 | 0.701 | 0.598 | 0.699 |
| D | 36641 | 0.741 | 0.828 | 0.744 | 0.836 |

## Tier 2.3 — Domain shift (real − synthetic), fine granularity

Mean paired Δ (fine): -0.214 · coarse: -0.219

| Band | mean Δ (real − synth), fine |
|------|------------------------------|
| A | -0.009 |
| B | -0.396 |
| C | -0.397 |
| D | -0.220 |

> Watchdog (strategy §3.1): a large/systematic real−synth gap is the signal to revise the `mixed` default.

## Tier 2.3b — Within look-alike group confusion

Overall within-group fine-confusion rate: 0.186

| Look-alike group | matched | confused | rate |
|------------------|---------|----------|------|
| sciurus | 1729 | 198 | 0.115 |
| elephant | 1643 | 845 | 0.514 |
| canis | 1628 | 579 | 0.356 |
| tragelaphus | 1247 | 216 | 0.173 |
| equine_unstriped | 1142 | 271 | 0.237 |
| marmota | 1118 | 67 | 0.060 |
| panthera_rosette | 949 | 225 | 0.237 |
| lepus | 892 | 95 | 0.107 |
| gazelle | 878 | 227 | 0.259 |
| cervus | 865 | 62 | 0.072 |
| ursus | 858 | 164 | 0.191 |
| sylvilagus | 841 | 107 | 0.127 |
| zebra | 816 | 66 | 0.081 |
| ovis | 778 | 77 | 0.099 |
| odocoileus | 708 | 66 | 0.093 |
| lynx_caracal_cluster | 583 | 90 | 0.154 |
| macaca | 564 | 46 | 0.082 |
| felis | 504 | 97 | 0.192 |
| bison | 502 | 66 | 0.131 |
| nasua | 463 | 39 | 0.084 |
| macropus | 405 | 0 | 0.000 |
| sus | 395 | 62 | 0.157 |
| bos | 376 | 42 | 0.112 |
| connochaetes | 366 | 48 | 0.131 |
| tapirus | 354 | 125 | 0.353 |
| hyaena | 254 | 102 | 0.402 |
| capra | 251 | 13 | 0.052 |
| hippotragus | 231 | 12 | 0.052 |
| kobus | 214 | 1 | 0.005 |
| leopardus | 0 | 0 | — |

---
*Per-class (225-row) table, per-band COCO-12 vectors and the confusion pairs are emitted as CSV/JSON artifacts alongside this file.*
