# Model Evaluation Report

**Checkpoint:** `scripts/training/yolo26n/model_exports/yolo26n-kd-bs16-20260916-101612/best.pt`  
**max_det:** 100 · **synthetic:** True


## Tier 1 — Headline

Default cell `G=fine · D=mixed · B=all` (cross-model ranking number), with the `D=real` breakout (public-comparison anchor) alongside.

| Metric | mixed (headline) | real (breakout) |
|--------|------------------|-----------------|
| map | 0.560 | 0.479 |
| map_50 | 0.624 | 0.550 |
| map_75 | 0.589 | 0.511 |
| map_small | — | — |
| map_medium | 0.343 | 0.346 |
| map_large | 0.605 | 0.522 |
| mar_1 | 0.703 | 0.635 |
| mar_10 | 0.834 | 0.789 |
| mar_100 | 0.838 | 0.793 |
| mar_small | — | — |
| mar_medium | 0.665 | 0.666 |
| mar_large | 0.867 | 0.821 |

**Public-comparison analog** — class-agnostic `mAP_detect`: mixed 0.770 (mAP50 0.891), real 0.745 (mAP50 0.879).


**Statistical hygiene** — headline mixed mAP 0.560 → 0.554 excluding the 9 test-limited (<30 real img) classes. Count-weighted (micro) mixed mAP: 0.561.

## Tier 2.1 — Granularity gap decomposition (mixed, all classes)

| Level | mAP | mAP50 |
|-------|-----|-------|
| detect (localisation only) | 0.770 | 0.891 |
| coarse (look-alikes merged) | 0.569 | 0.639 |
| fine (full 225-way) | 0.560 | 0.624 |

Δ_coarse (cross-group cost) = 0.201 · Δ_fine (look-alike cost) = 0.009 (mAP).

## Tier 2.2 — Band × granularity grid


**Domain: mixed**

| Band | n_img | mAP_fine | mAP50_fine | mAP_coarse | mAP50_coarse |
|------|-------|----------|------------|------------|--------------|
| A | 8294 | 0.493 | 0.508 | 0.492 | 0.511 |
| B | 9304 | 0.659 | 0.735 | 0.668 | 0.751 |
| C | 14253 | 0.584 | 0.658 | 0.563 | 0.642 |
| D | 42775 | 0.713 | 0.803 | 0.714 | 0.810 |

**Domain: real**

| Band | n_img | mAP_fine | mAP50_fine | mAP_coarse | mAP50_coarse |
|------|-------|----------|------------|------------|--------------|
| A | 5794 | 0.239 | 0.266 | 0.236 | 0.265 |
| B | 8004 | 0.565 | 0.661 | 0.584 | 0.687 |
| C | 12953 | 0.572 | 0.678 | 0.570 | 0.680 |
| D | 36625 | 0.706 | 0.812 | 0.710 | 0.821 |

## Tier 2.3 — Domain shift (real − synthetic), fine granularity

Mean paired Δ (fine): -0.420 · coarse: -0.411

| Band | mean Δ (real − synth), fine |
|------|------------------------------|
| A | -0.808 |
| B | -0.490 |
| C | -0.378 |
| D | -0.261 |

> Watchdog (strategy §3.1): a large/systematic real−synth gap is the signal to revise the `mixed` default.

## Tier 2.3b — Within look-alike group confusion

Overall within-group fine-confusion rate: 0.203

| Look-alike group | matched | confused | rate |
|------------------|---------|----------|------|
| elephant | 2994 | 1474 | 0.492 |
| equine_unstriped | 2245 | 751 | 0.335 |
| canis | 1957 | 729 | 0.373 |
| tragelaphus | 1892 | 413 | 0.218 |
| sciurus | 1765 | 214 | 0.121 |
| zebra | 1669 | 170 | 0.102 |
| gazelle | 1666 | 748 | 0.449 |
| ovis | 1663 | 42 | 0.025 |
| cervus | 1424 | 116 | 0.081 |
| marmota | 1160 | 68 | 0.059 |
| ursus | 994 | 199 | 0.200 |
| odocoileus | 965 | 115 | 0.119 |
| lepus | 947 | 95 | 0.100 |
| bison | 917 | 154 | 0.168 |
| panthera_rosette | 883 | 242 | 0.274 |
| sylvilagus | 853 | 116 | 0.136 |
| connochaetes | 838 | 76 | 0.091 |
| bos | 825 | 78 | 0.095 |
| macaca | 693 | 54 | 0.078 |
| macropus | 687 | 38 | 0.055 |
| lynx_caracal_cluster | 607 | 108 | 0.178 |
| sus | 570 | 61 | 0.107 |
| nasua | 548 | 60 | 0.109 |
| felis | 527 | 79 | 0.150 |
| tapirus | 397 | 38 | 0.096 |
| hyaena | 370 | 43 | 0.116 |
| capra | 342 | 5 | 0.015 |
| hippotragus | 332 | 15 | 0.045 |
| kobus | 254 | 0 | 0.000 |
| leopardus | 101 | 6 | 0.059 |

---
*Per-class (225-row) table, per-band COCO-12 vectors and the confusion pairs are emitted as CSV/JSON artifacts alongside this file.*
