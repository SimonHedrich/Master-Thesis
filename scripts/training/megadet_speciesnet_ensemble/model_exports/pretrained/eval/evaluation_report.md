# Model Evaluation Report

**Checkpoint:** `megadet_speciesnet_ensemble`  
**max_det:** 100 · **synthetic:** True


## Tier 1 — Headline

Default cell `G=fine · D=mixed · B=all` (cross-model ranking number), with the `D=real` breakout (public-comparison anchor) alongside.

| Metric | mixed (headline) | real (breakout) |
|--------|------------------|-----------------|
| map | 0.487 | 0.445 |
| map_50 | 0.487 | 0.445 |
| map_75 | 0.487 | 0.445 |
| map_small | — | — |
| map_medium | 0.310 | 0.313 |
| map_large | 0.523 | 0.482 |
| mar_1 | 0.473 | 0.429 |
| mar_10 | 0.549 | 0.516 |
| mar_100 | 0.550 | 0.517 |
| mar_small | — | — |
| mar_medium | 0.369 | 0.373 |
| mar_large | 0.585 | 0.553 |

**Public-comparison analog** — class-agnostic `mAP_detect`: mixed 0.952 (mAP50 0.952), real 0.947 (mAP50 0.947).


**Statistical hygiene** — headline mixed mAP 0.487 → 0.487 excluding the 9 test-limited (<30 real img) classes. Count-weighted (micro) mixed mAP: 0.509.

## Tier 2.1 — Granularity gap decomposition (mixed, all classes)

| Level | mAP | mAP50 |
|-------|-----|-------|
| detect (localisation only) | 0.952 | 0.952 |
| coarse (look-alikes merged) | 0.514 | 0.514 |
| fine (full 225-way) | 0.487 | 0.487 |

Δ_coarse (cross-group cost) = 0.438 · Δ_fine (look-alike cost) = 0.027 (mAP).

## Tier 2.2 — Band × granularity grid


**Domain: mixed**

| Band | n_img | mAP_fine | mAP50_fine | mAP_coarse | mAP50_coarse |
|------|-------|----------|------------|------------|--------------|
| A | 8294 | 0.397 | 0.397 | 0.455 | 0.455 |
| B | 9304 | 0.454 | 0.454 | 0.566 | 0.566 |
| C | 14253 | 0.473 | 0.473 | 0.525 | 0.525 |
| D | 42775 | 0.603 | 0.603 | 0.619 | 0.619 |

**Domain: real**

| Band | n_img | mAP_fine | mAP50_fine | mAP_coarse | mAP50_coarse |
|------|-------|----------|------------|------------|--------------|
| A | 5794 | 0.324 | 0.324 | 0.373 | 0.373 |
| B | 8004 | 0.408 | 0.408 | 0.525 | 0.525 |
| C | 12953 | 0.515 | 0.515 | 0.556 | 0.556 |
| D | 36625 | 0.551 | 0.551 | 0.569 | 0.569 |

## Tier 2.3 — Domain shift (real − synthetic), fine granularity

Mean paired Δ (fine): -0.223 · coarse: -0.223

| Band | mean Δ (real − synth), fine |
|------|------------------------------|
| A | -0.233 |
| B | -0.255 |
| C | -0.266 |
| D | -0.205 |

> Watchdog (strategy §3.1): a large/systematic real−synth gap is the signal to revise the `mixed` default.

## Tier 2.3b — Within look-alike group confusion

Overall within-group fine-confusion rate: 0.220

| Look-alike group | matched | confused | rate |
|------------------|---------|----------|------|
| elephant | 3022 | 1292 | 0.428 |
| canis | 2664 | 1114 | 0.418 |
| equine_unstriped | 2539 | 462 | 0.182 |
| tragelaphus | 2298 | 243 | 0.106 |
| gazelle | 2099 | 701 | 0.334 |
| sciurus | 1861 | 271 | 0.146 |
| zebra | 1658 | 153 | 0.092 |
| cervus | 1483 | 550 | 0.371 |
| odocoileus | 1442 | 125 | 0.087 |
| ovis | 1367 | 87 | 0.064 |
| ursus | 991 | 157 | 0.158 |
| lepus | 961 | 156 | 0.162 |
| bos | 949 | 208 | 0.219 |
| panthera_rosette | 938 | 95 | 0.101 |
| sylvilagus | 924 | 213 | 0.231 |
| marmota | 913 | 192 | 0.210 |
| bison | 832 | 218 | 0.262 |
| connochaetes | 761 | 50 | 0.066 |
| macaca | 757 | 159 | 0.210 |
| felis | 730 | 87 | 0.119 |
| capra | 693 | 17 | 0.025 |
| lynx_caracal_cluster | 654 | 181 | 0.277 |
| sus | 604 | 75 | 0.124 |
| tapirus | 551 | 193 | 0.350 |
| macropus | 549 | 6 | 0.011 |
| nasua | 534 | 215 | 0.403 |
| hyaena | 468 | 118 | 0.252 |
| leopardus | 193 | 55 | 0.285 |
| kobus | 152 | 7 | 0.046 |
| hippotragus | 73 | 19 | 0.260 |

---
*Per-class (225-row) table, per-band COCO-12 vectors and the confusion pairs are emitted as CSV/JSON artifacts alongside this file.*
