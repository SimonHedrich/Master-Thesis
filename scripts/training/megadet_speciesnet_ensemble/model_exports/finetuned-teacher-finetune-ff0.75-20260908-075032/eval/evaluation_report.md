# Model Evaluation Report

**Checkpoint:** `scripts/training/teacher_finetune/model_exports/teacher-finetune-ff0.75-20260908-075032/best.pt`  
**max_det:** 100 · **synthetic:** True


## Tier 1 — Headline

Default cell `G=fine · D=mixed · B=all` (cross-model ranking number), with the `D=real` breakout (public-comparison anchor) alongside.

| Metric | mixed (headline) | real (breakout) |
|--------|------------------|-----------------|
| map | 0.567 | 0.553 |
| map_50 | 0.567 | 0.553 |
| map_75 | 0.567 | 0.553 |
| map_small | — | — |
| map_medium | 0.433 | 0.437 |
| map_large | 0.603 | 0.592 |
| mar_1 | 0.520 | 0.487 |
| mar_10 | 0.625 | 0.608 |
| mar_100 | 0.626 | 0.610 |
| mar_small | — | — |
| mar_medium | 0.495 | 0.500 |
| mar_large | 0.659 | 0.645 |

**Public-comparison analog** — class-agnostic `mAP_detect`: mixed 0.969 (mAP50 0.969), real 0.965 (mAP50 0.965).


**Statistical hygiene** — headline mixed mAP 0.567 → 0.588 excluding the 9 test-limited (<30 real img) classes. Count-weighted (micro) mixed mAP: 0.621.

## Tier 2.1 — Granularity gap decomposition (mixed, all classes)

| Level | mAP | mAP50 |
|-------|-----|-------|
| detect (localisation only) | 0.969 | 0.969 |
| coarse (look-alikes merged) | 0.590 | 0.590 |
| fine (full 225-way) | 0.567 | 0.567 |

Δ_coarse (cross-group cost) = 0.379 · Δ_fine (look-alike cost) = 0.023 (mAP).

## Tier 2.2 — Band × granularity grid


**Domain: mixed**

| Band | n_img | mAP_fine | mAP50_fine | mAP_coarse | mAP50_coarse |
|------|-------|----------|------------|------------|--------------|
| A | 8294 | 0.027 | 0.027 | 0.142 | 0.142 |
| B | 9304 | 0.654 | 0.654 | 0.718 | 0.718 |
| C | 14253 | 0.653 | 0.653 | 0.690 | 0.690 |
| D | 42775 | 0.840 | 0.840 | 0.852 | 0.852 |

**Domain: real**

| Band | n_img | mAP_fine | mAP50_fine | mAP_coarse | mAP50_coarse |
|------|-------|----------|------------|------------|--------------|
| A | 5794 | 0.014 | 0.014 | 0.095 | 0.095 |
| B | 8004 | 0.599 | 0.599 | 0.670 | 0.670 |
| C | 12953 | 0.631 | 0.631 | 0.662 | 0.662 |
| D | 36625 | 0.822 | 0.822 | 0.835 | 0.835 |

## Tier 2.3 — Domain shift (real − synthetic), fine granularity

Mean paired Δ (fine): -0.108 · coarse: -0.113

| Band | mean Δ (real − synth), fine |
|------|------------------------------|
| A | -0.039 |
| B | -0.298 |
| C | -0.212 |
| D | -0.075 |

> Watchdog (strategy §3.1): a large/systematic real−synth gap is the signal to revise the `mixed` default.

## Tier 2.3b — Within look-alike group confusion

Overall within-group fine-confusion rate: 0.144

| Look-alike group | matched | confused | rate |
|------------------|---------|----------|------|
| elephant | 2774 | 1014 | 0.366 |
| equine_unstriped | 2711 | 525 | 0.194 |
| canis | 2465 | 705 | 0.286 |
| tragelaphus | 2306 | 267 | 0.116 |
| sciurus | 1977 | 150 | 0.076 |
| gazelle | 1731 | 315 | 0.182 |
| ovis | 1708 | 81 | 0.047 |
| zebra | 1621 | 126 | 0.078 |
| cervus | 1555 | 116 | 0.075 |
| marmota | 1230 | 27 | 0.022 |
| odocoileus | 1140 | 41 | 0.036 |
| panthera_rosette | 1058 | 105 | 0.099 |
| ursus | 1044 | 182 | 0.174 |
| lepus | 1042 | 66 | 0.063 |
| sylvilagus | 1009 | 94 | 0.093 |
| bison | 979 | 140 | 0.143 |
| bos | 831 | 129 | 0.155 |
| connochaetes | 810 | 40 | 0.049 |
| macaca | 798 | 156 | 0.195 |
| lynx_caracal_cluster | 699 | 94 | 0.134 |
| macropus | 662 | 1 | 0.002 |
| felis | 652 | 140 | 0.215 |
| sus | 635 | 72 | 0.113 |
| nasua | 613 | 38 | 0.062 |
| capra | 563 | 13 | 0.023 |
| tapirus | 544 | 135 | 0.248 |
| hyaena | 398 | 135 | 0.339 |
| hippotragus | 363 | 14 | 0.039 |
| kobus | 163 | 2 | 0.012 |
| leopardus | 1 | 1 | 1.000 |

---
*Per-class (225-row) table, per-band COCO-12 vectors and the confusion pairs are emitted as CSV/JSON artifacts alongside this file.*
