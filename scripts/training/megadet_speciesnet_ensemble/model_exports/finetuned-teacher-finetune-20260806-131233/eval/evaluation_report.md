# Model Evaluation Report

**Checkpoint:** `scripts/training/teacher_finetune/model_exports/teacher-finetune-20260806-131233/best.pt`  
**max_det:** 100 · **synthetic:** True


## Tier 1 — Headline

Default cell `G=fine · D=mixed · B=all` (cross-model ranking number), with the `D=real` breakout (public-comparison anchor) alongside.

| Metric | mixed (headline) | real (breakout) |
|--------|------------------|-----------------|
| map | 0.549 | 0.536 |
| map_50 | 0.549 | 0.536 |
| map_75 | 0.549 | 0.536 |
| map_small | — | — |
| map_medium | 0.412 | 0.416 |
| map_large | 0.586 | 0.575 |
| mar_1 | 0.505 | 0.475 |
| mar_10 | 0.606 | 0.590 |
| mar_100 | 0.607 | 0.592 |
| mar_small | — | — |
| mar_medium | 0.473 | 0.477 |
| mar_large | 0.641 | 0.628 |

**Public-comparison analog** — class-agnostic `mAP_detect`: mixed 0.969 (mAP50 0.969), real 0.964 (mAP50 0.964).


**Statistical hygiene** — headline mixed mAP 0.549 → 0.571 excluding the 9 test-limited (<30 real img) classes. Count-weighted (micro) mixed mAP: 0.603.

## Tier 2.1 — Granularity gap decomposition (mixed, all classes)

| Level | mAP | mAP50 |
|-------|-----|-------|
| detect (localisation only) | 0.969 | 0.969 |
| coarse (look-alikes merged) | 0.572 | 0.572 |
| fine (full 225-way) | 0.549 | 0.549 |

Δ_coarse (cross-group cost) = 0.397 · Δ_fine (look-alike cost) = 0.023 (mAP).

## Tier 2.2 — Band × granularity grid


**Domain: mixed**

| Band | n_img | mAP_fine | mAP50_fine | mAP_coarse | mAP50_coarse |
|------|-------|----------|------------|------------|--------------|
| A | 8294 | 0.013 | 0.013 | 0.130 | 0.130 |
| B | 9304 | 0.630 | 0.630 | 0.695 | 0.695 |
| C | 14253 | 0.634 | 0.634 | 0.667 | 0.667 |
| D | 42775 | 0.823 | 0.823 | 0.837 | 0.837 |

**Domain: real**

| Band | n_img | mAP_fine | mAP50_fine | mAP_coarse | mAP50_coarse |
|------|-------|----------|------------|------------|--------------|
| A | 5794 | 0.008 | 0.008 | 0.092 | 0.092 |
| B | 8004 | 0.574 | 0.574 | 0.649 | 0.649 |
| C | 12953 | 0.605 | 0.605 | 0.632 | 0.632 |
| D | 36625 | 0.801 | 0.801 | 0.816 | 0.816 |

## Tier 2.3 — Domain shift (real − synthetic), fine granularity

Mean paired Δ (fine): -0.116 · coarse: -0.121

| Band | mean Δ (real − synth), fine |
|------|------------------------------|
| A | -0.015 |
| B | -0.308 |
| C | -0.195 |
| D | -0.100 |

> Watchdog (strategy §3.1): a large/systematic real−synth gap is the signal to revise the `mixed` default.

## Tier 2.3b — Within look-alike group confusion

Overall within-group fine-confusion rate: 0.146

| Look-alike group | matched | confused | rate |
|------------------|---------|----------|------|
| elephant | 2764 | 1051 | 0.380 |
| equine_unstriped | 2544 | 363 | 0.143 |
| tragelaphus | 2364 | 230 | 0.097 |
| canis | 2360 | 696 | 0.295 |
| sciurus | 1935 | 162 | 0.084 |
| gazelle | 1686 | 342 | 0.203 |
| ovis | 1682 | 76 | 0.045 |
| zebra | 1631 | 139 | 0.085 |
| cervus | 1527 | 153 | 0.100 |
| marmota | 1231 | 28 | 0.023 |
| odocoileus | 1083 | 55 | 0.051 |
| panthera_rosette | 1045 | 87 | 0.083 |
| ursus | 1036 | 198 | 0.191 |
| lepus | 1033 | 76 | 0.074 |
| sylvilagus | 984 | 104 | 0.106 |
| bison | 966 | 158 | 0.164 |
| connochaetes | 804 | 35 | 0.044 |
| bos | 768 | 105 | 0.137 |
| macaca | 767 | 148 | 0.193 |
| lynx_caracal_cluster | 689 | 96 | 0.139 |
| macropus | 661 | 1 | 0.002 |
| sus | 624 | 72 | 0.115 |
| felis | 618 | 136 | 0.220 |
| nasua | 605 | 28 | 0.046 |
| tapirus | 515 | 124 | 0.241 |
| capra | 501 | 12 | 0.024 |
| hyaena | 419 | 186 | 0.444 |
| hippotragus | 359 | 9 | 0.025 |
| kobus | 163 | 4 | 0.025 |
| leopardus | 3 | 0 | 0.000 |

---
*Per-class (225-row) table, per-band COCO-12 vectors and the confusion pairs are emitted as CSV/JSON artifacts alongside this file.*
