# Model Evaluation Report

**Checkpoint:** `scripts/training/teacher_finetune/model_exports/teacher-finetune-ff0.75-bs64-20260909-231034/best.pt`  
**max_det:** 100 · **synthetic:** True


## Tier 1 — Headline

Default cell `G=fine · D=mixed · B=all` (cross-model ranking number), with the `D=real` breakout (public-comparison anchor) alongside.

| Metric | mixed (headline) | real (breakout) |
|--------|------------------|-----------------|
| map | 0.663 | 0.599 |
| map_50 | 0.663 | 0.599 |
| map_75 | 0.663 | 0.599 |
| map_small | — | — |
| map_medium | 0.446 | 0.450 |
| map_large | 0.703 | 0.641 |
| mar_1 | 0.609 | 0.529 |
| mar_10 | 0.718 | 0.654 |
| mar_100 | 0.719 | 0.656 |
| mar_small | — | — |
| mar_medium | 0.515 | 0.520 |
| mar_large | 0.754 | 0.692 |

**Public-comparison analog** — class-agnostic `mAP_detect`: mixed 0.969 (mAP50 0.969), real 0.964 (mAP50 0.964).


**Statistical hygiene** — headline mixed mAP 0.663 → 0.660 excluding the 9 test-limited (<30 real img) classes. Count-weighted (micro) mixed mAP: 0.656.

## Tier 2.1 — Granularity gap decomposition (mixed, all classes)

| Level | mAP | mAP50 |
|-------|-----|-------|
| detect (localisation only) | 0.969 | 0.969 |
| coarse (look-alikes merged) | 0.682 | 0.682 |
| fine (full 225-way) | 0.663 | 0.663 |

Δ_coarse (cross-group cost) = 0.287 · Δ_fine (look-alike cost) = 0.018 (mAP).

## Tier 2.2 — Band × granularity grid


**Domain: mixed**

| Band | n_img | mAP_fine | mAP50_fine | mAP_coarse | mAP50_coarse |
|------|-------|----------|------------|------------|--------------|
| A | 8294 | 0.447 | 0.447 | 0.485 | 0.485 |
| B | 9304 | 0.702 | 0.702 | 0.751 | 0.751 |
| C | 14253 | 0.629 | 0.629 | 0.644 | 0.644 |
| D | 42775 | 0.798 | 0.798 | 0.816 | 0.816 |

**Domain: real**

| Band | n_img | mAP_fine | mAP50_fine | mAP_coarse | mAP50_coarse |
|------|-------|----------|------------|------------|--------------|
| A | 5794 | 0.182 | 0.182 | 0.236 | 0.236 |
| B | 8004 | 0.638 | 0.638 | 0.700 | 0.700 |
| C | 12953 | 0.639 | 0.640 | 0.666 | 0.666 |
| D | 36625 | 0.821 | 0.821 | 0.834 | 0.834 |

## Tier 2.3 — Domain shift (real − synthetic), fine granularity

Mean paired Δ (fine): -0.203 · coarse: -0.207

| Band | mean Δ (real − synth), fine |
|------|------------------------------|
| A | -0.675 |
| B | -0.342 |
| C | -0.033 |
| D | -0.017 |

> Watchdog (strategy §3.1): a large/systematic real−synth gap is the signal to revise the `mixed` default.

## Tier 2.3b — Within look-alike group confusion

Overall within-group fine-confusion rate: 0.129

| Look-alike group | matched | confused | rate |
|------------------|---------|----------|------|
| elephant | 2885 | 1056 | 0.366 |
| equine_unstriped | 2694 | 458 | 0.170 |
| canis | 2510 | 649 | 0.259 |
| tragelaphus | 2374 | 226 | 0.095 |
| sciurus | 1926 | 118 | 0.061 |
| ovis | 1704 | 29 | 0.017 |
| zebra | 1626 | 115 | 0.071 |
| gazelle | 1619 | 264 | 0.163 |
| cervus | 1540 | 105 | 0.068 |
| marmota | 1223 | 28 | 0.023 |
| odocoileus | 1149 | 49 | 0.043 |
| lepus | 1049 | 66 | 0.063 |
| ursus | 1043 | 172 | 0.165 |
| panthera_rosette | 1040 | 98 | 0.094 |
| sylvilagus | 1009 | 87 | 0.086 |
| bison | 984 | 171 | 0.174 |
| connochaetes | 814 | 22 | 0.027 |
| bos | 782 | 106 | 0.136 |
| macaca | 759 | 135 | 0.178 |
| macropus | 715 | 19 | 0.027 |
| lynx_caracal_cluster | 702 | 79 | 0.113 |
| felis | 641 | 110 | 0.172 |
| sus | 639 | 63 | 0.099 |
| nasua | 611 | 35 | 0.057 |
| tapirus | 543 | 21 | 0.039 |
| capra | 541 | 11 | 0.020 |
| hyaena | 437 | 79 | 0.181 |
| hippotragus | 368 | 16 | 0.043 |
| leopardus | 168 | 19 | 0.113 |
| kobus | 166 | 2 | 0.012 |

---
*Per-class (225-row) table, per-band COCO-12 vectors and the confusion pairs are emitted as CSV/JSON artifacts alongside this file.*
