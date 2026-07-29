# Model Evaluation Report

**Checkpoint:** `scripts/training/yolo26n/model_exports/yolo26n-smoke-20260701-223023/best.pt`  
**max_det:** 100 · **synthetic:** False

> ⚠️ Synthetic test set was not supplied; 'mixed' == 'real', and the domain-shift / synthetic-only sections are omitted.


## Tier 1 — Headline

Default cell `G=fine · D=mixed · B=all` (cross-model ranking number), with the `D=real` breakout (public-comparison anchor) alongside.

| Metric | mixed (headline) | real (breakout) |
|--------|------------------|-----------------|
| map | 0.001 | 0.001 |
| map_50 | 0.004 | 0.004 |
| map_75 | 0.000 | 0.000 |
| map_small | — | — |
| map_medium | 0.000 | 0.000 |
| map_large | 0.001 | 0.001 |
| mar_1 | 0.009 | 0.009 |
| mar_10 | 0.012 | 0.012 |
| mar_100 | 0.012 | 0.012 |
| mar_small | — | — |
| mar_medium | 0.000 | 0.000 |
| mar_large | 0.013 | 0.013 |

**Public-comparison analog** — class-agnostic `mAP_detect`: mixed 0.008 (mAP50 0.033), real 0.008 (mAP50 0.033).


**Statistical hygiene** — headline mixed mAP 0.001 → — excluding the 33 test-limited (<30 real img) classes. Count-weighted (micro) mixed mAP: 0.001.

## Tier 2.1 — Granularity gap decomposition (mixed, all classes)

| Level | mAP | mAP50 |
|-------|-----|-------|
| detect (localisation only) | 0.008 | 0.033 |
| coarse (look-alikes merged) | 0.001 | 0.005 |
| fine (full 225-way) | 0.001 | 0.004 |

Δ_coarse (cross-group cost) = 0.008 · Δ_fine (look-alike cost) = 0.000 (mAP).

## Tier 2.2 — Band × granularity grid


**Domain: mixed**

| Band | n_img | mAP_fine | mAP50_fine | mAP_coarse | mAP50_coarse |
|------|-------|----------|------------|------------|--------------|
| A | 3 | 0.000 | 0.000 | 0.000 | 0.000 |
| B | 3 | 0.000 | 0.000 | 0.000 | 0.000 |
| C | 8 | 0.000 | 0.000 | 0.000 | 0.000 |
| D | 26 | 0.001 | 0.011 | 0.002 | 0.011 |

**Domain: real**

| Band | n_img | mAP_fine | mAP50_fine | mAP_coarse | mAP50_coarse |
|------|-------|----------|------------|------------|--------------|
| A | 3 | 0.000 | 0.000 | 0.000 | 0.000 |
| B | 3 | 0.000 | 0.000 | 0.000 | 0.000 |
| C | 8 | 0.000 | 0.000 | 0.000 | 0.000 |
| D | 26 | 0.001 | 0.011 | 0.002 | 0.011 |

## Tier 2.3b — Within look-alike group confusion

Overall within-group fine-confusion rate: —

| Look-alike group | matched | confused | rate |
|------------------|---------|----------|------|
| bison | 0 | 0 | — |
| bos | 0 | 0 | — |
| canis | 0 | 0 | — |
| capra | 0 | 0 | — |
| sciurus | 0 | 0 | — |
| sus | 0 | 0 | — |
| sylvilagus | 0 | 0 | — |
| cervus | 0 | 0 | — |
| tapirus | 0 | 0 | — |
| connochaetes | 0 | 0 | — |
| tragelaphus | 0 | 0 | — |
| ursus | 0 | 0 | — |
| zebra | 0 | 0 | — |
| elephant | 0 | 0 | — |
| equine_unstriped | 0 | 0 | — |
| felis | 0 | 0 | — |
| gazelle | 0 | 0 | — |
| hippotragus | 0 | 0 | — |
| hyaena | 0 | 0 | — |
| kobus | 0 | 0 | — |
| leopardus | 0 | 0 | — |
| lepus | 0 | 0 | — |
| lynx_caracal_cluster | 0 | 0 | — |
| macaca | 0 | 0 | — |
| macropus | 0 | 0 | — |
| marmota | 0 | 0 | — |
| nasua | 0 | 0 | — |
| odocoileus | 0 | 0 | — |
| ovis | 0 | 0 | — |
| panthera_rosette | 0 | 0 | — |

---
*Per-class (225-row) table, per-band COCO-12 vectors and the confusion pairs are emitted as CSV/JSON artifacts alongside this file.*
