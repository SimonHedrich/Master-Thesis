# Model Evaluation Report

**Checkpoint:** `/home/debian/Master-Thesis/scripts/training/yolov5s/model_exports/best.pt`  
**max_det:** 100 · **synthetic:** True


## Tier 1 — Headline

Default cell `G=fine · D=mixed · B=all` (cross-model ranking number), with the `D=real` breakout (public-comparison anchor) alongside.

| Metric | mixed (headline) | real (breakout) |
|--------|------------------|-----------------|
| map | 0.254 | 0.228 |
| map_50 | 0.295 | 0.272 |
| map_75 | 0.275 | 0.249 |
| map_small | — | — |
| map_medium | 0.130 | 0.131 |
| map_large | 0.288 | 0.261 |
| mar_1 | 0.329 | 0.299 |
| mar_10 | 0.357 | 0.332 |
| mar_100 | 0.357 | 0.332 |
| mar_small | — | — |
| mar_medium | 0.217 | 0.219 |
| mar_large | 0.394 | 0.368 |

**Public-comparison analog** — class-agnostic `mAP_detect`: mixed 0.651 (mAP50 0.783), real 0.611 (mAP50 0.754).


**Statistical hygiene** — headline mixed mAP 0.254 → 0.264 excluding the 9 test-limited (<30 real img) classes. Count-weighted (micro) mixed mAP: 0.261.

## Tier 2.1 — Granularity gap decomposition (mixed, all classes)

| Level | mAP | mAP50 |
|-------|-----|-------|
| detect (localisation only) | 0.651 | 0.783 |
| coarse (look-alikes merged) | 0.279 | 0.326 |
| fine (full 225-way) | 0.254 | 0.295 |

Δ_coarse (cross-group cost) = 0.372 · Δ_fine (look-alike cost) = 0.025 (mAP).

## Tier 2.2 — Band × granularity grid


**Domain: mixed**

| Band | n_img | mAP_fine | mAP50_fine | mAP_coarse | mAP50_coarse |
|------|-------|----------|------------|------------|--------------|
| A | 8299 | 0.000 | 0.000 | 0.060 | 0.064 |
| B | 9306 | 0.013 | 0.015 | 0.098 | 0.111 |
| C | 14257 | 0.252 | 0.287 | 0.266 | 0.309 |
| D | 42784 | 0.502 | 0.582 | 0.524 | 0.611 |

**Domain: real**

| Band | n_img | mAP_fine | mAP50_fine | mAP_coarse | mAP50_coarse |
|------|-------|----------|------------|------------|--------------|
| A | 5799 | 0.000 | 0.000 | 0.027 | 0.031 |
| B | 8006 | 0.006 | 0.008 | 0.065 | 0.077 |
| C | 12957 | 0.115 | 0.150 | 0.149 | 0.193 |
| D | 36634 | 0.472 | 0.564 | 0.488 | 0.586 |

## Tier 2.3 — Domain shift (real − synthetic), fine granularity

Mean paired Δ (fine): -0.212 · coarse: -0.247

| Band | mean Δ (real − synth), fine |
|------|------------------------------|
| A | 0.000 |
| B | -0.022 |
| C | -0.274 |
| D | -0.327 |

> Watchdog (strategy §3.1): a large/systematic real−synth gap is the signal to revise the `mixed` default.

## Tier 2.3b — Within look-alike group confusion

Overall within-group fine-confusion rate: 0.275

| Look-alike group | matched | confused | rate |
|------------------|---------|----------|------|
| equus | 2321 | 735 | 0.317 |
| elephant | 1789 | 1109 | 0.620 |
| panthera | 1632 | 395 | 0.242 |
| sciurus | 1580 | 375 | 0.237 |
| canis | 1477 | 823 | 0.557 |
| marmota | 1049 | 122 | 0.116 |
| tragelaphus | 987 | 296 | 0.300 |
| ovis | 949 | 118 | 0.124 |
| cervus | 914 | 202 | 0.221 |
| lepus | 841 | 130 | 0.155 |
| ursus | 838 | 197 | 0.235 |
| sylvilagus | 726 | 163 | 0.225 |
| odocoileus | 626 | 109 | 0.174 |
| bison | 572 | 129 | 0.226 |
| connochaetes | 508 | 89 | 0.175 |
| macaca | 503 | 90 | 0.179 |
| bos | 498 | 62 | 0.124 |
| macropus | 465 | 3 | 0.006 |
| lynx_caracal_cluster | 462 | 176 | 0.381 |
| felis | 446 | 79 | 0.177 |
| nasua | 431 | 61 | 0.142 |
| sus | 384 | 59 | 0.154 |
| hyaena | 210 | 60 | 0.286 |
| hippotragus | 194 | 69 | 0.356 |
| capra | 124 | 18 | 0.145 |
| kobus | 108 | 2 | 0.019 |
| tapirus | 16 | 11 | 0.688 |
| leopardus | 0 | 0 | — |

---
*Per-class (225-row) table, per-band COCO-12 vectors and the confusion pairs are emitted as CSV/JSON artifacts alongside this file.*
