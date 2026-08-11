# Model Evaluation Report

**Checkpoint:** `/home/debian/Master-Thesis/scripts/synthetic_model_comparison/training/model_exports/yolo26n-gpt-image-2-low-full-seed42-20260806-170240/best.pt`  
**max_det:** 100


## Headline — real-test mAP (fine, 12-way)

| Metric | value |
|--------|-------|
| map | 0.141 |
| map_50 | 0.240 |
| map_75 | 0.150 |
| map_small | — |
| map_medium | 0.037 |
| map_large | 0.153 |
| mar_1 | 0.369 |
| mar_10 | 0.446 |
| mar_100 | 0.464 |
| mar_small | — |
| mar_medium | 0.160 |
| mar_large | 0.499 |

## Per-class AP

| Class | Band | Real test images | Test-limited | AP |
|-------|------|-------------------|--------------|----|
| american black bear | D | 2097 |  | 0.327 |
| aye-aye | A | 29 | yes | 0.049 |
| grevy's zebra | B | 224 |  | 0.189 |
| kinkajou | A | 160 |  | 0.016 |
| lion | D | 2097 |  | 0.160 |
| mountain zebra | D | 467 |  | 0.219 |
| pangolin family | A | 52 |  | 0.037 |
| plains zebra | D | 2079 |  | 0.297 |
| red fox | D | 2150 |  | 0.305 |
| ringtail | A | 186 |  | 0.015 |
| saiga | A | 50 |  | 0.025 |
| water deer | A | 151 |  | 0.057 |

> Classes flagged test-limited have <30 real test images — lean on Axes A/B (qualitative rubric, teacher-recognition proxy) for those (`06_evaluation-methodology.md`).

## Within look-alike group confusion

Overall within-group fine-confusion rate: 0.147

| Look-alike group | matched | confused | rate |
|------------------|---------|----------|------|
| zebra | 2028 | 449 | 0.221 |
| ursus | 1032 | 0 | 0.000 |
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
*The per-class table and confusion pairs are emitted as CSV/JSON artifacts alongside this file.*
