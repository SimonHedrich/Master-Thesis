# Model Evaluation Report

**Checkpoint:** `/home/debian/Master-Thesis/scripts/synthetic_model_comparison/training/model_exports/yolo26n-realvisxl-lightning-maxlen-seed43-20260803-021129/best.pt`  
**max_det:** 100


## Headline — real-test mAP (fine, 12-way)

| Metric | value |
|--------|-------|
| map | 0.024 |
| map_50 | 0.052 |
| map_75 | 0.020 |
| map_small | — |
| map_medium | 0.001 |
| map_large | 0.026 |
| mar_1 | 0.160 |
| mar_10 | 0.233 |
| mar_100 | 0.245 |
| mar_small | — |
| mar_medium | 0.018 |
| mar_large | 0.272 |

## Per-class AP

| Class | Band | Real test images | Test-limited | AP |
|-------|------|-------------------|--------------|----|
| american black bear | D | 2097 |  | 0.071 |
| aye-aye | A | 29 | yes | 0.001 |
| grevy's zebra | B | 224 |  | 0.006 |
| kinkajou | A | 160 |  | 0.002 |
| lion | D | 2097 |  | 0.027 |
| mountain zebra | D | 467 |  | 0.053 |
| pangolin family | A | 52 |  | 0.006 |
| plains zebra | D | 2079 |  | 0.081 |
| red fox | D | 2150 |  | 0.030 |
| ringtail | A | 186 |  | 0.001 |
| saiga | A | 50 |  | 0.002 |
| water deer | A | 151 |  | 0.005 |

> Classes flagged test-limited have <30 real test images — lean on Axes A/B (qualitative rubric, teacher-recognition proxy) for those (`06_evaluation-methodology.md`).

## Within look-alike group confusion

Overall within-group fine-confusion rate: 0.409

| Look-alike group | matched | confused | rate |
|------------------|---------|----------|------|
| zebra | 1617 | 930 | 0.575 |
| ursus | 657 | 0 | 0.000 |
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
