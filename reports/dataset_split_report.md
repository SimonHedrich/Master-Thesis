# Dataset Split Report

**Generated:** 2026-06-09 07:50 UTC  
**Seed:** 42  
**Scoring weights:** area=0.30 · edge=0.25 · single=0.20 · conf=0.25

---

## Band Summary

| Band | Pool threshold | Classes | Train | Val | Test | Surplus | Hard excluded |
|------|---------------|---------|-------|-----|------|---------|---------------|
| A | < 150 | 51 | 0 | 0 | 5,808 | 0 | 99 |
| B | 150–249 | 26 | 2,210 | 514 | 8,025 | 0 | 163 |
| C | 250–399 | 26 | 4,420 | 780 | 12,965 | 0 | 352 |
| D | ≥ 400 | 122 | 137,561 | 11,125 | 36,641 | 231,107 | 5,834 |
| **Total** | | **225** | **144,191** | **12,419** | **63,439** | **231,107** | **6,448** |

---

## Source Distribution by Split

| Source | Train | Val | Test |
|--------|-------|-----|------|
| gbif | 18,761 | 1,207 | 8,403 |
| images_cv | 1,229 | 170 | 3,209 |
| inaturalist | 112,540 | 10,393 | 46,578 |
| openimages | 4,173 | 239 | 1,982 |
| wikimedia | 7,488 | 410 | 3,267 |

---

## Val Set Representativeness

Val images are sampled from the 30th–70th percentile Q range (representative quality, not cherry-picked).

| Band | All-active mean Q | Val mean Q | Δ |
|------|-------------------|------------|---|
| A | 0.825 | — | — |
| B | 0.828 | 0.802 | -0.026 |
| C | 0.816 | 0.766 | -0.049 |
| D | 0.856 | 0.905 | +0.049 |

---

## Test-Limited Classes (< 30 real test images)

These classes have fewer than 30 real test images; interpret per-class test metrics with caution.

- **african civet** (Band A): 28 test images
- **aye-aye** (Band A): 29 test images
- **clouded leopard** (Band A): 25 test images
- **domestic pig** (Band A): 15 test images
- **drill** (Band A): 23 test images
- **giant armadillo** (Band A): 11 test images
- **hog badger genus** (Band A): 18 test images
- **human** (Band A): 5 test images
- **malay tapir** (Band A): 21 test images
- **mouflon** (Band A): 6 test images

---

