# Low-Confidence Box Contamination Report

**Thresholds:** md_conf∈[0.1, 0.5)  sn_score≥0.3  tolerance=family

**Sources:** gbif, inaturalist, wikimedia, openimages, images_cv

> Images already present in `multi_animal_contamination_review.json` (the ≥ 0.5 pipeline) are excluded from this report.

---

## Summary

| Metric | Count |
|---|---:|
| Total records scanned | 465,130 |
| Excluded (already in ≥ 0.5 review) | 11,354 |
| Expected class not in 225 (skipped) | 464 |
| Images with ≥1 low-conf detection evaluated | 21,026 |
| **Flagged images** (≥1 confident mismatch) | **1,929** |
| Uncertain-only images (low-confidence mismatch) | 807 |
| Consistent-only images | 5,103 |

> **Flagging rate:** 1,929 / 21,026 images = 9.2%  (reference: ≥ 0.5 tier was ~7.5%)

## Breakdown by Source

| Source | Flagged Images |
|---|---:|
| inaturalist | 1,342 |
| gbif | 231 |
| wikimedia | 175 |
| openimages | 118 |
| images_cv | 63 |

## Offending Boxes — Match Level Breakdown

| Match Level | Count |
|---|---:|
| order | 750 |
| class | 1,487 |
| no_match | 761 |

## Offending Box Verdict Breakdown

| Verdict | Count |
|---|---:|
| flag | 2,078 |
| uncertain | 920 |

## Top 30 Contaminated Classes

| Rank | Class | Flagged Images |
|---:|---|---:|
| 1 | eared seals | 196 |
| 2 | sea otter | 73 |
| 3 | macaque species | 53 |
| 4 | gorilla species | 50 |
| 5 | domestic horse | 47 |
| 6 | elephant seal | 40 |
| 7 | hippopotamus | 38 |
| 8 | white-tailed deer | 35 |
| 9 | callithrix species | 32 |
| 10 | north american river otter | 31 |
| 11 | lion | 30 |
| 12 | eastern gray squirrel | 29 |
| 13 | kangaroo family | 27 |
| 14 | squirrel family | 27 |
| 15 | red fox | 26 |
| 16 | african wild dog | 25 |
| 17 | eulemur species | 25 |
| 18 | muridae family | 25 |
| 19 | african elephant | 24 |
| 20 | baboon genus | 24 |
| 21 | impala | 24 |
| 22 | bighorn sheep | 23 |
| 23 | domestic donkey | 23 |
| 24 | capybara | 22 |
| 25 | llama genus | 22 |
| 26 | eastern grey kangaroo | 21 |
| 27 | ring-tailed lemur | 21 |
| 28 | domestic cat | 21 |
| 29 | northern raccoon | 21 |
| 30 | howler monkey genus | 20 |
