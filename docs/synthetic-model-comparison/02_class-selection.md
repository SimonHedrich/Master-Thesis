# Class Selection for the Comparison Subset

**Date:** 2026-07-14
**Status:** Proposed subset (tunable) with rationale
**Data sources:** `reports/class_split_counts.csv`,
`reports/dataset_split_summary.json`, `reports/lookalike_groups_v2.csv`,
`data/gbif/metadata/GBIF_image_counts_v1.csv`,
`docs/plans/2026-06-11_lookalike-groups-review.md`

---

## 1. The three characteristics we want represented

The subset must let us observe three distinct phenomena. **They partly conflict**
(see §3), so the subset is chosen to cover all three *as a set*, not to satisfy
all three in every class.

1. **Unusual / long-tail species** likely **under-represented in foundation-model
   pretraining.** These stress the generator's *prior knowledge*: can it render a
   species it has barely seen, from a text description? Proxy for
   "under-represented" = low GBIF image count (GBIF counts are capped ≈500, so a
   low count is genuinely rare, not a sampling artifact).
2. **Robust real test set (>100 real test images).** Needed so downstream mAP on
   the **real** test set is statistically meaningful, not noise. 170 of 225
   classes clear this bar; the rare species mostly do **not**.
3. **Very similar (look-alike) species** — the three zebra species — to test
   whether a classifier trained on synthetic data can learn the *fine-grained*
   diagnostic markers (stripe patterns) that separate them. The incumbent
   observation is that local models likely **cannot** render these correctly;
   this is the sharpest test of "is the synthetic data good enough to teach
   species-level discrimination?"

## 2. Key facts that shape the choice

- **Zebras (all three present, genus *Equus*):** the frozen look-alike table puts
  them in one `zebra` group precisely because they are confusable. Diagnostic
  markers: **Grévy's** = narrow, dense stripes + white belly + large rounded
  ears; **plains** = broad stripes with brown "shadow" stripes; **mountain** =
  gridiron pattern on rump + dewlap.
- Rare species tend to have **tiny** real test sets (Band A: 6–149 images), so
  characteristics 1 and 2 pull against each other.
- A few species are **both** unusual **and** have >100 real test images — these
  are the most valuable picks because they de-confound rarity from test-set size.

## 3. The central tension (state this in the thesis)

> The rarest species (best for testing "not in pretraining") are exactly the
> ones with the fewest real test images (worst for downstream statistical
> power). We handle this by (a) including a few rare species that *do* clear the
> >100-test bar, and (b) reporting rare-but-test-limited species mainly on the
> **qualitative rubric + auto-proxy** metrics (which have per-image sample size),
> not leaning on their downstream mAP.

## 4. Proposed subset (12 classes)

`test_real` = real held-out test images; GBIF = GBIF image count (rarity proxy).

### Bucket 1 — Fine-grained look-alike group (the discrimination test)
| Class | Band | test_real | GBIF | Why |
|-------|------|-----------|------|-----|
| plains zebra | D | 500 | high | broad + shadow stripes; also a robust-test anchor |
| Grévy's zebra | B | 98 | low-ish | narrow dense stripes, white belly — the rare zebra |
| mountain zebra | D | 93 | mid | gridiron rump + dewlap |

Only Grévy's zebra is currently in the 76-class **synthetic-train** set (Band B);
all three have synthetic **test** images and real test images. Evaluate this
group at **fine** granularity (per-species AP), not just the coarse `zebra` group
— that is the whole point.

### Bucket 2 — Common, robust real test, in pretraining (upper-baseline anchors)
| Class | Band | test_real | Why |
|-------|------|-----------|-----|
| red fox | D | 500 | ubiquitous; any decent generator should nail it |
| American black bear | D | 500 | large, common, well-represented |
| lion | D | 500 | iconic; sanity ceiling for all models |

(plains zebra from Bucket 1 also serves here.) These calibrate the scale: if a
model can't do these well, its rare-species output is meaningless.

### Bucket 3 — Rare / long-tail, likely under-represented in pretraining
| Class | Band | test_real | GBIF | Why |
|-------|------|-----------|------|-----|
| kinkajou | A | 125 | 294 | **rare + robust test** — de-confounds rarity vs test size |
| water deer | A | 133 | 204 | **rare + robust test**; distinctive tusks, no antlers |
| ringtail | A | 123 | — | **rare + robust test**; distinctive banded tail |
| saiga | A | 47 | 58 | iconic bizarre proboscis — the textbook "not in pretraining" case |
| aye-aye | A | 29 | 29 | extreme morphology (elongated finger, ears) — stress test |
| pangolin family | A | 45 | 9–63 | scaled body — very rare; tests exotic texture |

Buckets 3's last three (saiga, aye-aye, pangolin) are **test-limited** — they are
included for the qualitative/auto-proxy axes and as vivid case studies, not for
their downstream mAP.

## 5. Optional additions (if budget allows)

- **A second fine-grained group that is also rare: the three tapirs** — Malay
  (test 21), Baird's (93), lowland (143). "Identical body plan," and Malay tapir
  has a unique black-and-white saddle. This doubles as look-alike **and** unusual,
  and lowland tapir clears the >100-test bar. Strong candidate if you want a
  second discrimination group.
- **panthera_rosette** (leopard/jaguar/snow leopard) as a second look-alike group
  with big test sets (386/87/163) if you prefer common-species discrimination.

## 6. Why this subset is defensible to the professor

- It is **purposive, not convenient**: each class is included for a stated reason
  tied to a hypothesis (prior knowledge, statistical power, fine-grained
  discrimination).
- It spans the **full difficulty range** (ubiquitous → bizarre-rare), so the
  comparison can show *where* models diverge, not just an average.
- It reuses the project's **frozen** look-alike table and split, so nothing is
  cherry-picked to flatter a model.

## 7. Practical notes

- All 12 already have synthetic **test** images and real train/test images in the
  existing split — no new *test* data needed.
- For generation we only need to make **synthetic train** images per generator
  for these 12 classes (the incumbent Gemini train images already exist for the
  Band A/B ones; regenerate the Band D zebra/anchors' synthetic set for
  apples-to-apples, or note they were real-only in production).
- Freeze this list in a small CSV (e.g. `reports/model_comparison_classes.csv`)
  before generating, mirroring the project's freeze-then-run discipline.
