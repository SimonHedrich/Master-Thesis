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
| Class | Band | test_real | test_real (this exp.)\* | GBIF | Why |
|-------|------|-----------|--------------------------|------|-----|
| plains zebra | D | 478 | 2,075 | high | broad + shadow stripes; also a robust-test anchor |
| Grévy's zebra | B | 119 | 224 | low-ish | narrow dense stripes, white belly — the rare zebra |
| mountain zebra | D | 93 | 467 | mid | gridiron rump + dewlap |

Only Grévy's zebra is currently in the 76-class **synthetic-train** set (Band B);
all three have synthetic **test** images and real test images. Evaluate this
group at **fine** granularity (per-species AP), not just the coarse `zebra` group
— that is the whole point.

\* See §4a for what "this exp." means and why it differs from the base `test_real`.

### Bucket 2 — Common, robust real test, in pretraining (upper-baseline anchors)
| Class | Band | test_real | test_real (this exp.)\* | Why |
|-------|------|-----------|--------------------------|-----|
| red fox | D | 500 | 2,149 | ubiquitous; any decent generator should nail it |
| American black bear | D | 500 | 2,097 | large, common, well-represented |
| lion | D | 497 | 2,097 | iconic; sanity ceiling for all models |

(plains zebra from Bucket 1 also serves here — test_real (this exp.) = 2,075.)
These calibrate the scale: if a model can't do these well, its rare-species
output is meaningless.

### 4a. Test-set expansion for Band B/C/D classes (this experiment only)

This experiment never trains on real images — for all 12 classes it trains
only on freshly generated **synthetic** images per generator (§7). The base
production split, by contrast, reserves real images for training/validation
for Band B/C/D classes (Band A classes are already 100% real-as-test, so
nothing changes for Bucket 3 below). Since this experiment's training doesn't
touch those reserved real images, they're free to add to its real test set —
substantially increasing statistical power, most valuably for the Bucket 2
anchor classes, where the reserved-but-idle real pool is large (e.g. red fox:
1,499 train + 150 val ordinarily withheld from testing).

**Rule:** for Band B/C/D classes in this experiment,
`test_real (this exp.) = base train_real + base val_real + base test_real`
— i.e. the full real pool actually allocated in the current split. This
excludes the documented, unallocated **surplus** some large Band D pools still
carry (see caveats below).

| Class | Band | train_real | val_real | base test_real | = test_real (this exp.) |
|-------|------|------------|----------|-----------------|----------------------------|
| Grévy's zebra | B | 85 | 20 | 119 | 224 |
| mountain zebra | D | 342 | 32 | 93 | 467 |
| plains zebra | D | 1,497 | 100 | 478 | 2,075 |
| lion | D | 1,500 | 100 | 497 | 2,097 |
| American black bear | D | 1,497 | 100 | 500 | 2,097 |
| red fox | D | 1,499 | 150 | 500 | 2,149 |

**Caveats:**
- **Quality-distribution shift.** The base split's train images are
  greedy top-Q selections and val is mid-Q (30th–70th percentile), while the
  original test images were assigned by unbiased stratified-random sampling
  (`docs/plans/2026-05-25_dataset-split-real-image-selection.md` §3). Folding
  train+val into test therefore skews the enlarged test set toward higher
  image quality than a purely random sample. This does **not** bias the
  *relative* generator-vs-generator comparison — every generator is scored on
  the identical enlarged set — but it means this expanded test set is
  specific to this comparison experiment, not the project's general
  real-test benchmark.
- **Surplus excluded.** Very large Band D pools (e.g. red fox, pool ≈8,900)
  still have thousands of real images never assigned to train/val/test at all
  ("surplus", see `reports/dataset_split_summary.json`); those are not folded
  in here. Using them would maximize power further but is left as a possible
  future extension, not part of this experiment.
- **Data source.** Counts above are read from the live split manifests
  (`data/real/annotations_{train,val,test}.json`) rather than
  `reports/class_split_counts.csv`, which is a formula-recomputed target that
  disagrees slightly for a few classes (e.g. `class_split_counts.csv` lists
  plains zebra test_real=500 vs. the manifest's 478) — the manifest is what
  this experiment actually loads images from, and it already reflects the
  multi-animal contamination filtering (`scripts/dataset_quality/14-*`,
  `15-*`) applied on 2026-06-24/29, so no further exclusion is needed.

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
  existing split — no new *test* data needed. For Band B/C/D classes, the real
  test set actually used by this experiment is the *expanded* set defined in
  §4a (base train+val+test), not just the base split's test_real.
- For generation we only need to make **synthetic train** images per generator
  for these 12 classes (the incumbent Gemini train images already exist for the
  Band A/B ones; regenerate the Band D zebra/anchors' synthetic set for
  apples-to-apples, or note they were real-only in production).
- Freeze this list in a small CSV (e.g. `reports/model_comparison_classes.csv`)
  before generating, mirroring the project's freeze-then-run discipline.
