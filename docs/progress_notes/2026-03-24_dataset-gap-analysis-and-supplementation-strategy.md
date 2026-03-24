# Progress Notes – 24.03.2026

## LILA BC Dataset Filtering Results

The [`scripts/download_lila_bc.py`](../../scripts/download_lila_bc.py) pipeline was run against three LILA BC camera trap datasets, filtering for species matching the 225-class student model label set ([`resources/2026-03-19_student_model_labels.txt`](../../resources/2026-03-19_student_model_labels.txt)). All three datasets are licensed under CDLA-Permissive — commercially safe for model training.

### Pipeline Summary

| Dataset | Total Images | Categories Matched | Images Filtered | With BBoxes |
|---------|:-----------:|:-----------------:|:---------------:|:-----------:|
| Snapshot Serengeti v2.1 | 7.2M | 39 / 61 | 1,637,554 | 62,540 |
| Snapshot Safari 2024 Expansion | 4.0M | 62 / 131 | 1,140,640 | 0 |
| WCS Camera Traps | 1.4M | 137 / 676 | 435,909 | 110,069 |
| **Total** | **12.6M** | | **3,214,103** | **172,609** |

**115 of 225 target classes** are covered by at least one image from LILA BC.

Output: `data/lila_bc/filtered_images_225.json` (3.2M entries with image URLs, matched labels, and bounding box annotations where available).

### Top Species by Image Count

**Snapshot Serengeti** (dominated by Serengeti plains game):

| Species | Images |
|---------|-------:|
| Common wildebeest | 530,429 |
| Thomson's gazelle | 321,743 |
| Plains zebra | 292,904 |
| African buffalo | 61,274 |
| Hartebeest | 56,773 |
| African elephant | 53,565 |
| Grant's gazelle | 46,407 |
| Giraffe | 44,024 |
| Impala | 42,956 |

**Snapshot Safari 2024** (broader African coverage):

| Species | Images |
|---------|-------:|
| Impala | 227,033 |
| African elephant | 118,953 |
| Common wildebeest | 118,323 |
| Plains zebra | 111,474 |
| Baboon genus | 64,559 |
| Gemsbok | 54,950 |
| Greater kudu | 48,879 |
| Thomson's gazelle | 45,836 |
| Giraffe | 44,378 |
| Springbok | 41,260 |

**WCS Camera Traps** (global, neotropical focus):

| Species | Images |
|---------|-------:|
| Human | 162,501 |
| Domestic cattle | 22,182 |
| Impala | 22,181 |
| Agouti genus | 21,149 |
| Plains zebra | 20,337 |
| Cephalophus (duikers) | 15,160 |
| Ocelot | 14,948 |
| African elephant | 13,394 |
| Collared peccary | 10,828 |
| Jaguar | 9,435 |

### Coverage Gaps — 110 Classes Still Uncovered

The 110 remaining uncovered classes fall into distinct geographic/taxonomic groups, each requiring a different supplementary data source:

**North American wildlife (30+ species):**
American black bear, brown bear, grey wolf, bobcat, Canada lynx, moose, elk, mule deer, white-tailed deer, pronghorn, mountain goat, bighorn sheep, American bison, coyote (matched in WCS but via Serengeti "jackal"), red fox, grey fox, ringtail, striped skunk, wolverine, fisher, American badger, river otter, mink, woodchuck, prairie dog, chipmunks, all NA squirrel species, muskrat, North American porcupine.

> **Source:** Danielle's existing GBIF dataset (already has many of these), GBIF API export (CC0/CC-BY), Caltech Camera Traps (CCT-20).

**European wildlife (~15 species):**
Eurasian badger, Eurasian lynx, Eurasian otter, European roe deer, red deer, sika deer, common fallow deer, reindeer, European bison, European rabbit, European hare, mouflon, chamois, Eurasian red squirrel.

> **Source:** GBIF API export (CC0/CC-BY), Open Images V7.

**Australian marsupials + monotremes (~8 species):**
Red kangaroo, eastern grey kangaroo, swamp wallaby, red-necked wallaby, quokka, koala, common wombat, short-beaked echidna.

> **Source:** Danielle's GBIF dataset (has some), GBIF API export.

**Domestic animals (~8 species):**
Domestic cat, domestic dog, domestic goat, domestic pig, domestic donkey, domestic water buffalo, llama genus, dingo.

> **Source:** COCO 2017 (cat, dog, horse, cow, sheep — high-quality bboxes), Open Images V7 (broader mammal coverage), GBIF API.

**Marine mammals (~5 entries):**
Pinniped clade, eared seals, elephant seal genus, walrus, sea otter.

> **Source:** GBIF API export, Open Images V7 ("sea lion" class).

**Asian / rare species (~15 species):**
Giant panda, snow leopard, red panda, sloth bear, Japanese macaque, raccoon dog, golden jackal, Asiatic wild ass, water deer, yak, nilgai, chital, sambar.

> **Source:** WCS Camera Traps (some matched but with few images), GBIF API export.

**Primates (~8 entries):**
Gorilla genus, ring-tailed lemur, drill, callicebus genus, callithrix genus, saguinus genus, gorilla, bornean orangutan.

> **Source:** GBIF API export, Danielle's GBIF dataset.

**Other (~10 entries):**
Hedgehog family, opossum family, pangolin family, muridae family, cricetidae family, old world porcupine family, squirrel family, rhinoceros family, pikas genus, beaver genus.

> These are family/genus-level fallback labels. Coverage will come indirectly from species-level matches in other sources.

### Key Observations

1. **Extreme class imbalance.** Common wildebeest alone has 530k+ images from Serengeti — while many target species have zero. A `--max-per-class` cap will be essential when downloading images to avoid drowning the training set in wildebeest and zebra.

2. **Bounding box coverage is sparse.** Only 172k of 3.2M filtered images have bounding boxes (~5%). The rest have image-level labels only. MegaDetector will need to be run on the downloaded images to generate bounding boxes for training.

3. **Camera trap modality.** All three datasets contain a mix of daylight color and nighttime IR images. The validation and test sets must be restricted to daylight color images to match the AX Visio deployment environment (see [`docs/dataset-supplementation-plan.md`](../dataset-supplementation-plan.md), Section 4.2).

4. **Taxonomy matching worked well.** The LILA taxonomy mapping CSV resolved Serengeti's informal category names (e.g., `"gazellegrants"` → *Nanger granti* → "Grant's gazelle"). WCS uses scientific names directly, matching 137 of 676 categories. Unmatched categories are mostly birds, reptiles, and species outside our mammal label set.

5. **Danielle's 66k GBIF dataset fills a different niche.** It covers many NA/EU/Australian species that LILA BC lacks. The two sources are highly complementary: LILA BC provides African + neotropical depth, GBIF provides global breadth.

---

## Supplementation Strategy for the 110 Uncovered Classes

Cross-referencing the coverage gaps above with the findings in [`docs/supplementary-dataset-research.md`](../supplementary-dataset-research.md), the following real-image data sources can fill most or all of the remaining 110 classes.

### Source 1: iNaturalist Open Data on AWS S3 (highest impact)

**Bucket:** `s3://inaturalist-open-data/` (public, no auth required)

**License strategy:** The iNaturalist *competition* datasets are NOT commercially usable (custom academic license). However, individual iNaturalist photos are uploaded with per-user Creative Commons licenses. Photos with CC0 or CC-BY licenses are exported to both GBIF and an AWS S3 bucket. By filtering the S3 metadata for `photo_license IN ('CC0', 'CC-BY')`, we can legally reconstruct a massive commercially safe image corpus.

**S3 bucket structure:**
- `metadata/` — Monthly snapshot tarballs containing tab-separated CSV files:
  - `photos.csv` — `photo_id`, `photo_license`, `extension` (used to construct image URLs)
  - `observations.csv` — `taxon_id`, observation metadata
  - `taxa.csv` — Full taxonomic hierarchy (kingdom → species), maps `taxon_id` to scientific names
  - `observers.csv` — Observer name/login (needed for CC-BY attribution)
- `photos/{photo_id}/{size}.{extension}` — Actual image files (sizes: original/2048px, large/1024px, medium/500px, small/240px)

**Why this is the single highest-impact source:**
- iNaturalist has **global citizen science coverage** — unlike LILA BC which is concentrated on African + neotropical camera traps
- Covers **all gap categories**: NA wildlife, European species, Australian marsupials, Asian mammals, primates, marine mammals, domestic animals
- The iNaturalist 2021 competition alone had 246 mammal species with 69k training images — the S3 export is far larger (400M+ total photos across all taxa)
- Photos are predominantly **daylight color images** taken by citizen scientists, closely matching the AX Visio deployment environment
- CC0/CC-BY filtering ensures commercial safety

**Estimated yield:** 500–10,000+ images per gap species depending on how common the species is on iNaturalist.

**Script:** [`scripts/download_inaturalist.py`](../../scripts/download_inaturalist.py)

### Source 2: Additional LILA BC datasets (NACTI)

The North American Camera Trap Images (NACTI) dataset on LILA BC was not included in the initial three-dataset processing run but could directly fill the **30+ North American species gap** — the largest single gap group. Licensed under CDLA-Permissive.

> **Action:** Add NACTI to the `DATASETS` registry in `scripts/download_lila_bc.py` and rerun the metadata pipeline.

### Source 3: Open Images V7 + COCO 2017

- **Open Images V7** (CC-BY): 16M bounding boxes, includes mammal classes like Jaguar, Kangaroo, Leopard, Elephant, Rhinoceros, Sea Lion, Cattle. Limitation: colloquial labels only ("Elephant" not *Loxodonta africana*), requires manual taxonomy mapping.
- **COCO 2017** (CC-BY): 10 mammal classes with pristine bounding boxes — cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe. Ideal for domestic animal gap classes.

Both are high-quality daylight images with native bounding boxes, complementing the camera trap data well.

### Source 4: Wikimedia Commons API (rare species)

For persistently rare species (aye-aye, drill, saiga, water deer) where even iNaturalist and GBIF yield <50 images, the Wikimedia Commons API allows programmatic extraction filtered by license (CC0, CC-BY, Public Domain). Must drop CC-BY-SA to avoid copyleft risk. Low volume but high taxonomic precision.

### Source 5: BioCLIP 2 pseudo-labeling (last resort for real images)

BioCLIP 2 model weights are MIT-licensed. Can be deployed as a zero-shot classifier on large pools of unlabeled but commercially safe nature photography (Unsplash CC0, Wikimedia Commons). High-confidence predictions (>0.95) get routed through MegaDetector for bounding box generation. This creates a fully automated, legally clean data mining pipeline for species where no pre-labeled data exists.

### Recommended execution order

| Priority | Source | Gap groups filled | Script |
|:--------:|--------|-------------------|--------|
| 1 | **iNaturalist Open Data (S3)** | All 110 classes globally | `scripts/download_inaturalist.py` |
| 2 | **NACTI (LILA BC)** | 30+ NA species | `scripts/download_lila_bc.py` (add dataset) |
| 3 | **Open Images V7 + COCO** | Domestic, megafauna, marine | *(planned)* |
| 4 | **Wikimedia Commons** | Rare species (<50 images) | *(planned)* |
| 5 | **BioCLIP 2 pseudo-labeling** | Persistently data-starved species | *(planned)* |

After all real-image sources are exhausted, synthetic data generation via LoRA-guided diffusion (see [`docs/dataset-supplementation-plan.md`](../dataset-supplementation-plan.md), Step 5) can fill any remaining long-tail gaps — but only for training, never for evaluation.

### Next Steps

1. **Run iNaturalist S3 pipeline:** `python scripts/download_inaturalist.py metadata` → `download`
2. **Download LILA BC images** with class balancing: `python scripts/download_lila_bc.py download --max-per-class 500`
3. **Run MegaDetector** on downloaded images lacking bounding boxes
4. **Add NACTI** to the LILA BC pipeline and rerun
5. **Integrate Open Images V7 + COCO** for domestic species and megafauna
6. **Merge all sources** with Danielle's dataset → unified dataset assembly (see [`docs/dataset-supplementation-plan.md`](../dataset-supplementation-plan.md), Step 6)
