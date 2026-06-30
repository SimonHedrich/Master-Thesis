# Dataset Supplementation Plan

This document outlines the concrete steps to build a training-ready dataset for the student model, based on the quality analysis of Danielle's GBIF/SpeciesNet dataset ([`scripts/analyze_dataset_quality.py`](../scripts/analyze_dataset_quality.py)) and the supplementary dataset research ([`docs/supplementary-dataset-research.md`](supplementary-dataset-research.md)).

---

## Current State

Danielle's dataset (`resources/SNPredictions_all.json` + `resources/GBIFImages/images/`):

| Metric | Value |
|--------|------:|
| Images on disk | 66,881 |
| Entries in JSON but missing from disk | 24,410 |
| Species-level confident predictions | 35,594 (53%) |
| Rolled-up predictions (genus/family/order) | 19,420 (29%) |
| Detector-only / blank / no result | 11,867 (18%) |
| Filename vs. prediction disagreement | ~23% |
| 225-class labels with 0 images | 31 |
| 225-class labels with <50 images | 94 |
| 480-class labels with 0 images | 170 |

**Key problems:**
1. **Coverage gaps** — 31 of 225 target classes have zero images, 94 have fewer than 50
2. **Label noise** — ~23% of predictions disagree with filename-inferred species
3. **Missing files** — 24k images referenced in JSON are not on disk
4. **Modality** — all images are daylight color photos from GBIF (good for binocular use case, but limited diversity)
5. **No bounding boxes** — the dataset has SpeciesNet detections but these come from MegaDetector, not ground-truth annotations

---

## Minimum Per-Class Image Threshold

Based on transfer-learning and knowledge-distillation literature, the following per-class instance counts are the working targets for this pipeline.

> **Instance** = one annotated animal occurrence (bounding box). For iNaturalist/GBIF images that typically contain one animal, images ≈ instances.

| Scenario | Minimum instances/class |
|---|---|
| KD from SpeciesNet teacher (primary pipeline) | ~200–400 |
| Direct fine-tuning of student (baseline) | ~400–800 |
| Visually similar or long-tail species | 800+ |

**Rationale:**
- Fine-tuning a COCO-pretrained backbone requires far less data than training from scratch, because the backbone already encodes textures, edges, and animal-like shapes.
- Knowledge distillation from SpeciesNet further lowers the threshold: the teacher's soft labels carry domain-relevant signal that effectively amplifies each training image.
- Nano-scale student models (1–3 M parameters) overfit less readily than large models and therefore need proportionally fewer examples per class.

**Class inclusion threshold:** A GBIF image count of **≥ 300** is used as a proxy for including a species in the target class set. This correlates with iNaturalist/supplementary availability and leaves headroom for quality filtering.

Classes that remain below ~200 instances after all real-data supplementation steps (Steps 1–4) are candidates for synthetic generation or pseudo-labeling (Step 5).

---

## Special Case: Human Class

The `human / homo sapiens` class requires a separate sourcing strategy from all other 224 classes because the standard quality-filter pipeline is incompatible with it: MegaDetector classifies humans in a dedicated detection class, separate from its "animal" class. Every human image therefore fails the pipeline's "no animal detected" check and is rejected. The 14 GBIF human images on disk all had zero images after filtering for this reason.

### Sources investigated

**iNaturalist open data (S3 export) — not viable**

The iNaturalist public S3 bulk export (`s3://inaturalist-open-data/`) deliberately excludes Homo sapiens observations, almost certainly for privacy reasons (the export does not publish human location data). The taxon `Homo sapiens` exists in `taxa.csv` (taxon_id 43584) but zero observations appear in `observations.csv`.

**Open Images V7 — tried and rejected**

Open Images V7 has a `Person` class with ground-truth bounding boxes. However, after downloading with filters (person bbox area ≥ 5%, ≤ 2 person bboxes, no vehicle annotations), the images were dominated by close-up portraits and low-quality shots. OI annotations tend toward tightly framed, subject-centred photography that does not match the AX Visio deployment scenario.

**COCO 2017 (first attempt) — tried and rejected**

COCO 2017 was the first source attempted. The initial implementation filtered for:
- No animal-supercategory annotations
- Ranked by fewest indoor-supercategory annotations

This produced unsuitable images because the `vehicle` supercategory was never excluded, the minimum person area threshold was 100 absolute px² (≈ 0.03% of a typical image), and there were no crowd, edge-margin, or person-count checks. The resulting images were dominated by urban street scenes with buses and motorcycles, crowd shots, and tiny background pedestrians.

**COCO 2017 (revised, strict filters) — adopted**

After adding the full set of selection and quality filters described below, COCO 2017 yields images that are suitable for the AX Visio use case: single or small groups of people visible at natural outdoor distances, not cropped or obscured.

### Adopted pipeline: `scripts/download_coco_humans.py`

Output: `data/coco_humans/images/human/coco_{image_id}.jpg`  
Catalog: `data/coco_humans/metadata_catalog.csv` (same schema as Open Images)  
Pipeline isolation: `data/coco_humans/` is not referenced by `1-filter_dataset_quality.py`, `2-analyse_dataset_coverage.py`, or `rename_dataset_images.py`. Human images are excluded from the standard quality filter by design.

#### Selection filters (applied before download, from COCO annotation JSON)

| Filter | Value | Problem addressed |
|---|---|---|
| Exclude `vehicle` supercategory | hard exclude | Buses, motorcycles, cars dominating the frame |
| Exclude `animal` supercategory | hard exclude | Animals in the image |
| Exclude `iscrowd = 1` annotations | hard exclude | Crowd shots (COCO crowd flag) |
| Max total person annotations | ≤ 3 | Large groups of people |
| Person bbox normalized area (floor) | ≥ 5% of image | Tiny background pedestrians |
| Person bbox normalized area (ceiling) | ≤ 60% of image | Extreme close-ups |
| Edge margin | ≥ 2% from all four edges | Cropped subjects at frame boundary |
| Person bbox aspect ratio (h/w) | ≥ 0.5 | Horizontal / lying-down crops |
| Indoor score (soft rank) | ascending | Prefer images without furniture/appliance/electronic annotations |
| Person area (secondary sort) | descending | Among equally outdoor images, prefer the most prominent person |

#### Post-download quality checks (same thresholds as `1-filter_dataset_quality.py`)

Applied via `cv2` immediately after each download. Failed images are deleted and marked as failed (not retried):

| Check | Threshold | Rationale |
|---|---|---|
| Minimum resolution | shorter side ≥ 256 px | Filter thumbnails or corrupted downloads |
| Blur (Laplacian variance) | ≥ 100 on 512 × 512 grayscale thumbnail | Discard motion-blurred or out-of-focus frames |
| Grayscale / IR detection | mean HSV saturation ≥ 15 | Discard near-grayscale or infrared images |

#### Usage

```bash
# Fresh start (deletes progress + catalog from any previous run)
python scripts/download_coco_humans.py --reset --target 50 --workers 4  # smoke test first

# Full run (or resume after interruption)
python scripts/download_coco_humans.py --target 2000
```

---

## Step 1: Download and Process LILA BC Datasets

> **Decision: Not used.** After downloading a representative sample and reviewing the images, LILA BC camera trap data was found to be unsuitable for this use case. The images predominantly show animals at long range, partially occluded by vegetation, with motion blur from fast-moving triggers, and a large fraction captured under infrared night-vision conditions. The recognisable animal content per image is consistently low, making these images a poor match for the AX Visio binocular deployment scenario, which requires clear, daylight, close-to-mid-range shots. The dataset was removed and is not included in the training pipeline.

**Priority: Highest. Impact: Covers most gap species. Risk: Lowest (CDLA-Permissive license).**

The LILA BC (Labeled Information Library of Alexandria) ecosystem hosts camera trap datasets under the CDLA-Permissive license, which explicitly permits commercial model training without copyleft obligations.<sup>[14][17]</sup> Three datasets are particularly valuable:

### 1a. Snapshot Safari / Serengeti

- **Source:** [lila.science/datasets/snapshot-serengeti](https://lila.science/datasets/snapshot-serengeti/) and [lila.science/datasets/snapshot-safari-2024-expansion](https://lila.science/datasets/snapshot-safari-2024-expansion/)
- **License:** CDLA-Permissive + CC-BY 4.0 — **commercially SAFE**
- **Scale:** ~11M images combined (Serengeti 7.1M + Safari Expansion 4M)
- **Annotations:** Image-level species labels; Serengeti has ~150k bounding boxes for 78k images
- **Gap coverage:** African antelopes (bongo, nyala, roan, kob, dik-dik, klipspringer), hyenas (brown hyena), meerkat, and other African species missing from our dataset
- **Caveat:** ~76% of images are blank/empty triggers. Many are grayscale IR night images. Need to filter for (a) non-empty, (b) target species, and (c) preferably daylight color images

### 1b. WCS Camera Traps

- **Source:** [lila.science/datasets/wcscameratraps](https://lila.science/datasets/wcscameratraps/)
- **License:** CDLA-Permissive — **commercially SAFE**
- **Scale:** ~1.4M images, ~675 species, 12 countries
- **Annotations:** 375k ground-truth bounding boxes across 300k images — this is the most valuable annotation source
- **Gap coverage:** Asian mammals (Asiatic wild ass, water deer, macaques, sloth bear), neotropical species (sloths, tapirs), and general diversity
- **Caveat:** Heavy class imbalance (white-lipped peccary alone: ~95k images). Mixed IR/color

### 1c. Caltech Camera Traps (CCT-20)

- **Source:** [lila.science/datasets/caltech-camera-traps](https://lila.science/datasets/caltech-camera-traps/)
- **License:** CDLA-Permissive — **commercially SAFE**
- **Scale:** 243k images, 21 categories, SW United States
- **Annotations:** 66k bounding boxes
- **Gap coverage:** Common North American species (opossum, raccoon, coyote, bobcat, deer)
- **Caveat:** Limited to SW US ecosystems. Useful for NA species reinforcement, not for filling exotic gaps

### Processing pipeline

All LILA BC datasets use the **COCO Camera Traps** JSON format. The processing script should:

1. Download the annotation JSON files (small) and image archives (large) for each dataset
2. Parse annotations and filter to images containing target species from our 225-class or 480-class label sets
3. Map dataset-local category names to our SpeciesNet taxonomy using LILA BC's [taxonomy mapping CSV](https://lila.science/taxonomy-mapping-for-camera-trap-data-sets/)
4. Extract bounding box annotations where available; flag images without boxes for MegaDetector processing
5. Filter out empty/blank images
6. Output a unified annotation file in YOLO or COCO format

---

## Step 2: GBIF API Export with License Filtering

**Priority: High. Impact: Fills remaining gaps (marine mammals, primates, domestic animals). Risk: Low if license filtering is strict.**

The iNaturalist competition datasets are **not commercially usable** (custom academic license prohibiting commercial use).<sup>[5]</sup> However, iNaturalist users can apply CC0 or CC-BY licenses to their uploads, and these are exported to GBIF.<sup>[6][7]</sup> By querying the GBIF API directly and filtering for `license=CC0` or `license=CC_BY_4_0`, we can build a commercially safe image corpus for gap species.

### Target species for GBIF export

These are the 31 zero-image classes from our 225-class list that are unlikely to be well-covered by LILA BC camera trap data:

**Marine mammals:** sea otter, walrus, elephant seals, eared seals, pinnipeds
**Primates:** aye-aye, ring-tailed lemur, patas monkey, drill, Japanese macaque
**Asian mammals:** Asiatic wild ass, water deer, yak, water buffalo, sloth bear
**Other:** meerkat, saiga, Eurasian lynx, domestic pig, sloths, Hoffmann's two-toed sloth, brown-throated sloth

### Processing pipeline

1. Query GBIF Occurrence API per target species (by taxon key), filtering for:
   - `mediaType=StillImage`
   - `license=CC0_1_0` or `license=CC_BY_4_0`
   - `hasCoordinate=true` (helps with geographic diversity)
2. Download images (respect GBIF rate limits)
3. Run MegaDetector v5 to generate bounding boxes
4. Filter: keep only images where MegaDetector detects exactly 1 animal with confidence > 0.5
5. Generate attribution manifest for CC-BY images (photographer name, URL, license)

---

## Step 3: Open Images V7 + COCO for Domestic Animals

**Priority: Medium. Impact: High-quality daylight bboxes for domestic species. Risk: Low (CC-BY).**

### Open Images V7

- **Source:** [storage.googleapis.com/openimages](https://storage.googleapis.com/openimages/web/factsfigures_v7.html)
- **License:** CC-BY 2.0 (images), CC-BY 4.0 (annotations) — **commercially SAFE for model training**
- **Relevant classes:** Mammal (95k images), plus specific classes: Jaguar, Kangaroo, Leopard, Elephant, Rhinoceros, Sea lion, Cattle, Dog, Cat, Horse, Sheep, Pig, Bear, Deer, Squirrel, Rabbit, Fox, Hedgehog, Monkey
- **Key limitation:** Labels use colloquial names (e.g., "Elephant"), not species-level taxonomy. Needs manual or model-based remapping to SpeciesNet taxonomy

### COCO 2017

- **Source:** [cocodataset.org](https://cocodataset.org/)
- **License:** CC-BY 4.0 (annotations), Flickr ToU (images) — **SAFE for model training**
- **Relevant mammal classes (10):** cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe (+ person)
- **Value:** Pristine bounding box + segmentation annotations, complex daylight backgrounds — ideal for domestic animal classes

### Processing pipeline

1. Download class-specific subsets (not the full dataset — use the OID or FiftyOne tools to filter)
2. Map labels to SpeciesNet taxonomy (e.g., COCO "cow" → *Bos taurus*, Open Images "Elephant" → *Loxodonta africana* or *Elephas maximus* — may need secondary classifier)
3. Convert to unified annotation format

---

## Step 4: Address Label Noise in Existing Dataset

**Priority: Medium-high. Impact: Improves training signal quality. Risk: None.**

The 23% filename-vs-prediction disagreement in Danielle's dataset indicates significant label noise. Training on noisy labels degrades model performance, especially for a small model with limited capacity.

### Approach

1. **Flag disagreements:** The dataset analysis script already identifies filename-inferred species vs. SpeciesNet prediction mismatches. Generate a list of all disagreeing images
2. **Apply confident learning:** Use [cleanlab](https://github.com/cleanlab/cleanlab) to identify likely label errors based on prediction confidence distributions
3. **Triage strategy:**
   - Images where SpeciesNet confidence > 0.9 AND filename agrees: **high-trust training data**
   - Images where SpeciesNet confidence > 0.9 BUT filename disagrees: likely **filename is wrong** (GBIF metadata error) — use SpeciesNet label
   - Images where SpeciesNet confidence < 0.5: **exclude from training** or use only for pre-training with soft labels
   - Rolled-up predictions (genus/family level): usable for hierarchical training but not for species-level supervision
4. **Weight samples:** During training, assign higher loss weight to human-verified datasets (COCO, Open Images, WCS with ground-truth boxes) than to SpeciesNet-inferred labels

---

## Step 5: Synthetic Data for Persistently Rare Species

**Priority: Lower (after steps 1–4). Impact: Fills the long tail. Risk: Medium (quality control needed).**

After combining all real-data sources, some species will likely remain data-starved (estimated <50 images). The research identifies three strategies:

### 5a. LoRA-guided diffusion generation

For species with 10–50 real images, train a LoRA adapter on Stable Diffusion XL (or a commercially licensed model like Adobe Firefly) to generate synthetic training images. Key constraint from the thesis design: **train on real+synthetic, evaluate strictly on real photographs only.**

### 5b. BioCLIP 2 pseudo-labeling

BioCLIP 2 model weights are MIT-licensed.<sup>[44]</sup> Use it as a zero-shot classifier on large pools of unlabeled but commercially safe images (e.g., CC0 nature photography from Unsplash, Wikimedia Commons). High-confidence predictions (>0.95) can be treated as pseudo-labels and passed through MegaDetector for bbox generation.

### 5c. Targeted Wikimedia Commons scraping

The Wikimedia Commons API allows filtering by license (CC0, CC-BY, Public Domain). For rare species like the aye-aye or drill, this can yield a small but high-quality set of images. CC-BY-SA images must be excluded to avoid copyleft risk.

**Implemented reconnaissance pipeline (see [`docs/progress_notes/2026-03-30_wikimedia-category-crawling.md`](../docs/progress_notes/2026-03-30_wikimedia-category-crawling.md)):**

1. `scripts/crawl_wikimedia_categories.py` — crawls the Wikimedia Commons category hierarchy for all 225 labels (up to depth 2, capped at 5000 categories per label), recording category names and file counts into `reports/wikimedia_categories/`. No images are downloaded at this stage.
2. `scripts/filter_wikimedia_categories.py` — applies keyword-based cascade filtering to remove non-photographic categories (artwork, anatomy, maps, stamps, taxidermy, fossils, etc.), producing `reports/wikimedia_categories_filtered/`.

The filtered output feeds a planned image downloader that will enumerate files per category, apply license filtering (CC0 / CC-BY / Public Domain only), download images, and pass them through MegaDetector v5 for bounding box generation.

---

## Step 6: Unified Dataset Assembly and Splitting

**Priority: Final step before training.**

### Taxonomy harmonization

All data sources use different label schemes. The unified pipeline must:
- Map all labels to the SpeciesNet taxonomy UUIDs used in our 225-class and 480-class label files
- Use LILA BC's [taxonomy mapping CSV](https://lila.science/taxonomy-mapping-for-camera-trap-data-sets/) for camera trap datasets
- Map COCO/Open Images colloquial labels manually (small fixed mapping table)
- Handle hierarchical labels: if a source says "bear" but doesn't specify species, map to *Ursidae* family

### Modality handling (IR vs. daylight)

Camera trap datasets contain mixed IR (grayscale, glowing eyes) and daylight color images. For the AX Visio use case:
- **Test and validation sets:** Must be daylight color images only (to match deployment conditions)
- **Training set:** May include IR images for shape/pose diversity, but apply aggressive augmentation (color jitter, random grayscale, channel dropout) to prevent the model from relying on IR-specific artifacts

### Train / validation / test split

- Split by **location/camera ID** for camera trap data (prevents background overfitting)
- Split by **source** for citizen science data (some sources to train, others to test)
- Hold-out test set: exclusively citizen science (GBIF) and Open Images photography — these most closely simulate binocular usage
- Target ratio: 80% train / 10% val / 10% test

### Estimated final dataset size

| Source | Est. usable images |
|--------|-------------------:|
| Danielle's GBIF dataset (high-confidence subset) | ~35,000 |
| Snapshot Safari/Serengeti (filtered to target species) | ~50,000–100,000 |
| WCS Camera Traps (filtered) | ~30,000–60,000 |
| Open Images V7 (mammal subsets) | ~20,000–40,000 |
| COCO 2017 (mammal classes) | ~10,000–15,000 |
| GBIF API export (gap species, CC0/CC-BY) | ~5,000–15,000 |
| CCT-20 (NA species) | ~5,000–10,000 |
| Synthetic / pseudo-labeled (if needed) | ~5,000–10,000 |
| **Total estimated** | **~160,000–300,000** |

This sits within the optimal range for a 1–3M parameter YOLO-nano architecture as recommended by the research report.<sup>[11]</sup>

---

## Output Files

| File | Description |
|------|-------------|
| [`scripts/analyze_dataset_quality.py`](../scripts/analyze_dataset_quality.py) | Quality analysis of Danielle's dataset |
| [`reports/class_counts_225.csv`](../reports/class_counts_225.csv) | Per-class image counts for the 225-class label set |
| [`reports/class_counts_480.csv`](../reports/class_counts_480.csv) | Per-class image counts for the 480-class label set |
| `scripts/download_lila_bc.py` | Download and process LILA BC datasets (Step 1) |
| *(planned)* `scripts/gbif_gap_export.py` | GBIF API export for gap species (Step 2) |
| `scripts/crawl_wikimedia_categories.py` | Crawl Wikimedia Commons category trees for all 225 labels (Step 5c recon) |
| `scripts/filter_wikimedia_categories.py` | Filter raw category trees to remove non-photographic content (Step 5c recon) |
| `reports/wikimedia_categories/` | Raw Wikimedia category trees, one .txt per label |
| `reports/wikimedia_categories_filtered/` | Filtered category trees (non-photographic categories removed) |
| *(planned)* `scripts/assemble_dataset.py` | Unified dataset assembly and splitting (Step 6) |

---

## References

Citation numbers refer to [`docs/supplementary-dataset-research.md`](supplementary-dataset-research.md).

- [5] iNaturalist competition rules — non-commercial only
- [6][7] iNaturalist license options — CC0/CC-BY available per user
- [11] Optimal dataset size for YOLOv8 training
- [14] LILA BC — CDLA-Permissive license
- [17] Snapshot Serengeti — CDLA-Permissive / CC-BY 4.0
- [44] BioCLIP 2 model weights — MIT license
