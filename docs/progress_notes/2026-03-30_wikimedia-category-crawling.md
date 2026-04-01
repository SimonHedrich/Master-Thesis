# Progress Notes – 30.03.2026

## Wikimedia Commons Category Crawling and Filtering

### Motivation

The gap analysis from [2026-03-24](2026-03-24_dataset-gap-analysis-and-supplementation-strategy.md) identified 110 of the 225 target species as uncovered by LILA BC camera trap data. The primary gap-filling strategy is the iNaturalist Open Data S3 export. However, a subset of species — mainly rare or charismatic mammals (aye-aye, drill, saiga, water deer, Eurasian lynx) — are expected to yield fewer than 50 images even from iNaturalist, due to low observation frequency.

For these persistently data-starved species, Wikimedia Commons is a viable supplementary source. Wikimedia images are available under CC0, CC-BY, and Public Domain licenses; CC-BY-SA must be excluded to avoid copyleft. Unlike GBIF or iNaturalist, which filter by taxon, Wikimedia organises content into a hierarchical category tree rooted at species or genus names. This tree mixes genuinely useful wildlife photographs with large amounts of non-photographic content: illustrations, anatomy diagrams, maps, stamps, artworks, fossils, taxidermy, and more. Directly downloading from these categories without inspection would pollute the training set.

The crawling and filtering pipeline was therefore designed as a two-step offline reconnaissance process:

1. **Crawl** the Wikimedia category hierarchy for all 225 labels and record category names with file counts — no images are downloaded at this stage.
2. **Filter** the resulting category trees automatically using keyword-based rules to remove categories that are structurally non-photographic.

The filtered output is a curated list of category paths per species that can be used directly by a downstream image downloader.

---

### Implementation

#### Step 1: `scripts/crawl_wikimedia_categories.py`

Crawls the Wikimedia Commons category hierarchy for every label in the 225-class label file ([`resources/2026-03-19_student_model_labels.txt`](../../resources/2026-03-19_student_model_labels.txt)).

**Root category construction**

For each label, root Wikimedia categories are derived from the taxonomic fields in the label file:

- Species-level labels → `Category:Genus_species` (e.g., `Category:Acinonyx_jubatus`)
- Genus-level labels → `Category:Genus`, plus one `Category:Genus_species` entry per species listed in `resources/genus_species_mapping.csv`
- Family-level labels → `Category:Family`, plus per-species entries from `resources/family_species_mapping.csv`

**Tree crawl**

Starting from each root category, the script recursively fetches subcategories and their file counts using the Wikimedia API's `generator=categorymembers` + `prop=categoryinfo` combined query (one API call per node). The crawl goes up to `--max-depth` levels deep (default 2) and caps at `--max-categories` per label (default 5000) to prevent runaway traversal into large, tangentially related category trees (e.g., `Category:Lions_in_art` branches into hundreds of country-specific subcategories).

Rate limiting and retry logic with exponential back-off handle 429/503 responses from the Wikimedia API.

**Output format**

One `.txt` file per label in `reports/wikimedia_categories/`, named after the sanitised common name. Each file contains an indented category tree:

```
# Lion | Panthera leo
Category:Panthera_leo  (512 files)
  Category:Panthera leo in art  (87 files)
    Category:Paintings of Panthera leo  (12 files)
  Category:Panthera leo skulls  (4 files)
  Category:Lions in Serengeti  (203 files)
    ...
```

The `--resume` flag skips already-completed label files for interrupted runs.

---

#### Step 2: `scripts/filter_wikimedia_categories.py`

Reads all `.txt` files from `reports/wikimedia_categories/` and applies two types of exclusion:

1. **Zero-file removal** — lines with `(0 files)` are dropped. The category exists but contains no directly attached files; its subtree may still be useful, so children are *not* cascaded.

2. **Keyword cascade removal** — lines whose category name contains any entry from the `FILTER_KEYWORDS` list are dropped *together with all their indented subcategories*. The cascade ends when the indentation returns to the same or higher level (i.e., when we step back out of the excluded subtree).

**Filter keyword groups**

The keyword list covers the following content types, each documented with a rationale comment in the script:

| Group | Example keywords | Rationale |
|-------|-----------------|-----------|
| Artwork / cultural | `" in art"`, `"illustration"`, `"engraving"`, `"sculpture"`, `"heraldry"`, `"in mythology"` | Non-photographic; would train model on drawings and paintings |
| Anatomy / body parts | `"anatomy"`, `"skull"`, `" bones"`, `" teeth"`, `" paws"`, `"fur-skin"` | Isolated body parts are not useful for whole-animal detection |
| Maps | `"distribution map"`, `"range map"` | Geographic diagrams, not photographs |
| Philately / numismatics | `" stamps"`, `" coins"`, `"banknote"` | Animal depictions on currency / postage |
| Feces / spoor | `"feces"`, `" dung"`, `" tracks"`, `"footprint"` | Indirect signs of presence, not the animal itself |
| Museum / taxidermy | `"taxidermy"`, `"museum specimen"` | Stuffed specimens in museum settings |
| Dead / killed animals | `"(dead)"`, `" carcass"`, `"hunting trophy"`, `"roadkill"`, `"poaching"` | Not representative of live animals in their environment |
| Audio | `"audio file"`, `"pronunciation"` | Non-image content |
| Human interactions | `"riding on"`, `"(clothing)"`, `" pelts"`, `"people with"` | Human activities rather than animal appearance |
| Food / products | `"as food"`, `" meat"`, `"ivory"`, `" products"` | Processed animal products |
| Molecular / biochemical | `" proteins"`, `"ribbon diagram"` | Protein structure diagrams (common for domestic species) |
| Fossils / paleontology | `"fossil"`, `" mummies"`, `"mummified"` | Prehistoric or preserved specimens |
| Other non-photographic | `"size comparison"`, `"information graphic"`, `"in advertisement"`, `"cladogram"` | Diagrams, infographics |

**Output**

Filtered `.txt` files are written to `reports/wikimedia_categories_filtered/`. The script also prints a per-file summary of kept vs. removed category lines with a breakdown by removal reason (zero-file / keyword match / cascaded).

---

### Outputs

| Path | Description |
|------|-------------|
| `reports/wikimedia_categories/` | Raw category trees, one `.txt` per label (225 files) |
| `reports/wikimedia_categories_filtered/` | Filtered trees with non-photographic categories removed |

---

#### Step 3: `scripts/scrape_wikimedia_file_list.py`

Enumerates the actual file titles within each filtered category by calling the Wikimedia API, and writes one `.jsonl` manifest per label.

**Input:** `reports/wikimedia_categories_filtered/*.txt`

**Logic**

For each filtered `.txt` file, the script:

1. Parses the header line to extract the common name and scientific name.
2. Extracts every `Category:…` line whose file count is greater than zero.
3. For each category, paginates `action=query&generator=categorymembers&gcmtype=file&gcmlimit=500` until all file titles have been collected.
4. Deduplicates titles within a label (the same image can appear in multiple subcategories; the first-seen category is kept).
5. Writes results atomically via a `.tmp` rename to avoid partial files on interruption.

**Resume behaviour:** labels whose `.jsonl` already exists are skipped by default. Use `--force` to re-scrape.

**Output format** — one JSON record per line in `reports/wikimedia_file_manifests/{label_name}.jsonl`:

```json
{"title": "File:Panthera_leo_01.jpg", "category": "Category:Panthera_leo",
 "label": "lion", "scientific": "Panthera leo",
 "genus": "Panthera", "species": "leo", "label_dir": "lion"}
```

---

#### Step 4: `scripts/download_wikimedia_images.py`

Downloads images from the `.jsonl` manifests and writes full per-image metadata to a CSV catalog.

**Input:** `reports/wikimedia_file_manifests/*.jsonl`

**Output:**

| Path | Description |
|------|-------------|
| `data/wikimedia/images/{label_dir}/{original_filename}` | Downloaded images; original Wikimedia filenames preserved |
| `data/wikimedia/metadata.csv` | One row per downloaded image, 23 metadata columns |

**Logic**

1. Loads all `.jsonl` manifests and deduplicates file titles globally across labels (a file claimed by two labels is assigned to the first).
2. Checks which destination files already exist on disk and reports the count of pending vs. already-done images upfront.
3. Fetches `imageinfo + extmetadata` in batches of 50 titles per API call (`prop=imageinfo&iiprop=url|extmetadata|size|mime|timestamp|user|canonicaltitle`).
4. Applies two filters:
   - **License:** keeps only CC0, CC-BY (all versions), and Public Domain. All CC-BY-SA, CC-BY-NC, and unrecognised licenses are discarded. Uses the shared `WIKI_SAFE_LICENSES` set from `download_supplementary.py`.
   - **Minimum width:** images narrower than `--min-width` (default 300 px) are skipped.
5. Downloads the image to a `.tmp` file first, then renames it atomically — interrupted downloads leave no corrupt files.
6. Appends a metadata row immediately after each successful download (CSV opened in append mode; header written only when the file is new).
7. Displays a `tqdm` progress bar with ETA based on the total pending count determined at startup.

**metadata.csv columns:**

`filename`, `title`, `url`, `description_url`, `label`, `scientific_name`, `genus`, `species`, `wikimedia_category`, `label_dir`, `width`, `height`, `mime`, `size_bytes`, `upload_timestamp`, `uploader`, `license_short`, `license_url`, `artist`, `image_description`, `date_taken`, `gps_lat`, `gps_lon`

**Resume behaviour:** files already present on disk are skipped; the CSV is appended to without re-downloading.

---

### Outputs

| Path | Description |
|------|-------------|
| `reports/wikimedia_categories/` | Raw category trees, one `.txt` per label (225 files) |
| `reports/wikimedia_categories_filtered/` | Filtered trees with non-photographic categories removed |
| `reports/wikimedia_file_manifests/` | Per-label `.jsonl` file lists (Step 3 output) |
| `data/wikimedia/images/{label_dir}/` | Downloaded images, original filenames kept (Step 4 output) |
| `data/wikimedia/metadata.csv` | Full image metadata catalog (Step 4 output) |

### Usage

```bash
# Step 3 — enumerate file titles (resumable, ~0.2 s between API calls)
python scripts/scrape_wikimedia_file_list.py --rate-limit 0.2

# Step 4 — download images and write metadata (resumable, ~0.5 s between API calls)
python scripts/download_wikimedia_images.py --rate-limit 0.5

# Both scripts are safe to Ctrl+C and restart; already-completed work is skipped.
```

### Next Steps

The downloaded images feed into the MegaDetector v5 filtering pass (see [`docs/dataset-supplementation-plan.md`](../dataset-supplementation-plan.md), Step 5c):

1. Run MegaDetector v5 over `data/wikimedia/images/` to generate bounding boxes.
2. Keep only images where MegaDetector detects exactly one animal with confidence > 0.5.

This pipeline is targeted at the ~20–30 species that remain under-represented after the iNaturalist and LILA BC downloads are complete.
