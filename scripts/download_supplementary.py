"""
Download supplementary images from Open Images V7, COCO 2017, Wikimedia Commons, and Flickr.

This script fills gaps in the primary datasets (LILA BC + iNaturalist) by scraping
images from four additional sources. It searches using both common and scientific
names, expanding genus-level labels to all constituent species via the genus mapping CSV.

Sources:
    1. Open Images V7  — ~600 boxable classes, CSV-based annotations on GCS
    2. COCO 2017       — 80 categories, high-quality bounding boxes
    3. Wikimedia Commons — MediaWiki API, broad coverage of rare species
    4. Flickr           — REST API, large CC-licensed photo corpus

Output structure:
    data/supplementary/
    ├── metadata/                # Cached annotation files
    ├── images/{label_name}/     # One dir per label
    ├── download_progress.json   # Resume tracking
    └── metadata_catalog.csv     # Per-image metadata

Usage:
    python scripts/download_supplementary.py openimages [--max-per-class 200]
    python scripts/download_supplementary.py coco [--max-per-class 200]
    python scripts/download_supplementary.py wikimedia [--max-per-class 50]
    python scripts/download_supplementary.py flickr --api-key KEY [--max-per-class 100]
    python scripts/download_supplementary.py status

Requirements:
    pip install requests tqdm
"""

import argparse
import csv
import json
import os
import re
import signal
import sys
import time
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

# ── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
LABELS_225 = REPO_ROOT / "resources" / "2026-03-19_student_model_labels.txt"
GENUS_SPECIES_MAP = REPO_ROOT / "reports" / "genus_species_mapping.csv"
FAMILY_SPECIES_MAP = REPO_ROOT / "reports" / "family_species_mapping.csv"
SUPP_DIR = REPO_ROOT / "data" / "supplementary"
METADATA_DIR = SUPP_DIR / "metadata"
IMAGES_DIR = SUPP_DIR / "images"
PROGRESS_FILE = SUPP_DIR / "download_progress.json"
CATALOG_FILE = SUPP_DIR / "metadata_catalog.csv"

# ── Open Images V7 URLs ──────────────────────────────────────────────────────

OI_CLASS_DESC_URL = "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv"
OI_BBOX_URL = "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv"
OI_IMAGE_URL = "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv"
OI_IMAGE_DL_BASE = "https://s3.amazonaws.com/open-images-dataset/train"

# Manual overrides: our label name -> Open Images display name
OPENIMAGES_OVERRIDES = {
    "puma": "Cougar",
    "domestic cattle": "Cattle",
    "domestic cat": "Cat",
    "domestic dog": "Dog",
    "domestic horse": "Horse",
    "domestic sheep": "Sheep",
    "brown bear": "Bear",
    "american black bear": "Bear",
    "asiatic black bear": "Bear",
    "sloth bear": "Bear",
    "sun bear": "Bear",
    "spectacled bear": "Bear",
    "giant panda": "Giant panda",
    "grey wolf": "Dog",
    "red fox": "Fox",
    "grey fox": "Fox",
    "plains zebra": "Zebra",
    "grevy's zebra": "Zebra",
    "mountain zebra": "Zebra",
    "african elephant": "Elephant",
    "asian elephant": "Elephant",
    "hippopotamus": "Hippopotamus",
    "giraffe": "Giraffe",
    "lion": "Lion",
    "tiger": "Tiger",
    "leopard": "Leopard",
    "jaguar": "Jaguar",
    "cheetah": "Cheetah",
    "european rabbit": "Rabbit",
    "eastern cottontail": "Rabbit",
    "domestic goat": "Goat",
    "alpine ibex": "Goat",
    "wild boar": "Pig",
    "domestic pig": "Pig",
    "northern raccoon": "Raccoon",
    "koala": "Koala bear",
    "kangaroo family": "Kangaroo",
    "red kangaroo": "Kangaroo",
    "eastern grey kangaroo": "Kangaroo",
    "european roe deer": "Deer",
    "white-tailed deer": "Deer",
    "mule deer": "Deer",
    "elk": "Deer",
    "red deer": "Deer",
    "moose": "Deer",
    "reindeer": "Deer",
    "brown-throated sloth": "Sloth",
    "hoffmann's two-toed sloth": "Sloth",
    "eurasian red squirrel": "Squirrel",
    "eastern gray squirrel": "Squirrel",
    "eastern fox squirrel": "Squirrel",
    "red squirrel": "Squirrel",
    "chipmunk genus": "Squirrel",
    "squirrel family": "Squirrel",
    "wolverine": "Weasel",
    "american mink": "Weasel",
    "sea otter": "Otter",
    "eurasian otter": "Otter",
    "north american river otter": "Otter",
    "giant otter": "Otter",
    "domestic donkey": "Mule",
    "capybara": "Capybara",
}

# ── COCO Label Mapping ────────────────────────────────────────────────────────

# COCO category name -> list of our label common_names that map to it
COCO_LABEL_MAP = {
    "cat": ["domestic cat", "wild cat"],
    "dog": ["domestic dog", "dingo"],
    "horse": ["domestic horse"],
    "sheep": ["domestic sheep", "mouflon", "bighorn sheep"],
    "cow": ["domestic cattle", "yak", "domestic water buffalo"],
    "elephant": ["african elephant", "asian elephant"],
    "bear": [
        "brown bear", "american black bear", "asiatic black bear",
        "spectacled bear", "sloth bear", "sun bear", "giant panda",
    ],
    "zebra": ["plains zebra", "grevy's zebra", "mountain zebra"],
    "giraffe": ["giraffe"],
}

# COCO license IDs that are commercially safe
COCO_SAFE_LICENSES = {4, 5, 6, 7, 8}
# 4=CC-BY, 5=CC-BY-SA, 6=CC-BY-ND, 7=No known copyright, 8=US Gov

# ── Wikimedia / Flickr Config ────────────────────────────────────────────────

WIKI_API = "https://commons.wikimedia.org/w/api.php"
WIKI_SAFE_LICENSES = {
    "cc0", "cc-zero", "public domain", "pd",
    "cc by 4.0", "cc by 3.0", "cc by 2.0", "cc by 2.5",
    "cc-by-4.0", "cc-by-3.0", "cc-by-2.0", "cc-by-2.5",
    "cc by", "attribution",
}

FLICKR_API = "https://www.flickr.com/services/rest/"
# Flickr license IDs: 1=CC-BY-NC-SA, 2=CC-BY-NC, 3=CC-BY-NC-ND,
# 4=CC-BY, 5=CC-BY-SA, 6=CC-BY-ND, 7=No known, 9=CC0, 10=PDM
FLICKR_SAFE_LICENSES = "4,5,6,7,9,10"

USER_AGENT = "MasterThesis-WildlifeDetection/1.0 (https://github.com/simonhedrich; wildlife-detection-research) python-requests"


# ── Shared Utilities ─────────────────────────────────────────────────────────


def sanitize_dirname(name):
    """Convert a label name to a filesystem-safe directory name."""
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9_\- ]", "", name)
    name = name.replace(" ", "_")
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def load_target_labels(label_path):
    """Load the 225-label file. Returns list of dicts with uuid, genus, species, common_name."""
    labels = []
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(";")
            if len(parts) < 7:
                continue
            labels.append({
                "uuid": parts[0],
                "order": parts[2],
                "family": parts[3],
                "genus": parts[4],
                "species": parts[5],
                "common_name": parts[6],
            })
    return labels


def load_genus_species_mapping(csv_path):
    """Load genus_species_mapping.csv. Returns {genus_label: [species_rows]}."""
    mapping = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["genus_label"].strip()
            if label not in mapping:
                mapping[label] = []
            mapping[label].append({
                "genus_scientific": row["genus_scientific"].strip(),
                "species_scientific": row["species_scientific"].strip(),
                "species_common_name": row.get("species_common_name", "").strip(),
            })
    return mapping


def load_family_species_mapping(csv_path):
    """Load family_species_mapping.csv. Returns {family_label: [species_rows]}."""
    mapping = {}
    if not Path(csv_path).exists():
        return mapping
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["family_label"].strip()
            if label not in mapping:
                mapping[label] = []
            mapping[label].append({
                "family_scientific": row["family_scientific"].strip(),
                "genus_scientific": row.get("genus_scientific", "").strip(),
                "species_scientific": row["species_scientific"].strip(),
                "species_common_name": row.get("species_common_name", "").strip(),
            })
    return mapping


def build_search_keywords(labels, genus_map):
    """Build search keywords for each of the 225 labels.

    Returns dict: {common_name: {"search_terms": [...], "dir_name": str, ...}}
    """
    keywords = {}
    for entry in labels:
        cn = entry["common_name"]
        terms = [cn]

        # Add scientific name for species-level labels
        if entry["species"]:
            sci = f"{entry['genus']} {entry['species']}"
            terms.append(sci)
        elif entry["genus"]:
            # Genus-level label: add genus name
            terms.append(entry["genus"])

        # For genus-level labels, expand using genus_species_mapping.csv
        if cn in genus_map:
            for sp_row in genus_map[cn]:
                if sp_row["species_scientific"] and sp_row["species_scientific"] not in terms:
                    terms.append(sp_row["species_scientific"])
                if sp_row["species_common_name"] and sp_row["species_common_name"] not in terms:
                    terms.append(sp_row["species_common_name"])
        # Also try matching by genus scientific name for family-level labels
        elif entry["genus"] and not entry["species"]:
            for label_key, species_list in genus_map.items():
                if species_list and species_list[0]["genus_scientific"].lower() == entry["genus"].lower():
                    for sp_row in species_list:
                        if sp_row["species_scientific"] and sp_row["species_scientific"] not in terms:
                            terms.append(sp_row["species_scientific"])
                        if sp_row["species_common_name"] and sp_row["species_common_name"] not in terms:
                            terms.append(sp_row["species_common_name"])

        keywords[cn] = {
            "uuid": entry["uuid"],
            "common_name": cn,
            "genus": entry["genus"],
            "species": entry["species"],
            "search_terms": terms,
            "dir_name": sanitize_dirname(cn),
        }
    return keywords


class ProgressTracker:
    """Track download progress per source and label for resume support.

    Structure: {source: {label: {count: N, image_ids: [id1, id2, ...]}}}
    Saves atomically to avoid corruption on interrupt.
    """

    def __init__(self, path):
        self.path = Path(path)
        self._dirty = 0
        self._save_interval = 50
        if self.path.exists():
            with open(self.path) as f:
                self.data = json.load(f)
        else:
            self.data = {}

    def _ensure(self, source, label):
        if source not in self.data:
            self.data[source] = {}
        if label not in self.data[source]:
            self.data[source][label] = {"count": 0, "image_ids": []}

    def get_count(self, source, label):
        self._ensure(source, label)
        return self.data[source][label]["count"]

    def get_image_ids(self, source, label):
        self._ensure(source, label)
        return set(self.data[source][label]["image_ids"])

    def is_label_done(self, source, label, max_per_class):
        return self.get_count(source, label) >= max_per_class

    def record_download(self, source, label, image_id):
        self._ensure(source, label)
        self.data[source][label]["image_ids"].append(str(image_id))
        self.data[source][label]["count"] += 1
        self._dirty += 1
        if self._dirty >= self._save_interval:
            self.save()

    def save(self):
        if self._dirty == 0:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self.data, f)
        os.replace(tmp, self.path)
        self._dirty = 0

    def get_summary(self):
        summary = {}
        for source, labels in self.data.items():
            total_images = sum(v["count"] for v in labels.values())
            summary[source] = {
                "labels_hit": len([v for v in labels.values() if v["count"] > 0]),
                "total_images": total_images,
            }
        return summary


def append_to_catalog(rows):
    """Append metadata rows to the catalog CSV."""
    CATALOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    write_header = not CATALOG_FILE.exists()
    with open(CATALOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "source", "label", "filename", "license", "scientific_name",
                "url", "search_term", "bbox",
            ])
        for row in rows:
            writer.writerow(row)


def download_image(url, dest_path, timeout=30):
    """Download a single image. Returns True on success."""
    if dest_path.exists():
        return True
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        if len(resp.content) < 100:
            return False
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "wb") as f:
            f.write(resp.content)
        return True
    except Exception:
        return False


class RateLimiter:
    """Simple rate limiter with minimum interval between calls."""

    def __init__(self, min_interval=0.5):
        self.min_interval = min_interval
        self.last_request = 0.0

    def wait(self):
        elapsed = time.time() - self.last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request = time.time()


# Global progress tracker for signal handler
_progress = None


def _signal_handler(signum, frame):
    if _progress is not None:
        print("\nInterrupted — saving progress...")
        _progress.save()
    sys.exit(1)


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ── Open Images V7 ───────────────────────────────────────────────────────────


def _download_oi_metadata():
    """Download Open Images metadata CSVs if not present."""
    oi_dir = METADATA_DIR / "openimages"
    oi_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "class-descriptions.csv": OI_CLASS_DESC_URL,
        "train-annotations-bbox.csv": OI_BBOX_URL,
        "train-images.csv": OI_IMAGE_URL,
    }

    for name, url in files.items():
        dest = oi_dir / name
        if dest.exists():
            print(f"  Already have: {name}")
            continue
        print(f"  Downloading {name}...")
        resp = requests.get(url, stream=True, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=name
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                pbar.update(len(chunk))
    return oi_dir


def _load_oi_class_map(oi_dir):
    """Load Open Images class descriptions. Returns {display_name_lower: MID}."""
    class_map = {}
    with open(oi_dir / "class-descriptions.csv", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                mid, display_name = row[0].strip(), row[1].strip()
                class_map[display_name.lower()] = mid
    return class_map


def _map_labels_to_oi(keywords, class_map):
    """Map our 225 labels to Open Images MIDs.

    Returns {our_label_name: MID} and logs unmapped labels.
    """
    label_to_mid = {}
    for label_name, info in keywords.items():
        # Check manual override first
        override = OPENIMAGES_OVERRIDES.get(label_name)
        if override and override.lower() in class_map:
            label_to_mid[label_name] = class_map[override.lower()]
            continue

        # Try exact match on common name
        if label_name.lower() in class_map:
            label_to_mid[label_name] = class_map[label_name.lower()]
            continue

        # Try scientific name
        sci = f"{info['genus']} {info['species']}".strip().lower()
        if sci in class_map:
            label_to_mid[label_name] = class_map[sci]
            continue

    return label_to_mid


def cmd_openimages(args):
    """Download images from Open Images V7."""
    global _progress
    progress = ProgressTracker(PROGRESS_FILE)
    _progress = progress

    print("── Open Images V7 ──\n")

    # Load labels and keywords
    labels = load_target_labels(LABELS_225)
    genus_map = load_genus_species_mapping(GENUS_SPECIES_MAP)
    keywords = build_search_keywords(labels, genus_map)
    print(f"Loaded {len(keywords)} labels")

    # Download metadata
    print("\n── Downloading metadata ──")
    oi_dir = _download_oi_metadata()

    # Map labels to OI classes
    print("\n── Mapping labels to Open Images classes ──")
    class_map = _load_oi_class_map(oi_dir)
    label_to_mid = _map_labels_to_oi(keywords, class_map)
    print(f"  Mapped {len(label_to_mid)} of {len(keywords)} labels to Open Images classes")

    if not label_to_mid:
        print("  No labels mapped. Exiting.")
        return

    # Reverse map: MID -> [label_names]
    mid_to_labels = {}
    for label_name, mid in label_to_mid.items():
        mid_to_labels.setdefault(mid, []).append(label_name)

    # Parse bbox CSV to find images per MID
    print("\n── Parsing bounding box annotations ──")
    bbox_path = oi_dir / "train-annotations-bbox.csv"
    # Collect image_ids per label (with bbox info)
    label_images = {label: [] for label in label_to_mid}
    target_mids = set(label_to_mid.values())

    row_count = 0
    with open(bbox_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_count += 1
            mid = row.get("LabelName", "")
            if mid not in target_mids:
                continue
            image_id = row.get("ImageID", "")
            bbox = f"{row.get('XMin','')},{row.get('YMin','')},{row.get('XMax','')},{row.get('YMax','')}"

            for label_name in mid_to_labels.get(mid, []):
                label_images[label_name].append({
                    "image_id": image_id,
                    "bbox": bbox,
                })
            if row_count % 5_000_000 == 0:
                print(f"    ...{row_count:,} bbox rows scanned")

    print(f"  Scanned {row_count:,} bbox rows")

    # Deduplicate and limit per label
    max_per = args.max_per_class
    to_download = []
    for label_name, images in label_images.items():
        existing_ids = progress.get_image_ids("openimages", label_name)
        if progress.is_label_done("openimages", label_name, max_per):
            continue
        seen = set()
        needed = max_per - progress.get_count("openimages", label_name)
        for img in images:
            iid = img["image_id"]
            if iid in seen or iid in existing_ids:
                continue
            seen.add(iid)
            to_download.append((label_name, img))
            if len(seen) >= needed:
                break

    print(f"\n  Images to download: {len(to_download):,}")

    if not to_download:
        print("  Nothing to download.")
        progress.save()
        return

    # Download
    successes = 0
    failures = 0

    def _dl_oi(item):
        label_name, img = item
        iid = img["image_id"]
        dir_name = keywords[label_name]["dir_name"]
        dest = IMAGES_DIR / dir_name / f"oi_{iid}.jpg"
        url = f"{OI_IMAGE_DL_BASE}/{iid}.jpg"
        ok = download_image(url, dest)
        return ok, label_name, iid, img.get("bbox", ""), url

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_dl_oi, item): item for item in to_download}
        with tqdm(total=len(futures), desc="Open Images", unit="img") as pbar:
            for future in as_completed(futures):
                ok, label_name, iid, bbox, url = future.result()
                if ok:
                    progress.record_download("openimages", label_name, iid)
                    append_to_catalog([(
                        "openimages", label_name, f"oi_{iid}.jpg", "CC-BY-2.0",
                        "", url, "", bbox,
                    )])
                    successes += 1
                else:
                    failures += 1
                pbar.update(1)

    progress.save()
    print(f"\nDone: {successes:,} downloaded, {failures:,} failed")


# ── COCO 2017 ────────────────────────────────────────────────────────────────


def _download_coco_metadata():
    """Download and extract COCO 2017 annotations if not present."""
    coco_dir = METADATA_DIR / "coco"
    coco_dir.mkdir(parents=True, exist_ok=True)

    train_json = coco_dir / "instances_train2017.json"
    val_json = coco_dir / "instances_val2017.json"

    if train_json.exists() and val_json.exists():
        print("  Already have COCO annotations")
        return coco_dir

    zip_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    zip_path = coco_dir / "annotations_trainval2017.zip"

    if not zip_path.exists():
        print("  Downloading COCO annotations (~252 MB)...")
        resp = requests.get(zip_url, stream=True, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(zip_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc="COCO annotations"
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                pbar.update(len(chunk))

    print("  Extracting...")
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if "instances_" in member and member.endswith(".json"):
                # Extract to coco_dir with flat name
                fname = Path(member).name
                dest = coco_dir / fname
                if not dest.exists():
                    with zf.open(member) as src, open(dest, "wb") as dst:
                        dst.write(src.read())
                    print(f"    Extracted: {fname}")

    return coco_dir


def cmd_coco(args):
    """Download images from COCO 2017."""
    global _progress
    progress = ProgressTracker(PROGRESS_FILE)
    _progress = progress

    print("── COCO 2017 ──\n")

    labels = load_target_labels(LABELS_225)
    label_by_cn = {e["common_name"]: e for e in labels}
    print(f"Loaded {len(labels)} labels")

    # Download metadata
    print("\n── Downloading metadata ──")
    coco_dir = _download_coco_metadata()

    # Build reverse map: our label -> COCO category name(s)
    our_label_to_coco = {}
    for coco_cat, our_labels in COCO_LABEL_MAP.items():
        for our_label in our_labels:
            if our_label in label_by_cn:
                our_label_to_coco.setdefault(our_label, []).append(coco_cat)

    print(f"  {len(our_label_to_coco)} of our labels have COCO mappings")

    # Load COCO annotation JSONs
    all_images = {}   # image_id -> image_info
    all_licenses = {} # license_id -> license_info
    label_image_ids = {label: [] for label in our_label_to_coco}

    for split in ["train2017", "val2017"]:
        json_path = coco_dir / f"instances_{split}.json"
        if not json_path.exists():
            print(f"  WARNING: {json_path.name} not found, skipping")
            continue

        print(f"  Loading {split}...")
        with open(json_path) as f:
            coco_data = json.load(f)

        # Build category name -> id map
        cat_name_to_id = {}
        for cat in coco_data["categories"]:
            cat_name_to_id[cat["name"]] = cat["id"]

        # License map
        for lic in coco_data.get("licenses", []):
            all_licenses[lic["id"]] = lic

        # Image map
        for img in coco_data["images"]:
            all_images[img["id"]] = {
                "file_name": img["file_name"],
                "url": img.get("coco_url", ""),
                "flickr_url": img.get("flickr_url", ""),
                "license": img.get("license", 0),
                "split": split,
            }

        # Find matching annotations
        target_cat_ids = {}
        for our_label, coco_cats in our_label_to_coco.items():
            for coco_cat in coco_cats:
                if coco_cat in cat_name_to_id:
                    target_cat_ids.setdefault(cat_name_to_id[coco_cat], []).append(our_label)

        for ann in coco_data["annotations"]:
            cat_id = ann["category_id"]
            if cat_id not in target_cat_ids:
                continue
            img_id = ann["image_id"]
            bbox = ann.get("bbox", [])
            bbox_str = ",".join(str(round(v, 1)) for v in bbox) if bbox else ""

            for our_label in target_cat_ids[cat_id]:
                label_image_ids[our_label].append({
                    "image_id": img_id,
                    "bbox": bbox_str,
                })

    # Filter by license and deduplicate
    max_per = args.max_per_class
    to_download = []

    for our_label, images in label_image_ids.items():
        existing_ids = progress.get_image_ids("coco", our_label)
        if progress.is_label_done("coco", our_label, max_per):
            continue

        seen = set()
        needed = max_per - progress.get_count("coco", our_label)
        for img in images:
            iid = img["image_id"]
            if iid in seen or str(iid) in existing_ids:
                continue
            img_info = all_images.get(iid)
            if not img_info:
                continue
            if img_info["license"] not in COCO_SAFE_LICENSES:
                continue
            seen.add(iid)
            to_download.append((our_label, img, img_info))
            if len(seen) >= needed:
                break

    print(f"\n  Images to download: {len(to_download):,}")

    if not to_download:
        print("  Nothing to download.")
        progress.save()
        return

    # Download
    keywords = build_search_keywords(labels, load_genus_species_mapping(GENUS_SPECIES_MAP))
    successes = 0
    failures = 0

    def _dl_coco(item):
        our_label, img, img_info = item
        iid = img["image_id"]
        dir_name = keywords[our_label]["dir_name"]
        dest = IMAGES_DIR / dir_name / f"coco_{iid}.jpg"
        url = img_info["url"] or img_info["flickr_url"]
        ok = download_image(url, dest)
        lic_id = img_info.get("license", 0)
        lic_name = all_licenses.get(lic_id, {}).get("name", f"license-{lic_id}")
        return ok, our_label, iid, img.get("bbox", ""), url, lic_name

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_dl_coco, item): item for item in to_download}
        with tqdm(total=len(futures), desc="COCO 2017", unit="img") as pbar:
            for future in as_completed(futures):
                ok, our_label, iid, bbox, url, lic_name = future.result()
                if ok:
                    progress.record_download("coco", our_label, iid)
                    append_to_catalog([(
                        "coco", our_label, f"coco_{iid}.jpg", lic_name,
                        "", url, "", bbox,
                    )])
                    successes += 1
                else:
                    failures += 1
                pbar.update(1)

    progress.save()
    print(f"\nDone: {successes:,} downloaded, {failures:,} failed")


# ── Wikimedia Commons ─────────────────────────────────────────────────────────


def _wiki_search(term, rate_limiter, limit=50):
    """Search Wikimedia Commons for images matching a term. Returns list of titles."""
    rate_limiter.wait()
    params = {
        "action": "query",
        "list": "search",
        "srsearch": f'"{term}" filetype:bitmap',
        "srnamespace": 6,
        "srlimit": limit,
        "format": "json",
    }
    try:
        resp = requests.get(WIKI_API, params=params, timeout=30,
                            headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        data = resp.json()
        results = data.get("query", {}).get("search", [])
        return [r["title"] for r in results]
    except Exception as e:
        print(f"    Wiki search error for '{term}': {e}")
        return []


def _wiki_get_image_info(titles, rate_limiter):
    """Get image info (URL, license, size) for a batch of Wikimedia file titles.

    Returns list of dicts with keys: title, url, license, width, page_id, descriptionurl
    """
    if not titles:
        return []
    rate_limiter.wait()
    params = {
        "action": "query",
        "titles": "|".join(titles[:50]),
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|size|mime",
        "iiurlwidth": 1024,
        "format": "json",
    }
    try:
        resp = requests.get(WIKI_API, params=params, timeout=30,
                            headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        data = resp.json()
        pages = data.get("query", {}).get("pages", {})
    except Exception as e:
        print(f"    Wiki imageinfo error: {e}")
        return []

    results = []
    for page_id, page in pages.items():
        if int(page_id) < 0:
            continue
        ii = page.get("imageinfo", [{}])
        if not ii:
            continue
        info = ii[0]
        mime = info.get("mime", "")
        if mime not in ("image/jpeg", "image/png"):
            continue
        width = info.get("width", 0)
        if width < 300:
            continue

        # Extract license
        ext = info.get("extmetadata", {})
        lic_short = ext.get("LicenseShortName", {}).get("value", "").strip()
        if not lic_short:
            continue
        if lic_short.lower() not in WIKI_SAFE_LICENSES:
            continue

        url = info.get("thumburl") or info.get("url", "")
        if not url:
            continue

        results.append({
            "page_id": str(page_id),
            "title": page.get("title", ""),
            "url": url,
            "license": lic_short,
            "width": width,
            "descriptionurl": info.get("descriptionurl", ""),
        })
    return results


def cmd_wikimedia(args):
    """Download images from Wikimedia Commons."""
    global _progress
    progress = ProgressTracker(PROGRESS_FILE)
    _progress = progress
    rate_limiter = RateLimiter(min_interval=args.rate_limit)

    print("── Wikimedia Commons ──\n")

    labels = load_target_labels(LABELS_225)
    genus_map = load_genus_species_mapping(GENUS_SPECIES_MAP)
    keywords = build_search_keywords(labels, genus_map)
    print(f"Loaded {len(keywords)} labels")

    max_per = args.max_per_class

    # Sort labels by least coverage first
    label_order = sorted(
        keywords.keys(),
        key=lambda l: progress.get_count("wikimedia", l),
    )

    total_downloaded = 0
    total_skipped = 0

    for label_name in tqdm(label_order, desc="Labels", unit="label"):
        if progress.is_label_done("wikimedia", label_name, max_per):
            continue

        info = keywords[label_name]
        dir_name = info["dir_name"]
        existing_ids = progress.get_image_ids("wikimedia", label_name)
        current_count = progress.get_count("wikimedia", label_name)
        needed = max_per - current_count

        for term in info["search_terms"]:
            if needed <= 0:
                break

            titles = _wiki_search(term, rate_limiter)
            if not titles:
                continue

            # Batch get image info
            for i in range(0, len(titles), 50):
                batch = titles[i:i + 50]
                image_infos = _wiki_get_image_info(batch, rate_limiter)

                for img_info in image_infos:
                    if needed <= 0:
                        break
                    pid = img_info["page_id"]
                    if pid in existing_ids:
                        total_skipped += 1
                        continue

                    ext = "jpg" if "jpeg" in img_info["url"].lower() or img_info["url"].lower().endswith(".jpg") else "png"
                    fname = f"wiki_{pid}.{ext}"
                    dest = IMAGES_DIR / dir_name / fname

                    ok = download_image(img_info["url"], dest)
                    if ok:
                        progress.record_download("wikimedia", label_name, pid)
                        existing_ids.add(pid)
                        append_to_catalog([(
                            "wikimedia", label_name, fname, img_info["license"],
                            "", img_info["url"], term, "",
                        )])
                        needed -= 1
                        total_downloaded += 1

    progress.save()
    print(f"\nDone: {total_downloaded:,} downloaded, {total_skipped:,} duplicates skipped")


# ── Flickr ────────────────────────────────────────────────────────────────────


def _flickr_search(api_key, term, rate_limiter, page=1, per_page=100):
    """Search Flickr for CC-licensed photos matching a term.

    Returns list of photo dicts with id, url, license, owner.
    """
    rate_limiter.wait()
    params = {
        "method": "flickr.photos.search",
        "api_key": api_key,
        "text": term,
        "media": "photos",
        "content_type": 1,
        "license": FLICKR_SAFE_LICENSES,
        "extras": "url_l,url_m,url_o,license,owner_name,tags",
        "per_page": per_page,
        "page": page,
        "format": "json",
        "nojsoncallback": 1,
        "sort": "relevance",
    }
    try:
        resp = requests.get(FLICKR_API, params=params, timeout=30,
                            headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        data = resp.json()
        if data.get("stat") != "ok":
            print(f"    Flickr API error for '{term}': {data.get('message', 'unknown')}")
            return [], 0
        photos = data.get("photos", {})
        total_pages = photos.get("pages", 0)
        return photos.get("photo", []), total_pages
    except Exception as e:
        print(f"    Flickr search error for '{term}': {e}")
        return [], 0


LICENSE_NAMES = {
    0: "All Rights Reserved",
    1: "CC-BY-NC-SA-2.0",
    2: "CC-BY-NC-2.0",
    3: "CC-BY-NC-ND-2.0",
    4: "CC-BY-2.0",
    5: "CC-BY-SA-2.0",
    6: "CC-BY-ND-2.0",
    7: "No known copyright",
    8: "US Government Work",
    9: "CC0-1.0",
    10: "Public Domain Mark",
}


def cmd_flickr(args):
    """Download images from Flickr."""
    global _progress
    progress = ProgressTracker(PROGRESS_FILE)
    _progress = progress
    rate_limiter = RateLimiter(min_interval=1.0)

    api_key = args.api_key
    if not api_key:
        print("ERROR: --api-key is required for Flickr downloads.")
        print("  Get a free key at: https://www.flickr.com/services/apps/create/")
        sys.exit(1)

    print("── Flickr ──\n")

    labels = load_target_labels(LABELS_225)
    genus_map = load_genus_species_mapping(GENUS_SPECIES_MAP)
    keywords = build_search_keywords(labels, genus_map)
    print(f"Loaded {len(keywords)} labels")

    max_per = args.max_per_class

    # Sort labels by least coverage first
    label_order = sorted(
        keywords.keys(),
        key=lambda l: progress.get_count("flickr", l),
    )

    total_downloaded = 0
    total_skipped = 0

    for label_name in tqdm(label_order, desc="Labels", unit="label"):
        if progress.is_label_done("flickr", label_name, max_per):
            continue

        info = keywords[label_name]
        dir_name = info["dir_name"]
        existing_ids = progress.get_image_ids("flickr", label_name)
        current_count = progress.get_count("flickr", label_name)
        needed = max_per - current_count

        for term in info["search_terms"]:
            if needed <= 0:
                break

            photos, total_pages = _flickr_search(api_key, term, rate_limiter)
            if not photos:
                continue

            # Process up to 3 pages of results per search term
            max_pages = min(total_pages, 3)
            page = 1

            while page <= max_pages and needed > 0:
                if page > 1:
                    photos, _ = _flickr_search(api_key, term, rate_limiter, page=page)

                for photo in photos:
                    if needed <= 0:
                        break

                    photo_id = str(photo.get("id", ""))
                    if not photo_id or photo_id in existing_ids:
                        total_skipped += 1
                        continue

                    # Get download URL: prefer url_l (1024px), fallback url_m (500px)
                    url = photo.get("url_l") or photo.get("url_m") or photo.get("url_o")
                    if not url:
                        continue

                    fname = f"flickr_{photo_id}.jpg"
                    dest = IMAGES_DIR / dir_name / fname

                    ok = download_image(url, dest)
                    if ok:
                        lic_id = int(photo.get("license", 0))
                        lic_name = LICENSE_NAMES.get(lic_id, f"license-{lic_id}")
                        progress.record_download("flickr", label_name, photo_id)
                        existing_ids.add(photo_id)
                        append_to_catalog([(
                            "flickr", label_name, fname, lic_name,
                            "", url, term, "",
                        )])
                        needed -= 1
                        total_downloaded += 1

                page += 1

    progress.save()
    print(f"\nDone: {total_downloaded:,} downloaded, {total_skipped:,} duplicates skipped")


# ── Status Command ────────────────────────────────────────────────────────────


def cmd_status(args):
    """Show download coverage report."""
    print("── Supplementary Image Coverage ──\n")

    if not PROGRESS_FILE.exists():
        print("No downloads yet. Run a source command first.")
        return

    progress = ProgressTracker(PROGRESS_FILE)
    labels = load_target_labels(LABELS_225)
    all_label_names = {e["common_name"] for e in labels}

    summary = progress.get_summary()

    # Per-source table
    print(f"{'Source':<16} {'Labels hit':>12} {'Total images':>14}")
    print("─" * 44)
    combined_labels = set()
    combined_total = 0
    for source in ["openimages", "coco", "wikimedia", "flickr"]:
        if source in summary:
            s = summary[source]
            print(f"{source:<16} {s['labels_hit']:>12} {s['total_images']:>14,}")
            combined_total += s["total_images"]
            # Count labels with > 0 images
            for label, info in progress.data.get(source, {}).items():
                if info["count"] > 0:
                    combined_labels.add(label)
        else:
            print(f"{source:<16} {'—':>12} {'—':>14}")

    print("─" * 44)
    print(f"{'Combined':<16} {len(combined_labels):>12} {combined_total:>14,}")

    # Labels with 0 images
    uncovered = all_label_names - combined_labels
    if uncovered:
        print(f"\nLabels with 0 images ({len(uncovered)}):")
        for name in sorted(uncovered)[:30]:
            print(f"  - {name}")
        if len(uncovered) > 30:
            print(f"  ... and {len(uncovered) - 30} more")

    # Labels with < 10 images
    low_coverage = []
    for label_name in all_label_names:
        total = 0
        for source in ["openimages", "coco", "wikimedia", "flickr"]:
            total += progress.get_count(source, label_name)
        if 0 < total < 10:
            low_coverage.append((label_name, total))

    if low_coverage:
        low_coverage.sort(key=lambda x: x[1])
        print(f"\nLabels with < 10 images ({len(low_coverage)}):")
        for name, count in low_coverage[:30]:
            print(f"  - {name} ({count})")
        if len(low_coverage) > 30:
            print(f"  ... and {len(low_coverage) - 30} more")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Download supplementary images from Open Images, COCO, Wikimedia, and Flickr",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download from COCO 2017 (fast, small source)
    python scripts/download_supplementary.py coco --max-per-class 200

    # Download from Open Images V7 (needs ~2GB metadata download first)
    python scripts/download_supplementary.py openimages --max-per-class 200

    # Download from Wikimedia Commons (rate-limited, good for rare species)
    python scripts/download_supplementary.py wikimedia --max-per-class 50

    # Download from Flickr (requires free API key)
    python scripts/download_supplementary.py flickr --api-key YOUR_KEY --max-per-class 100

    # Check coverage across all sources
    python scripts/download_supplementary.py status
        """,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # openimages
    p_oi = subparsers.add_parser("openimages", help="Download from Open Images V7")
    p_oi.add_argument("--max-per-class", type=int, default=500)
    p_oi.add_argument("--workers", type=int, default=8)

    # coco
    p_coco = subparsers.add_parser("coco", help="Download from COCO 2017")
    p_coco.add_argument("--max-per-class", type=int, default=500)
    p_coco.add_argument("--workers", type=int, default=8)

    # wikimedia
    p_wiki = subparsers.add_parser("wikimedia", help="Download from Wikimedia Commons")
    p_wiki.add_argument("--max-per-class", type=int, default=500)
    p_wiki.add_argument("--workers", type=int, default=4)
    p_wiki.add_argument("--rate-limit", type=float, default=0.5,
                        help="Seconds between API calls (default: 0.5)")

    # flickr
    p_flickr = subparsers.add_parser("flickr", help="Download from Flickr")
    p_flickr.add_argument("--api-key", type=str, default=os.environ.get("FLICKR_API_KEY", ""),
                          help="Flickr API key (or set FLICKR_API_KEY env var)")
    p_flickr.add_argument("--max-per-class", type=int, default=100)
    p_flickr.add_argument("--workers", type=int, default=8)

    # status
    subparsers.add_parser("status", help="Show coverage report")

    args = parser.parse_args()

    if args.command == "openimages":
        cmd_openimages(args)
    elif args.command == "coco":
        cmd_coco(args)
    elif args.command == "wikimedia":
        cmd_wikimedia(args)
    elif args.command == "flickr":
        cmd_flickr(args)
    elif args.command == "status":
        cmd_status(args)


if __name__ == "__main__":
    main()



"""
python scripts/download_supplementary.py openimages
── Open Images V7 ──

Loaded 225 labels

── Downloading metadata ──
  Already have: class-descriptions.csv
  Downloading train-annotations-bbox.csv...
train-annotations-bbox.csv: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2.26G/2.26G [01:16<00:00, 29.6MB/s]
  Downloading train-images.csv...
train-images.csv: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 638M/638M [00:19<00:00, 32.4MB/s]

── Mapping labels to Open Images classes ──
  Mapped 107 of 225 labels to Open Images classes

── Parsing bounding box annotations ──
  Scanned 14,610,229 bbox rows

  Images to download: 25,802
Open Images: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 25802/25802 [38:28<00:00, 11.18img/s]

Done: 25,800 downloaded, 2 failed
"""