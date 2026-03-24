"""
Download and process LILA BC camera trap datasets for the student model.

Targets three datasets (all CDLA-Permissive — commercially safe):
  1. Snapshot Serengeti v2.1  — African species, 7.1M images
  2. Snapshot Safari 2024 Expansion — African species, 4M images
  3. WCS Camera Traps — Global, 1.4M images, 375k bounding boxes

Pipeline:
  1. Download annotation JSON files (small, ~100MB total)
  2. Load the target label set (225-class or 480-class)
  3. Build a taxonomy mapping from LILA category names → target labels
  4. Filter annotations to images containing target species only
  5. Download only the filtered images (avoids downloading millions of blanks)
  6. Export a unified annotation file in YOLO format

Usage:
    # Step 1: Download metadata and build the filtered image list
    python scripts/download_lila_bc.py metadata

    # Step 2: Download the filtered images
    python scripts/download_lila_bc.py download [--max-per-class 500] [--dataset serengeti|safari|wcs|all]

    # Step 3: Export to YOLO format
    python scripts/download_lila_bc.py export [--label-set 225|480]

Requirements:
    pip install requests tqdm
"""

import argparse
import csv
import json
import os
import sys
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin

import ijson
import requests
from tqdm import tqdm

# ── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
LABELS_225 = REPO_ROOT / "resources" / "2026-03-19_student_model_labels.txt"
LABELS_480 = REPO_ROOT / "resources" / "2026-03-20_student_model_labels_extended.txt"
LILA_DIR = REPO_ROOT / "data" / "lila_bc"
METADATA_DIR = LILA_DIR / "metadata"
IMAGES_DIR = LILA_DIR / "images"
EXPORT_DIR = LILA_DIR / "yolo_export"
TAXONOMY_MAPPING_PATH = METADATA_DIR / "lila-taxonomy-mapping_release.csv"

# ── Dataset Registry ─────────────────────────────────────────────────────────

DATASETS = {
    "serengeti": {
        "name": "Snapshot Serengeti v2.1",
        "taxonomy_name": "Snapshot Serengeti",  # name used in LILA taxonomy CSV
        "metadata_url": "https://storage.googleapis.com/public-datasets-lila/snapshotserengeti-v-2-0/SnapshotSerengeti_S1-11_v2_1.json.zip",
        "bbox_url": "https://storage.googleapis.com/public-datasets-lila/snapshotserengeti-v-2-0/SnapshotSerengetiBboxes_20190903.json.zip",
        "image_base_url": "https://lilawildlife.blob.core.windows.net/lila-wildlife/snapshotserengeti-unzipped/",
        "metadata_file": "SnapshotSerengeti_S1-11_v2.1.json",
        "bbox_file": "SnapshotSerengetiBboxes_20190903.json",
    },
    "safari": {
        "name": "Snapshot Safari 2024 Expansion",
        "taxonomy_name": "Snapshot Safari",  # covers all Snapshot Safari sub-datasets
        "metadata_url": "https://storage.googleapis.com/public-datasets-lila/snapshot-safari-2024-expansion/snapshot_safari_2024_metadata.zip",
        "bbox_url": None,
        "image_base_url": "https://lilawildlife.blob.core.windows.net/lila-wildlife/snapshot-safari-2024-expansion/",
        "metadata_file": "snapshot_safari_2024_metadata.json",
        "bbox_file": None,
    },
    "wcs": {
        "name": "WCS Camera Traps",
        "taxonomy_name": "WCS Camera Traps",
        "metadata_url": "https://storage.googleapis.com/public-datasets-lila/wcs/wcs_camera_traps.json.zip",
        "bbox_url": "https://storage.googleapis.com/public-datasets-lila/wcs/wcs_20220205_bboxes_with_classes.zip",
        "image_base_url": "https://lilawildlife.blob.core.windows.net/lila-wildlife/wcs-unzipped/",
        "metadata_file": "wcs_camera_traps.json",
        "bbox_file": "wcs_20220205_bboxes_with_classes.json",
    },
}

TAXONOMY_MAPPING_URL = (
    "https://lila.science/public/lila-taxonomy-mapping_release.csv"
)


# ── Label Set Loading ────────────────────────────────────────────────────────


def load_target_labels(label_path):
    """Load a label file and return lookup structures for matching."""
    labels = {}
    genus_lookup = {}  # genus -> label (genus-level entries)
    family_lookup = {}  # family -> label (family-level entries)
    species_lookup = {}  # "genus species" -> label (species-level entries)

    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(";")
            if len(parts) < 7:
                continue

            entry = {
                "uuid": parts[0],
                "genus": parts[4],
                "species": parts[5],
                "common_name": parts[6],
                "full_line": line,
            }

            labels[parts[0]] = entry

            if parts[5]:  # species level
                species_lookup[f"{parts[4]} {parts[5]}".lower()] = entry
            elif parts[4]:  # genus level
                genus_lookup[parts[4].lower()] = entry
            elif parts[3]:  # family level
                family_lookup[parts[3].lower()] = entry

    return {
        "labels": labels,
        "species": species_lookup,
        "genus": genus_lookup,
        "family": family_lookup,
    }


# ── Taxonomy Mapping ─────────────────────────────────────────────────────────


def load_taxonomy_mapping(csv_path):
    """Load the LILA taxonomy mapping CSV.

    Returns a dict mapping (dataset_name_lower, query_lower) -> row dict.
    The row dict contains: dataset_name, query, taxonomy_level,
    scientific_name, common_name, genus, species, family, order, ...
    """
    mapping = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ds = row.get("dataset_name", "").strip()
            cat = row.get("query", "").strip()
            if ds and cat:
                mapping[(ds.lower(), cat.lower())] = row
    return mapping


def match_category_to_label(category_name, scientific_name, target_lookup):
    """Try to match a LILA category to a target label.

    Args:
        category_name: The dataset's local category name (e.g., "impala")
        scientific_name: The mapped scientific name (e.g., "aepyceros melampus")
        target_lookup: Dict with 'species', 'genus', 'family' sub-dicts

    Returns:
        (matched_label, match_type) or (None, None)
    """
    sci = scientific_name.lower().strip() if scientific_name else ""

    # 1. Exact species match
    if sci and sci in target_lookup["species"]:
        return target_lookup["species"][sci], "species"

    # 2. Genus match (first word of scientific name)
    if sci:
        genus = sci.split()[0] if " " in sci else sci
        if genus in target_lookup["genus"]:
            return target_lookup["genus"][genus], "genus"

    # 3. Common name heuristic — try matching category_name to common names
    cat_lower = category_name.lower().strip()
    for level in ("species", "genus", "family"):
        for key, label in target_lookup[level].items():
            if label["common_name"].lower() == cat_lower:
                return label, f"common_name_{level}"

    return None, None


# ── Metadata Download ────────────────────────────────────────────────────────


def download_file(url, dest_path, description=""):
    """Download a file with progress bar. Skips if already exists."""
    if dest_path.exists():
        print(f"  ✓ Already downloaded: {dest_path.name}")
        return

    print(f"  ↓ Downloading {description or dest_path.name}...")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    with open(dest_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest_path.name
    ) as pbar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


def download_and_extract_zip(url, dest_dir, description=""):
    """Download a ZIP and extract its contents. Skips if already done."""
    zip_name = url.split("/")[-1]
    zip_path = dest_dir / zip_name

    download_file(url, zip_path, description)

    # Check if THIS zip's contents are already extracted
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        all_extracted = all((dest_dir / m).exists() for m in members)
        if all_extracted:
            print(f"  ✓ Already extracted: {members}")
            return

        print(f"  ⇢ Extracting {zip_name}...")
        zf.extractall(dest_dir)
    print(f"  ✓ Extracted {members} to {dest_dir}")


def cmd_metadata(args):
    """Download all metadata files and the taxonomy mapping."""
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    # Taxonomy mapping
    print("\n── Taxonomy Mapping ──")
    download_file(
        TAXONOMY_MAPPING_URL,
        TAXONOMY_MAPPING_PATH,
        "LILA taxonomy mapping CSV",
    )

    # Dataset metadata
    datasets_to_process = (
        [args.dataset] if args.dataset != "all" else DATASETS.keys()
    )

    for ds_key in datasets_to_process:
        ds = DATASETS[ds_key]
        ds_dir = METADATA_DIR / ds_key
        ds_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n── {ds['name']} ──")

        # Main metadata
        download_and_extract_zip(
            ds["metadata_url"], ds_dir, f"{ds['name']} metadata"
        )

        # Bounding boxes (if available)
        if ds["bbox_url"]:
            download_and_extract_zip(
                ds["bbox_url"], ds_dir, f"{ds['name']} bounding boxes"
            )

    # Build filtered image list
    print("\n── Building Filtered Image List ──")
    build_filtered_list(args)


def _sanitize_json(filepath):
    """Replace NaN/Infinity with null in a JSON file (in-place).

    LILA BC datasets exported from Python/pandas sometimes contain bare NaN
    or Infinity tokens which are invalid JSON.  This does a single-pass
    regex replacement and writes a sanitized copy with a .clean.json suffix.
    Skips if the sanitized file already exists.

    Returns the path to the sanitized file (or the original if clean).
    """
    # Handle files with compound extensions like .v2.1.json
    stem = filepath.name
    if stem.endswith(".clean.json"):
        return filepath  # already sanitized
    # Strip .json to get the base, then add .clean.json
    if stem.endswith(".json"):
        base = stem[:-5]
    else:
        base = stem
    clean_path = filepath.parent / (base + ".clean.json")

    if clean_path.exists():
        return clean_path

    # Always sanitize — the quick-check approach misses NaN that appears
    # deep in multi-GB files.  The cost is one extra pass on first run,
    # but the .clean.json is reused on all subsequent runs.
    print(f"    Sanitizing {filepath.name} (replacing NaN/Infinity → null)...")
    print(f"    This is a one-time operation; {clean_path.name} will be reused.")
    import re

    bytes_written = 0
    file_size = filepath.stat().st_size
    with open(filepath, "r", encoding="utf-8", errors="replace") as fin, \
         open(clean_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = re.sub(r':\s*NaN\b', ": null", line)
            line = re.sub(r':\s*-?Infinity\b', ": null", line)
            fout.write(line)
            bytes_written += len(line)
            if bytes_written % 500_000_000 < len(line):
                print(f"      ...{bytes_written / 1e9:.1f} / {file_size / 1e9:.1f} GB")

    print(f"    Wrote sanitized file: {clean_path.name} ({bytes_written / 1e9:.1f} GB)")
    return clean_path


def _stream_json_array(filepath, key):
    """Stream a top-level JSON array using ijson to avoid loading the entire
    file into memory.  Yields dicts one at a time.

    Automatically sanitizes the file if it contains NaN/Infinity values.

    Args:
        filepath: Path to a JSON file.
        key: The top-level key whose value is an array (e.g. "images").
    """
    filepath = _sanitize_json(filepath)
    with open(filepath, "rb") as f:
        for item in ijson.items(f, f"{key}.item"):
            yield item


def _resolve_json_path(ds_dir, expected_name, exclude_patterns=(), must_exist=True):
    """Find a JSON file in ds_dir, checking the expected name first, then
    falling back to a glob.  Skips .zip files and names matching any
    exclude_pattern."""
    path = ds_dir / expected_name
    if path.exists():
        return path

    candidates = [
        f for f in ds_dir.glob("*.json")
        if not any(p in f.name.lower() for p in exclude_patterns)
    ]
    if candidates:
        return candidates[0]

    if must_exist:
        return None
    return None


def build_filtered_list(args):
    """Parse metadata with streaming JSON, match to target labels, and write
    a filtered image list.  Designed for multi-GB metadata files."""
    # Check if output already exists
    output_path = LILA_DIR / f"filtered_images_{args.label_set}.json"
    if output_path.exists():
        print(f"  ✓ Filtered list already exists: {output_path}")
        print(f"    Delete it to regenerate.")
        return

    label_path = LABELS_225 if args.label_set == "225" else LABELS_480
    target = load_target_labels(label_path)
    print(f"  Loaded {len(target['labels'])} target labels from {label_path.name}")

    # Try to load taxonomy mapping if available
    tax_mapping = {}
    if TAXONOMY_MAPPING_PATH.exists():
        tax_mapping = load_taxonomy_mapping(TAXONOMY_MAPPING_PATH)
        print(f"  Loaded {len(tax_mapping)} taxonomy mapping entries")

    datasets_to_process = (
        [args.dataset] if args.dataset != "all" else DATASETS.keys()
    )

    all_filtered = []
    overall_stats = Counter()

    for ds_key in datasets_to_process:
        ds = DATASETS[ds_key]
        ds_dir = METADATA_DIR / ds_key

        # ── Find metadata JSON ───────────────────────────────────────
        meta_path = _resolve_json_path(
            ds_dir, ds["metadata_file"],
            exclude_patterns=("bbox", "split"),
        )
        if meta_path is None:
            print(f"  ✗ Metadata not found for {ds['name']}, skipping")
            continue

        print(f"\n  Parsing {ds['name']} ({meta_path.name}, {meta_path.stat().st_size / 1e9:.1f} GB)...")

        # ── Stream categories (small array — load fully) ─────────────
        print(f"    Streaming categories...")
        categories = {}
        for cat in _stream_json_array(meta_path, "categories"):
            categories[cat["id"]] = cat["name"]
        print(f"    {len(categories)} categories loaded")

        # ── Match categories to target labels ────────────────────────
        tax_ds_name = ds.get("taxonomy_name", ds["name"]).lower()

        # Build a quick lookup: category_name_lower -> taxonomy row
        # for this dataset (and related Snapshot * datasets)
        ds_tax = {}
        for (ds_name_l, query_l), row in tax_mapping.items():
            # Match the exact dataset name, or any Snapshot * variant for safari
            if ds_name_l == tax_ds_name or (
                tax_ds_name.startswith("snapshot") and ds_name_l.startswith("snapshot")
            ):
                if query_l not in ds_tax:
                    ds_tax[query_l] = row

        cat_to_label = {}
        for cat_id, cat_name in categories.items():
            matched = None

            # Strategy 1: Look up in LILA taxonomy mapping CSV
            tax_row = ds_tax.get(cat_name.lower())
            if tax_row:
                sci = tax_row.get("species", "") or ""
                if not sci and tax_row.get("genus"):
                    sci = tax_row["genus"]
                matched, mtype = match_category_to_label(
                    cat_name, sci, target
                )

            # Strategy 2: Category name IS a scientific name (WCS uses "puma concolor")
            if not matched:
                matched, mtype = match_category_to_label(
                    cat_name, cat_name, target
                )

            if matched:
                cat_to_label[cat_id] = (matched, mtype)

        print(f"    Matched {len(cat_to_label)} / {len(categories)} categories to target labels")
        if cat_to_label:
            sample = list(cat_to_label.items())[:5]
            for cid, (lab, mt) in sample:
                print(f"      e.g. '{categories[cid]}' → '{lab['common_name']}' ({mt})")

        # ── Stream annotations → build image_id → category mapping ───
        # Only keep image IDs that have a matching category.
        print(f"    Streaming annotations...")
        # img_id → list of matching category_ids
        img_matched_cats = defaultdict(list)
        ann_count = 0
        for ann in _stream_json_array(meta_path, "annotations"):
            ann_count += 1
            cat_id = ann.get("category_id")
            if cat_id in cat_to_label:
                img_matched_cats[ann["image_id"]].append(cat_id)
            if ann_count % 5_000_000 == 0:
                print(f"      ...{ann_count:,} annotations scanned, {len(img_matched_cats):,} images matched so far")
        print(f"    {ann_count:,} annotations scanned, {len(img_matched_cats):,} images with target species")

        # ── Load bounding boxes if available ──────────────────────────
        bboxes_by_image = defaultdict(list)
        if ds["bbox_file"]:
            bbox_path = _resolve_json_path(
                ds_dir, ds["bbox_file"],
                exclude_patterns=("split",),
            )
            # Make sure we found a JSON, not a ZIP
            if bbox_path and bbox_path.suffix == ".zip":
                bbox_path = None
            if bbox_path and bbox_path.exists():
                print(f"    Streaming bounding boxes from {bbox_path.name}...")
                bbox_count = 0
                for ann in _stream_json_array(bbox_path, "annotations"):
                    if "bbox" in ann:
                        img_id = ann["image_id"]
                        # Only keep bboxes for images we care about
                        if img_id in img_matched_cats:
                            bboxes_by_image[img_id].append(ann)
                            bbox_count += 1
                print(f"    {bbox_count:,} bboxes loaded for {len(bboxes_by_image):,} matched images")
            else:
                print(f"    No extracted bbox JSON found (check that zip was extracted)")

        # ── Stream images → build filtered entries ───────────────────
        print(f"    Streaming images and building filtered list...")
        ds_filtered = []
        ds_stats = Counter()
        img_count = 0

        for img_info in _stream_json_array(meta_path, "images"):
            img_count += 1
            img_id = img_info["id"]

            if img_id not in img_matched_cats:
                continue

            # Pick the first matching category
            cat_id = img_matched_cats[img_id][0]
            matched_label, match_type = cat_to_label[cat_id]
            cat_name = categories.get(cat_id, "unknown")

            # Skip empty/blank
            if cat_name.lower() in ("empty", "blank", "nothing", ""):
                continue

            img_bboxes = bboxes_by_image.get(img_id, [])

            entry = {
                "dataset": ds_key,
                "image_id": img_id,
                "file_name": img_info.get("file_name", ""),
                "width": img_info.get("width"),
                "height": img_info.get("height"),
                "target_label": matched_label["common_name"],
                "target_uuid": matched_label["uuid"],
                "source_category": cat_name,
                "match_type": match_type,
                "has_bbox": len(img_bboxes) > 0,
                "bboxes": [
                    {
                        "bbox": b["bbox"],
                        "category_id": b.get("category_id"),
                    }
                    for b in img_bboxes
                ],
                "image_url": ds["image_base_url"] + img_info.get("file_name", ""),
                "location": img_info.get("location"),
            }
            ds_filtered.append(entry)
            ds_stats[matched_label["common_name"]] += 1

            if img_count % 2_000_000 == 0:
                print(f"      ...{img_count:,} images scanned, {len(ds_filtered):,} matched")

        print(f"    {img_count:,} images scanned")
        print(f"    Filtered: {len(ds_filtered)} images matching target labels")
        print(f"    Top 10 classes:")
        for name, count in ds_stats.most_common(10):
            print(f"      {count:>6}  {name}")

        all_filtered.extend(ds_filtered)
        overall_stats.update(ds_stats)

    # Write filtered list
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ijson returns Decimal objects for numbers — convert to float for JSON
    import decimal

    class _DecimalEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, decimal.Decimal):
                return float(o)
            return super().default(o)

    with open(output_path, "w") as f:
        json.dump(all_filtered, f, cls=_DecimalEncoder)

    print(f"\n── Summary ──")
    print(f"  Total filtered images: {len(all_filtered)}")
    print(f"  Unique classes matched: {len(overall_stats)}")
    print(f"  Written to: {output_path}")

    # Coverage report
    covered = set(overall_stats.keys())
    all_labels = set(l["common_name"] for l in target["labels"].values())
    uncovered = all_labels - covered
    print(f"\n  Labels covered: {len(covered)} / {len(all_labels)}")
    if uncovered:
        print(f"  Still uncovered ({len(uncovered)}):")
        for name in sorted(uncovered):
            print(f"    - {name}")


# ── Image Download ───────────────────────────────────────────────────────────


def download_image(url, dest_path):
    """Download a single image. Returns True on success."""
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        with open(dest_path, "wb") as f:
            f.write(resp.content)
        return True
    except Exception:
        return False


def cmd_download(args):
    """Download filtered images from LILA BC."""
    filtered_path = LILA_DIR / f"filtered_images_{args.label_set}.json"
    if not filtered_path.exists():
        print(f"✗ Filtered list not found: {filtered_path}")
        print("  Run 'python scripts/download_lila_bc.py metadata' first.")
        sys.exit(1)

    with open(filtered_path) as f:
        filtered = json.load(f)

    print(f"Loaded {len(filtered)} filtered images")

    # Apply max-per-class limit
    if args.max_per_class:
        class_counts = Counter()
        limited = []
        for entry in filtered:
            label = entry["target_label"]
            if class_counts[label] < args.max_per_class:
                limited.append(entry)
                class_counts[label] += 1
        print(f"Limited to {args.max_per_class} per class: {len(limited)} images")
        filtered = limited

    # Filter by dataset if specified
    if args.dataset != "all":
        filtered = [e for e in filtered if e["dataset"] == args.dataset]
        print(f"Filtered to dataset '{args.dataset}': {len(filtered)} images")

    # Download
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    to_download = []
    already_exists = 0

    for entry in filtered:
        fname = f"{entry['dataset']}_{entry['file_name'].replace('/', '_')}"
        dest = IMAGES_DIR / fname
        if dest.exists():
            already_exists += 1
        else:
            to_download.append((entry["image_url"], dest))

    print(f"Already downloaded: {already_exists}")
    print(f"To download: {len(to_download)}")

    if not to_download:
        print("Nothing to download.")
        return

    successes = 0
    failures = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_image, url, dest): (url, dest)
            for url, dest in to_download
        }
        with tqdm(total=len(futures), desc="Downloading", unit="img") as pbar:
            for future in as_completed(futures):
                if future.result():
                    successes += 1
                else:
                    failures += 1
                pbar.update(1)

    print(f"\nDownloaded: {successes}, Failed: {failures}")


# ── YOLO Export ──────────────────────────────────────────────────────────────


def cmd_export(args):
    """Export filtered images with annotations to YOLO format."""
    filtered_path = LILA_DIR / f"filtered_images_{args.label_set}.json"
    if not filtered_path.exists():
        print(f"✗ Filtered list not found: {filtered_path}")
        sys.exit(1)

    with open(filtered_path) as f:
        filtered = json.load(f)

    # Build class index
    label_path = LABELS_225 if args.label_set == "225" else LABELS_480
    target = load_target_labels(label_path)
    class_names = sorted(set(l["common_name"] for l in target["labels"].values()))
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Write class names
    names_path = EXPORT_DIR / "classes.txt"
    with open(names_path, "w") as f:
        for name in class_names:
            f.write(name + "\n")
    print(f"Wrote {len(class_names)} class names to {names_path}")

    # Write YOLO annotations
    labels_dir = EXPORT_DIR / "labels"
    labels_dir.mkdir(exist_ok=True)

    exported = 0
    no_bbox = 0

    for entry in filtered:
        fname = f"{entry['dataset']}_{entry['file_name'].replace('/', '_')}"
        img_path = IMAGES_DIR / fname
        if not img_path.exists():
            continue

        label_name = img_path.stem + ".txt"
        label_path = labels_dir / label_name

        class_idx = class_to_idx.get(entry["target_label"])
        if class_idx is None:
            continue

        if entry["bboxes"] and entry.get("width") and entry.get("height"):
            w_img = entry["width"]
            h_img = entry["height"]

            with open(label_path, "w") as f:
                for bbox_info in entry["bboxes"]:
                    x, y, w, h = bbox_info["bbox"]
                    # COCO format [x, y, w, h] (absolute) → YOLO [cx, cy, w, h] (relative)
                    cx = (x + w / 2) / w_img
                    cy = (y + h / 2) / h_img
                    rw = w / w_img
                    rh = h / h_img
                    # Clamp to [0, 1]
                    cx = max(0, min(1, cx))
                    cy = max(0, min(1, cy))
                    rw = max(0, min(1, rw))
                    rh = max(0, min(1, rh))
                    f.write(f"{class_idx} {cx:.6f} {cy:.6f} {rw:.6f} {rh:.6f}\n")
            exported += 1
        else:
            # No bounding box — write whole-image box as placeholder
            # (these should later be processed through MegaDetector)
            with open(label_path, "w") as f:
                f.write(f"{class_idx} 0.5 0.5 1.0 1.0\n")
            no_bbox += 1
            exported += 1

    print(f"Exported {exported} YOLO annotations ({no_bbox} without real bboxes)")
    print(f"Labels directory: {labels_dir}")

    # Write dataset YAML for YOLO training
    yaml_path = EXPORT_DIR / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"# LILA BC filtered dataset for {args.label_set}-class model\n")
        f.write(f"path: {EXPORT_DIR}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n")
        f.write(f"test: images/test\n\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write(f"names:\n")
        for i, name in enumerate(class_names):
            f.write(f"  {i}: {name}\n")
    print(f"Wrote YOLO dataset config to {yaml_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Download and process LILA BC camera trap datasets"
    )
    parser.add_argument(
        "--label-set",
        choices=["225", "480"],
        default="225",
        help="Target label set (default: 225)",
    )
    parser.add_argument(
        "--dataset",
        choices=["serengeti", "safari", "wcs", "all"],
        default="all",
        help="Which dataset to process (default: all)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # metadata command
    sub_meta = subparsers.add_parser(
        "metadata", help="Download metadata and build filtered image list"
    )

    # download command
    sub_dl = subparsers.add_parser("download", help="Download filtered images")
    sub_dl.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Max images per class (default: no limit)",
    )
    sub_dl.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel download workers (default: 8)",
    )

    # export command
    sub_exp = subparsers.add_parser(
        "export", help="Export to YOLO annotation format"
    )

    args = parser.parse_args()

    if args.command == "metadata":
        cmd_metadata(args)
    elif args.command == "download":
        cmd_download(args)
    elif args.command == "export":
        cmd_export(args)


if __name__ == "__main__":
    main()
