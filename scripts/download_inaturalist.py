"""
Download and filter iNaturalist Open Data from AWS S3 for the student model.

iNaturalist hosts a public S3 bucket (s3://inaturalist-open-data/) containing
400M+ citizen science wildlife photos with metadata. Unlike the iNaturalist
competition datasets (which prohibit commercial use), individual photos carry
per-user Creative Commons licenses. By filtering for CC0 and CC-BY licenses,
we can legally build a commercially safe training corpus.

Data source:
    Bucket:   s3://inaturalist-open-data/ (public, no auth required)
    Region:   us-east-1
    Docs:     https://github.com/inaturalist/inaturalist-open-data

Metadata files (tab-separated, gzipped):
    photos.csv       — photo_id, observation_id, photo_license, extension, ...
    observations.csv — observation_id, taxon_id, latitude, longitude, ...
    taxa.csv         — taxon_id, ancestry, rank_level, rank, name, ...
    observers.csv    — observer_id, login, name (for CC-BY attribution)

Image URL pattern:
    https://inaturalist-open-data.s3.amazonaws.com/photos/{photo_id}/{size}.{ext}
    Sizes: original (2048px), large (1024px), medium (500px), small (240px)

Pipeline:
    1. metadata  — Download and decompress the metadata CSVs from S3
    2. filter    — Load taxa.csv + observations.csv + photos.csv, match against
                   the 225-class (or 480-class) label set, filter for CC0/CC-BY
                   licenses, and write a filtered image list JSON
    3. download  — Download filtered images with class-balanced sampling

Commercial license safety:
    - CC0:    No restrictions — public domain
    - CC-BY:  Attribution required — we generate an attribution manifest
    - CC-BY-NC, CC-BY-SA, etc.: EXCLUDED (not commercially safe)

Usage:
    # Step 1: Download metadata CSVs (~3 GB compressed, ~20 GB uncompressed)
    python scripts/download_inaturalist.py metadata

    # Step 2: Filter for target species with commercial licenses
    python scripts/download_inaturalist.py filter [--label-set 225|480]

    # Step 3: Download the filtered images
    python scripts/download_inaturalist.py download [--max-per-class 500] [--size medium]

Requirements:
    pip install requests tqdm
    Optional: pip install boto3   (for faster S3 downloads; falls back to HTTPS)
"""

import argparse
import csv
import gzip
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

# ── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
LABELS_225 = REPO_ROOT / "resources" / "2026-03-19_student_model_labels.txt"
LABELS_480 = REPO_ROOT / "resources" / "2026-03-20_student_model_labels_extended.txt"
INAT_DIR = REPO_ROOT / "data" / "inaturalist"
METADATA_DIR = INAT_DIR / "metadata"
IMAGES_DIR = INAT_DIR / "images"

# ── S3 / HTTPS Configuration ────────────────────────────────────────────────

S3_BUCKET = "inaturalist-open-data"
S3_REGION = "us-east-1"
S3_BASE_URL = f"https://{S3_BUCKET}.s3.amazonaws.com"

# Metadata files we need (in order of dependency)
METADATA_FILES = [
    "taxa.csv.gz",
    "observations.csv.gz",
    "photos.csv.gz",
    "observers.csv.gz",
]

# Commercially safe licenses (case-insensitive matching)
SAFE_LICENSES = {"cc0", "cc-by"}

# iNaturalist image sizes (width in pixels)
IMAGE_SIZES = {
    "original": 2048,
    "large": 1024,
    "medium": 500,
    "small": 240,
}

# ── Label Set Loading ────────────────────────────────────────────────────────
# Reuses the same label file format as download_lila_bc.py:
#   UUID;mammalia;order;family;genus;species;common_name


def load_target_labels(label_path):
    """Load a label file and return lookup structures for matching.

    Returns a dict with sub-dicts for species, genus, family, and order-level
    matching, plus a reverse lookup from common_name to entry.
    """
    labels = {}
    species_lookup = {}   # "genus species" -> entry
    genus_lookup = {}     # "genus" -> entry
    family_lookup = {}    # "family" -> entry
    order_lookup = {}     # "order" -> entry
    common_lookup = {}    # "common_name_lower" -> entry

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
                "order": parts[2],
                "family": parts[3],
                "genus": parts[4],
                "species": parts[5],
                "common_name": parts[6],
                "full_line": line,
            }

            labels[parts[0]] = entry
            common_lookup[parts[6].lower()] = entry

            if parts[5]:  # species level
                species_lookup[f"{parts[4]} {parts[5]}".lower()] = entry
            elif parts[4]:  # genus level
                genus_lookup[parts[4].lower()] = entry
            elif parts[3]:  # family level
                family_lookup[parts[3].lower()] = entry
            elif parts[2]:  # order level
                order_lookup[parts[2].lower()] = entry

    return {
        "labels": labels,
        "species": species_lookup,
        "genus": genus_lookup,
        "family": family_lookup,
        "order": order_lookup,
        "common": common_lookup,
    }


# ── Taxonomy Matching ────────────────────────────────────────────────────────


def match_taxon_to_label(taxon_name, taxon_rank, ancestry_names, target):
    """Match an iNaturalist taxon to a target label.

    Strategy (in order of preference):
        1. Exact species match (rank == "species")
        2. Subspecies → match parent species (rank == "subspecies")
        3. Genus match (rank == "genus" or first word of species name)
        4. Family match (from ancestry)
        5. Order match (from ancestry)

    Args:
        taxon_name:      Scientific name from taxa.csv (e.g., "Ursus arctos")
        taxon_rank:       Rank string ("species", "genus", "family", etc.)
        ancestry_names:   Dict of {rank: name} from the taxon's ancestry
        target:           Target label lookup from load_target_labels()

    Returns:
        (matched_entry, match_type) or (None, None)
    """
    name_lower = taxon_name.lower().strip()

    # 1. Species-level match
    if taxon_rank in ("species", "hybrid"):
        if name_lower in target["species"]:
            return target["species"][name_lower], "species"

    # 2. Subspecies → try the parent species (first two words)
    if taxon_rank == "subspecies":
        parts = name_lower.split()
        if len(parts) >= 2:
            parent = f"{parts[0]} {parts[1]}"
            if parent in target["species"]:
                return target["species"][parent], "subspecies"

    # 3. Genus match
    genus = name_lower.split()[0] if name_lower else ""
    if taxon_rank in ("species", "subspecies", "hybrid") and genus:
        if genus in target["genus"]:
            return target["genus"][genus], "genus"
    if taxon_rank == "genus" and name_lower in target["genus"]:
        return target["genus"][name_lower], "genus"

    # 4. Family match (from ancestry or directly if rank is family)
    family = ancestry_names.get("family", "").lower()
    if taxon_rank == "family":
        family = name_lower
    if family and family in target["family"]:
        return target["family"][family], "family"

    # 5. Order match (from ancestry)
    order = ancestry_names.get("order", "").lower()
    if taxon_rank == "order":
        order = name_lower
    if order and order in target["order"]:
        return target["order"][order], "order"

    return None, None


# ── Metadata Download ────────────────────────────────────────────────────────


def download_s3_file(filename, dest_dir, use_boto3=False):
    """Download a file from the iNaturalist S3 bucket.

    Tries boto3 first (faster, supports multipart), falls back to HTTPS.
    Skips if the file already exists locally.
    """
    gz_path = dest_dir / filename
    csv_path = dest_dir / filename.replace(".gz", "")

    # Skip if already decompressed
    if csv_path.exists():
        print(f"  Already decompressed: {csv_path.name}")
        return csv_path

    # Skip download if .gz already exists
    if not gz_path.exists():
        if use_boto3:
            try:
                import boto3
                from botocore import UNSIGNED
                from botocore.config import Config

                s3 = boto3.client(
                    "s3", region_name=S3_REGION, config=Config(signature_version=UNSIGNED)
                )
                obj = s3.head_object(Bucket=S3_BUCKET, Key=filename)
                total_size = obj["ContentLength"]

                print(f"  Downloading {filename} via boto3 ({total_size / 1e9:.1f} GB)...")
                with tqdm(total=total_size, unit="B", unit_scale=True, desc=filename) as pbar:
                    s3.download_file(
                        S3_BUCKET,
                        filename,
                        str(gz_path),
                        Callback=lambda bytes_transferred: pbar.update(bytes_transferred),
                    )
            except ImportError:
                print("  boto3 not available, falling back to HTTPS...")
                use_boto3 = False
            except Exception as e:
                print(f"  boto3 download failed ({e}), falling back to HTTPS...")
                if gz_path.exists():
                    gz_path.unlink()
                use_boto3 = False

        if not use_boto3:
            url = f"{S3_BASE_URL}/{filename}"
            print(f"  Downloading {filename} via HTTPS...")
            resp = requests.get(url, stream=True)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))

            with open(gz_path, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, desc=filename
            ) as pbar:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
                    pbar.update(len(chunk))
    else:
        print(f"  Already downloaded: {gz_path.name}")

    # Decompress (or rename if the server returned plain text despite .gz URL)
    # Check the first two bytes for the gzip magic number (0x1f 0x8b)
    with open(gz_path, "rb") as f:
        magic = f.read(2)

    if magic == b"\x1f\x8b":
        print(f"  Decompressing {filename}...")
        with gzip.open(gz_path, "rb") as f_in, open(csv_path, "wb") as f_out:
            while True:
                chunk = f_in.read(1024 * 1024 * 64)  # 64 MB chunks
                if not chunk:
                    break
                f_out.write(chunk)
        print(f"  Decompressed: {csv_path.name} ({csv_path.stat().st_size / 1e9:.1f} GB)")
    else:
        # File was served uncompressed (AWS sometimes does transparent decompression)
        print(f"  File already uncompressed, renaming {gz_path.name} → {csv_path.name}")
        gz_path.rename(csv_path)

    return csv_path


def cmd_metadata(args):
    """Download all metadata CSVs from S3."""
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    use_boto3 = args.use_boto3
    if use_boto3:
        try:
            import boto3  # noqa: F401
            print("Using boto3 for S3 downloads (faster).")
        except ImportError:
            print("boto3 not installed. Using HTTPS (slower but works).")
            print("  Install with: pip install boto3")
            use_boto3 = False

    for filename in METADATA_FILES:
        print(f"\n── {filename} ──")
        download_s3_file(filename, METADATA_DIR, use_boto3=use_boto3)

    print("\nAll metadata files downloaded.")
    print(f"Directory: {METADATA_DIR}")


# ── Taxonomy Loading ─────────────────────────────────────────────────────────


def load_taxa(taxa_path):
    """Load taxa.csv into lookup structures.

    Returns:
        taxon_by_id: dict of taxon_id -> {name, rank, ancestry, ancestry_names}
        mammal_ids:  set of taxon_ids that are mammals (class Mammalia)

    The ancestry field in taxa.csv contains a slash-separated list of ancestor
    taxon_ids (e.g., "48460/1/2/355675/40151/848317").  We resolve these to
    build ancestry_names = {rank: name} for each taxon.
    """
    print("  Loading taxa.csv...")
    taxon_by_id = {}
    row_count = 0

    with open(taxa_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            row_count += 1
            tid = int(row["taxon_id"])
            taxon_by_id[tid] = {
                "name": row.get("name", ""),
                "rank": row.get("rank", ""),
                "ancestry": row.get("ancestry", ""),
                "active": row.get("active", "t"),
            }
            if row_count % 500_000 == 0:
                print(f"    ...{row_count:,} taxa loaded")

    print(f"  {row_count:,} taxa loaded")

    # Resolve ancestry to find Mammalia subtree
    # First, find the taxon_id for "Mammalia"
    mammalia_id = None
    for tid, t in taxon_by_id.items():
        if t["name"].lower() == "mammalia" and t["rank"] == "class":
            mammalia_id = tid
            break

    if mammalia_id is None:
        print("  WARNING: Could not find Mammalia taxon_id in taxa.csv")
        print("  Will match by name only (slower)")
        return taxon_by_id, set()

    print(f"  Mammalia taxon_id: {mammalia_id}")

    # Find all taxa whose ancestry includes Mammalia
    mammalia_str = str(mammalia_id)
    mammal_ids = set()
    for tid, t in taxon_by_id.items():
        ancestry = t["ancestry"]
        # Ancestry is slash-separated: "48460/1/2/355675/40151"
        # Check if mammalia_id appears in the ancestry chain
        if mammalia_str in ancestry.split("/"):
            mammal_ids.add(tid)
        elif tid == mammalia_id:
            mammal_ids.add(tid)

    print(f"  {len(mammal_ids):,} mammal taxa found")

    # Build ancestry_names for mammal taxa only (saves memory)
    print("  Resolving ancestry names for mammal taxa...")
    for tid in mammal_ids:
        t = taxon_by_id[tid]
        ancestry_names = {}
        for ancestor_id_str in t["ancestry"].split("/"):
            if not ancestor_id_str:
                continue
            try:
                ancestor_id = int(ancestor_id_str)
            except ValueError:
                continue
            ancestor = taxon_by_id.get(ancestor_id)
            if ancestor and ancestor["rank"]:
                ancestry_names[ancestor["rank"]] = ancestor["name"]
        # Also include the taxon itself
        if t["rank"]:
            ancestry_names[t["rank"]] = t["name"]
        t["ancestry_names"] = ancestry_names

    return taxon_by_id, mammal_ids


# ── Filter Command ───────────────────────────────────────────────────────────


def cmd_filter(args):
    """Filter iNaturalist data for target species with commercial licenses.

    This is the core pipeline step. It processes ~400M photos and ~200M
    observations to find commercially licensed images of target mammal species.

    Processing strategy (memory-efficient):
        1. Load taxa.csv fully (small: ~2M rows) → find mammal taxon_ids
        2. Match mammal taxon_ids to target labels → build taxon_id→label map
        3. Stream observations.csv → build observation_id→taxon_id map
           (only for observations with matching taxon_ids)
        4. Stream photos.csv → filter by license AND observation match
        5. Write filtered results to JSON
    """
    output_path = INAT_DIR / f"filtered_images_{args.label_set}.json"
    if output_path.exists() and not args.force:
        print(f"Filtered list already exists: {output_path}")
        print("  Delete it or use --force to regenerate.")
        return

    # Check that metadata files exist
    for filename in METADATA_FILES:
        csv_name = filename.replace(".gz", "")
        csv_path = METADATA_DIR / csv_name
        if not csv_path.exists():
            print(f"Metadata file not found: {csv_path}")
            print("  Run 'python scripts/download_inaturalist.py metadata' first.")
            sys.exit(1)

    # Load target labels
    label_path = LABELS_225 if args.label_set == "225" else LABELS_480
    target = load_target_labels(label_path)
    print(f"Loaded {len(target['labels'])} target labels from {label_path.name}")

    # ── Step 1: Load taxonomy and find mammal taxa ───────────────────────
    print("\n── Step 1: Load taxonomy ──")
    taxa_path = METADATA_DIR / "taxa.csv"
    taxon_by_id, mammal_ids = load_taxa(taxa_path)

    # ── Step 2: Match mammal taxa to target labels ───────────────────────
    print("\n── Step 2: Match taxa to target labels ──")
    taxon_to_label = {}  # taxon_id -> (entry, match_type)
    matched_taxa_count = 0

    for tid in mammal_ids:
        t = taxon_by_id[tid]
        # Only match species, subspecies, and genus ranks (avoid noisy
        # higher-rank matches that would pull in too many unrelated images)
        if t["rank"] not in ("species", "subspecies", "genus", "hybrid"):
            continue
        ancestry_names = t.get("ancestry_names", {})
        entry, match_type = match_taxon_to_label(
            t["name"], t["rank"], ancestry_names, target
        )
        if entry:
            taxon_to_label[tid] = (entry, match_type)
            matched_taxa_count += 1

    print(f"  {matched_taxa_count} iNaturalist taxa matched to target labels")

    # Show coverage
    matched_labels = set()
    for entry, _ in taxon_to_label.values():
        matched_labels.add(entry["common_name"])
    all_labels = set(l["common_name"] for l in target["labels"].values())
    print(f"  Target labels covered: {len(matched_labels)} / {len(all_labels)}")

    uncovered = all_labels - matched_labels
    if uncovered:
        print(f"  Uncovered ({len(uncovered)}):")
        for name in sorted(uncovered)[:20]:
            print(f"    - {name}")
        if len(uncovered) > 20:
            print(f"    ... and {len(uncovered) - 20} more")

    # Show some example matches
    sample_matches = list(taxon_to_label.items())[:10]
    print(f"\n  Example matches:")
    for tid, (entry, mtype) in sample_matches:
        t = taxon_by_id[tid]
        print(f"    {t['name']} ({t['rank']}) → {entry['common_name']} ({mtype})")

    # ── Step 3: Stream observations to find matching observation_ids ─────
    print("\n── Step 3: Stream observations.csv ──")
    matching_taxon_ids = set(taxon_to_label.keys())
    obs_to_taxon = {}  # observation_id -> taxon_id (only matching ones)
    obs_count = 0

    obs_path = METADATA_DIR / "observations.csv"
    with open(obs_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            obs_count += 1
            try:
                taxon_id = int(row["taxon_id"])
            except (ValueError, KeyError):
                continue

            if taxon_id in matching_taxon_ids:
                obs_to_taxon[row["observation_uuid"]] = taxon_id

            if obs_count % 10_000_000 == 0:
                print(f"    ...{obs_count:,} observations scanned, {len(obs_to_taxon):,} matched")

    print(f"  {obs_count:,} observations scanned")
    print(f"  {len(obs_to_taxon):,} observations match target taxa")

    # ── Step 4: Stream photos and filter by license + observation match ──
    print("\n── Step 4: Stream photos.csv (filtering by license + species) ──")
    matching_obs_ids = set(obs_to_taxon.keys())
    filtered = []
    class_counts = Counter()
    license_counts = Counter()
    photo_count = 0
    skipped_license = 0

    photos_path = METADATA_DIR / "photos.csv"
    with open(photos_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            photo_count += 1
            obs_id = row.get("observation_uuid", "")

            if obs_id not in matching_obs_ids:
                if photo_count % 50_000_000 == 0:
                    print(f"    ...{photo_count:,} photos scanned, {len(filtered):,} kept")
                continue

            # Check license
            license_str = (row.get("license") or "").strip().lower()
            license_counts[license_str] += 1

            if license_str not in SAFE_LICENSES:
                skipped_license += 1
                continue

            # Build entry
            taxon_id = obs_to_taxon[obs_id]
            entry, match_type = taxon_to_label[taxon_id]
            taxon = taxon_by_id[taxon_id]

            photo_id = row.get("photo_id", "")
            extension = row.get("extension", "jpg") or "jpg"

            record = {
                "photo_id": photo_id,
                "observation_id": obs_id,
                "taxon_id": taxon_id,
                "scientific_name": taxon["name"],
                "taxon_rank": taxon["rank"],
                "target_label": entry["common_name"],
                "target_uuid": entry["uuid"],
                "match_type": match_type,
                "license": license_str,
                "extension": extension,
                "image_url": f"{S3_BASE_URL}/photos/{photo_id}/medium.{extension}",
            }
            filtered.append(record)
            class_counts[entry["common_name"]] += 1

            if photo_count % 50_000_000 == 0:
                print(f"    ...{photo_count:,} photos scanned, {len(filtered):,} kept")

    print(f"  {photo_count:,} photos scanned")
    print(f"  {len(filtered):,} photos kept (commercially safe + target species)")
    print(f"  {skipped_license:,} photos skipped (unsafe license)")

    print(f"\n  License distribution (for matched observations):")
    for lic, count in license_counts.most_common():
        safe = "SAFE" if lic in SAFE_LICENSES else "EXCLUDED"
        print(f"    {lic or '(none)':>12}: {count:>10,}  [{safe}]")

    # ── Write output ─────────────────────────────────────────────────────
    print(f"\n── Writing filtered list ──")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(filtered, f)
    print(f"  Written {len(filtered):,} entries to {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1e6:.1f} MB")

    # ── Coverage report ──────────────────────────────────────────────────
    print(f"\n── Coverage Report ──")
    covered = set(class_counts.keys())
    print(f"  Target labels with images: {len(covered)} / {len(all_labels)}")

    print(f"\n  Top 20 classes:")
    for name, count in class_counts.most_common(20):
        print(f"    {count:>8,}  {name}")

    print(f"\n  Bottom 20 classes:")
    for name, count in class_counts.most_common()[-20:]:
        print(f"    {count:>8,}  {name}")

    still_uncovered = all_labels - covered
    if still_uncovered:
        print(f"\n  Still uncovered ({len(still_uncovered)}):")
        for name in sorted(still_uncovered):
            print(f"    - {name}")

    # ── Write attribution manifest stub ──────────────────────────────────
    cc_by_count = sum(1 for r in filtered if r["license"] == "cc-by")
    if cc_by_count > 0:
        print(f"\n  {cc_by_count:,} images use CC-BY license (attribution required)")
        print(f"  Attribution manifest will be generated during download step.")


# ── Image Download ───────────────────────────────────────────────────────────


def download_image(url, dest_path):
    """Download a single image. Returns (success, url) tuple."""
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        with open(dest_path, "wb") as f:
            f.write(resp.content)
        return True
    except Exception:
        return False


def cmd_download(args):
    """Download filtered images from iNaturalist S3."""
    filtered_path = INAT_DIR / f"filtered_images_{args.label_set}.json"
    if not filtered_path.exists():
        print(f"Filtered list not found: {filtered_path}")
        print("  Run 'python scripts/download_inaturalist.py filter' first.")
        sys.exit(1)

    print(f"Loading filtered list from {filtered_path.name}...")
    with open(filtered_path) as f:
        filtered = json.load(f)
    print(f"Loaded {len(filtered):,} entries")

    # Apply max-per-class limit (with shuffling for diversity)
    if args.max_per_class:
        # Shuffle to avoid always getting the same subset
        import random
        random.seed(42)  # Reproducible
        random.shuffle(filtered)

        class_counts = Counter()
        limited = []
        for entry in filtered:
            label = entry["target_label"]
            if class_counts[label] < args.max_per_class:
                limited.append(entry)
                class_counts[label] += 1
        print(f"Limited to {args.max_per_class} per class: {len(limited):,} images")
        filtered = limited

    # Adjust image size in URLs if needed
    size = args.size
    if size != "medium":
        print(f"Using image size: {size} ({IMAGE_SIZES.get(size, '?')}px)")
        adjusted = []
        for entry in filtered:
            entry = dict(entry)
            # Replace /medium. with /{size}.
            entry["image_url"] = entry["image_url"].replace("/medium.", f"/{size}.")
            adjusted.append(entry)
        filtered = adjusted

    # Prepare download list
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    to_download = []
    already_exists = 0
    attribution_entries = []

    for entry in filtered:
        ext = entry.get("extension", "jpg")
        fname = f"inat_{entry['photo_id']}.{ext}"
        dest = IMAGES_DIR / fname

        if dest.exists():
            already_exists += 1
        else:
            to_download.append((entry["image_url"], dest, entry))

        # Track CC-BY images for attribution
        if entry.get("license") == "cc-by":
            attribution_entries.append(entry)

    print(f"Already downloaded: {already_exists:,}")
    print(f"To download: {len(to_download):,}")

    if not to_download:
        print("Nothing to download.")
    else:
        # Download with progress bar
        successes = 0
        failures = 0

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(download_image, url, dest): (url, dest)
                for url, dest, _ in to_download
            }
            with tqdm(total=len(futures), desc="Downloading", unit="img") as pbar:
                for future in as_completed(futures):
                    if future.result():
                        successes += 1
                    else:
                        failures += 1
                    pbar.update(1)

        print(f"\nDownloaded: {successes:,}, Failed: {failures:,}")

    # Write attribution manifest for CC-BY images
    if attribution_entries:
        _write_attribution_manifest(attribution_entries)


def _write_attribution_manifest(entries):
    """Write a CC-BY attribution manifest for legally required credits.

    This manifest documents all CC-BY licensed images used in training,
    including the photo URL and observation ID. Observer names are resolved
    during the download step if observers.csv is available.
    """
    manifest_path = INAT_DIR / "attribution_manifest.csv"
    print(f"\nWriting CC-BY attribution manifest ({len(entries):,} entries)...")

    # Try to load observer names
    observer_names = {}
    observer_path = METADATA_DIR / "observers.csv"
    if observer_path.exists():
        print("  Loading observer names from observers.csv...")
        with open(observer_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                oid = row.get("observer_id", "")
                name = row.get("name", "") or row.get("login", "")
                if oid and name:
                    observer_names[oid] = name

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "photo_id", "observation_id", "license", "species",
            "image_url", "observation_url",
        ])
        for entry in entries:
            pid = entry["photo_id"]
            oid = entry["observation_id"]
            writer.writerow([
                pid,
                oid,
                entry["license"],
                entry.get("scientific_name", ""),
                entry["image_url"],
                f"https://www.inaturalist.org/observations/{oid}",
            ])

    print(f"  Written to: {manifest_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Download and filter iNaturalist Open Data for the student model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download metadata CSVs (~3 GB compressed)
    python scripts/download_inaturalist.py metadata

    # Filter for 225-class targets with commercial licenses
    python scripts/download_inaturalist.py filter

    # Download images (max 500 per class, medium resolution)
    python scripts/download_inaturalist.py download --max-per-class 500

    # Download larger images for higher quality training
    python scripts/download_inaturalist.py download --max-per-class 500 --size large
        """,
    )
    parser.add_argument(
        "--label-set",
        choices=["225", "480"],
        default="225",
        help="Target label set (default: 225)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # metadata command
    sub_meta = subparsers.add_parser(
        "metadata",
        help="Download metadata CSVs from S3",
    )
    sub_meta.add_argument(
        "--use-boto3",
        action="store_true",
        default=False,
        help="Use boto3 for faster S3 downloads (requires: pip install boto3)",
    )

    # filter command
    sub_filter = subparsers.add_parser(
        "filter",
        help="Filter metadata for target species with commercial licenses",
    )
    sub_filter.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Regenerate even if filtered list already exists",
    )

    # download command
    sub_dl = subparsers.add_parser(
        "download",
        help="Download filtered images",
    )
    sub_dl.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Max images per class (default: no limit)",
    )
    sub_dl.add_argument(
        "--size",
        choices=list(IMAGE_SIZES.keys()),
        default="medium",
        help="Image size to download (default: medium/500px)",
    )
    sub_dl.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel download workers (default: 8)",
    )

    args = parser.parse_args()

    if args.command == "metadata":
        cmd_metadata(args)
    elif args.command == "filter":
        cmd_filter(args)
    elif args.command == "download":
        cmd_download(args)


if __name__ == "__main__":
    main()


"""
python scripts/download_inaturalist.py filter        
Loaded 225 target labels from 2026-03-19_student_model_labels.txt

── Step 1: Load taxonomy ──
  Loading taxa.csv...
    ...500,000 taxa loaded
    ...1,000,000 taxa loaded
    ...1,500,000 taxa loaded
  1,626,690 taxa loaded
  Mammalia taxon_id: 40151
  16,698 mammal taxa found
  Resolving ancestry names for mammal taxa...

── Step 2: Match taxa to target labels ──
  7158 iNaturalist taxa matched to target labels
  Target labels covered: 221 / 225
  Uncovered (4):
    - dingo
    - domestic goat
    - domestic pig
    - pinniped clade

  Example matches:
    Cervus canadensis alashanicus (subspecies) → elk (subspecies)
    Vulpes vulpes alpherakyi (subspecies) → red fox (subspecies)
    Cervus canadensis sibiricus (subspecies) → elk (subspecies)
    Cervus canadensis macneilli (subspecies) → elk (subspecies)
    Callosciurus erythraeus styani (subspecies) → squirrel family (family)
    Callosciurus erythraeus thai (subspecies) → squirrel family (family)
    Apodemus agrarius coreae (subspecies) → muridae family (family)
    Apodemus agrarius chejuensis (subspecies) → muridae family (family)
    Tamiasciurus hudsonicus ventorum (subspecies) → red squirrel (subspecies)
    Saguinus ursula (species) → saguinus species (genus)

── Step 3: Stream observations.csv ──
    ...10,000,000 observations scanned, 237,643 matched
    ...20,000,000 observations scanned, 422,879 matched
    ...30,000,000 observations scanned, 624,321 matched
    ...40,000,000 observations scanned, 788,927 matched
    ...50,000,000 observations scanned, 980,832 matched
    ...60,000,000 observations scanned, 1,112,908 matched
    ...70,000,000 observations scanned, 1,305,972 matched
    ...80,000,000 observations scanned, 1,452,075 matched
    ...90,000,000 observations scanned, 1,595,156 matched
    ...100,000,000 observations scanned, 1,801,897 matched
    ...110,000,000 observations scanned, 1,938,817 matched
    ...120,000,000 observations scanned, 2,072,142 matched
    ...130,000,000 observations scanned, 2,257,371 matched
    ...140,000,000 observations scanned, 2,442,147 matched
    ...150,000,000 observations scanned, 2,579,210 matched
    ...160,000,000 observations scanned, 2,720,019 matched
    ...170,000,000 observations scanned, 2,891,540 matched
    ...180,000,000 observations scanned, 3,114,562 matched
    ...190,000,000 observations scanned, 3,261,714 matched
    ...200,000,000 observations scanned, 3,403,693 matched
    ...210,000,000 observations scanned, 3,548,180 matched
    ...220,000,000 observations scanned, 3,708,941 matched
    ...230,000,000 observations scanned, 3,924,930 matched
  233,286,559 observations scanned
  4,003,990 observations match target taxa

── Step 4: Stream photos.csv (filtering by license + species) ──
    ...50,000,000 photos scanned, 0 kept
    ...100,000,000 photos scanned, 0 kept
    ...150,000,000 photos scanned, 0 kept
    ...200,000,000 photos scanned, 0 kept
    ...250,000,000 photos scanned, 0 kept
    ...300,000,000 photos scanned, 0 kept
    ...350,000,000 photos scanned, 0 kept
    ...400,000,000 photos scanned, 0 kept
  413,168,476 photos scanned
  0 photos kept (commercially safe + target species)
  0 photos skipped (unsafe license)

  License distribution (for matched observations):

── Writing filtered list ──
  Written 0 entries to /Users/simonhedrich/Code/Master-Thesis/data/inaturalist/filtered_images_225.json
  File size: 0.0 MB

── Coverage Report ──
  Target labels with images: 0 / 225

  Top 20 classes:

  Bottom 20 classes:

  Still uncovered (225):
    - aardvark
    - aardwolf
    - african buffalo
    - african civet
    - african elephant
    - african wild dog
    - agouti genus
    - alpine ibex
    - alpine marmot
    - american badger
    - american bison
    - american black bear
    - american mink
    - arizona black-tailed prairie dog
    - asian elephant
    - asiatic black bear
    - asiatic wild ass
    - ateles species
    - aye-aye
    - baboon genus
    - baird's tapir
    - bat-eared fox
    - beaver genus
    - bighorn sheep
    - binturong
    - black wildebeest
    - black-backed jackal
    - blackbuck
    - blesbok
    - bobcat
    - bongo
    - bornean orangutan
    - brown bear
    - brown hyaena
    - brown-throated sloth
    - bushbuck
    - california ground squirrel
    - callicebus genus
    - callithrix species
    - canada lynx
    - capybara
    - caracal
    - cebus species
    - cephalophus species
    - cercopithecus species
    - cheetah
    - chimpanzee
    - chipmunk genus
    - chital
    - clouded leopard
    - collared peccary
    - colobus species
    - common duiker
    - common eland
    - common fallow deer
    - common warthog
    - common wildebeest
    - common wombat
    - cottontail rabbits genus
    - coyote
    - cricetidae family
    - dhole
    - dingo
    - domestic cat
    - domestic cattle
    - domestic dog
    - domestic donkey
    - domestic goat
    - domestic horse
    - domestic pig
    - domestic sheep
    - domestic water buffalo
    - drill
    - dromedary camel
    - eared seals
    - eastern cottontail
    - eastern fox squirrel
    - eastern gray squirrel
    - eastern grey kangaroo
    - elephant seal
    - elk
    - eulemur species
    - eurasian badger
    - eurasian lynx
    - eurasian otter
    - eurasian red squirrel
    - european bison
    - european hare
    - european rabbit
    - european roe deer
    - fisher
    - fossa
    - gemsbok
    - genet genus
    - gerenuk
    - giant anteater
    - giant armadillo
    - giant otter
    - giant panda
    - giraffe
    - glaucomys species
    - golden jackal
    - golden mantled ground squirrel
    - gorilla species
    - grant's gazelle
    - greater kudu
    - grevy's zebra
    - grey fox
    - grey wolf
    - hares and jackrabbits genus
    - hartebeest
    - hedgehog family
    - hippopotamus
    - hoffmann's two-toed sloth
    - hog badger genus
    - honey badger
    - howler monkey genus
    - human
    - impala
    - jaguar
    - japanese macaque
    - kangaroo family
    - kinkajou
    - kirk's dik-dik
    - klipspringer
    - koala
    - kob
    - leaf monkeys genus
    - leopard
    - leopard cat
    - leopardus species
    - lion
    - llama genus
    - lowland tapir
    - lycalopex species
    - macaque species
    - malay tapir
    - maned wolf
    - mangabeys genus
    - martes species
    - meerkat
    - mongoose family
    - moose
    - mouflon
    - mountain goat
    - mountain zebra
    - mule deer
    - muntjac genus
    - muridae family
    - muskrat
    - nilgai
    - nine-banded armadillo
    - north american porcupine
    - north american river otter
    - northern chamois
    - northern raccoon
    - nutria
    - nyala
    - ocelot
    - old world porcupine family
    - opossum family
    - pangolin family
    - patas monkey
    - pikas genus
    - pinniped clade
    - plains zebra
    - pronghorn
    - puma
    - quokka
    - raccoon dog
    - rattus genus
    - red brocket
    - red deer
    - red fox
    - red kangaroo
    - red panda
    - red river hog
    - red squirrel
    - red-necked wallaby
    - reedbuck genus
    - reindeer
    - rhinoceros family
    - ring-tailed lemur
    - ringtail
    - roan antelope
    - rock hyrax
    - sable antelope
    - saguinus species
    - saiga
    - saimiri species
    - sambar
    - sea otter
    - serval
    - short-beaked echidna
    - sika deer
    - sloth bear
    - snow leopard
    - south american coati
    - spectacled bear
    - spilogale species
    - spotted hyaena
    - springbok
    - squirrel family
    - steenbok
    - striped hyaena
    - striped skunk
    - sun bear
    - swamp wallaby
    - tayra
    - thomson's gazelle
    - tiger
    - vervet monkey
    - walrus
    - water deer
    - waterbuck
    - weasel species
    - western gray squirrel
    - white-nosed coati
    - white-tailed deer
    - wild boar
    - wild cat
    - wolverine
    - woodchuck
    - yak
    - yellow-bellied marmot
"""