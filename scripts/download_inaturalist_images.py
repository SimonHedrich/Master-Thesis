"""
Download iNaturalist images from the condensed metadata.

Reads the condensed photos.csv (produced by analyze_inaturalist_metadata.py condense)
and downloads images from the iNaturalist S3 bucket. By default, only commercially
safe images (CC0, CC-BY) are downloaded.

Images are saved in an ImageFolder-compatible directory structure:
    data/inaturalist/images/{target_class}/inat_{photo_id}.{ext}

The download is resumable by default — existing files are skipped. Use --from-scratch
to delete everything and start over.

Usage:
    # Download commercially safe images (default)
    python scripts/download_inaturalist_images.py

    # Download ALL images regardless of license
    python scripts/download_inaturalist_images.py --all-licenses

    # Limit per class for balanced datasets
    python scripts/download_inaturalist_images.py --max-per-class 500

    # Larger images
    python scripts/download_inaturalist_images.py --size large

    # Start fresh (deletes existing images)
    python scripts/download_inaturalist_images.py --from-scratch

Prerequisites:
    python scripts/analyze_inaturalist_metadata.py condense
"""

import argparse
import csv
import json
import random
import shutil
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

# ── Paths & Config ───────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
CONDENSED_PHOTOS = REPO_ROOT / "data" / "inaturalist" / "metadata" / "condensed" / "photos.csv"
IMAGES_DIR = REPO_ROOT / "data" / "inaturalist" / "images"
PROGRESS_FILE = IMAGES_DIR / ".download_progress.json"

S3_BASE_URL = "https://inaturalist-open-data.s3.amazonaws.com"
SAFE_LICENSES = {"cc0", "cc-by", "cc-by-sa"}
IMAGE_SIZES = {"original": 2048, "large": 1024, "medium": 500, "small": 240}


# ── Load photo list ──────────────────────────────────────────────────────────


def load_photos(all_licenses, max_per_class):
    """Load the condensed photos CSV, apply license and class filters.

    Returns:
        list of dicts with keys: photo_id, extension, license, target_class, observer_id
    """
    if not CONDENSED_PHOTOS.exists():
        print(f"ERROR: {CONDENSED_PHOTOS} not found.")
        print("  Run: python scripts/analyze_inaturalist_metadata.py condense")
        sys.exit(1)

    print(f"Loading {CONDENSED_PHOTOS.name}...")
    photos = []
    skipped_license = 0

    with open(CONDENSED_PHOTOS, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            license_str = (row.get("license") or "").strip().lower()
            if not all_licenses and license_str not in SAFE_LICENSES:
                skipped_license += 1
                continue
            photos.append({
                "photo_id": row["photo_id"],
                "extension": row.get("extension", "jpg") or "jpg",
                "license": license_str,
                "target_class": row["target_class"],
                "observer_id": row.get("observer_id", ""),
            })

    print(f"  {len(photos):,} photos loaded" +
          (f" ({skipped_license:,} skipped, non-commercial license)" if skipped_license else ""))

    if max_per_class:
        random.seed(42)
        random.shuffle(photos)
        class_counts = Counter()
        limited = []
        for p in photos:
            if class_counts[p["target_class"]] < max_per_class:
                limited.append(p)
                class_counts[p["target_class"]] += 1
        print(f"  Limited to {max_per_class} per class: {len(limited):,} photos")
        photos = limited

    return photos


# ── Download ─────────────────────────────────────────────────────────────────


def download_image(url, dest_path):
    """Download a single image. Returns (success, photo_id)."""
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "wb") as f:
            f.write(resp.content)
        return True
    except Exception:
        return False


def build_download_list(photos, size):
    """Build list of (url, dest_path, photo) tuples, skipping existing files.

    Returns:
        to_download: list of (url, dest_path, photo)
        already_exists: int
    """
    to_download = []
    already_exists = 0

    for p in photos:
        class_dir = p["target_class"].replace("/", "_")
        fname = f"inat_{p['photo_id']}.{p['extension']}"
        dest = IMAGES_DIR / class_dir / fname

        if dest.exists():
            already_exists += 1
            continue

        url = f"{S3_BASE_URL}/photos/{p['photo_id']}/{size}.{p['extension']}"
        to_download.append((url, dest, p))

    return to_download, already_exists


def save_progress(total, downloaded, failed, skipped, failed_ids):
    """Save download progress to JSON file."""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump({
            "total_planned": total,
            "downloaded": downloaded,
            "failed": failed,
            "skipped_existing": skipped,
            "failed_photo_ids": failed_ids,
        }, f, indent=2)


def write_attribution_manifest(photos):
    """Write CC-BY attribution manifest for images that require credit."""
    cc_by = [p for p in photos if p["license"] == "cc-by"]
    if not cc_by:
        return

    # Load observer names if available
    observers_path = REPO_ROOT / "data" / "inaturalist" / "metadata" / "observers.csv"
    observer_names = {}
    if observers_path.exists():
        print("  Loading observer names for attribution...")
        with open(observers_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            # Only load observers we actually need
            needed_ids = {p["observer_id"] for p in cc_by}
            for row in reader:
                oid = row.get("observer_id", "")
                if oid in needed_ids:
                    observer_names[oid] = row.get("name", "") or row.get("login", "")

    manifest_path = IMAGES_DIR / "attribution_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["photo_id", "observer_id", "observer_name", "license",
                         "target_class", "image_url", "observation_url"])
        for p in cc_by:
            writer.writerow([
                p["photo_id"],
                p["observer_id"],
                observer_names.get(p["observer_id"], ""),
                p["license"],
                p["target_class"],
                f"{S3_BASE_URL}/photos/{p['photo_id']}/medium.{p['extension']}",
                f"https://www.inaturalist.org/photos/{p['photo_id']}",
            ])

    print(f"  Attribution manifest: {manifest_path.relative_to(REPO_ROOT)} ({len(cc_by):,} CC-BY entries)")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Download iNaturalist images from condensed metadata.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--all-licenses",
        action="store_true",
        default=False,
        help="Download all images, not just commercially safe (CC0 + CC-BY)",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Max images per class (for balanced datasets)",
    )
    parser.add_argument(
        "--size",
        choices=list(IMAGE_SIZES.keys()),
        default="medium",
        help="Image size to download (default: medium/500px)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel download workers (default: 8)",
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        default=False,
        help="Delete existing images and start from scratch",
    )
    args = parser.parse_args()

    # From scratch: wipe images directory
    if args.from_scratch and IMAGES_DIR.exists():
        print(f"Deleting {IMAGES_DIR}...")
        shutil.rmtree(IMAGES_DIR)

    # Load and filter photos
    photos = load_photos(args.all_licenses, args.max_per_class)
    if not photos:
        print("No photos to download.")
        return

    # Class summary
    class_counts = Counter(p["target_class"] for p in photos)
    print(f"  {len(class_counts)} classes, "
          f"min {min(class_counts.values()):,}/class, "
          f"max {max(class_counts.values()):,}/class")

    # Build download list (skip existing)
    to_download, already_exists = build_download_list(photos, args.size)
    print(f"\n  Already downloaded: {already_exists:,}")
    print(f"  To download:        {len(to_download):,}")

    if not to_download:
        print("\nNothing to download — all files already exist.")
        write_attribution_manifest(photos)
        return

    # Download with progress bar
    successes = 0
    failures = 0
    failed_ids = []

    print(f"\nDownloading {len(to_download):,} images ({args.size}/{IMAGE_SIZES[args.size]}px) "
          f"with {args.workers} workers...\n")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for url, dest, photo in to_download:
            future = executor.submit(download_image, url, dest)
            futures[future] = photo

        with tqdm(total=len(futures), desc="Downloading", unit="img") as pbar:
            for future in as_completed(futures):
                if future.result():
                    successes += 1
                else:
                    failures += 1
                    failed_ids.append(futures[future]["photo_id"])
                pbar.update(1)

    # Summary
    print(f"\nDownloaded: {successes:,}")
    print(f"Failed:     {failures:,}")
    print(f"Skipped:    {already_exists:,}")
    print(f"Total:      {successes + already_exists:,} / {len(photos):,}")

    # Save progress
    save_progress(len(photos), successes, failures, already_exists, failed_ids)
    print(f"Progress saved to {PROGRESS_FILE.relative_to(REPO_ROOT)}")

    # Attribution manifest
    write_attribution_manifest(photos)


if __name__ == "__main__":
    main()
