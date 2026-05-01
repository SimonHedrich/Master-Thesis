"""Convert all existing downloaded images to JPEG and update metadata files.

One-shot migration script. Safe to re-run — already-converted images are skipped.

What it does:
  1. Scans all four image directories for non-JPEG files.
  2. Converts each file to JPEG using Pillow (first frame for GIFs, white
     background for transparent images). Deletes the original on success.
  3. Renames bare .jpeg files to .jpg (they are already JPEG, no re-encode).
  4. Updates metadata files to record the local filename/extension:
       - data/inaturalist/metadata/condensed/photos.csv  → adds local_extension = "jpg"
       - data/wikimedia/metadata.csv                     → adds local_filename  = stem + ".jpg"

Usage:
    python scripts/convert_existing_images_to_jpg.py
    python scripts/convert_existing_images_to_jpg.py --dry-run
    python scripts/convert_existing_images_to_jpg.py --quality 85
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _image_utils import save_as_jpg

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"

IMAGE_DIRS = [
    DATA_ROOT / "inaturalist" / "images",
    DATA_ROOT / "wikimedia" / "images",
    DATA_ROOT / "openimages" / "images",
]

# Extensions that need full re-encoding
CONVERT_EXTS = {".png", ".gif", ".webp", ".tiff", ".tif", ".bmp"}
# Extensions that are already JPEG — just rename
RENAME_EXTS = {".jpeg"}


def _scan_images(image_dirs):
    """Yield (path, action) for all non-.jpg image files.

    action is either "convert" or "rename".
    """
    for base_dir in image_dirs:
        if not base_dir.exists():
            continue
        for p in base_dir.rglob("*"):
            if not p.is_file():
                continue
            ext = p.suffix.lower()
            if ext in CONVERT_EXTS:
                yield p, "convert"
            elif ext in RENAME_EXTS:
                yield p, "rename"


def _convert_image(path: Path, quality: int, dry_run: bool) -> bool:
    """Convert *path* to .jpg. Returns True on success."""
    dest = path.with_suffix(".jpg")
    if dest.exists():
        # Already converted — remove the stale original if present
        if not dry_run:
            path.unlink()
        return True
    if dry_run:
        return True
    try:
        data = path.read_bytes()
        save_as_jpg(data, dest, quality=quality)
        path.unlink()
        return True
    except Exception as exc:
        print(f"  ERROR converting {path}: {exc}", file=sys.stderr)
        return False


def _rename_image(path: Path, dry_run: bool) -> bool:
    """Rename .jpeg → .jpg. Returns True on success."""
    dest = path.with_suffix(".jpg")
    if dest.exists():
        if not dry_run:
            path.unlink()
        return True
    if dry_run:
        return True
    try:
        path.rename(dest)
        return True
    except Exception as exc:
        print(f"  ERROR renaming {path}: {exc}", file=sys.stderr)
        return False


# ── Metadata updaters ─────────────────────────────────────────────────────────


def update_inaturalist_photos_csv(dry_run: bool) -> int:
    """Add local_extension = 'jpg' column to condensed photos.csv.

    Returns number of rows updated (0 if file absent or already has column).
    """
    tsv_path = DATA_ROOT / "inaturalist" / "metadata" / "condensed" / "photos.csv"
    if not tsv_path.exists():
        return 0

    with open(tsv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames or []
        if "local_extension" in fieldnames:
            print(f"  {tsv_path.name}: already has local_extension column, skipping")
            return 0
        rows = list(reader)

    new_fields = list(fieldnames) + ["local_extension"]
    print(f"  {tsv_path.name}: adding local_extension to {len(rows):,} rows")

    if dry_run:
        return len(rows)

    tmp = tsv_path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=new_fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            row["local_extension"] = "jpg"
            writer.writerow(row)
    tmp.rename(tsv_path)
    return len(rows)


def update_wikimedia_metadata_csv(dry_run: bool) -> int:
    """Add local_filename column to wikimedia metadata.csv.

    Returns number of rows updated.
    """
    csv_path = DATA_ROOT / "wikimedia" / "metadata.csv"
    if not csv_path.exists():
        return 0

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "local_filename" in fieldnames:
            print(f"  {csv_path.name}: already has local_filename column, skipping")
            return 0
        rows = list(reader)

    # Insert local_filename right after filename
    idx = fieldnames.index("filename") + 1
    new_fields = list(fieldnames[:idx]) + ["local_filename"] + list(fieldnames[idx:])
    print(f"  {csv_path.name}: adding local_filename to {len(rows):,} rows")

    if dry_run:
        return len(rows)

    tmp = csv_path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=new_fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            row["local_filename"] = Path(row["filename"]).stem + ".jpg"
            writer.writerow(row)
    tmp.rename(csv_path)
    return len(rows)


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Convert all downloaded images to JPEG and update metadata files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without modifying any files",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=92,
        help="JPEG quality for conversions (default: 92)",
    )
    args = parser.parse_args()

    if args.dry_run:
        print("DRY RUN — no files will be modified\n")

    # ── Image conversion ──────────────────────────────────────────────────────
    print("── Scanning image directories ──")
    to_process = list(_scan_images(IMAGE_DIRS))

    if not to_process:
        print("  No non-JPEG images found.")
    else:
        conversions = [(p, a) for p, a in to_process if a == "convert"]
        renames = [(p, a) for p, a in to_process if a == "rename"]
        print(f"  {len(conversions):,} files to convert, {len(renames):,} files to rename (.jpeg → .jpg)")

        if args.dry_run:
            for p, _ in conversions[:10]:
                print(f"    would convert: {p.relative_to(REPO_ROOT)}")
            if len(conversions) > 10:
                print(f"    ... and {len(conversions) - 10:,} more")
        else:
            converted_ok = 0
            converted_err = 0
            renamed_ok = 0
            renamed_err = 0

            print("\n  Converting …")
            for i, (p, action) in enumerate(to_process, 1):
                if i % 1000 == 0:
                    print(f"    {i:,} / {len(to_process):,} …")
                if action == "convert":
                    if _convert_image(p, args.quality, dry_run=False):
                        converted_ok += 1
                    else:
                        converted_err += 1
                else:
                    if _rename_image(p, dry_run=False):
                        renamed_ok += 1
                    else:
                        renamed_err += 1

            print(f"\n  Converted : {converted_ok:,}  (errors: {converted_err})")
            print(f"  Renamed   : {renamed_ok:,}  (errors: {renamed_err})")

    # ── Metadata updates ──────────────────────────────────────────────────────
    print("\n── Updating metadata files ──")

    n = update_inaturalist_photos_csv(dry_run=args.dry_run)
    if n:
        print(f"  ✓ photos.csv: {n:,} rows {'would be' if args.dry_run else ''} updated")

    n = update_wikimedia_metadata_csv(dry_run=args.dry_run)
    if n:
        print(f"  ✓ wikimedia/metadata.csv: {n:,} rows {'would be' if args.dry_run else ''} updated")

    print("\nDone.")


if __name__ == "__main__":
    main()
