#!/usr/bin/env python3
"""
Organize the flat GBIFImages directory into data/gbif/ with per-class subdirectories,
and move GBIF metadata files alongside the images.

Source layout (before):
    resources/GBIFImages/images/{genus}_{species}_{GBIF_ID}.jpg  (flat, 66 881 files)
    resources/SNPredictions_all.json
    resources/GBIF_image_counts_v1.csv

Target layout (after):
    data/gbif/images/{common_name}/{genus}_{species}_{GBIF_ID}.jpg
    data/gbif/images/_unmatched/  (images not assignable to any class)
    data/gbif/metadata/SNPredictions_all.json
    data/gbif/metadata/GBIF_image_counts_v1.csv

Class mapping uses a four-step fallback chain per image:
  1. Species exact match:  "genus species" → class_counts_225.csv (level=species)
  2. Genus match:          "genus"         → class_counts_225.csv (level=genus)
  3. Species→family:       "genus species" → family_species_mapping.csv species_scientific → family_label
  4. Genus→family:         "genus"         → family_species_mapping.csv genus_scientific   → family_label

reports/genus_species_mapping.csv is intentionally not used: every genus it references
already has a level=genus entry in class_counts_225.csv, making it redundant.

Usage (run from the repository root):
    python scripts/gbif/2-organize_gbif.py            # move files (default)
    python scripts/gbif/2-organize_gbif.py --copy     # copy instead of move
    python scripts/gbif/2-organize_gbif.py --dry-run  # print actions without touching files
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

SRC_IMAGES   = REPO_ROOT / "resources" / "GBIFImages" / "images"
DST_IMAGES   = REPO_ROOT / "data" / "gbif" / "images"
DST_META     = REPO_ROOT / "data" / "gbif" / "metadata"

METADATA_FILES = [
    REPO_ROOT / "resources" / "SNPredictions_all.json",
    REPO_ROOT / "resources" / "GBIF_image_counts_v1.csv",
]

CLASS_CSV  = REPO_ROOT / "reports" / "class_counts_225.csv"
FAMILY_CSV = REPO_ROOT / "reports" / "family_species_mapping.csv"


def build_lookups(
    class_csv: Path,
    family_csv: Path,
) -> tuple[dict, dict, dict, dict]:
    """Build four lookup dicts from the class and family mapping CSVs.

    Returns:
        species_lut       — "genus species" → common_name  (class_counts level=species)
        genus_lut         — "genus"         → common_name  (class_counts level=genus)
        species_to_family — "genus species" → family_label (family_species_mapping, species col)
        genus_to_family   — "genus"         → family_label (family_species_mapping, genus col)
    """
    species_lut: dict[str, str] = {}
    genus_lut: dict[str, str] = {}
    valid_classes: set[str] = set()

    with class_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sci    = row["scientific_name"].strip().lower()
            common = row["common_name"].strip().lower()
            level  = row["level"].strip().lower()
            valid_classes.add(common)
            if level == "genus":
                genus_lut[sci] = common
            elif level == "species":
                species_lut[sci] = common
            # level=family entries are intentionally skipped here; they are only
            # reachable via the family_species_mapping fallback below.

    species_to_family: dict[str, str] = {}
    genus_to_family: dict[str, str] = {}

    with family_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fl = row["family_label"].strip().lower()
            gs = row["genus_scientific"].strip().lower()
            ss = row["species_scientific"].strip().lower()
            if fl not in valid_classes:
                continue  # family not in the target 225 classes — skip
            if gs:
                genus_to_family.setdefault(gs, fl)    # keep first occurrence
            if ss:
                species_to_family.setdefault(ss, fl)

    return species_lut, genus_lut, species_to_family, genus_to_family


def class_dir_for(
    stem: str,
    species_lut: dict,
    genus_lut: dict,
    species_to_family: dict,
    genus_to_family: dict,
) -> str:
    """Return the target class directory name for an image filename stem.

    Fallback chain:
      1. species exact match (class_counts)
      2. genus match         (class_counts)
      3. species → family    (family_species_mapping)
      4. genus   → family    (family_species_mapping)
      5. _unmatched
    """
    parts = stem.split("_")
    if len(parts) < 2:
        return "_unmatched"
    g, s = parts[0], parts[1]
    sci = f"{g} {s}"

    if sci in species_lut:
        return species_lut[sci]
    if g in genus_lut:
        return genus_lut[g]
    if sci in species_to_family:
        return species_to_family[sci]
    if g in genus_to_family:
        return genus_to_family[g]
    return "_unmatched"


def transfer(src: Path, dst: Path, copy: bool, dry_run: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return  # idempotent: skip already-transferred files
    if dry_run:
        action = "COPY" if copy else "MOVE"
        print(f"  [{action}] {src.name} → {dst.parent.name}/")
        return
    if copy:
        shutil.copy2(src, dst)
    else:
        shutil.move(str(src), dst)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--copy",    action="store_true", help="Copy files instead of moving them")
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions without touching files")
    args = parser.parse_args()

    for path, label in [(SRC_IMAGES, "source images dir"), (CLASS_CSV, "class CSV"), (FAMILY_CSV, "family mapping CSV")]:
        if not path.exists():
            sys.exit(f"ERROR: {label} not found: {path}")

    species_lut, genus_lut, species_to_family, genus_to_family = build_lookups(CLASS_CSV, FAMILY_CSV)
    print(
        f"Loaded {len(species_lut)} species, {len(genus_lut)} genus, "
        f"{len(species_to_family)} species→family, {len(genus_to_family)} genus→family entries."
    )

    images = sorted(SRC_IMAGES.glob("*.jpg"))
    total  = len(images)
    print(f"Found {total} images in {SRC_IMAGES}")

    counts: dict[str, int] = {}
    for i, src in enumerate(images, 1):
        class_name = class_dir_for(src.stem, species_lut, genus_lut, species_to_family, genus_to_family)
        dst = DST_IMAGES / class_name / src.name
        transfer(src, dst, copy=args.copy, dry_run=args.dry_run)
        counts[class_name] = counts.get(class_name, 0) + 1
        if i % 5000 == 0:
            print(f"  {i}/{total} processed…")

    # Move metadata files
    if not args.dry_run:
        DST_META.mkdir(parents=True, exist_ok=True)
    for meta_src in METADATA_FILES:
        if not meta_src.exists():
            print(f"  [SKIP] metadata file not found: {meta_src}")
            continue
        meta_dst = DST_META / meta_src.name
        if meta_dst.exists():
            print(f"  [SKIP] metadata already at destination: {meta_dst}")
            continue
        if args.dry_run:
            action = "COPY" if args.copy else "MOVE"
            print(f"  [{action}] {meta_src.name} → data/gbif/metadata/")
        elif args.copy:
            shutil.copy2(meta_src, meta_dst)
            print(f"  [COPY] {meta_src.name} → data/gbif/metadata/")
        else:
            shutil.move(str(meta_src), meta_dst)
            print(f"  [MOVE] {meta_src.name} → data/gbif/metadata/")

    # Summary
    unmatched = counts.pop("_unmatched", 0)
    matched   = sum(counts.values())
    print(f"\nDone.")
    print(f"  Matched:   {matched:>6} images across {len(counts)} class directories")
    print(f"  Unmatched: {unmatched:>6} images in _unmatched/")
    print(f"  Total:     {matched + unmatched:>6}")
    if unmatched:
        print(f"\nReview unmatched images at: {DST_IMAGES / '_unmatched'}")


if __name__ == "__main__":
    main()
