"""Rename dataset images to a consistent naming scheme.

New filename format:  {prefix}_{species_dir}_{index:05d}.jpg
Species directory names are normalized: spaces replaced with underscores.

Images within each species directory are sorted alphabetically by their
current filename, giving stable and deterministic index numbers.
The script is fully idempotent: already-renamed files are skipped.

Source prefixes:
  gbif        →  gbif_
  inaturalist →  inat_
  images_cv   →  cv_
  openimages  →  oi_

Files updated to reflect the new paths:
  data/{source}/filter_results.jsonl
    - filepath field updated to new relative path
    - for openimages: also fixes the legacy "supplementary_openimages" prefix
  data/openimages/metadata_catalog.csv
    - filename field updated to new filename

Not updated (not needed to preserve existing filter results):
  data/gbif/metadata/SNPredictions_all.json
    Contains only bare filenames used to re-run the gbif metadata stage.
    filter_results.jsonl already carries all results.
  data/inaturalist/metadata/photos.csv
    Contains photo_id integers, not local filenames.

Safety:
  --dry-run  prints all planned changes without touching any file.
  Aborts if target files already exist on disk (indicates a partial run
  from a previous interrupted execution — resolve manually first).

Usage:
    python scripts/rename_dataset_images.py --dry-run        # preview
    python scripts/rename_dataset_images.py                  # rename all
    python scripts/rename_dataset_images.py --source gbif    # one source
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# legacy_rel_images: old path prefix used in filter_results.jsonl when the
# images directory was subsequently renamed on disk.
SOURCES: dict[str, dict] = {
    "gbif": {
        "images_dir":    REPO_ROOT / "data" / "gbif" / "images",
        "filter_results": REPO_ROOT / "data" / "gbif" / "filter_results.jsonl",
        "prefix":        "gbif",
    },
    "inaturalist": {
        "images_dir":    REPO_ROOT / "data" / "inaturalist" / "images",
        "filter_results": REPO_ROOT / "data" / "inaturalist" / "filter_results.jsonl",
        "prefix":        "inat",
    },
    "images_cv": {
        "images_dir":    REPO_ROOT / "data" / "images_cv" / "images",
        "filter_results": REPO_ROOT / "data" / "images_cv" / "filter_results.jsonl",
        "prefix":        "cv",
    },
    "openimages": {
        "images_dir":    REPO_ROOT / "data" / "openimages" / "images",
        "filter_results": REPO_ROOT / "data" / "openimages" / "filter_results.jsonl",
        "prefix":        "oi",
        "legacy_rel_images": "data/supplementary_openimages/images",
    },
}

OI_CATALOG = REPO_ROOT / "data" / "openimages" / "metadata_catalog.csv"


# ── Plan building ─────────────────────────────────────────────────────────────

def build_plan(
    source_name: str,
    cfg: dict,
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]], dict[str, str]]:
    """Scan the images directory and compute the full rename plan.

    Returns:
      file_moves:  [(old_path, new_path), ...]   — only pairs that differ
      dir_renames: [(old_dir, new_dir), ...]     — only dirs whose name changes
      rel_map:     {old_rel_posix: new_rel_posix} — for filter_results update
    """
    images_dir = cfg["images_dir"]
    prefix = cfg["prefix"]
    legacy_rel = cfg.get("legacy_rel_images")

    if not images_dir.exists():
        print(f"  WARNING: {images_dir} not found — skipping {source_name}")
        return [], [], {}

    file_moves: list[tuple[Path, Path]] = []
    dir_renames: list[tuple[Path, Path]] = []
    rel_map: dict[str, str] = {}

    for species_dir in sorted(images_dir.iterdir()):
        if not species_dir.is_dir():
            continue

        new_dir_name = species_dir.name.replace(" ", "_").lstrip("_")
        new_dir = images_dir / new_dir_name

        if new_dir_name != species_dir.name:
            dir_renames.append((species_dir, new_dir))

        images = sorted(
            [p for p in species_dir.iterdir()
             if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
            key=lambda p: p.name,
        )

        for idx, old_path in enumerate(images, start=1):
            new_name = f"{prefix}_{new_dir_name}_{idx:05d}.jpg"
            new_path = new_dir / new_name

            if old_path != new_path:
                file_moves.append((old_path, new_path))

            # Build old relative path as it appears in filter_results.jsonl.
            # For openimages the directory was renamed after the filter run,
            # so filter_results.jsonl still uses the legacy prefix.
            if legacy_rel:
                old_rel = f"{legacy_rel}/{species_dir.name}/{old_path.name}"
            else:
                old_rel = old_path.relative_to(REPO_ROOT).as_posix()

            new_rel = new_path.relative_to(REPO_ROOT).as_posix()
            if old_rel != new_rel:
                rel_map[old_rel] = new_rel

    return file_moves, dir_renames, rel_map


def validate_plan(file_moves: list[tuple[Path, Path]]) -> bool:
    """Return False if the plan has duplicate targets or pre-existing target files.

    A pre-existing target that differs from its source indicates a partial
    rename from an interrupted previous run.
    """
    targets = [new for _, new in file_moves]
    dup_targets = [str(p) for p, n in Counter(targets).items() if n > 1]
    if dup_targets:
        print(f"  ERROR: {len(dup_targets)} duplicate target path(s) in plan!")
        for p in dup_targets[:5]:
            print(f"    {p}")
        return False

    conflicts = [(old, new) for old, new in file_moves if new.exists() and old != new]
    if conflicts:
        print(f"  ERROR: {len(conflicts)} target file(s) already exist on disk.")
        print("  This usually indicates a previous interrupted run.")
        print("  Delete the partially-renamed files and re-run, or use --source")
        print("  to process only the unaffected sources.")
        for old, new in conflicts[:5]:
            print(f"    {old.name}  →  {new.name}  ← already exists")
        return False

    return True


# ── Apply ─────────────────────────────────────────────────────────────────────

def apply_file_moves(file_moves: list[tuple[Path, Path]]) -> int:
    """Create target directories, then rename/move all files. Returns moved count."""
    if not file_moves:
        return 0
    for d in {new.parent for _, new in file_moves}:
        d.mkdir(parents=True, exist_ok=True)

    moved = 0
    for old_path, new_path in tqdm(file_moves, desc="  files", unit="file", leave=False):
        if not old_path.exists():
            if new_path.exists():
                continue  # already done in a previous complete run
            print(f"\n  WARNING: source missing and target absent: {old_path}")
            continue
        old_path.rename(new_path)
        moved += 1
    return moved


def remove_old_dirs(dir_renames: list[tuple[Path, Path]]) -> int:
    """Remove old species dirs that were renamed (should be empty now)."""
    removed = 0
    for old_dir, _ in dir_renames:
        if not old_dir.exists():
            removed += 1
            continue
        remaining = list(old_dir.iterdir())
        if remaining:
            print(f"  WARNING: old dir not empty after rename: {old_dir} "
                  f"({len(remaining)} item(s) remain)")
            continue
        old_dir.rmdir()
        removed += 1
    return removed


# ── Metadata updates ──────────────────────────────────────────────────────────

def update_filter_results(jsonl_path: Path, rel_map: dict[str, str]) -> int:
    """Rewrite filter_results.jsonl with updated filepath fields.

    Writes atomically via a .tmp sidecar. Returns count of updated entries.
    """
    if not jsonl_path.exists() or not rel_map:
        return 0

    entries = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    updated = 0
    for entry in entries:
        new_fp = rel_map.get(entry.get("filepath", ""))
        if new_fp:
            entry["filepath"] = new_fp
            updated += 1

    if updated:
        tmp = jsonl_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
        tmp.rename(jsonl_path)

    return updated


GBIF_PREDICTIONS_JSON = REPO_ROOT / "data" / "gbif" / "metadata" / "SNPredictions_all.json"


def update_gbif_predictions(file_moves: list[tuple[Path, Path]]) -> int:
    """Update the filepath field in SNPredictions_all.json to reflect renamed files.

    Returns count of updated predictions.
    """
    if not GBIF_PREDICTIONS_JSON.exists() or not file_moves:
        return 0

    bare_map = {old.name: new.name for old, new in file_moves}

    with open(GBIF_PREDICTIONS_JSON, encoding="utf-8") as f:
        data = json.load(f)

    updated = 0
    for pred in data.get("predictions", []):
        new_name = bare_map.get(pred.get("filepath", ""))
        if new_name:
            pred["filepath"] = new_name
            updated += 1

    if updated:
        tmp = GBIF_PREDICTIONS_JSON.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        tmp.rename(GBIF_PREDICTIONS_JSON)

    return updated


def update_oi_catalog(file_moves: list[tuple[Path, Path]]) -> int:
    """Update the filename column in openimages/metadata_catalog.csv.

    Returns count of updated rows.
    """
    if not OI_CATALOG.exists() or not file_moves:
        return 0

    filename_map = {old.name: new.name for old, new in file_moves}

    with open(OI_CATALOG, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    updated = 0
    for row in rows:
        new_fn = filename_map.get(row.get("filename", ""))
        if new_fn:
            row["filename"] = new_fn
            updated += 1

    if updated:
        tmp = OI_CATALOG.with_suffix(".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        tmp.rename(OI_CATALOG)

    return updated


# ── Orchestration ─────────────────────────────────────────────────────────────

def process_source(source_name: str, cfg: dict, dry_run: bool) -> None:
    tag = "[DRY RUN] " if dry_run else ""
    print(f"\n{'═' * 60}")
    print(f"  {tag}{source_name}")
    print(f"{'═' * 60}")

    print("  Building rename plan …")
    file_moves, dir_renames, rel_map = build_plan(source_name, cfg)

    if not file_moves and not dir_renames:
        print("  Nothing to rename — already up to date.")
        return

    print(f"  Files to rename  : {len(file_moves):>8,}")
    print(f"  Dirs to rename   : {len(dir_renames):>8}")
    print(f"  filter_results   : {len(rel_map):>8,}  entries to update")

    if dry_run:
        print()
        print("  Sample renames (first 5 files):")
        for old, new in file_moves[:5]:
            print(f"    {old.relative_to(REPO_ROOT)}")
            print(f"    → {new.relative_to(REPO_ROOT)}")
        if len(file_moves) > 5:
            print(f"    … and {len(file_moves) - 5:,} more")
        if dir_renames:
            print("  Directory renames:")
            for old, new in dir_renames[:5]:
                print(f"    {old.name}  →  {new.name}")
            if len(dir_renames) > 5:
                print(f"    … and {len(dir_renames) - 5} more")
        return

    if not validate_plan(file_moves):
        print("  Aborting this source.")
        return

    moved = apply_file_moves(file_moves)
    print(f"  Files renamed    : {moved:>8,}")

    removed = remove_old_dirs(dir_renames)
    if removed:
        print(f"  Old dirs removed : {removed:>8}")

    fr_updated = update_filter_results(cfg["filter_results"], rel_map)
    print(f"  filter_results   : {fr_updated:>8,}  entries updated")

    if source_name == "gbif":
        gbif_updated = update_gbif_predictions(file_moves)
        print(f"  SNPredictions    : {gbif_updated:>8,}  filepath fields updated")

    if source_name == "openimages":
        oi_updated = update_oi_catalog(file_moves)
        print(f"  metadata_catalog : {oi_updated:>8,}  rows updated")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        choices=list(SOURCES),
        default=None,
        help="Process only this source (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview planned changes without modifying any files",
    )
    args = parser.parse_args()

    sources = {args.source: SOURCES[args.source]} if args.source else SOURCES
    for name, cfg in sources.items():
        process_source(name, cfg, dry_run=args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
