"""
Analyze and optionally delete unused dataset images to free disk space.

Reads filter_results.jsonl from all sources, cross-references with the
dataset_split_manifest.json to identify every on-disk image that is not
used in the final dataset, then reports disk savings per tier/category
before prompting for interactive deletion.

Categories:
  d_surplus    — Band D images that passed quality filtering but were not
                 selected for any split (pool exceeded allocation)
  abc_excluded — Band A/B/C images that passed quality filtering but were
                 excluded during split assignment (bbox area / margin failures)
  quality_failed — Images that were rejected by the filtering pipeline
                   (may be partially gone if 3-remove_failed_images.py ran)

Usage:
    uv run python scripts/dataset_quality/13-cleanup_unused_images.py           # analysis only
    uv run python scripts/dataset_quality/13-cleanup_unused_images.py --execute # analysis + interactive deletion
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
MANIFEST = REPO_ROOT / "reports/dataset_split_manifest.json"
SUMMARY = REPO_ROOT / "reports/dataset_split_summary.json"
FILTER_JSONL = {
    "inaturalist": REPO_ROOT / "data/inaturalist/filter_results.jsonl",
    "gbif": REPO_ROOT / "data/gbif/filter_results.jsonl",
    "wikimedia": REPO_ROOT / "data/wikimedia/filter_results.jsonl",
    "openimages": REPO_ROOT / "data/openimages/filter_results.jsonl",
    "images_cv": REPO_ROOT / "data/images_cv/filter_results.jsonl",
}


# ── helpers ──────────────────────────────────────────────────────────────────

def fmt_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} PB"


def fmt_count(n: int) -> str:
    return f"{n:>9,}"


def _folder_key(name: str) -> str:
    """Normalize a class name to match on-disk folder naming conventions.

    Folder names use underscores for spaces and drop apostrophes, e.g.
    "baird's tapir" → "bairds_tapir".
    """
    return name.replace(" ", "_").replace("'", "")


def build_folder_to_class(class_bands: dict) -> dict:
    """Map folder-style names back to canonical class names."""
    return {_folder_key(cls): cls for cls in class_bands}


def class_from_path(filepath: str, folder_to_class: dict) -> str:
    parts = Path(filepath).parts
    try:
        folder = parts[parts.index("images") + 1]
    except (ValueError, IndexError):
        return "unknown"
    return folder_to_class.get(folder, "unknown")


# ── loading ───────────────────────────────────────────────────────────────────

def load_used_paths() -> set:
    print(f"Loading manifest … ({MANIFEST.stat().st_size // 1024**2} MB)", flush=True)
    with open(MANIFEST) as f:
        data = json.load(f)
    used = {item["filepath"] for item in data["splits"]}
    print(f"  {len(used):,} images assigned to dataset splits")
    return used


def load_class_bands() -> dict:
    with open(SUMMARY) as f:
        data = json.load(f)
    return {cls: info["band"] for cls, info in data.items()}


# ── analysis ──────────────────────────────────────────────────────────────────

def analyze(used_paths: set, class_bands: dict) -> dict:
    """
    Returns:
        {
          "d_surplus":      [{"filepath", "size", "class", "band"}, ...],
          "abc_excluded":   [{"filepath", "size", "class", "band"}, ...],
          "quality_failed": [{"filepath", "size", "class", "band", "stage_failed"}, ...],
        }
    """
    folder_to_class = build_folder_to_class(class_bands)
    results = {"d_surplus": [], "abc_excluded": [], "quality_failed": []}
    total_records = 0

    for source, jsonl_path in FILTER_JSONL.items():
        if not jsonl_path.exists():
            print(f"  [SKIP] {jsonl_path} not found")
            continue

        count = 0
        with open(jsonl_path) as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                fp = rec.get("filepath", "")
                if fp in used_paths:
                    count += 1
                    continue

                full = REPO_ROOT / fp
                if not full.exists():
                    count += 1
                    continue

                size = full.stat().st_size
                cls = class_from_path(fp, folder_to_class)
                band = class_bands.get(cls, "unknown")

                if not rec.get("passed", False):
                    results["quality_failed"].append(
                        {"filepath": fp, "size": size, "class": cls, "band": band,
                         "stage_failed": rec.get("stage_failed", "unknown")}
                    )
                elif band == "D":
                    results["d_surplus"].append(
                        {"filepath": fp, "size": size, "class": cls, "band": band}
                    )
                else:
                    results["abc_excluded"].append(
                        {"filepath": fp, "size": size, "class": cls, "band": band}
                    )
                count += 1

        total_records += count
        print(f"  {source}: {count:,} records processed")

    print(f"  Total records scanned: {total_records:,}")
    return results


# ── reporting ─────────────────────────────────────────────────────────────────

def band_breakdown(items: list) -> dict:
    by_band = defaultdict(lambda: {"count": 0, "size": 0})
    for item in items:
        b = item["band"]
        by_band[b]["count"] += 1
        by_band[b]["size"] += item["size"]
    return by_band


def print_report(results: dict) -> None:
    ds = results["d_surplus"]
    ab = results["abc_excluded"]
    qf = results["quality_failed"]

    ds_size = sum(i["size"] for i in ds)
    ab_size = sum(i["size"] for i in ab)
    qf_size = sum(i["size"] for i in qf)

    print()
    print("=" * 62)
    print("  DATASET CLEANUP ANALYSIS")
    print("=" * 62)

    # Category 1
    print()
    print("Category 1 — Band D surplus")
    print("  (quality-passed, not selected for any split)")
    if ds:
        bb = band_breakdown(ds)
        print(f"  {fmt_count(len(ds))} images  |  {fmt_size(ds_size)}")
        n_classes = len({i['class'] for i in ds})
        print(f"  across {n_classes} classes")
    else:
        print("  (none found on disk)")

    # Category 2
    print()
    print("Category 2 — Hard-excluded (A/B/C)")
    print("  (passed quality filter, excluded during split assignment)")
    ab_known = [i for i in ab if i["band"] != "unknown"]
    ab_unknown = [i for i in ab if i["band"] == "unknown"]
    if ab_known:
        bb = band_breakdown(ab_known)
        for band in sorted(bb):
            d = bb[band]
            print(f"  Band {band}: {fmt_count(d['count'])} images  |  {fmt_size(d['size'])}")
    if ab_unknown:
        sz = sum(i["size"] for i in ab_unknown)
        print(f"  Outside 225-class taxonomy: {fmt_count(len(ab_unknown))} images  |  {fmt_size(sz)}")
    if ab:
        print(f"  Total : {fmt_count(len(ab))} images  |  {fmt_size(ab_size)}")
    else:
        print("  (none found on disk)")

    # Category 3
    print()
    print("Category 3 — Quality-failed")
    print("  (rejected during filtering pipeline; may overlap with prior runs of")
    print("   3-remove_failed_images.py if it was already executed)")
    qf_known = [i for i in qf if i["band"] != "unknown"]
    qf_unknown = [i for i in qf if i["band"] == "unknown"]
    if qf_known:
        bb = band_breakdown(qf_known)
        for band in sorted(bb):
            d = bb[band]
            print(f"  Band {band}: {fmt_count(d['count'])} images  |  {fmt_size(d['size'])}")
    if qf_unknown:
        sz = sum(i["size"] for i in qf_unknown)
        print(f"  Outside 225-class taxonomy: {fmt_count(len(qf_unknown))} images  |  {fmt_size(sz)}")
    if qf:
        print(f"  Total : {fmt_count(len(qf))} images  |  {fmt_size(qf_size)}")
    else:
        print("  (none found on disk)")

    # Options summary
    print()
    print("─" * 62)
    print("  DELETION OPTIONS")
    print("─" * 62)

    opt1_n = len(ds)
    opt1_s = ds_size

    opt2_n = len(ds) + len(ab)
    opt2_s = ds_size + ab_size

    opt3_n = len(qf)
    opt3_s = qf_size

    opt4_n = opt2_n + opt3_n
    opt4_s = opt2_s + opt3_s

    print(f"  1  Band D surplus only              {fmt_count(opt1_n)} images  {fmt_size(opt1_s)}")
    print(f"  2  Band D surplus + hard-excluded   {fmt_count(opt2_n)} images  {fmt_size(opt2_s)}")
    print(f"  3  Quality-failed only              {fmt_count(opt3_n)} images  {fmt_size(opt3_s)}")
    print(f"  4  Everything                       {fmt_count(opt4_n)} images  {fmt_size(opt4_s)}")
    print(f"  5  Exit without deleting")
    print()


# ── deletion ──────────────────────────────────────────────────────────────────

def delete_files(items: list, label: str) -> None:
    total = len(items)
    if total == 0:
        print(f"  Nothing to delete for {label}.")
        return

    deleted = 0
    errors = 0
    report_every = max(1, total // 20)

    print(f"\nDeleting {label} ({total:,} files) …")
    for i, item in enumerate(items, 1):
        path = REPO_ROOT / item["filepath"]
        try:
            path.unlink()
            deleted += 1
        except FileNotFoundError:
            pass
        except OSError as e:
            print(f"  [ERROR] {path}: {e}")
            errors += 1

        if i % report_every == 0 or i == total:
            pct = i / total * 100
            print(f"  {i:>7,} / {total:,}  ({pct:.0f}%)  deleted={deleted:,}  errors={errors}")

    freed = sum(it["size"] for it in items)
    print(f"Done. Deleted {deleted:,} files ({fmt_size(freed)} freed). Errors: {errors}.")


def run_deletion(choice: int, results: dict) -> None:
    ds = results["d_surplus"]
    ab = results["abc_excluded"]
    qf = results["quality_failed"]

    if choice == 1:
        delete_files(ds, "Band D surplus")
    elif choice == 2:
        delete_files(ds, "Band D surplus")
        delete_files(ab, "hard-excluded (A/B/C)")
    elif choice == 3:
        delete_files(qf, "quality-failed")
    elif choice == 4:
        delete_files(ds, "Band D surplus")
        delete_files(ab, "hard-excluded (A/B/C)")
        delete_files(qf, "quality-failed")


# ── main ──────────────────────────────────────────────────────────────────────

def main(execute: bool) -> None:
    if not MANIFEST.exists():
        sys.exit(f"ERROR: manifest not found at {MANIFEST}")
    if not SUMMARY.exists():
        sys.exit(f"ERROR: summary not found at {SUMMARY}")

    used_paths = load_used_paths()
    class_bands = load_class_bands()

    print(f"\nScanning filter_results.jsonl files …")
    results = analyze(used_paths, class_bands)

    print_report(results)

    if not execute:
        print("Analysis only — no files deleted.")
        print("Re-run with --execute to enable the interactive deletion menu.\n")
        return

    while True:
        raw = input("Choose option (1-5): ").strip()
        if raw in ("1", "2", "3", "4", "5"):
            choice = int(raw)
            break
        print("  Please enter a number between 1 and 5.")

    if choice == 5:
        print("Exiting without deleting.")
        return

    labels = {
        1: "Band D surplus only",
        2: "Band D surplus + hard-excluded (A/B/C)",
        3: "Quality-failed only",
        4: "Everything (surplus + hard-excluded + quality-failed)",
    }
    print(f"\nSelected: {labels[choice]}")
    confirm = input("Type 'yes' to confirm deletion: ").strip().lower()
    if confirm != "yes":
        print("Aborted — no files deleted.")
        return

    run_deletion(choice, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Report and optionally delete unused dataset images."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Enable interactive deletion menu (default: analysis/dry-run only)",
    )
    args = parser.parse_args()
    main(execute=args.execute)
