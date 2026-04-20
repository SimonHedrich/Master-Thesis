"""
Remove failed images from iNaturalist for classes with 'excellent' coverage status.

Reads filter_results.jsonl to identify failed images, cross-references with
coverage_analysis.csv to restrict deletion to excellent-status classes only.
Use --all-classes to remove failed images from every class regardless of status.

Usage:
    python scripts/dataset_quality/3-remove_failed_images.py                        # dry run, excellent only
    python scripts/dataset_quality/3-remove_failed_images.py --execute              # delete, excellent only
    python scripts/dataset_quality/3-remove_failed_images.py --all-classes          # dry run, all classes
    python scripts/dataset_quality/3-remove_failed_images.py --all-classes --execute  # delete, all classes
"""

import argparse
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).parent.parent.parent
JSONL = REPO_ROOT / "data/inaturalist/filter_results.jsonl"
COVERAGE_CSV = REPO_ROOT / "reports/coverage_analysis.csv"


def main(execute: bool, all_classes: bool) -> None:
    # ── 1. Load class filter ──────────────────────────────────────
    if all_classes:
        excellent = None  # sentinel: no filtering
        print("Targeting all classes (--all-classes).")
    else:
        df = pd.read_csv(COVERAGE_CSV)
        excellent = set(df[df["status"] == "excellent"]["common_name"].str.strip().str.lower())
        print(f"Excellent classes: {len(excellent)}")

    # ── 2. Collect failed iNaturalist images ──────────────────────
    candidates: list[Path] = []
    with open(JSONL) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("passed"):
                continue
            parts = Path(entry["filepath"]).parts
            try:
                cls = parts[parts.index("images") + 1].replace("_", " ")
            except (ValueError, IndexError):
                continue
            if excellent is not None and cls not in excellent:
                continue
            candidates.append(REPO_ROOT / entry["filepath"])

    total = len(candidates)
    total_bytes = sum(p.stat().st_size for p in candidates if p.exists())
    print(f"Failed images to remove: {total:,}")
    print(f"Disk space to free:      {total_bytes / 1024**3:.2f} GB")

    if not execute:
        print("\nDry run — no files deleted. Re-run with --execute to delete.")
        return

    # ── 3. Delete ─────────────────────────────────────────────────
    print("\nDeleting files...")
    deleted = 0
    errors = 0
    report_every = max(1, total // 20)  # progress every 5%

    for i, path in enumerate(candidates, 1):
        try:
            path.unlink()
            deleted += 1
        except FileNotFoundError:
            pass  # already gone
        except OSError as e:
            print(f"  [ERROR] {path}: {e}")
            errors += 1

        if i % report_every == 0 or i == total:
            pct = i / total * 100
            print(f"  {i:>7,} / {total:,}  ({pct:.0f}%)  deleted={deleted:,}  errors={errors}")

    print(f"\nDone. Deleted {deleted:,} files ({total_bytes / 1024**3:.2f} GB freed). Errors: {errors}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove failed iNaturalist images for excellent-status classes.")
    parser.add_argument("--execute", action="store_true",
                        help="Actually delete files (default is dry run)")
    parser.add_argument("--all-classes", action="store_true",
                        help="Remove failed images from all classes, not just excellent-status ones")
    args = parser.parse_args()
    main(execute=args.execute, all_classes=args.all_classes)
