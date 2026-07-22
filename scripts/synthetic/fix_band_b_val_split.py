#!/usr/bin/env python3
"""
One-time migration: retroactively assign val split to Band B images in index.jsonl.

Selects 20 images per Band B class as validation using the same "basic behaviour"
criteria as Band A and the synthetic test set (eye_level + three_quarter_front,
medium distance, no occlusion):

  - All 15 eye_level + medium + none images        → val (group 1, images 001–015)
  - First 5 three_quarter_front + medium + none    → val (group 7 partial, images 074–078)
  - All other Band B images                         → train (unchanged)

Band A images are not touched. All fields other than `split` are preserved as-is
(including `status: "generated"`).

Usage:
    uv run python -m scripts.synthetic.fix_band_b_val_split --dry-run  # preview
    uv run python -m scripts.synthetic.fix_band_b_val_split            # apply
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT  = Path(__file__).resolve().parent.parent.parent
INDEX_FILE = REPO_ROOT / "data" / "synthetic" / "index.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch Band B val split in index.jsonl")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print changes without writing")
    args = parser.parse_args()

    records: list[dict] = []
    with open(INDEX_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Track how many three_quarter_front+medium+none have been marked val per class
    tqf_val_count: dict[str, int] = defaultdict(int)

    before_counts: Counter = Counter()
    after_counts:  Counter = Counter()
    changes: list[tuple[str, str, str]] = []  # (filename, old_split, new_split)

    for rec in records:
        band  = rec.get("band", "").upper()
        old   = rec["split"]
        before_counts[(band, old)] += 1

        if band != "B":
            after_counts[(band, old)] += 1
            continue

        shot     = rec.get("shot_type", "")
        distance = rec.get("distance", "")
        occlusion= rec.get("occlusion", "none")
        cls      = rec["class"]

        new = old  # default: unchanged

        if shot == "eye_level" and distance == "medium" and occlusion == "none":
            new = "val"
        elif shot == "three_quarter_front" and distance == "medium" and occlusion == "none":
            if tqf_val_count[cls] < 5:
                new = "val"
                tqf_val_count[cls] += 1

        if new != old:
            changes.append((rec["filename"], old, new))
            rec["split"] = new

        after_counts[(band, new)] += 1

    # Summary
    print(f"Total records   : {len(records)}")
    print(f"Band B changes  : {len(changes)}")
    print()
    print("Before:")
    for k, v in sorted(before_counts.items()):
        print(f"  {k}: {v}")
    print()
    print("After:")
    for k, v in sorted(after_counts.items()):
        print(f"  {k}: {v}")

    # Verify 20 val per Band B class
    val_per_class: Counter = Counter()
    for rec in records:
        if rec.get("band", "").upper() == "B" and rec["split"] == "val":
            val_per_class[rec["class"]] += 1

    print()
    print("Val images per Band B class:")
    all_ok = True
    for cls in sorted(val_per_class):
        cnt = val_per_class[cls]
        flag = "" if cnt == 20 else " ← UNEXPECTED"
        print(f"  {cls:<30s} {cnt}{flag}")
        if cnt != 20:
            all_ok = False
    if all_ok:
        print(f"  All {len(val_per_class)} Band B classes have exactly 20 val images.")
    else:
        print("  WARNING: some classes do not have exactly 20 val images!")

    if args.dry_run:
        print("\n[dry-run] No changes written.")
        return

    # Write updated index
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(records)} records to {INDEX_FILE}")


if __name__ == "__main__":
    main()
