"""Step 2: Assign classes to tiers and produce the manual review queue.

Reads reports/class_distribution.csv (output of Script 8) and
reports/classes_225.csv, then:

  1. Detects anomalies: ghost classes (apostrophe-normalization artifacts from
     Script 8), pseudo-classes (unmatched, human), and canonical classes with
     zero images across all sources.
  2. Builds a prioritised review queue for all Tier 1 and Tier 2 canonical
     classes that have at least one trusted quality-pass image.
  3. Sorts the queue by trusted_quality_pass ascending — smallest class first
     so reviewers can complete one class per session before moving to the next.
  4. Assigns a review_priority label (P1/P2/P3) based on tsn_fail_reason to
     signal label-error risk. All images in trusted_quality_pass are reviewed
     regardless of priority; priority only indicates scrutiny level.

Does not modify any JSONL or source data files. Does not require Docker.

Usage:
    python scripts/dataset_quality/9-manual_review_queue.py
    python scripts/dataset_quality/9-manual_review_queue.py \\
        --dist-csv reports/class_distribution.csv \\
        --classes-csv reports/classes_225.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_DIST_CSV  = REPO_ROOT / "reports" / "class_distribution.csv"
_CLS_CSV   = REPO_ROOT / "reports" / "classes_225.csv"
_OUT_CSV   = REPO_ROOT / "reports" / "manual_review_queue.csv"
_OUT_MD    = REPO_ROOT / "reports" / "manual_review_queue.md"

# Names that are never wildlife training classes.
_PSEUDO_CLASSES = {"unmatched", "human"}

# review_priority label per tsn_fail_reason
_PRIORITY_MAP: dict[str, tuple[str, str]] = {
    "family_mismatch_high_confidence": (
        "P1 HIGH",
        "classifier confident it sees a different family — check every image for label errors",
    ),
    "match_level_no_match": (
        "P2 MED",
        "classifier finds no 225-class match — verify species identity for each image",
    ),
}
_PRIORITY_DEFAULT = (
    "P3 LOW",
    "classifier uncertain but not contradicting — verify each image, expect most to be correct",
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _strip_apostrophe(name: str) -> str:
    """Remove ASCII and Unicode right-single-quote for ghost-class detection."""
    return name.replace("'", "").replace("’", "")


def load_canonical(path: Path) -> set[str]:
    with open(path, encoding="utf-8", newline="") as f:
        return {row["common_name"].strip().lower() for row in csv.DictReader(f)}


def load_distribution(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# ── Row classification ────────────────────────────────────────────────────────

def classify_rows(
    rows: list[dict],
    canonical_set: set[str],
) -> tuple[list[dict], list[tuple[str, str, int]], list[tuple[str, int]]]:
    """Partition distribution rows into canonical, ghost, and anomaly groups.

    Returns:
        canonical_rows — rows whose class name is an exact canonical match
        ghost_pairs    — [(ghost_name, canonical_name, trusted_quality_pass), ...]
        anomaly_rows   — [(name, effective_pool), ...] for pseudo/unknown entries
    """
    norm_to_canonical = {_strip_apostrophe(c): c for c in canonical_set}

    canonical_rows: list[dict] = []
    ghost_pairs: list[tuple[str, str, int]] = []
    anomaly_rows: list[tuple[str, int]] = []

    for row in rows:
        name = row["class"]
        tqp  = int(row["trusted_quality_pass"])
        pool = int(row["effective_pool"])

        if name in canonical_set:
            canonical_rows.append(row)
        elif _strip_apostrophe(name) in norm_to_canonical:
            # Directory name lost apostrophe; images belong to the canonical class.
            ghost_pairs.append((name, norm_to_canonical[_strip_apostrophe(name)], tqp))
        elif name in _PSEUDO_CLASSES:
            anomaly_rows.append((name, pool))
        else:
            anomaly_rows.append((name, pool))

    return canonical_rows, ghost_pairs, anomaly_rows


def find_zero_pool_classes(
    canonical_rows: list[dict],
    canonical_set: set[str],
) -> list[str]:
    seen = {row["class"] for row in canonical_rows}
    return sorted(c for c in canonical_set if c not in seen)


# ── Queue construction ────────────────────────────────────────────────────────

def assign_priority(fail_reason: str) -> tuple[str, str]:
    return _PRIORITY_MAP.get(fail_reason, _PRIORITY_DEFAULT)


def build_queue(canonical_rows: list[dict]) -> list[dict]:
    queue: list[dict] = []
    for row in canonical_rows:
        tier = int(row["tier"])
        tqp  = int(row["trusted_quality_pass"])
        if tier not in (1, 2) or tqp == 0:
            continue
        priority_label, review_notes = assign_priority(row["trusted_sn_fail_reason"])
        queue.append({
            "class":                row["class"],
            "tier":                 tier,
            "effective_pool":       int(row["effective_pool"]),
            "trusted_quality_pass": tqp,
            "tsn_fail_reason":      row["trusted_sn_fail_reason"],
            "review_priority":      priority_label,
            "review_notes":         review_notes,
        })
    queue.sort(key=lambda r: r["trusted_quality_pass"])
    return queue


# ── Output ────────────────────────────────────────────────────────────────────

def write_csv(queue: list[dict], path: Path) -> None:
    if not queue:
        print("Queue is empty — no CSV written.")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(queue[0].keys()))
        writer.writeheader()
        writer.writerows(queue)
    print(f"CSV  → {path.relative_to(REPO_ROOT)}")


def write_md(
    queue: list[dict],
    ghost_pairs: list[tuple[str, str, int]],
    anomaly_rows: list[tuple[str, int]],
    zero_pool: list[str],
    path: Path,
) -> None:
    lines: list[str] = [
        "# Manual Review Queue\n",
        f"**Date:** {date.today()}  ",
        f"**Input:** `reports/class_distribution.csv`  ",
        "**Mode:** STATS ONLY — no data files modified\n",
        "---\n",
        "## Anomalies\n",
        "Entries that are not valid canonical wildlife classes. Exclude from all",
        "downstream training and evaluation pipelines.\n",
    ]

    # Ghost classes
    lines += [
        f"### Ghost classes ({len(ghost_pairs)})\n",
        "Class names where directory names on disk dropped apostrophes that appear",
        "in the canonical 225-class list. Images belong to their canonical counterpart.",
        "**Fix:** restore apostrophes in Script 8 `process_source` (line 140) so that",
        "`Path(...).parent.name` is mapped through the canonical common-name lookup",
        "before being stored as a class key.\n",
        "| Ghost name (in CSV) | Canonical name | Trusted images |",
        "|---|---|---:|",
    ]
    for ghost, canonical, count in sorted(ghost_pairs):
        lines.append(f"| `{ghost}` | `{canonical}` | {count:,} |")

    # Pseudo-classes
    lines += [
        "",
        f"### Pseudo-classes ({len(anomaly_rows)})\n",
        "| Name | Effective pool |",
        "|---|---:|",
    ]
    for name, pool in sorted(anomaly_rows):
        lines.append(f"| `{name}` | {pool:,} |")

    # Zero-pool canonical classes
    lines += [
        "",
        f"### Zero-pool canonical classes ({len(zero_pool)})\n",
    ]
    if zero_pool:
        lines += [
            "Present in `classes_225.csv` but absent from the distribution CSV —",
            "zero images found across all five sources. Synthetic generation is the",
            "only training signal; real evaluation images must be sourced separately.\n",
            "| Class |",
            "|---|",
        ]
        for c in zero_pool:
            lines.append(f"| `{c}` |")
    else:
        lines.append("None — every canonical class has at least one image.\n")

    lines += ["", "---\n"]

    # Workload summary
    tier_stats: dict[int, dict] = {1: {"classes": 0, "images": 0}, 2: {"classes": 0, "images": 0}}
    prio_stats: dict[str, dict] = defaultdict(lambda: {"classes": 0, "images": 0})

    for row in queue:
        t = row["tier"]
        p = row["review_priority"]
        tier_stats[t]["classes"] += 1
        tier_stats[t]["images"]  += row["trusted_quality_pass"]
        prio_stats[p]["classes"] += 1
        prio_stats[p]["images"]  += row["trusted_quality_pass"]

    total_cls = sum(v["classes"] for v in tier_stats.values())
    total_img = sum(v["images"]  for v in tier_stats.values())

    lines += [
        "## Workload Summary\n",
        "All `trusted_quality_pass` images (SN-pass and SN-fail) are reviewed for",
        "each queued class. Work through one class completely before starting the next.\n",
        "| | Classes | Images to review |",
        "|---|---:|---:|",
    ]
    for t in (1, 2):
        ts = tier_stats[t]
        lines.append(f"| Tier {t} | {ts['classes']:,} | {ts['images']:,} |")
    lines.append(f"| **Total** | **{total_cls:,}** | **{total_img:,}** |")

    lines += [
        "",
        "By review priority (informational only — priority signals label-error risk,",
        "not whether images are skipped):\n",
        "| Priority | Classes | Images |",
        "|---|---:|---:|",
    ]
    for p in sorted(prio_stats.keys()):
        ps = prio_stats[p]
        lines.append(f"| {p} | {ps['classes']:,} | {ps['images']:,} |")

    lines += ["", "---\n"]

    # Queue table
    lines += [
        "## Review Queue (Tier 1 + Tier 2)\n",
        "Sorted by `trusted_quality_pass` ascending — smallest class first.",
        "All `trusted_quality_pass` images must be reviewed regardless of priority.\n",
        "| Class | Tier | eff_pool | tq_pass | tsn_fail_reason | Priority |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in queue:
        lines.append(
            f"| {row['class']} "
            f"| {row['tier']} "
            f"| {row['effective_pool']:,} "
            f"| {row['trusted_quality_pass']:,} "
            f"| {row['tsn_fail_reason']} "
            f"| {row['review_priority']} |"
        )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"MD   → {path.relative_to(REPO_ROOT)}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dist-csv", type=Path, default=_DIST_CSV, metavar="PATH",
                        help="class_distribution.csv from Script 8.")
    parser.add_argument("--classes-csv", type=Path, default=_CLS_CSV, metavar="PATH",
                        help="Canonical 225-class list.")
    parser.add_argument("--output-csv", type=Path, default=_OUT_CSV, metavar="PATH")
    parser.add_argument("--output-md",  type=Path, default=_OUT_MD,  metavar="PATH")
    args = parser.parse_args()

    print(f"Loading canonical 225 classes from {args.classes_csv.name} …")
    canonical_set = load_canonical(args.classes_csv)
    print(f"  {len(canonical_set)} classes loaded.")

    print(f"Loading class distribution from {args.dist_csv.name} …")
    rows = load_distribution(args.dist_csv)
    print(f"  {len(rows)} rows loaded.")

    print("Classifying rows …")
    canonical_rows, ghost_pairs, anomaly_rows = classify_rows(rows, canonical_set)
    print(
        f"  Canonical: {len(canonical_rows)}"
        f"  Ghosts: {len(ghost_pairs)}"
        f"  Anomalies: {len(anomaly_rows)}"
    )

    zero_pool = find_zero_pool_classes(canonical_rows, canonical_set)
    if zero_pool:
        print(f"  Zero-pool canonical classes ({len(zero_pool)}):")
        for c in zero_pool:
            print(f"    - {c}")
    else:
        print("  Zero-pool canonical classes: none")

    print("Building review queue …")
    queue = build_queue(canonical_rows)
    print(f"  {len(queue)} classes queued for manual review.")

    total_images = sum(r["trusted_quality_pass"] for r in queue)
    print(f"  Total images to review: {total_images:,}")

    write_csv(queue, args.output_csv)
    write_md(queue, ghost_pairs, anomaly_rows, zero_pool, args.output_md)
    print("\nDone.")


if __name__ == "__main__":
    main()
