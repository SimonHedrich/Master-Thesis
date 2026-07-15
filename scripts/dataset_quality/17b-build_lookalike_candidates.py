"""Mine SpeciesNet confusion patterns from manually-rejected multi-animal images.

Background
----------
The manual contamination review (script 14b / reports/contamination_review_decisions.jsonl)
asked human reviewers to classify each flagged multi-animal image as one of two
outcomes:

  ``reject``  — the image genuinely shows only ONE animal species.  The secondary
                MegaDetector boxes are SpeciesNet false-positives.  In other words:
                SpeciesNet looked at a bounding box that is actually part of the
                expected animal (or background) and predicted a different species.

  ``confirm`` — the image really does contain a second, different animal — the flag
                was correct and the image is genuine multi-species contamination.

The ``reject`` images are the interesting case for this script: because SpeciesNet
mislabelled a region of the *same* animal (or its environmental context) as a
different species, the predicted species is a known **confusion partner** for the
expected class.  These confusion partners are exactly the candidates for look-alike
pair curation.

What the script does
--------------------
1. **Replay the decisions JSONL** to obtain the final, deduplicated decision for
   every reviewed filepath.  The log is append-only; an ``undo`` entry erases the
   preceding decision for that filepath.  Only the latest decision per filepath is
   kept.

2. **Load the review JSON** (reports/multi_animal_contamination_review.json).
   Retain only entries whose filepath's latest decision is ``reject``.

3. **Extract confusion pairs**: for each retained image, iterate the
   ``offending_boxes`` list and keep boxes whose ``verdict == "flag"``.  Each such
   box yields a tuple ``(expected_class, pred_common, pred_scientific, match_level)``.

4. **Aggregate**: count ``(expected_class, pred_common)`` occurrences across all
   retained images.  For each pair, record ``pred_scientific`` (taken from the most
   frequent occurrence) and a representative ``match_level`` (the most common
   match_level value seen for that pair).

5. **Per-class image count**: separately count, for each ``expected_class``, the
   number of *distinct* rejected multi-animal images that contributed at least one
   flagged box.  This is the image-level support count used in the Markdown header.

6. **Filter**: retain only ``expected_class`` values whose rejected-image count is
   strictly greater than ``--min-images`` (default 3).  This removes classes with
   too little evidence to trust the confusion signal.

7. **Output — CSV** (``reports/lookalike_candidates.csv``):
   One row per ``(expected_class, pred_common)`` pair, sorted first by
   ``expected_class`` (by its rejected-image count descending), then by pair
   ``count`` descending within each class.  Columns::

       expected_class, predicted_common, predicted_scientific,
       count, match_level, review_decision

   ``review_decision`` is left blank — it is intended for later manual annotation
   (values: ``lookalike``, ``different``, ``skip``).

8. **Output — Markdown** (``reports/lookalike_candidates.md``):
   One ``##`` section per ``expected_class``, sorted by rejected-image count
   descending.  Each section contains a Markdown table of confusion pairs sorted
   by count descending.

CLI
---
    uv run python scripts/dataset_quality/17b-build_lookalike_candidates.py
    uv run python scripts/dataset_quality/17b-build_lookalike_candidates.py --min-images 5
    uv run python scripts/dataset_quality/17b-build_lookalike_candidates.py --dry-run

Options
-------
--decisions-jsonl PATH
    Path to the decisions JSONL (default: reports/contamination_review_decisions.jsonl)
--review-json PATH
    Path to the review JSON (default: reports/multi_animal_contamination_review.json)
--output-csv PATH
    Override path for the candidate CSV (default: reports/lookalike_candidates.csv)
--output-md PATH
    Override path for the candidate Markdown (default: reports/lookalike_candidates.md)
--min-images N
    Minimum number of distinct rejected images per expected_class to include
    (threshold is strictly greater than N; default: 3)
--dry-run
    Compute the full analysis and print a summary, but write no output files.

See also
--------
docs/plans/2026-06-09_flag-cross-species-contamination-multi-box.md
    — original contamination-flagging design spec
scripts/dataset_quality/14-flag_multi_animal_contamination.py
    — produced multi_animal_contamination_review.json
scripts/dataset_quality/14b-review_contamination.py
    — the review server that wrote contamination_review_decisions.jsonl
scripts/dataset_quality/16-build_lookalike_groups.py
    — uses curated look-alike groups for evaluation grouping; this script
      feeds the curation pipeline for that table
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ── Repo paths ────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DECISIONS_JSONL = REPO_ROOT / "reports" / "contamination_review_decisions.jsonl"
DEFAULT_REVIEW_JSON     = REPO_ROOT / "reports" / "multi_animal_contamination_review.json"
DEFAULT_OUTPUT_CSV      = REPO_ROOT / "reports" / "lookalike_candidates.csv"
DEFAULT_OUTPUT_MD       = REPO_ROOT / "reports" / "lookalike_candidates.md"

CSV_COLUMNS = [
    "expected_class",
    "predicted_common",
    "predicted_scientific",
    "count",
    "match_level",
    "review_decision",
]

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(levelname)s %(message)s",
    level=logging.INFO,
    stream=sys.stderr,
)
log = logging.getLogger(__name__)


# ── Step 1: replay the decisions JSONL ───────────────────────────────────────

def replay_decisions(jsonl_path: Path) -> dict[str, str]:
    """Return ``{filepath: latest_decision}`` by replaying the append-only log.

    Rules:
      - Each line is a JSON object with keys ``filepath``, ``decision``, ``ts``.
      - ``decision`` is ``"reject"``, ``"confirm"``, or ``"undo"``.
      - ``undo`` removes the filepath's current decision (if any); subsequent
        entries for the same filepath start fresh.
      - Lines are processed in file order (earliest first), so the last
        non-undo entry wins.
    """
    decisions: dict[str, str] = {}
    n_lines = 0
    n_undo = 0
    with open(jsonl_path, encoding="utf-8") as fh:
        for raw_line in fh:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            n_lines += 1
            rec = json.loads(raw_line)
            fp  = rec["filepath"]
            dec = rec["decision"]
            if dec == "undo":
                decisions.pop(fp, None)
                n_undo += 1
            else:
                decisions[fp] = dec
    log.info(
        "Decisions JSONL: %d lines, %d undo entries → %d unique filepaths",
        n_lines, n_undo, len(decisions),
    )
    return decisions


# ── Step 2-5: extract confusion pairs from rejected images ───────────────────

def extract_pairs(
    decisions: dict[str, str],
    review: dict[str, dict],
) -> tuple[
    Counter,           # pair_counts: (expected_class, pred_common) → int
    dict,              # pair_scientific: (expected_class, pred_common) → Counter of sci names
    dict,              # pair_match_levels: (expected_class, pred_common) → Counter of levels
    Counter,           # class_image_counts: expected_class → n distinct rejected images
    int,               # total rejected images processed (in the review JSON)
]:
    """Extract and aggregate confusion pairs from rejected multi-animal images.

    Only ``offending_boxes`` with ``verdict == "flag"`` contribute to pairs.
    Images that are in the decisions dict as ``reject`` but have no flagged
    offending boxes are counted toward ``class_image_counts`` only if they
    actually contribute at least one flagged box.

    Returns five objects (see type hints above).
    """
    pair_counts:      Counter = Counter()
    pair_scientific:  dict[tuple, Counter] = defaultdict(Counter)
    pair_match_levels: dict[tuple, Counter] = defaultdict(Counter)
    class_image_counts: Counter = Counter()

    rejected_set = {fp for fp, d in decisions.items() if d == "reject"}
    total_processed = 0

    for fp, info in review.items():
        if fp not in rejected_set:
            continue
        total_processed += 1

        expected = info.get("expected_class", "")
        contributed = False

        for box in info.get("offending_boxes", []):
            if box.get("verdict") != "flag":
                continue
            pred_common    = box.get("pred_common", "")
            pred_scientific = box.get("pred_scientific", "")
            match_level    = box.get("match_level", "")

            key = (expected, pred_common)
            pair_counts[key] += 1
            pair_scientific[key][pred_scientific] += 1
            pair_match_levels[key][match_level] += 1
            contributed = True

        if contributed:
            class_image_counts[expected] += 1

    log.info(
        "Processed %d rejected images; found %d distinct (class, predicted) pairs "
        "across %d expected classes.",
        total_processed, len(pair_counts), len(class_image_counts),
    )
    return pair_counts, pair_scientific, pair_match_levels, class_image_counts, total_processed


# ── Step 6-7: build the rows list ────────────────────────────────────────────

def build_rows(
    pair_counts:       Counter,
    pair_scientific:   dict,
    pair_match_levels: dict,
    class_image_counts: Counter,
    min_images:        int,
) -> list[dict]:
    """Build sorted CSV rows, applying the per-class image-count filter.

    Filter: include only ``expected_class`` values where
    ``class_image_counts[expected_class] > min_images``.

    Sort order: classes sorted by rejected-image count descending; within each
    class, pairs sorted by count descending.
    """
    # Classes that pass the filter, sorted by image count descending
    passing_classes = [
        cls for cls, img_count in class_image_counts.items()
        if img_count > min_images
    ]
    passing_classes.sort(key=lambda c: -class_image_counts[c])

    passing_set = set(passing_classes)
    rows: list[dict] = []

    # Group pairs by expected_class
    class_pairs: dict[str, list[tuple]] = defaultdict(list)
    for (expected, pred_common), count in pair_counts.items():
        if expected not in passing_set:
            continue
        sci   = pair_scientific[(expected, pred_common)].most_common(1)[0][0]
        level = pair_match_levels[(expected, pred_common)].most_common(1)[0][0]
        class_pairs[expected].append((pred_common, sci, count, level))

    for cls in passing_classes:
        pairs = class_pairs.get(cls, [])
        # Sort by count descending
        pairs.sort(key=lambda t: -t[2])
        for pred_common, sci, count, level in pairs:
            rows.append({
                "expected_class":    cls,
                "predicted_common":  pred_common,
                "predicted_scientific": sci,
                "count":             count,
                "match_level":       level,
                "review_decision":   "",
            })

    return rows


# ── Output writers ────────────────────────────────────────────────────────────

def write_csv(rows: list[dict], output_path: Path) -> None:
    """Write the candidate table to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    log.info("Wrote %d rows to %s", len(rows), output_path)


def write_markdown(
    rows: list[dict],
    class_image_counts: Counter,
    output_path: Path,
) -> None:
    """Write the per-class Markdown report."""
    # Determine class order from the rows (already sorted)
    seen_classes: list[str] = []
    for row in rows:
        cls = row["expected_class"]
        if cls not in seen_classes:
            seen_classes.append(cls)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("# Lookalike Candidates\n\n")
        fh.write(
            "SpeciesNet confusion pairs mined from manually-rejected multi-animal images.\n"
            "Each pair shows what SpeciesNet predicted on a bounding box that belongs to\n"
            "the expected class (or its environment).  High counts indicate a systematic\n"
            "confusion that may warrant a curated look-alike group.\n\n"
            "The `review_decision` column in the companion CSV is for manual annotation:\n"
            "`lookalike` | `different` | `skip`.\n\n"
        )

        # Group rows by class for Markdown table generation
        class_rows: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            class_rows[row["expected_class"]].append(row)

        for cls in seen_classes:
            n_images = class_image_counts[cls]
            fh.write(f"## {cls}  ({n_images} rejected multi-animal images)\n\n")
            fh.write("| Predicted as | Scientific | Count | Match level |\n")
            fh.write("|---|---|---:|---|\n")
            for row in class_rows[cls]:
                fh.write(
                    f"| {row['predicted_common']} "
                    f"| {row['predicted_scientific']} "
                    f"| {row['count']} "
                    f"| {row['match_level']} |\n"
                )
            fh.write("\n")

    log.info("Wrote %d class sections to %s", len(seen_classes), output_path)


# ── Summary printer ───────────────────────────────────────────────────────────

def print_summary(
    rows: list[dict],
    class_image_counts: Counter,
    total_processed: int,
    min_images: int,
    passing_class_count: int,
    top_n: int = 15,
) -> None:
    """Print a human-readable summary to stdout."""
    print(f"\n{'─' * 60}")
    print("LOOKALIKE CANDIDATES SUMMARY")
    print(f"{'─' * 60}")
    print(f"  Total rejected images processed (in review JSON): {total_processed}")
    print(f"  --min-images threshold (strictly >):              {min_images}")
    print(f"  Candidate classes (pass threshold):               {passing_class_count}")
    print(f"  Total (class, predicted) pairs:                   {len(rows)}")
    print()
    print(f"  Top {top_n} pairs by count:")
    print(f"  {'Count':>5}  {'Expected class':<30}  {'Predicted as':<30}  Match")
    print(f"  {'─'*5}  {'─'*30}  {'─'*30}  {'─'*10}")
    for row in rows[:top_n]:
        print(
            f"  {row['count']:>5}  {row['expected_class']:<30}  "
            f"{row['predicted_common']:<30}  {row['match_level']}"
        )
    print(f"{'─' * 60}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mine SpeciesNet confusion pairs from manually-rejected multi-animal "
            "images and produce a lookalike candidate table."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--decisions-jsonl",
        type=Path,
        default=DEFAULT_DECISIONS_JSONL,
        metavar="PATH",
        help="Append-only decisions log (contamination_review_decisions.jsonl).",
    )
    parser.add_argument(
        "--review-json",
        type=Path,
        default=DEFAULT_REVIEW_JSON,
        metavar="PATH",
        help="Per-image review dict (multi_animal_contamination_review.json).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        metavar="PATH",
        help="Destination CSV path.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=DEFAULT_OUTPUT_MD,
        metavar="PATH",
        help="Destination Markdown path.",
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=3,
        metavar="N",
        help=(
            "Only include expected_class values with strictly more than N "
            "rejected multi-animal images.  Default 3 means >3."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute the analysis and print a summary, but write no files.",
    )
    return parser.parse_args(argv)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Step 1: replay decisions
    log.info("Loading decisions from %s", args.decisions_jsonl)
    decisions = replay_decisions(args.decisions_jsonl)

    # Step 2-5: extract pairs
    log.info("Loading review JSON from %s", args.review_json)
    with open(args.review_json, encoding="utf-8") as fh:
        review: dict[str, dict] = json.load(fh)
    log.info("Review JSON: %d entries", len(review))

    (
        pair_counts,
        pair_scientific,
        pair_match_levels,
        class_image_counts,
        total_processed,
    ) = extract_pairs(decisions, review)

    # Step 6-7: build rows (apply filter, sort)
    rows = build_rows(
        pair_counts, pair_scientific, pair_match_levels,
        class_image_counts, args.min_images,
    )

    passing_class_count = len({r["expected_class"] for r in rows})

    # Print summary regardless of --dry-run
    print_summary(
        rows, class_image_counts, total_processed,
        args.min_images, passing_class_count,
    )

    if args.dry_run:
        log.info("--dry-run: no files written.")
        return 0

    # Step 8: write outputs
    write_csv(rows, args.output_csv)
    write_markdown(rows, class_image_counts, args.output_md)

    log.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
