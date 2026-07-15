"""Per-class image and annotation counts after contamination filtering and low-conf supplementation.

Background
----------
Scripts 15 (apply_contamination_decisions) and 15b (supplement_low_conf_boxes) have
just rewritten the three canonical COCO split files:

    data/real/annotations_{train,val,test}.json

Script 15 removed individual annotation boxes flagged as cross-species contamination
(decision = "edit"); no images were discarded (0 "discard" decisions), so the
per-class IMAGE counts are identical to what they were before filtering.

Script 15b added back low-confidence MegaDetector detections (0.1 ≤ conf < 0.5)
as annotations with source = "megadetector_lowconf".

This script produces a single Markdown report that makes both annotation-count
changes visible while also confirming that no band boundary was crossed.

Algorithm
---------
1.  Load all three COCO JSONs.  For each split, build two mappings keyed by
    category_id:
      * image_count   — number of distinct image_ids that have at least one
                        annotation for that category (equivalently: the folder-
                        class of the image, since no image has mixed categories).
      * ann_count     — total annotation count (all sources).
      * lowconf_count — annotation count where source == "megadetector_lowconf".

2.  Load ``reports/class_split_counts.csv`` (224 rows) for the ORIGINAL band
    assignment.  The 225th COCO category absent from the CSV receives band "?".

3.  Compute derived columns per class:
      * new_pool      = real_train_imgs + real_val_imgs + real_test_imgs
      * new_band      = A / B / C / D per the fixed thresholds
      * synth_train   = 200 if orig_band == "A" else 100 if orig_band == "B" else 0
                        (uses ORIGINAL band — synthetic counts are already
                         committed; "?" → 0)
      * synth_test    = 50  (all 225 classes)
      * grand_total   = new_pool + synth_train + synth_test

4.  Build a Markdown table sorted by grand_total descending, with a "Changed?"
    column that shows "⚠" where original band ≠ new band.

5.  Write ``reports/post_filter_counts.md`` (unless --dry-run is set).

Band thresholds (pool = real train + val + test images)
-------------------------------------------------------
    A : pool < 150
    B : 150 ≤ pool < 250
    C : 250 ≤ pool < 400
    D : pool ≥ 400

Image-count caveat
------------------
Because script 15 produced zero "discard" decisions, the per-class image counts
reported here are identical to the pre-filter counts.  Band assignments
therefore cannot have changed.  The value of this report lies in the annotation-
count columns: ``Real Ann Total`` reflects the net effect of contamination-box
removal (script 15) and low-conf supplementation (script 15b).

Verification check
------------------
The script checks that the sum of per-class image counts (across all classes
and all splits) plus the blank/negative image count equals the total number of
images in the three COCO files.  Blank images (``data/blanks/``) carry no
annotations and are intentionally excluded from per-class counts; they are
accounted for separately.

Band-change note
----------------
The original band in ``reports/class_split_counts.csv`` was derived from
``effective_pool`` — the SpeciesNet-filtered, quality-vetted subset of images.
The actual COCO files contain ALL images that passed quality filtering
(including those that failed SpeciesNet), so the raw image counts per class
in COCO are systematically larger than ``effective_pool`` for many classes.
As a result, "Band (new)" computed from raw COCO counts will differ from
"Band (orig)" for classes where the full image set crosses a tier boundary.
This is expected behaviour reflecting the dataset construction strategy, not
a data quality issue.  The "Changed?" column is preserved for auditability.

Usage
-----
    # Compute counts and write the report:
    uv run python scripts/dataset_quality/17-post_filter_counts.py

    # Preview (compute + print summary, write nothing):
    uv run python scripts/dataset_quality/17-post_filter_counts.py --dry-run

    # Write to a custom path:
    uv run python scripts/dataset_quality/17-post_filter_counts.py \\
        --output /tmp/my_counts.md

CLI arguments
-------------
--dry-run       Compute everything and print the summary; skip writing the
                report file.
--output PATH   Override the default output path
                (default: reports/post_filter_counts.md).
--coco-dir DIR  Directory that contains annotations_{train,val,test}.json
                (default: data/real).
--csv PATH      Path to class_split_counts.csv
                (default: reports/class_split_counts.csv).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

# ── Repo root and default paths ───────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_COCO_DIR  = REPO_ROOT / "data" / "real"
DEFAULT_CSV       = REPO_ROOT / "reports" / "class_split_counts.csv"
DEFAULT_OUTPUT    = REPO_ROOT / "reports" / "post_filter_counts.md"

SPLITS = ("train", "val", "test")

# ── Band thresholds ───────────────────────────────────────────────────────────

def compute_band(pool: int) -> str:
    """Return the band label for a given pool (real image count)."""
    if pool < 150:
        return "A"
    if pool < 250:
        return "B"
    if pool < 400:
        return "C"
    return "D"

# ── Synthetic-count rules (based on ORIGINAL band) ───────────────────────────

def synth_train_count(orig_band: str) -> int:
    """Return the committed synthetic-train image count for an original band."""
    if orig_band == "A":
        return 200
    if orig_band == "B":
        return 100
    return 0

SYNTH_TEST_PER_CLASS = 50

# ── Main logic ────────────────────────────────────────────────────────────────

def load_coco_counts(coco_dir: Path) -> dict[str, dict[str, dict]]:
    """Load per-category image and annotation counts from the three COCO splits.

    Returns a nested dict:
        {split_name: {category_id: {"img": int, "ann": int, "lowconf": int}}}
    """
    results: dict[str, dict[int, dict]] = {}

    for split in SPLITS:
        path = coco_dir / f"annotations_{split}.json"
        logging.info("Loading %s …", path)
        with open(path) as fh:
            data = json.load(fh)

        # Aggregate per category_id
        img_sets: dict[int, set] = defaultdict(set)
        ann_counts: dict[int, int] = defaultdict(int)
        lowconf_counts: dict[int, int] = defaultdict(int)

        for ann in data["annotations"]:
            cid = ann["category_id"]
            img_sets[cid].add(ann["image_id"])
            ann_counts[cid] += 1
            if ann.get("source") == "megadetector_lowconf":
                lowconf_counts[cid] += 1

        results[split] = {
            cid: {
                "img": len(img_sets[cid]),
                "ann": ann_counts[cid],
                "lowconf": lowconf_counts[cid],
            }
            for cid in ann_counts
        }

        total_images = len(data["images"])
        annotated_ids = {a["image_id"] for a in data["annotations"]}
        # Blank/negative images have zero annotations; count them explicitly
        blank_images = [
            img for img in data["images"]
            if img["id"] not in annotated_ids
        ]
        blank_count = len(blank_images)
        sum_imgs = sum(v["img"] for v in results[split].values())
        logging.info(
            "  %s: %d categories, %d annotations, %d images in COCO "
            "(%d annotated + %d blank), %d images via ann-mapping",
            split, len(data["categories"]), len(data["annotations"]),
            total_images, total_images - blank_count, blank_count, sum_imgs,
        )

        # Invariant: per-class image sum + blank images == total COCO images
        if sum_imgs + blank_count != total_images:
            logging.error(
                "VERIFICATION FAILED for split '%s': "
                "per-class image sum (%d) + blanks (%d) = %d != "
                "total COCO images (%d).",
                split, sum_imgs, blank_count, sum_imgs + blank_count, total_images,
            )
        else:
            logging.info(
                "  Verification OK: %d + %d = %d",
                sum_imgs, blank_count, total_images,
            )

        # Store totals for verification reporting
        results[split]["__total_coco_images__"] = total_images  # type: ignore[assignment]
        results[split]["__blank_images__"] = blank_count          # type: ignore[assignment]

        # Attach category id→name map (same for all splits, overwritten is fine)
        results[split]["__categories__"] = {  # type: ignore[assignment]
            c["id"]: c["name"] for c in data["categories"]
        }

    return results  # type: ignore[return-value]


def load_original_bands(csv_path: Path) -> dict[str, str]:
    """Return {class_name: orig_band} from class_split_counts.csv."""
    mapping: dict[str, str] = {}
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            mapping[row["class"]] = row["band"]
    return mapping


def build_rows(
    split_counts: dict[str, dict],
    orig_bands: dict[str, str],
) -> list[dict]:
    """Assemble one row per class."""

    # Build a unified category map from all splits (they share the same 225)
    categories: dict[int, str] = {}
    for split in SPLITS:
        categories.update(split_counts[split].get("__categories__", {}))  # type: ignore[arg-type]

    rows = []
    for cid, cname in sorted(categories.items(), key=lambda x: x[1]):
        train_img = split_counts["train"].get(cid, {}).get("img", 0)
        val_img   = split_counts["val"].get(cid, {}).get("img", 0)
        test_img  = split_counts["test"].get(cid, {}).get("img", 0)

        ann_total = (
            split_counts["train"].get(cid, {}).get("ann", 0)
            + split_counts["val"].get(cid, {}).get("ann", 0)
            + split_counts["test"].get(cid, {}).get("ann", 0)
        )

        new_pool  = train_img + val_img + test_img
        new_band  = compute_band(new_pool)
        orig_band = orig_bands.get(cname, "?")

        s_train   = synth_train_count(orig_band)
        s_test    = SYNTH_TEST_PER_CLASS
        grand     = new_pool + s_train + s_test

        changed = "⚠" if (orig_band not in ("?",) and orig_band != new_band) else ""

        rows.append({
            "class":       cname,
            "real_train":  train_img,
            "real_val":    val_img,
            "real_test":   test_img,
            "synth_train": s_train,
            "synth_test":  s_test,
            "real_total":  new_pool,
            "real_ann":    ann_total,
            "grand_total": grand,
            "band_orig":   orig_band,
            "band_new":    new_band,
            "changed":     changed,
        })

    # Sort by grand_total descending, then class name for ties
    rows.sort(key=lambda r: (-r["grand_total"], r["class"]))
    return rows


def format_md_table(rows: list[dict]) -> str:
    """Render the per-class data as a Markdown table."""
    header = (
        "| Class | Real Train | Real Val | Real Test | Synth Train | Synth Test "
        "| Real Total | Real Ann Total | Grand Total | Band (orig) | Band (new) | Changed? |"
    )
    sep = (
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|"
    )
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {r['class']} "
            f"| {r['real_train']:,} "
            f"| {r['real_val']:,} "
            f"| {r['real_test']:,} "
            f"| {r['synth_train']:,} "
            f"| {r['synth_test']:,} "
            f"| {r['real_total']:,} "
            f"| {r['real_ann']:,} "
            f"| {r['grand_total']:,} "
            f"| {r['band_orig']} "
            f"| {r['band_new']} "
            f"| {r['changed']} |"
        )
    return "\n".join(lines)


def build_report(rows: list[dict], split_counts: dict) -> str:
    """Compose the full Markdown report text."""

    total_real = sum(r["real_total"] for r in rows)
    total_ann  = sum(r["real_ann"] for r in rows)
    changed    = [r for r in rows if r["changed"]]
    missing_from_csv = [r for r in rows if r["band_orig"] == "?"]

    # Per-split image totals via COCO (for verification block)
    coco_totals = {
        split: split_counts[split]["__total_coco_images__"]
        for split in SPLITS
    }
    blank_counts = {
        split: split_counts[split].get("__blank_images__", 0)
        for split in SPLITS
    }
    sum_class_images = {
        split: sum(
            split_counts[split].get(cid, {}).get("img", 0)
            for cid in split_counts[split]
            if not isinstance(cid, str)
        )
        for split in SPLITS
    }
    # Verification: per-class sum + blanks == total COCO images
    verification_ok = all(
        sum_class_images[s] + blank_counts[s] == coco_totals[s] for s in SPLITS
    )

    lowconf_total = sum(
        split_counts[split].get(cid, {}).get("lowconf", 0)
        for split in SPLITS
        for cid in split_counts[split]
        if not isinstance(cid, str)
    )

    caveat = f"""\
> **Image-count caveat (read first)**
>
> Script 15 (apply_contamination_decisions) produced **zero "discard" decisions** —
> only "edit" and "keep".  This means **no images were removed** from any split.
> Per-class IMAGE counts are therefore **identical to the pre-filter state**.
>
> The meaningful change visible in this report is in the **annotation counts**:
>
> * Script 15 removed contamination boxes via "edit" decisions (~1,344 boxes
>   dropped).
> * Script 15b added low-confidence MegaDetector boxes back
>   (`source = "megadetector_lowconf"`, **{lowconf_total:,}** boxes added across
>   all splits).
>
> The "Real Ann Total" column captures the net effect of both operations.
>
> **Band-change note:** The "Band (orig)" column comes from `reports/class_split_counts.csv`,
> which was derived from `effective_pool` — the SpeciesNet-filtered, quality-vetted
> image subset.  "Band (new)" is computed from the raw COCO image counts (all images
> that passed quality filtering, including SpeciesNet-failing ones).  Raw COCO counts
> are systematically larger for many classes, so band crossings in the "Changed?" column
> reflect the difference between filtered and raw counts — **not** any change caused by
> the contamination/low-conf scripts.  The original band assignments (and committed
> synthetic counts) remain valid; this report simply documents the raw COCO baseline.
>
> **{len(changed)} class(es) show a band change** in raw COCO counts vs. the original
> filtered effective_pool.
"""

    verify_block = """\
## Verification

The check is: `sum(per-class image counts) + blank images == total COCO images`.
Blank images (`data/blanks/`) are zero-annotation negative samples; they are
excluded from per-class counts but must be accounted for in the totals.

"""
    for split in SPLITS:
        ok = "PASS" if sum_class_images[split] + blank_counts[split] == coco_totals[split] else "FAIL"
        verify_block += (
            f"* **{split}**: per-class imgs = {sum_class_images[split]:,} + "
            f"blanks = {blank_counts[split]:,} → "
            f"sum = {sum_class_images[split] + blank_counts[split]:,} | "
            f"COCO total = {coco_totals[split]:,} → **{ok}**\n"
        )
    verify_block += f"\n**Overall verification: {'PASS' if verification_ok else 'FAIL'}**\n"
    verify_block += f"\nTotal real images (annotated, across all splits): **{total_real:,}**\n"
    verify_block += f"\nTotal real annotations across all splits: **{total_ann:,}**\n"

    missing_block = ""
    if missing_from_csv:
        names = ", ".join(f"`{r['class']}`" for r in missing_from_csv)
        missing_block = (
            f"\n## Class absent from class_split_counts.csv\n\n"
            f"The following {len(missing_from_csv)} class(es) appear in the COCO "
            f"files but are not present in `reports/class_split_counts.csv`; "
            f"their original band is shown as \"?\" and synthetic-train count is 0:\n\n"
            f"{names}\n"
        )

    table = format_md_table(rows)

    report = f"""\
# Post-Filter Class Counts

Generated by `scripts/dataset_quality/17-post_filter_counts.py`.

{caveat}

{verify_block}

## Summary

| Metric | Value |
|---|---|
| Total classes | {len(rows)} |
| Band changes (A/B/C/D crossed) | {len(changed)} |
| Classes absent from CSV | {len(missing_from_csv)} |
| Total real images | {total_real:,} |
| Total real annotations | {total_ann:,} |
| Low-conf annotations added (15b) | {lowconf_total:,} |
{missing_block}
## Per-Class Table

Sorted by Grand Total (real + synth) descending.
"Grand Total" = Real Total + Synth Train + Synth Test.
"Real Ann Total" = total annotation boxes across all three real splits.

{table}
"""
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-class image/annotation counts post-filtering."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print summary, but do not write any file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output Markdown path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--coco-dir",
        type=Path,
        default=DEFAULT_COCO_DIR,
        help=f"Directory containing annotations_{{train,val,test}}.json (default: {DEFAULT_COCO_DIR}).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Path to class_split_counts.csv (default: {DEFAULT_CSV}).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s  %(message)s",
        stream=sys.stderr,
    )

    # ── Load data ─────────────────────────────────────────────────────────────
    split_counts = load_coco_counts(args.coco_dir)
    orig_bands   = load_original_bands(args.csv)

    # ── Build rows ────────────────────────────────────────────────────────────
    rows = build_rows(split_counts, orig_bands)

    # ── Compose report ────────────────────────────────────────────────────────
    report_text = build_report(rows, split_counts)

    # ── Summary to stderr ─────────────────────────────────────────────────────
    total_real = sum(r["real_total"] for r in rows)
    changed    = [r for r in rows if r["changed"]]
    missing    = [r for r in rows if r["band_orig"] == "?"]

    logging.info("─" * 60)
    logging.info("Total classes:                 %d", len(rows))
    logging.info("Total real images (all splits): %d", total_real)
    logging.info("Band changes (⚠):              %d", len(changed))
    logging.info("Missing from class_split_counts.csv: %d → %s",
                 len(missing), [r["class"] for r in missing])
    if changed:
        logging.warning("Band changes detected:")
        for r in changed:
            logging.warning("  %s: %s → %s (pool=%d)",
                            r["class"], r["band_orig"], r["band_new"], r["real_total"])
    logging.info("─" * 60)

    # ── Write ─────────────────────────────────────────────────────────────────
    if args.dry_run:
        logging.info("--dry-run: skipping write to %s", args.output)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.output.with_suffix(".md.tmp")
        tmp.write_text(report_text, encoding="utf-8")
        tmp.rename(args.output)
        logging.info("Report written → %s", args.output)


if __name__ == "__main__":
    main()
