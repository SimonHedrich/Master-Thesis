"""Apply cross-species contamination decisions to the three COCO split JSONs.

Consumes a decisions file and rewrites ``data/real/annotations_{train,val,test}.json``
atomically, removing images or individual annotations as directed.

Decisions file format (``reports/multi_animal_contamination_decisions.json``)
-----------------------------------------------------------------------------
A JSON object keyed by ``file_name`` (byte-identical to COCO ``image.file_name``
and to the review JSON key), mapping to a decision record:

.. code-block:: json

    {
      "data/gbif/images/aardvark/gbif_aardvark_00009.jpg": {
        "decision": "discard"
      },
      "data/inaturalist/images/common_warthog/inat_common_warthog_00868.jpg": {
        "decision": "edit",
        "drop_detection_idx": [5]
      },
      "data/wikimedia/images/red_fox/fox_01.jpg": {
        "decision": "keep"
      }
    }

Decision values
---------------
discard
    Remove the image entry AND all its annotations from the split JSON.
edit
    Remove only the annotation(s) corresponding to the listed
    ``drop_detection_idx`` values.  The image and all other annotations are kept.
    Box matching uses the ``offending_boxes[*].bbox_norm`` field from the review
    JSON: each normalized [cx, cy, w, h] is converted to absolute xyxy using the
    image dimensions, then the COCO annotation with the highest IoU (> 0.5 threshold)
    AND the closest ``conf`` value is removed.  If no match is found at IoU > 0.5,
    a WARNING is printed and the annotation is left unchanged.
keep
    No-op; the record is passed through unchanged.

``--from-review`` default mode
-------------------------------
Derives decisions automatically from
``reports/multi_animal_contamination_review.json`` using the "prefer edit"
strategy from plan §10:

* If only *some* significant boxes are offending → ``edit`` (drop offending boxes).
* If *all* significant boxes are offending    → ``discard`` (no correct box remains).

This produces a runnable artifact without requiring a manual review pass first.

Band-safety assertion
---------------------
Loads ``reports/class_distribution.csv`` and checks whether any class would
drop below its current tier band after applying the decisions.  Tier boundaries:

  Tier 1 → < 100 image-level annotations
  Tier 2 → 100–499
  Tier 3 → 500–1499
  Tier 4 → 1500+

A class "leaves its band" if its projected post-edit annotation count falls
below the lower boundary of its current tier.  If any class would do so, a
loud WARNING listing those classes is printed and execution aborts unless
``--force`` is supplied.

Usage
-----
    # Derive decisions from review JSON, preview diff, write nothing:
    python scripts/dataset_quality/15-apply_contamination_decisions.py \\
        --from-review --dry-run

    # Derive decisions and apply them (with band check):
    python scripts/dataset_quality/15-apply_contamination_decisions.py \\
        --from-review

    # Use a human-reviewed decisions file:
    python scripts/dataset_quality/15-apply_contamination_decisions.py \\
        --decisions reports/multi_animal_contamination_decisions.json

    # Force apply even if a class would leave its band:
    python scripts/dataset_quality/15-apply_contamination_decisions.py \\
        --from-review --force
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

COCO_PATHS = {
    "train": REPO_ROOT / "data" / "real" / "annotations_train.json",
    "val":   REPO_ROOT / "data" / "real" / "annotations_val.json",
    "test":  REPO_ROOT / "data" / "real" / "annotations_test.json",
}

DEFAULT_DECISIONS_PATH = REPO_ROOT / "reports" / "multi_animal_contamination_decisions.json"
DEFAULT_REVIEW_JSON    = REPO_ROOT / "reports" / "multi_animal_contamination_review.json"
CLASS_DIST_PATH        = REPO_ROOT / "reports" / "class_distribution.csv"

# Lower bound (inclusive) of each tier's annotation range
TIER_LOWER_BOUNDS = {
    "1": 0,    # tier 1 has no lower bound — already the smallest tier
    "2": 100,
    "3": 500,
    "4": 1500,
}


# ── Utility ───────────────────────────────────────────────────────────────────

def _pct(num: int, denom: int) -> str:
    if denom == 0:
        return "  n/a"
    return f"{100 * num / denom:5.1f}%"


def _iou(box_a_xyxy: tuple[float, float, float, float],
         box_b_xyxy: tuple[float, float, float, float]) -> float:
    """Compute IoU between two [x1, y1, x2, y2] absolute boxes."""
    ix1 = max(box_a_xyxy[0], box_b_xyxy[0])
    iy1 = max(box_a_xyxy[1], box_b_xyxy[1])
    ix2 = min(box_a_xyxy[2], box_b_xyxy[2])
    iy2 = min(box_a_xyxy[3], box_b_xyxy[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = (box_a_xyxy[2] - box_a_xyxy[0]) * (box_a_xyxy[3] - box_a_xyxy[1])
    area_b = (box_b_xyxy[2] - box_b_xyxy[0]) * (box_b_xyxy[3] - box_b_xyxy[1])
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _bbox_norm_to_xyxy(
    bbox_norm: list[float],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    """Convert review-JSON bbox_norm [cx, cy, w, h] (normalized) to absolute xyxy."""
    cx_n, cy_n, w_n, h_n = bbox_norm
    cx = cx_n * width
    cy = cy_n * height
    w  = w_n  * width
    h  = h_n  * height
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


def _coco_bbox_to_xyxy(
    bbox: list[float],
) -> tuple[float, float, float, float]:
    """Convert COCO bbox [x, y, w, h] (absolute) to xyxy."""
    x, y, w, h = bbox
    return (x, y, x + w, y + h)


def _match_box_to_annotation(
    bbox_norm: list[float],
    offending_conf: float,
    width: int,
    height: int,
    candidates: list[dict],
    iou_threshold: float = 0.5,
) -> int | None:
    """Return the COCO annotation ``id`` best matching a review-JSON offending box.

    Matching strategy:
      1. Convert ``bbox_norm`` [cx, cy, w, h] normalized → absolute xyxy.
      2. For each candidate annotation, compute IoU against that xyxy box.
      3. Keep only candidates with IoU > ``iou_threshold``.
      4. Among those, pick the one whose ``conf`` is closest to ``offending_conf``.

    Returns None if no candidate has IoU > threshold (logged as a warning by
    the caller).
    """
    query_xyxy = _bbox_norm_to_xyxy(bbox_norm, width, height)

    best_id:   int | None = None
    best_iou:  float      = -1.0
    best_conf_diff: float = float("inf")

    for ann in candidates:
        ann_xyxy = _coco_bbox_to_xyxy(ann["bbox"])
        iou = _iou(query_xyxy, ann_xyxy)
        if iou <= iou_threshold:
            continue
        conf_diff = abs(ann.get("conf", 0.0) - offending_conf)
        # Primary sort: highest IoU; secondary sort: closest conf
        if iou > best_iou or (iou == best_iou and conf_diff < best_conf_diff):
            best_iou       = iou
            best_conf_diff = conf_diff
            best_id        = ann["id"]

    return best_id


# ── Decisions derivation from review JSON ────────────────────────────────────

def derive_decisions_from_review(review: dict[str, dict]) -> dict[str, dict]:
    """Generate a decisions dict from the review JSON using the 'prefer edit' rule.

    - All significant boxes are offending → discard.
    - Only some significant boxes are offending → edit (drop offending boxes).
    """
    decisions: dict[str, dict] = {}
    for filepath, entry in review.items():
        n_all      = entry.get("n_significant_boxes", len(entry.get("all_boxes", [])))
        n_offend   = len(entry.get("offending_boxes", []))
        if n_offend == 0:
            # Should not normally appear (review JSON only has flagged images)
            decisions[filepath] = {"decision": "keep"}
        elif n_offend >= n_all:
            decisions[filepath] = {"decision": "discard"}
        else:
            offend_idxs = [
                b["detection_idx"] for b in entry.get("offending_boxes", [])
            ]
            decisions[filepath] = {
                "decision": "edit",
                "drop_detection_idx": offend_idxs,
            }
    return decisions


# ── Band-safety check ──────────────────────────────────────────────────────────

def load_class_tier(path: Path) -> dict[str, str]:
    """Return {class_name: tier_str} from class_distribution.csv."""
    result: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            result[row["class"].strip().lower()] = row["tier"].strip()
    return result


def count_annotations_per_class(
    coco: dict,
) -> tuple[dict[str, int], dict[str, int]]:
    """Return (ann_count_by_class, img_count_by_class) for a COCO dict."""
    cats = {c["id"]: c["name"].lower() for c in coco["categories"]}
    img_to_cat: dict[int, str] = {}
    for img in coco["images"]:
        # class name is the directory name of file_name (e.g. data/gbif/images/<class>/...)
        class_name = Path(img["file_name"]).parent.name.lower().replace("_", " ")
        img_to_cat[img["id"]] = class_name

    ann_counts: dict[str, int] = defaultdict(int)
    img_sets:   dict[str, set] = defaultdict(set)

    for ann in coco["annotations"]:
        cat_id = ann["category_id"]
        class_name = cats.get(cat_id, "unknown")
        ann_counts[class_name] += 1
        img_sets[class_name].add(ann["image_id"])

    img_counts = {k: len(v) for k, v in img_sets.items()}
    return dict(ann_counts), img_counts


def check_band_violations(
    before_ann_counts: dict[str, int],
    after_ann_counts:  dict[str, int],
    tier_map: dict[str, str],
) -> list[tuple[str, str, int, int]]:
    """Return list of (class_name, tier, before_count, after_count) for band violations.

    A violation occurs when a class's annotation count after the edit drops
    below the lower bound of its current tier band.
    """
    violations: list[tuple[str, str, int, int]] = []
    for cls, tier in tier_map.items():
        lower = TIER_LOWER_BOUNDS.get(tier, 0)
        before = before_ann_counts.get(cls, 0)
        after  = after_ann_counts.get(cls, 0)
        if before >= lower and after < lower:
            violations.append((cls, tier, before, after))
    return violations


# ── Per-split processing ──────────────────────────────────────────────────────

def process_split(
    split: str,
    coco_path: Path,
    decisions: dict[str, dict],
    review: dict[str, dict],
    dry_run: bool,
) -> tuple[dict, dict, dict]:
    """Apply decisions to one COCO split.

    Returns
    -------
    (before_counts, after_counts, action_summary)
      before_counts  — {class_name: ann_count} before edits
      after_counts   — {class_name: ann_count} after edits (same structure)
      action_summary — {"discarded": int, "edited": int, "kept": int,
                        "ann_removed": int, "unmatched_boxes": int}
    """
    with coco_path.open(encoding="utf-8") as f:
        coco = json.load(f)

    cats = {c["id"]: c["name"].lower() for c in coco["categories"]}

    before_ann_counts, _ = count_annotations_per_class(coco)

    # Build lookup structures
    img_by_filename: dict[str, dict] = {
        img["file_name"]: img for img in coco["images"]
    }
    anns_by_image_id: dict[int, list[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image_id[ann["image_id"]].append(ann)

    # Track which image ids and annotation ids to remove
    discard_image_ids:     set[int] = set()
    remove_annotation_ids: set[int] = set()

    n_discarded    = 0
    n_edited       = 0
    n_kept         = 0
    n_ann_removed  = 0
    n_unmatched    = 0

    for filepath, dec in decisions.items():
        decision = dec.get("decision", "keep")
        img = img_by_filename.get(filepath)
        if img is None:
            continue  # image not in this split

        image_id = img["id"]

        if decision == "discard":
            discard_image_ids.add(image_id)
            n_discarded += 1

        elif decision == "edit":
            drop_idxs = set(dec.get("drop_detection_idx", []))
            review_entry = review.get(filepath, {})
            candidates = anns_by_image_id.get(image_id, [])

            # Map each drop_detection_idx → offending box in review JSON
            offending_by_idx = {
                b["detection_idx"]: b
                for b in review_entry.get("offending_boxes", [])
            }

            for didx in drop_idxs:
                box = offending_by_idx.get(didx)
                if box is None:
                    print(
                        f"  WARNING [{split}] {filepath}: detection_idx={didx} "
                        f"not found in review offending_boxes — skipping."
                    )
                    n_unmatched += 1
                    continue

                matched_id = _match_box_to_annotation(
                    bbox_norm=box["bbox_norm"],
                    offending_conf=box["megadetector_conf"],
                    width=img["width"],
                    height=img["height"],
                    candidates=candidates,
                )
                if matched_id is None:
                    print(
                        f"  WARNING [{split}] {filepath}: detection_idx={didx} "
                        f"bbox_norm={box['bbox_norm']} → no COCO ann with IoU>0.5 "
                        f"(conf={box['megadetector_conf']:.4f}) — not removed."
                    )
                    n_unmatched += 1
                else:
                    remove_annotation_ids.add(matched_id)
                    n_ann_removed += 1
            n_edited += 1

        else:  # keep or unknown
            n_kept += 1

    # Build output COCO
    new_images = [
        img for img in coco["images"]
        if img["id"] not in discard_image_ids
    ]
    new_anns = [
        ann for ann in coco["annotations"]
        if ann["image_id"] not in discard_image_ids
        and ann["id"] not in remove_annotation_ids
    ]

    new_coco = {
        "info":        coco.get("info", {}),
        "licenses":    coco.get("licenses", []),
        "categories":  coco["categories"],
        "images":      new_images,
        "annotations": new_anns,
    }

    after_ann_counts, _ = count_annotations_per_class(new_coco)

    action_summary = {
        "discarded":    n_discarded,
        "edited":       n_edited,
        "kept":         n_kept,
        "ann_removed":  n_ann_removed,
        "unmatched_boxes": n_unmatched,
        "images_before": len(coco["images"]),
        "images_after":  len(new_images),
        "anns_before":   len(coco["annotations"]),
        "anns_after":    len(new_anns),
    }

    if not dry_run:
        tmp_path = coco_path.with_suffix(".json.tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(new_coco, f)
        os.replace(tmp_path, coco_path)
        print(f"  [{split}] Written atomically → {coco_path}")

    return before_ann_counts, after_ann_counts, action_summary


# ── Diff printer ──────────────────────────────────────────────────────────────

def print_diff(
    split: str,
    action_summary: dict,
    before_ann: dict[str, int],
    after_ann:  dict[str, int],
    tier_map:   dict[str, str],
    violations: list[tuple[str, str, int, int]],
) -> None:
    sep = "─" * 66
    print(f"\n{sep}")
    print(f"[{split}]  images: {action_summary['images_before']:,} → "
          f"{action_summary['images_after']:,}  "
          f"(removed {action_summary['images_before'] - action_summary['images_after']:,})")
    print(f"  annotations: {action_summary['anns_before']:,} → "
          f"{action_summary['anns_after']:,}  "
          f"(removed {action_summary['anns_before'] - action_summary['anns_after']:,})")
    print(f"  decisions applied to this split: "
          f"discard={action_summary['discarded']}  "
          f"edit={action_summary['edited']}  "
          f"keep={action_summary['kept']}")
    if action_summary["unmatched_boxes"]:
        print(f"  WARNING: {action_summary['unmatched_boxes']} offending box(es) "
              f"could not be matched to a COCO annotation (see above).")

    # Per-class delta: only show classes with changes
    changed = {
        cls: (before_ann.get(cls, 0), after_ann.get(cls, 0))
        for cls in set(before_ann) | set(after_ann)
        if before_ann.get(cls, 0) != after_ann.get(cls, 0)
    }
    if changed:
        print(f"\n  Per-class annotation delta ({len(changed)} classes changed):")
        header = f"  {'Class':<35}  {'Before':>7}  {'After':>7}  {'Delta':>7}  Tier"
        print(header)
        print("  " + "─" * (len(header) - 2))
        for cls in sorted(changed):
            b, a = changed[cls]
            delta = a - b
            tier = tier_map.get(cls, "?")
            print(f"  {cls:<35}  {b:>7,}  {a:>7,}  {delta:>+7,}  {tier}")

    if violations:
        print(f"\n  *** BAND VIOLATION WARNING ***  {len(violations)} class(es) "
              f"would leave their tier band:")
        for cls, tier, b, a in violations:
            lb = TIER_LOWER_BOUNDS.get(tier, 0)
            print(f"    {cls:<35}  tier={tier}  "
                  f"{b:,} → {a:,}  (lower bound={lb})")
    else:
        print(f"\n  Band check: OK — no class leaves its tier band.")

    print(sep)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--decisions",
        type=Path,
        default=DEFAULT_DECISIONS_PATH,
        help=f"Path to decisions JSON (default: {DEFAULT_DECISIONS_PATH}). "
             "Required unless --from-review is supplied.",
    )
    parser.add_argument(
        "--from-review",
        action="store_true",
        help="Derive decisions automatically from the review JSON using the "
             "'prefer edit' rule (edit if only some boxes offend; discard if all "
             "boxes offend). Overwrites --decisions for the current run only.",
    )
    parser.add_argument(
        "--review-json",
        type=Path,
        default=DEFAULT_REVIEW_JSON,
        help=f"Review JSON used by --from-review (default: {DEFAULT_REVIEW_JSON}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print the before/after diff; write NO files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Proceed even if one or more classes would leave their tier band.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Load decisions ────────────────────────────────────────────────────────
    if args.from_review:
        if not args.review_json.exists():
            print(f"ERROR: review JSON not found: {args.review_json}", file=sys.stderr)
            sys.exit(1)
        print(f"Deriving decisions from review JSON: {args.review_json}")
        with args.review_json.open(encoding="utf-8") as f:
            review: dict[str, dict] = json.load(f)
        decisions = derive_decisions_from_review(review)
        print(
            f"  {len(decisions):,} decisions derived  "
            f"(discard={sum(1 for d in decisions.values() if d['decision']=='discard')}, "
            f"edit={sum(1 for d in decisions.values() if d['decision']=='edit')}, "
            f"keep={sum(1 for d in decisions.values() if d['decision']=='keep')})"
        )
    else:
        if not args.decisions.exists():
            print(
                f"ERROR: decisions file not found: {args.decisions}\n"
                "Supply --from-review to derive decisions automatically, or provide "
                "a decisions file with --decisions.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Loading decisions from {args.decisions}")
        with args.decisions.open(encoding="utf-8") as f:
            decisions: dict[str, dict] = json.load(f)
        # Load review for edit-mode box matching
        if args.review_json.exists():
            with args.review_json.open(encoding="utf-8") as f:
                review = json.load(f)
        else:
            review = {}
            print(
                "WARNING: review JSON not found; edit decisions cannot match "
                "offending boxes by bbox_norm.  Only conf+IoU from decisions file "
                "will be used.",
                file=sys.stderr,
            )

    # ── Load tier map ─────────────────────────────────────────────────────────
    if CLASS_DIST_PATH.exists():
        tier_map = load_class_tier(CLASS_DIST_PATH)
        print(f"Loaded tier map for {len(tier_map)} classes from {CLASS_DIST_PATH.name}")
    else:
        tier_map = {}
        print(
            f"WARNING: {CLASS_DIST_PATH} not found; band check disabled.",
            file=sys.stderr,
        )

    mode_str = "DRY-RUN (no files written)" if args.dry_run else "LIVE (will write files)"
    print(f"\nMode: {mode_str}")
    print(f"Force: {args.force}")
    print()

    # ── Process each split ────────────────────────────────────────────────────
    all_before: dict[str, int] = defaultdict(int)
    all_after:  dict[str, int] = defaultdict(int)
    all_violations: list[tuple[str, str, int, int]] = []

    for split, coco_path in COCO_PATHS.items():
        if not coco_path.exists():
            print(f"[{split}] COCO JSON not found, skipping: {coco_path}")
            continue

        before_ann, after_ann, summary = process_split(
            split, coco_path, decisions, review, dry_run=True  # always dry-run first
        )

        # Accumulate cross-split totals
        for cls, cnt in before_ann.items():
            all_before[cls] += cnt
        for cls, cnt in after_ann.items():
            all_after[cls] += cnt

        # Band check per split (using combined totals would double-count; check
        # per-split so the warning is specific)
        violations = check_band_violations(before_ann, after_ann, tier_map) if tier_map else []
        all_violations.extend(violations)

        print_diff(split, summary, before_ann, after_ann, tier_map, violations)

    # ── Cross-split band check ────────────────────────────────────────────────
    if tier_map:
        cross_violations = check_band_violations(all_before, all_after, tier_map)
        if cross_violations:
            print(
                f"\n{'='*66}\n"
                f"COMBINED (all splits) BAND VIOLATIONS  "
                f"({len(cross_violations)} class(es)):"
            )
            for cls, tier, b, a in cross_violations:
                lb = TIER_LOWER_BOUNDS.get(tier, 0)
                print(f"  {cls:<35}  tier={tier}  {b:,} → {a:,}  (lb={lb})")
            print('='*66)
            if not args.force and not args.dry_run:
                print(
                    "\nAborting — supply --force to proceed despite band violations.",
                    file=sys.stderr,
                )
                sys.exit(1)

    if args.dry_run:
        print("\nDry-run complete — no files were written.")
        return

    # ── Confirm band check passed or forced, then write ───────────────────────
    if all_violations and not args.force:
        print(
            "\nAborting — one or more classes would leave their tier band.  "
            "Use --force to override.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("\nApplying decisions (atomic writes) …")
    for split, coco_path in COCO_PATHS.items():
        if not coco_path.exists():
            continue
        _, _, summary = process_split(
            split, coco_path, decisions, review, dry_run=False
        )
        print(
            f"  [{split}] done — images: "
            f"{summary['images_before']:,} → {summary['images_after']:,}; "
            f"annotations: {summary['anns_before']:,} → {summary['anns_after']:,}"
        )

    print("\nAll splits updated successfully.")


if __name__ == "__main__":
    main()
