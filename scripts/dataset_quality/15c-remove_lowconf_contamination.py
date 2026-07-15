"""Remove contaminated low-confidence annotations from the three COCO split JSONs.

Applies the decisions produced by the 14b review session (run against
``reports/lowconf_contamination_review.json``) to remove mislabeled
``source="megadetector_lowconf"`` annotations that were added by script 15b.

Background
----------
Script 14c flagged images where MegaDetector detections at 0.1–0.5 confidence
are classified by SpeciesNet as a genuinely different mammal species (outside
the tolerance band).  Script 14b presented those images for manual review.
This script applies the resulting decisions, removing only the contaminated
low-confidence annotations.

Key difference from script 15
------------------------------
Script 15 targets original (≥ 0.5 conf) annotations.  This script targets
annotations with ``source="megadetector_lowconf"`` (added by script 15b).
To prevent accidents, the removal step verifies that a matched annotation has
``source="megadetector_lowconf"`` before deleting it.

Decisions file format
---------------------
Same schema as ``multi_animal_contamination_decisions.json`` (written by 14b):

.. code-block:: json

    {
      "data/gbif/images/lion/gbif_lion_00001.jpg": {
        "decision": "edit",
        "drop_detection_idx": [3]
      },
      "data/inaturalist/images/wolf/inat_wolf_00512.jpg": {
        "decision": "keep"
      }
    }

``discard`` is accepted but should be rare: it removes the image and ALL its
annotations (both ≥ 0.5 and lowconf tiers).  Use only for images that are
entirely wrong-class.

Usage
-----
    # Preview diff (no files written):
    uv run python scripts/dataset_quality/15c-remove_lowconf_contamination.py \\
        --decisions reports/lowconf_contamination_decisions.json --dry-run

    # Apply:
    uv run python scripts/dataset_quality/15c-remove_lowconf_contamination.py \\
        --decisions reports/lowconf_contamination_decisions.json --force

    # Derive decisions automatically from review JSON (no manual decisions file):
    uv run python scripts/dataset_quality/15c-remove_lowconf_contamination.py \\
        --from-review

See also
--------
    scripts/dataset_quality/14c-flag_lowconf_contamination.py
        — generates lowconf_contamination_review.json
    scripts/dataset_quality/14b-review_contamination.py
        — review UI; run with --review-json reports/lowconf_contamination_review.json
    scripts/dataset_quality/15b-supplement_low_conf_boxes.py
        — added the megadetector_lowconf annotations being cleaned here
    scripts/dataset_quality/17-post_filter_counts.py
        — re-run after this script to update the counts table
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

COCO_PATHS = {
    "train": REPO_ROOT / "data" / "real" / "annotations_train.json",
    "val":   REPO_ROOT / "data" / "real" / "annotations_val.json",
    "test":  REPO_ROOT / "data" / "real" / "annotations_test.json",
}

DEFAULT_DECISIONS_PATH = REPO_ROOT / "reports" / "lowconf_contamination_decisions.json"
DEFAULT_REVIEW_JSON    = REPO_ROOT / "reports" / "lowconf_contamination_review.json"
CLASS_DIST_PATH        = REPO_ROOT / "reports" / "class_distribution.csv"

LOWCONF_SOURCE = "megadetector_lowconf"

TIER_LOWER_BOUNDS = {
    "1": 0,
    "2": 100,
    "3": 500,
    "4": 1500,
}


# ── Utility ───────────────────────────────────────────────────────────────────

def _pct(num: int, denom: int) -> str:
    if denom == 0:
        return "  n/a"
    return f"{100 * num / denom:5.1f}%"


def _iou(
    box_a_xyxy: tuple[float, float, float, float],
    box_b_xyxy: tuple[float, float, float, float],
) -> float:
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
    cx_n, cy_n, w_n, h_n = bbox_norm
    cx = cx_n * width
    cy = cy_n * height
    w  = w_n  * width
    h  = h_n  * height
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


def _coco_bbox_to_xyxy(
    bbox: list[float],
) -> tuple[float, float, float, float]:
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
    """Return COCO annotation id best matching a review-JSON box.

    Only considers candidates with ``source="megadetector_lowconf"`` to prevent
    accidentally removing original (≥ 0.5 conf) annotations.
    """
    query_xyxy = _bbox_norm_to_xyxy(bbox_norm, width, height)

    best_id:        int | None = None
    best_iou:       float      = -1.0
    best_conf_diff: float      = float("inf")

    for ann in candidates:
        if ann.get("source") != LOWCONF_SOURCE:
            continue
        ann_xyxy = _coco_bbox_to_xyxy(ann["bbox"])
        iou = _iou(query_xyxy, ann_xyxy)
        if iou <= iou_threshold:
            continue
        conf_diff = abs(ann.get("conf", 0.0) - offending_conf)
        if iou > best_iou or (iou == best_iou and conf_diff < best_conf_diff):
            best_iou       = iou
            best_conf_diff = conf_diff
            best_id        = ann["id"]

    return best_id


# ── Decisions derivation from review JSON ─────────────────────────────────────

def derive_decisions_from_review(review: dict[str, dict]) -> dict[str, dict]:
    """Generate decisions from review JSON using the 'always edit' rule.

    For the low-conf tier a discard is almost never warranted: the contaminated
    box is simply removed, leaving the image intact with its ≥ 0.5 annotations.
    Only if all significant (low-conf) boxes are offending do we discard.
    """
    decisions: dict[str, dict] = {}
    for filepath, entry in review.items():
        n_all    = entry.get("n_significant_boxes", len(entry.get("all_boxes", [])))
        n_offend = len(entry.get("offending_boxes", []))
        if n_offend == 0:
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
    result: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            result[row["class"].strip().lower()] = row["tier"].strip()
    return result


def count_annotations_per_class(coco: dict) -> tuple[dict[str, int], dict[str, int]]:
    cats = {c["id"]: c["name"].lower() for c in coco["categories"]}
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
    violations: list[tuple[str, str, int, int]] = []
    for cls, tier in tier_map.items():
        lower  = TIER_LOWER_BOUNDS.get(tier, 0)
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

    Returns (before_ann_counts, after_ann_counts, action_summary).
    """
    with coco_path.open(encoding="utf-8") as f:
        coco = json.load(f)

    before_ann_counts, _ = count_annotations_per_class(coco)

    img_by_filename: dict[str, dict] = {
        img["file_name"]: img for img in coco["images"]
    }
    anns_by_image_id: dict[int, list[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image_id[ann["image_id"]].append(ann)

    discard_image_ids:     set[int] = set()
    remove_annotation_ids: set[int] = set()

    n_discarded   = 0
    n_edited      = 0
    n_kept        = 0
    n_ann_removed = 0
    n_unmatched   = 0
    n_wrong_src   = 0  # matched by IoU but source != megadetector_lowconf

    for filepath, dec in decisions.items():
        decision = dec.get("decision", "keep")
        img = img_by_filename.get(filepath)
        if img is None:
            continue

        image_id = img["id"]

        if decision == "discard":
            discard_image_ids.add(image_id)
            n_discarded += 1

        elif decision == "edit":
            drop_idxs    = set(dec.get("drop_detection_idx", []))
            review_entry = review.get(filepath, {})
            candidates   = anns_by_image_id.get(image_id, [])

            offending_by_idx = {
                b["detection_idx"]: b
                for b in review_entry.get("offending_boxes", [])
            }

            for didx in drop_idxs:
                box = offending_by_idx.get(didx)
                if box is None:
                    print(
                        f"  WARNING [{split}] {filepath}: detection_idx={didx} "
                        "not found in review offending_boxes — skipping."
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
                        f"bbox_norm={box['bbox_norm']} → no {LOWCONF_SOURCE} annotation "
                        f"with IoU>0.5 (conf={box['megadetector_conf']:.4f}) — not removed."
                    )
                    n_unmatched += 1
                else:
                    remove_annotation_ids.add(matched_id)
                    n_ann_removed += 1
            n_edited += 1

        else:
            n_kept += 1

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
        "discarded":      n_discarded,
        "edited":         n_edited,
        "kept":           n_kept,
        "ann_removed":    n_ann_removed,
        "unmatched_boxes": n_unmatched,
        "wrong_src":      n_wrong_src,
        "images_before":  len(coco["images"]),
        "images_after":   len(new_images),
        "anns_before":    len(coco["annotations"]),
        "anns_after":     len(new_anns),
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
              "could not be matched to a lowconf annotation (see above).")

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
              "would leave their tier band:")
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
             "'prefer edit' rule.",
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
        help="Proceed even if a class would leave its tier band.",
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
        if args.review_json.exists():
            with args.review_json.open(encoding="utf-8") as f:
                review = json.load(f)
        else:
            review = {}
            print(
                "WARNING: review JSON not found; edit decisions cannot match "
                "offending boxes by bbox_norm.",
                file=sys.stderr,
            )

    # ── Load tier map ─────────────────────────────────────────────────────────
    if CLASS_DIST_PATH.exists():
        tier_map = load_class_tier(CLASS_DIST_PATH)
        print(f"Loaded tier map for {len(tier_map)} classes from {CLASS_DIST_PATH.name}")
    else:
        tier_map = {}
        print(f"WARNING: {CLASS_DIST_PATH} not found; band check disabled.", file=sys.stderr)

    mode_str = "DRY-RUN (no files written)" if args.dry_run else "LIVE (will write files)"
    print(f"\nMode: {mode_str}")
    print(f"Force: {args.force}")
    print(f"Target annotation source: {LOWCONF_SOURCE!r}")
    print()

    # ── Backup before any write ───────────────────────────────────────────────
    if not args.dry_run:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_dir = REPO_ROOT / "data" / "real" / f"backup_pre_lowconf_contamination_{ts}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        for split, coco_path in COCO_PATHS.items():
            if coco_path.exists():
                shutil.copy2(coco_path, backup_dir / coco_path.name)
        print(f"Backup written to {backup_dir.relative_to(REPO_ROOT)}")

    # ── Process each split ────────────────────────────────────────────────────
    all_before: dict[str, int] = defaultdict(int)
    all_after:  dict[str, int] = defaultdict(int)
    all_violations: list[tuple[str, str, int, int]] = []

    for split, coco_path in COCO_PATHS.items():
        if not coco_path.exists():
            print(f"[{split}] COCO JSON not found, skipping: {coco_path}")
            continue

        before_ann, after_ann, summary = process_split(
            split, coco_path, decisions, review, dry_run=True
        )

        for cls, cnt in before_ann.items():
            all_before[cls] += cnt
        for cls, cnt in after_ann.items():
            all_after[cls] += cnt

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
    print(
        "\nNext step — update the counts table:\n"
        "  python3 scripts/dataset_quality/17-post_filter_counts.py"
    )


if __name__ == "__main__":
    main()
