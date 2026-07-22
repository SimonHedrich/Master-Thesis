"""Supplement COCO annotations with low-confidence MegaDetector "gray boxes".

Background
----------
The standard GT export pipeline keeps only MegaDetector detections with
confidence ≥ 0.5.  Boxes in the range 0.1 ≤ conf < 0.5 (shown as gray
boxes in the review UI) are excluded despite potentially representing real
animals.  This script adds those boxes back as supplementary annotations
AFTER script 15 has cleaned contamination decisions.

Why after script 15?
  Script 15 removes contaminating boxes from flagged images.  Running 15b
  AFTER ensures we never re-introduce a contamination box that was
  deliberately dropped, and the dedup logic (IoU > 0.5 against existing
  annotations) prevents double-counting any box that is already present.

Algorithm (per COCO image)
--------------------------
1. Look up the image's GT class from the folder name in ``file_name``
   (e.g. ``data/gbif/images/african_buffalo/foo.jpg`` → ``african_buffalo``
   → ``african buffalo`` → category_id lookup).  WARN + skip if not found.

2. Look up the ``speciesnet_results.jsonl`` record by matching ``filepath``
   to COCO ``images[*].file_name``.  Count as ``skip_not_in_jsonl`` if
   missing.

3. For each detection in the record:
   a. Skip if ``speciesnet_skipped`` is True.
   b. Keep only detections with ``conf_lower ≤ megadetector_conf < conf_upper``
      (defaults: 0.1–0.5).
   c. Convert ``bbox_norm`` [cx, cy, w, h] normalized → absolute xyxy via
      ``_bbox_norm_to_xyxy``, clamped to [0,W]×[0,H]; skip if degenerate
      (w ≤ 0 or h ≤ 0 after clamping).
   d. **Dedup check:** skip if IoU > 0.5 with ANY existing annotation in
      that image (prevents duplicating ≥ 0.5 boxes already present and makes
      re-runs idempotent).  New boxes accepted in this pass are added to the
      live dedup list so two overlapping low-conf boxes are not both added.
   e. **Contamination guard:** for images with an ``edit`` decision, recover
      the dropped contamination boxes from the review JSON
      (``offending_boxes`` whose ``detection_idx`` is in
      ``drop_detection_idx``).  Skip any low-conf box whose IoU > 0.5 with
      any of those dropped boxes.

4. Accepted boxes are appended to COCO annotations as::

       {
         "id": <max_existing_id + running_counter>,
         "image_id": <image_id>,
         "category_id": <category_id>,
         "bbox": [x, y, w, h],   # absolute COCO format
         "area": w * h,
         "iscrowd": 0,
         "source": "megadetector_lowconf",
         "conf": <megadetector_conf>
       }

5. ``images`` and ``categories`` are NEVER modified.
6. Each COCO split is written atomically (``<path>.json.tmp`` then
   ``os.replace``).

Species filter (--species-filter)
----------------------------------
``off`` (default) — add every surviving box regardless of SpeciesNet class.
``lenient`` / ``strict`` — not yet implemented; raise NotImplementedError.

Usage
-----
    # Preview without writing:
    uv run python scripts/dataset_quality/15b-supplement_low_conf_boxes.py --dry-run

    # Apply to all sources:
    uv run python scripts/dataset_quality/15b-supplement_low_conf_boxes.py

    # Restrict to a single source:
    uv run python scripts/dataset_quality/15b-supplement_low_conf_boxes.py --source gbif

    # Custom confidence window:
    uv run python scripts/dataset_quality/15b-supplement_low_conf_boxes.py \\
        --conf-lower 0.2 --conf-upper 0.45

    # Idempotency check (should add 0 boxes on second run):
    uv run python scripts/dataset_quality/15b-supplement_low_conf_boxes.py

See also
--------
    scripts/dataset_quality/15-apply_contamination_decisions.py
        — applied before this script; cleans contamination decisions.
    docs/plans/2026-06-09_flag-cross-species-contamination-multi-box.md
        — design spec for the contamination review workflow.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# ── Reuse SPECIESNET_RESULTS_PATHS from script 7 (same importlib trick as 14) ─

_s7_path = Path(__file__).resolve().parent / "7-filter_speciesnet.py"
_s7_spec = importlib.util.spec_from_file_location("filter_speciesnet_s7_15b", _s7_path)
_s7 = importlib.util.module_from_spec(_s7_spec)
sys.modules["filter_speciesnet_s7_15b"] = _s7
_s7_spec.loader.exec_module(_s7)

SPECIESNET_RESULTS_PATHS: dict[str, Path] = _s7.SPECIESNET_RESULTS_PATHS

# ── Constants ──────────────────────────────────────────────────────────────────

COCO_PATHS: dict[str, Path] = {
    "train": REPO_ROOT / "data" / "real" / "annotations_train.json",
    "val":   REPO_ROOT / "data" / "real" / "annotations_val.json",
    "test":  REPO_ROOT / "data" / "real" / "annotations_test.json",
}

DEFAULT_DECISIONS_PATH = REPO_ROOT / "reports" / "multi_animal_contamination_decisions.json"
DEFAULT_REVIEW_JSON    = REPO_ROOT / "reports" / "multi_animal_contamination_review.json"

DEFAULT_CONF_LOWER = 0.1
DEFAULT_CONF_UPPER = 0.5


# ── Geometry helpers (copied verbatim from script 15) ─────────────────────────

def _iou(
    box_a_xyxy: tuple[float, float, float, float],
    box_b_xyxy: tuple[float, float, float, float],
) -> float:
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
    """Convert bbox_norm [cx, cy, w, h] (normalized) to absolute xyxy."""
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


# ── Helper: build list of xyxy boxes for existing COCO annotations ────────────

def _existing_xyxy_list(
    anns: list[dict],
) -> list[tuple[float, float, float, float]]:
    """Return a list of xyxy tuples for existing annotations."""
    return [_coco_bbox_to_xyxy(ann["bbox"]) for ann in anns]


def _overlaps_any(
    candidate_xyxy: tuple[float, float, float, float],
    existing_xyxy: list[tuple[float, float, float, float]],
    iou_threshold: float = 0.5,
) -> bool:
    """Return True if candidate overlaps any existing box at IoU > threshold."""
    for ex in existing_xyxy:
        if _iou(candidate_xyxy, ex) > iou_threshold:
            return True
    return False


# ── Per-split processing ──────────────────────────────────────────────────────

def process_split(
    split: str,
    coco_path: Path,
    jsonl_index: dict[str, dict],         # filepath → jsonl record (all sources)
    decisions: dict[str, dict],            # filepath → decision
    review: dict[str, dict],              # filepath → review entry (may be empty)
    conf_lower: float,
    conf_upper: float,
    species_filter: str,
    dry_run: bool,
) -> dict:
    """Add low-conf boxes to one COCO split.

    Parameters
    ----------
    split
        One of ``train``, ``val``, ``test``.
    coco_path
        Path to the COCO JSON.
    jsonl_index
        Mapping from ``filepath`` to the parsed jsonl record (pre-built across
        all active sources so we need only one dict lookup per image).
    decisions
        Contamination decisions dict (from ``multi_animal_contamination_decisions.json``).
    review
        Review JSON dict (from ``multi_animal_contamination_review.json``).
        May be ``{}`` if the file is absent.
    conf_lower, conf_upper
        Confidence window (inclusive lower, exclusive upper).
    species_filter
        ``'off'`` only for now.
    dry_run
        If True, compute stats but write no files.

    Returns
    -------
    dict with keys:
        images_total, skip_no_category, skip_not_in_jsonl, skip_discard,
        skip_degenerate, skip_dedup, skip_contamination_guard,
        boxes_added, per_class_added  (dict[str, int])
    """
    if not coco_path.exists():
        print(f"  [{split}] COCO JSON not found — skipping: {coco_path}")
        return {}

    with coco_path.open(encoding="utf-8") as f:
        coco = json.load(f)

    categories = coco["categories"]
    # Build category_id lookup: folder_name (underscore, lower) → category_id
    # Category names use spaces; folder names use underscores.
    cats_by_normalized: dict[str, int] = {
        c["name"].lower().replace(" ", "_"): c["id"]
        for c in categories
    }
    # Also support space-form lookup (in case folder already uses spaces)
    cats_by_space: dict[str, int] = {
        c["name"].lower(): c["id"]
        for c in categories
    }
    cat_id_to_name: dict[int, str] = {c["id"]: c["name"] for c in categories}

    # Build per-image annotation index (mutable — we append to it)
    anns_by_image_id: dict[int, list[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image_id[ann["image_id"]].append(ann)

    # Primary category resolution: an image's existing GT annotations are the
    # folder-class (verified: 0 images carry >1 distinct category among their
    # annotations).  This is strictly more robust than folder-name parsing,
    # which fails for classes whose folder drops an apostrophe (e.g.
    # "baird's tapir" stored under "bairds_tapir").  Folder-name lookup remains
    # the fallback for images that have no existing annotation.
    img_existing_category: dict[int, int] = {
        image_id: anns[0]["category_id"]
        for image_id, anns in anns_by_image_id.items()
        if anns
    }

    # Maximum existing annotation id (for assigning new ids)
    max_ann_id = max((ann["id"] for ann in coco["annotations"]), default=0)

    # Counters
    images_total          = len(coco["images"])
    skip_no_category      = 0
    skip_not_in_jsonl     = 0
    skip_discard          = 0
    skip_degenerate       = 0
    skip_dedup            = 0
    skip_contamination_guard = 0
    boxes_added           = 0
    per_class_added: dict[str, int] = defaultdict(int)

    new_annotations: list[dict] = []

    for img in coco["images"]:
        image_id  = img["id"]
        file_name = img["file_name"]
        W         = img["width"]
        H         = img["height"]

        # ── 1. Resolve GT category ────────────────────────────────────────────
        # Prefer the image's existing-annotation category (reliable); fall back
        # to folder-name parsing only for images with no existing annotation.
        category_id = img_existing_category.get(image_id)
        if category_id is None:
            folder = Path(file_name).parent.name.lower()  # e.g. "african_buffalo"
            category_id = cats_by_normalized.get(folder) or cats_by_space.get(
                folder.replace("_", " ")
            )
        if category_id is None:
            print(
                f"  WARNING [{split}] Cannot resolve category for folder "
                f"'{folder}' (file_name={file_name!r}) — skipping image."
            )
            skip_no_category += 1
            continue

        class_name = cat_id_to_name[category_id]

        # ── 2. Look up jsonl record ───────────────────────────────────────────
        rec = jsonl_index.get(file_name)
        if rec is None:
            skip_not_in_jsonl += 1
            continue

        # ── 3. Guard: discard images should not be present post-15 ───────────
        dec = decisions.get(file_name, {})
        if dec.get("decision") == "discard":
            skip_discard += 1
            continue

        # ── 4. Build contamination-guard set for 'edit' images ────────────────
        guard_xyxy: list[tuple[float, float, float, float]] = []
        if dec.get("decision") == "edit":
            drop_idxs = set(dec.get("drop_detection_idx", []))
            review_entry = review.get(file_name)
            if review_entry is None:
                # Review JSON missing for this edit image — disable guard
                # (loud WARNING already printed at load time if review is absent)
                pass
            else:
                for ob in review_entry.get("offending_boxes", []):
                    if ob["detection_idx"] in drop_idxs:
                        guard_box = _bbox_norm_to_xyxy(ob["bbox_norm"], W, H)
                        guard_xyxy.append(guard_box)

        # ── 5. Build live dedup list (existing annotations for this image) ────
        existing_xyxy: list[tuple[float, float, float, float]] = _existing_xyxy_list(
            anns_by_image_id[image_id]
        )

        # ── 6. Iterate over detections ────────────────────────────────────────
        detections: list[dict] = rec.get("speciesnet_detections") or []

        for det in detections:
            # Skip if SpeciesNet flagged as skipped
            if det.get("speciesnet_skipped", False):
                continue

            conf = det.get("megadetector_conf", 0.0)

            # Confidence window filter
            if not (conf_lower <= conf < conf_upper):
                continue

            # Convert bbox to xyxy and clamp
            bbox_norm = det["bbox_norm"]
            x1, y1, x2, y2 = _bbox_norm_to_xyxy(bbox_norm, W, H)
            x1 = max(0.0, min(float(W), x1))
            y1 = max(0.0, min(float(H), y1))
            x2 = max(0.0, min(float(W), x2))
            y2 = max(0.0, min(float(H), y2))
            bw = x2 - x1
            bh = y2 - y1

            # Skip degenerate boxes
            if bw <= 0.0 or bh <= 0.0:
                skip_degenerate += 1
                continue

            candidate_xyxy = (x1, y1, x2, y2)

            # (a) Dedup against existing + already-accepted boxes
            if _overlaps_any(candidate_xyxy, existing_xyxy):
                skip_dedup += 1
                continue

            # (b) Contamination guard: skip if overlaps a dropped offending box
            if guard_xyxy and _overlaps_any(candidate_xyxy, guard_xyxy):
                skip_contamination_guard += 1
                continue

            # (c) Species filter
            if species_filter == "lenient":
                raise NotImplementedError(
                    "--species-filter lenient is not yet implemented.  "
                    "Use --species-filter off (default)."
                )
            elif species_filter == "strict":
                raise NotImplementedError(
                    "--species-filter strict is not yet implemented.  "
                    "Use --species-filter off (default)."
                )
            # species_filter == 'off': accept all surviving boxes

            # ── Accept: build COCO annotation ────────────────────────────────
            max_ann_id += 1
            new_ann = {
                "id":          max_ann_id,
                "image_id":    image_id,
                "category_id": category_id,
                "bbox":        [x1, y1, bw, bh],
                "area":        bw * bh,
                "iscrowd":     0,
                "source":      "megadetector_lowconf",
                "conf":        conf,
            }
            new_annotations.append(new_ann)
            # Add to live dedup list so overlapping low-conf boxes are skipped
            existing_xyxy.append(candidate_xyxy)
            anns_by_image_id[image_id].append(new_ann)

            boxes_added += 1
            per_class_added[class_name] += 1

    # ── Write output ──────────────────────────────────────────────────────────
    if not dry_run and new_annotations:
        coco["annotations"].extend(new_annotations)
        tmp_path = coco_path.with_suffix(".json.tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(coco, f)
        os.replace(tmp_path, coco_path)
        print(f"  [{split}] Written atomically → {coco_path}")
    elif dry_run:
        pass  # nothing to write
    else:
        print(f"  [{split}] No new annotations — file not rewritten.")

    return {
        "images_total":            images_total,
        "skip_no_category":        skip_no_category,
        "skip_not_in_jsonl":       skip_not_in_jsonl,
        "skip_discard":            skip_discard,
        "skip_degenerate":         skip_degenerate,
        "skip_dedup":              skip_dedup,
        "skip_contamination_guard": skip_contamination_guard,
        "boxes_added":             boxes_added,
        "per_class_added":         dict(per_class_added),
    }


# ── Count annotations per class (reused from script 15) ──────────────────────

def count_annotations_per_class(coco: dict) -> dict[str, int]:
    """Return {class_name: annotation_count} for a loaded COCO dict."""
    cats = {c["id"]: c["name"].lower() for c in coco["categories"]}
    ann_counts: dict[str, int] = defaultdict(int)
    for ann in coco["annotations"]:
        class_name = cats.get(ann["category_id"], "unknown")
        ann_counts[class_name] += 1
    return dict(ann_counts)


# ── Build global jsonl index from active sources ──────────────────────────────

def build_jsonl_index(sources: list[str]) -> dict[str, dict]:
    """Stream all active source jsonl files; return {filepath: record}.

    The COCO ``images[*].file_name`` is byte-identical to jsonl ``filepath``.
    Later sources overwrite earlier ones for the same filepath — in practice
    each image appears in exactly one source.
    """
    index: dict[str, dict] = {}
    for source in sources:
        sn_path = SPECIESNET_RESULTS_PATHS.get(source)
        if sn_path is None or not sn_path.exists():
            if sn_path is not None:
                print(f"  [{source}] speciesnet_results.jsonl not found — skipping.")
            continue
        n = 0
        with sn_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                fp = rec.get("filepath")
                if fp:
                    index[fp] = rec
                    n += 1
        print(f"  [{source}] loaded {n:,} records from jsonl.")
    return index


# ── Diff printer ──────────────────────────────────────────────────────────────

def print_split_report(split: str, stats: dict) -> None:
    if not stats:
        return
    sep = "─" * 66
    print(f"\n{sep}")
    print(f"[{split}]  images: {stats['images_total']:,}")
    print(f"  Boxes added:                     {stats['boxes_added']:,}")
    print(f"  Skip — category not found:       {stats['skip_no_category']:,}")
    print(f"  Skip — not in jsonl:             {stats['skip_not_in_jsonl']:,}")
    print(f"  Skip — discard image (guard):    {stats['skip_discard']:,}")
    print(f"  Skip — degenerate bbox:          {stats['skip_degenerate']:,}")
    print(f"  Skip — dedup (IoU > 0.5):        {stats['skip_dedup']:,}")
    print(f"  Skip — contamination guard:      {stats['skip_contamination_guard']:,}")

    per_class = stats.get("per_class_added", {})
    if per_class:
        print(f"\n  Per-class boxes added ({len(per_class)} classes):")
        header = f"  {'Class':<40}  {'Added':>7}"
        print(header)
        print("  " + "─" * (len(header) - 2))
        for cls in sorted(per_class):
            print(f"  {cls:<40}  {per_class[cls]:>7,}")
    print(sep)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--conf-lower",
        type=float,
        default=DEFAULT_CONF_LOWER,
        metavar="CONF",
        help=f"Lower bound of the confidence window (inclusive, default: {DEFAULT_CONF_LOWER}).",
    )
    parser.add_argument(
        "--conf-upper",
        type=float,
        default=DEFAULT_CONF_UPPER,
        metavar="CONF",
        help=f"Upper bound of the confidence window (exclusive, default: {DEFAULT_CONF_UPPER}).",
    )
    parser.add_argument(
        "--decisions",
        type=Path,
        default=DEFAULT_DECISIONS_PATH,
        help=f"Path to contamination decisions JSON (default: {DEFAULT_DECISIONS_PATH}).",
    )
    parser.add_argument(
        "--review-json",
        type=Path,
        default=DEFAULT_REVIEW_JSON,
        help=f"Path to contamination review JSON (default: {DEFAULT_REVIEW_JSON}).  "
             "Used only to recover dropped offending boxes for the contamination guard.",
    )
    parser.add_argument(
        "--species-filter",
        choices=["off", "lenient", "strict"],
        default="off",
        help="Species consistency filter.  'off' (default): add every surviving box.  "
             "'lenient' / 'strict': not yet implemented.",
    )
    parser.add_argument(
        "--source",
        choices=list(SPECIESNET_RESULTS_PATHS.keys()) + ["all"],
        default="all",
        help="Dataset source(s) to load jsonl from (default: all).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print the diff; write NO files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reserved for future use (currently a no-op; kept for CLI symmetry with script 15).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Validate confidence window ─────────────────────────────────────────────
    if not (0.0 <= args.conf_lower < args.conf_upper <= 1.0):
        print(
            f"ERROR: invalid confidence window [{args.conf_lower}, {args.conf_upper}).  "
            "Require 0 ≤ conf_lower < conf_upper ≤ 1.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Load decisions ─────────────────────────────────────────────────────────
    if not args.decisions.exists():
        print(f"ERROR: decisions file not found: {args.decisions}", file=sys.stderr)
        sys.exit(1)
    with args.decisions.open(encoding="utf-8") as f:
        decisions: dict[str, dict] = json.load(f)
    n_edit = sum(1 for d in decisions.values() if d.get("decision") == "edit")
    print(f"Loaded {len(decisions):,} decisions ({n_edit} edits) from {args.decisions.name}")

    # ── Load review JSON (for contamination guard) ────────────────────────────
    review: dict[str, dict] = {}
    if args.review_json.exists():
        with args.review_json.open(encoding="utf-8") as f:
            review = json.load(f)
        print(f"Loaded review JSON: {len(review):,} entries from {args.review_json.name}")
    else:
        if n_edit > 0:
            print(
                f"WARNING: review JSON not found at {args.review_json}  "
                f"but {n_edit} edit decisions exist.  "
                "Contamination guard DISABLED — low-conf boxes may overlap "
                "previously dropped contamination boxes.",
                file=sys.stderr,
            )
        else:
            print(
                f"WARNING: review JSON not found at {args.review_json}  "
                "(no edit decisions, so contamination guard is not needed).",
                file=sys.stderr,
            )

    # ── Determine active sources ───────────────────────────────────────────────
    if args.source == "all":
        active_sources = list(SPECIESNET_RESULTS_PATHS.keys())
    else:
        active_sources = [args.source]

    # ── Build jsonl index ──────────────────────────────────────────────────────
    print(f"\nBuilding jsonl index from sources: {active_sources} …")
    jsonl_index = build_jsonl_index(active_sources)
    print(f"  Total records indexed: {len(jsonl_index):,}")

    # ── Mode banner ────────────────────────────────────────────────────────────
    mode_str = "DRY-RUN (no files written)" if args.dry_run else "LIVE (will write files)"
    print(f"\nMode: {mode_str}")
    print(
        f"Confidence window: [{args.conf_lower}, {args.conf_upper})  "
        f"species_filter={args.species_filter}"
    )
    print()

    # ── Process each split ─────────────────────────────────────────────────────
    total_added = 0
    for split, coco_path in COCO_PATHS.items():
        stats = process_split(
            split=split,
            coco_path=coco_path,
            jsonl_index=jsonl_index,
            decisions=decisions,
            review=review,
            conf_lower=args.conf_lower,
            conf_upper=args.conf_upper,
            species_filter=args.species_filter,
            dry_run=args.dry_run,
        )
        print_split_report(split, stats)
        total_added += stats.get("boxes_added", 0)

    # ── Summary ────────────────────────────────────────────────────────────────
    sep = "=" * 66
    print(f"\n{sep}")
    print(f"Total low-conf boxes {'would be ' if args.dry_run else ''}added: {total_added:,}")
    if args.dry_run:
        print("Dry-run complete — no files were written.")
    else:
        print("All splits updated successfully.")
    print(sep)


if __name__ == "__main__":
    main()
