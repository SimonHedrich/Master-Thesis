"""Visualize YOLOv5s predictions vs. ground truth in FiftyOne.

Default mode
------------
Loads the GT subset JSON written by ``run_inference.py`` as a COCO detection
dataset, attaches the predictions JSON as a separate detections field, and
launches the FiftyOne app. Confidence is preserved on each detection so the
sidebar can filter low-conf boxes interactively.

Flagged-review mode (``--flagged-review``)
-------------------------------------------
Loads ONLY the images flagged in ``multi_animal_contamination_review.json``
and shows them side-by-side with SpeciesNet's per-box prediction so a reviewer
can judge contamination without wading through the full ~145 k train set.

Each sample gets two detections fields:

  ground_truth   — COCO GT boxes (category label)
  sn_prediction  — SpeciesNet result per box, labelled
                   "<pred_common> [<match_level>/<verdict>]"
                   with confidence = pred_top1_score.

Review workflow (FiftyOne sidebar):
  1. Open a sample.  Ground-truth boxes appear in one colour, SpeciesNet
     predictions in another.
  2. Tag samples:
       discard  →  remove the whole image and ALL its annotations
       edit     →  keep the image; only the offending box(es) will be dropped
       (leave untagged to keep unchanged)
  3. Export decisions::

       import fiftyone as fo
       import json, pathlib
       ds = fo.load_dataset("<name>")
       decisions = {}
       for s in ds.iter_samples():
           tags = list(s.tags)
           if "discard" in tags:
               decisions[s.filepath[len(str(IMAGE_ROOT))+1:]] = {"decision": "discard"}
           elif "edit" in tags:
               # drop_detection_idx is filled by the apply script from offending_boxes
               decisions[s.filepath[len(str(IMAGE_ROOT))+1:]] = {
                   "decision": "edit",
                   "drop_detection_idx": []   # leave empty; apply script uses review JSON
               }
       pathlib.Path("reports/multi_animal_contamination_decisions.json").write_text(
           json.dumps(decisions, indent=2))

  4. Run ``scripts/dataset_quality/15-apply_contamination_decisions.py
     --decisions reports/multi_animal_contamination_decisions.json``

``--dry-run`` builds the subset COCO + sn_prediction detections for the first
~50 flagged images, asserts no exceptions, and prints counts — safe to run in
any environment without a display.

Run instructions
----------------
From the repository root (dependencies are managed by uv):

    uv run python -m scripts.evaluation.visualize_fiftyone

Flagged-review mode::

    uv run python -m scripts.evaluation.visualize_fiftyone --flagged-review

Dry-run (headless / CI)::

    uv run python -m scripts.evaluation.visualize_fiftyone --flagged-review --dry-run
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on sys.path so `scripts.*` imports work with plain `uv run`
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
import logging
import os
import tempfile
from typing import Any

import fiftyone as fo
import fiftyone.utils.coco as fouc

import scripts.training.yolov5s.constants as constants

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = constants.REPO_ROOT / "scripts" / "evaluation" / "outputs"
DEFAULT_REVIEW_JSON = constants.REPO_ROOT / "reports" / "multi_animal_contamination_review.json"
DEFAULT_ANNOTATIONS_TRAIN = constants.REPO_ROOT / "data" / "real" / "annotations_train.json"


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Default-mode args
    p.add_argument(
        "--annotations",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "annotations_subset.json",
        help="COCO JSON with GT annotations (default mode) or full split JSON "
             "(flagged-review mode, overridden by --flagged-review default).",
    )
    p.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "predictions.json",
        help="COCO-style predictions JSON (default mode only).",
    )
    p.add_argument(
        "--data-path",
        type=Path,
        default=constants.IMAGE_ROOT,
        help="Image root that `file_name` fields in the COCO JSON resolve against.",
    )
    p.add_argument("--name", type=str, default="yolov5s_test_eval")
    p.add_argument("--port", type=int, default=5155)
    p.add_argument(
        "--address",
        type=str,
        default="0.0.0.0",
        help="Bind address. Default 0.0.0.0 so Tailscale peers can reach it; "
             "use 127.0.0.1 for local-only.",
    )

    # Flagged-review mode args
    p.add_argument(
        "--flagged-review",
        action="store_true",
        help="Enable flagged-review mode: load only contamination-flagged images "
             "and show SpeciesNet predictions alongside GT boxes.",
    )
    p.add_argument(
        "--review-json",
        type=Path,
        default=DEFAULT_REVIEW_JSON,
        help=f"Path to multi_animal_contamination_review.json "
             f"(default: {DEFAULT_REVIEW_JSON}).",
    )
    p.add_argument(
        "--split-annotations",
        type=Path,
        default=DEFAULT_ANNOTATIONS_TRAIN,
        help="Full split COCO JSON used in --flagged-review mode to locate GT "
             "annotations for flagged images (default: annotations_train.json).  "
             "Pass annotations_val.json or annotations_test.json to review those splits.",
    )

    # --dry-run
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the subset COCO and sn_prediction detections for the first "
             "~50 flagged images, assert no exceptions, print counts, exit without "
             "launching FiftyOne.  Safe in headless / CI environments.",
    )

    return p.parse_args()


def load_categories(annotations_path: Path) -> list[dict]:
    """Return the COCO ``categories`` array (list of ``{id, name, supercategory}`` dicts)."""
    with annotations_path.open() as f:
        coco = json.load(f)
    return coco["categories"]


def subset_coco_to_flagged(
    coco_path: Path,
    flagged_filepaths: set[str],
) -> dict[str, Any]:
    """Return a minimal COCO dict containing only flagged images and their annotations.

    Parameters
    ----------
    coco_path:
        Full split COCO JSON path (annotations_train.json etc.).
    flagged_filepaths:
        Set of ``file_name`` values that appear in the review JSON.

    Returns
    -------
    dict with keys ``info``, ``licenses``, ``categories``, ``images``,
    ``annotations`` — a valid COCO detection dataset covering only the flagged
    images.
    """
    with coco_path.open(encoding="utf-8") as f:
        coco = json.load(f)

    flagged_images = [
        img for img in coco["images"]
        if img["file_name"] in flagged_filepaths
    ]
    flagged_ids = {img["id"] for img in flagged_images}
    flagged_anns = [
        ann for ann in coco["annotations"]
        if ann["image_id"] in flagged_ids
    ]

    return {
        "info":        coco.get("info", {}),
        "licenses":    coco.get("licenses", []),
        "categories":  coco["categories"],
        "images":      flagged_images,
        "annotations": flagged_anns,
    }


def build_sn_detections_for_sample(
    file_name: str,
    review_entry: dict,
    img_width: int,
    img_height: int,
) -> fo.Detections:
    """Convert all_boxes from the review JSON into a ``fo.Detections`` object.

    Each box in ``all_boxes`` becomes one ``fo.Detection`` with:
      label  = "<pred_common> [<match_level>/<verdict>]"
      bounding_box = [x_top_left, y_top_left, w, h]  (FiftyOne normalized coords)
      confidence = pred_top1_score

    bbox_norm in the review JSON is ``[cx, cy, w, h]`` normalized (0–1).
    FiftyOne expects ``[x_top_left, y_top_left, w, h]`` normalized.
    """
    detections: list[fo.Detection] = []

    for box in review_entry.get("all_boxes", []):
        cx_n, cy_n, w_n, h_n = box["bbox_norm"]
        x_tl = cx_n - w_n / 2.0
        y_tl = cy_n - h_n / 2.0

        label = (
            f"{box['pred_common']} "
            f"[{box['match_level']}/{box['verdict']}]"
        )

        detections.append(
            fo.Detection(
                label=label,
                bounding_box=[x_tl, y_tl, w_n, h_n],
                confidence=float(box["pred_top1_score"]),
            )
        )

    return fo.Detections(detections=detections)


def load_flagged_dataset(
    review_json: Path,
    annotations: Path,
    data_path: Path,
    name: str,
    max_samples: int | None = None,
) -> tuple[fo.Dataset, int]:
    """Build and return a FiftyOne dataset containing only flagged images.

    Steps:
      1. Read review JSON → collect flagged file_name values.
      2. Subset the COCO JSON to those images → write temp file.
      3. Load subset via ``fo.Dataset.from_dir(COCODetectionDataset)``.
      4. Attach ``sn_prediction`` detections field from review JSON.

    Parameters
    ----------
    max_samples:
        If set, restrict to the first N flagged file_names (for ``--dry-run``).

    Returns
    -------
    (dataset, n_flagged_total) — the dataset and the total number of flagged
    images in the review JSON (before the max_samples cap).
    """
    logger.info("Loading review JSON from %s", review_json)
    with review_json.open(encoding="utf-8") as f:
        review: dict[str, dict] = json.load(f)

    n_flagged_total = len(review)
    flagged_filepaths = set(review.keys())

    if max_samples is not None:
        flagged_filepaths = set(list(flagged_filepaths)[:max_samples])
        logger.info("Dry-run: capping to %d flagged images", max_samples)

    logger.info(
        "Review JSON: %d flagged images; loading from %s",
        n_flagged_total,
        annotations,
    )

    # Build subset COCO dict
    subset_coco = subset_coco_to_flagged(annotations, flagged_filepaths)
    n_found = len(subset_coco["images"])
    logger.info(
        "Subset COCO: %d images found in %s (out of %d flagged keys)",
        n_found,
        annotations.name,
        len(flagged_filepaths),
    )

    # Write temp COCO file so FiftyOne can read it
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    tmp_coco_path = DEFAULT_OUTPUT_DIR / f"_flagged_review_{annotations.stem}.json"
    with tmp_coco_path.open("w", encoding="utf-8") as f:
        json.dump(subset_coco, f)
    logger.info("Wrote temporary subset COCO to %s", tmp_coco_path)

    # Load into FiftyOne
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=str(data_path),
        labels_path=str(tmp_coco_path),
        label_field="ground_truth",
        name=name,
        overwrite=True,
        include_id=True,
    )
    logger.info("FiftyOne dataset loaded: %d samples", len(dataset))

    # Build image metadata lookup for width/height
    img_meta: dict[str, tuple[int, int]] = {
        img["file_name"]: (img["width"], img["height"])
        for img in subset_coco["images"]
    }

    # Attach sn_prediction detections
    n_attached = 0
    with fo.ProgressBar() as pb:
        for sample in pb(dataset):
            # sample.filepath is absolute; strip data_path prefix to get file_name
            rel = str(Path(sample.filepath).relative_to(data_path))
            entry = review.get(rel)
            if entry is None:
                continue
            w, h = img_meta.get(rel, (1, 1))
            sample["sn_prediction"] = build_sn_detections_for_sample(rel, entry, w, h)
            sample.save()
            n_attached += 1

    logger.info("Attached sn_prediction to %d / %d samples", n_attached, len(dataset))
    return dataset, n_flagged_total


# ── Main entry points ──────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()

    if args.flagged_review:
        _run_flagged_review(args)
    else:
        _run_default(args)


def _run_default(args: argparse.Namespace) -> None:
    """Original default mode: load GT subset + predictions, launch app."""
    logger.info(
        "loading GT from %s (data root %s)", args.annotations, args.data_path
    )
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=str(args.data_path),
        labels_path=str(args.annotations),
        label_field="ground_truth",
        name=args.name,
        overwrite=True,
        include_id=True,
    )
    logger.info("loaded %d samples", len(dataset))

    categories = load_categories(args.annotations)
    logger.info(
        "attaching predictions from %s (%d classes)",
        args.predictions,
        len(categories),
    )
    fouc.add_coco_labels(
        dataset,
        "predictions",
        str(args.predictions),
        categories,
        coco_id_field="ground_truth_coco_id",
    )

    logger.info(
        "launching FiftyOne app on http://%s:%d", args.address, args.port
    )
    session = fo.launch_app(dataset, address=args.address, port=args.port)
    session.wait()


def _run_flagged_review(args: argparse.Namespace) -> None:
    """Flagged-review mode: load only contaminated images, show GT + SN preds."""
    # In dry-run mode, cap at 50 images to stay fast
    max_samples = 50 if args.dry_run else None

    dataset, n_flagged_total = load_flagged_dataset(
        review_json=args.review_json,
        annotations=args.split_annotations,
        data_path=args.data_path,
        name=args.name,
        max_samples=max_samples,
    )

    if args.dry_run:
        # Validate and print summary; do NOT launch FiftyOne
        # FiftyOne appends "_detections" when the label_field is "ground_truth"
        gt_field = "ground_truth_detections"
        n_samples = len(dataset)
        n_with_gt = sum(
            1 for s in dataset
            if s.has_field(gt_field)
               and s.get_field(gt_field) is not None
               and len(s.get_field(gt_field).detections) > 0
        )
        n_with_sn = sum(
            1 for s in dataset
            if s.has_field("sn_prediction")
               and s.get_field("sn_prediction") is not None
               and len(s.get_field("sn_prediction").detections) > 0
        )
        print("=" * 60)
        print("DRY-RUN RESULTS (flagged-review mode)")
        print(f"  Total flagged images in review JSON : {n_flagged_total:,}")
        print(f"  Loaded into FiftyOne dataset        : {n_samples}")
        print(f"  Samples with ground_truth detections: {n_with_gt}")
        print(f"  Samples with sn_prediction attached : {n_with_sn}")

        # Spot-check: count distinct verdicts across sn_prediction labels
        from collections import Counter
        verdict_counts: Counter = Counter()
        for sample in dataset:
            sn_field = sample.get_field("sn_prediction") if sample.has_field("sn_prediction") else None
            if sn_field is None:
                continue
            for det in sn_field.detections:
                # label format: "<name> [<match_level>/<verdict>]"
                if "[" in det.label and "]" in det.label:
                    verdict_part = det.label.split("[")[1].rstrip("]")
                    verdict_counts[verdict_part] += 1
        print(f"  sn_prediction label distribution    :")
        for label, cnt in sorted(verdict_counts.items(), key=lambda x: -x[1]):
            print(f"    {label:<30} {cnt}")
        print("=" * 60)
        print("Dry-run passed — no exceptions, no FiftyOne app launched.")
        return

    # Interactive mode: print instructions then launch
    print("\n" + "=" * 60)
    print("FLAGGED-REVIEW MODE")
    print(f"  {n_flagged_total:,} images flagged; {len(dataset)} loaded from")
    print(f"  {args.annotations}")
    print()
    print("  Fields:")
    print("    ground_truth  — COCO GT boxes (category label)")
    print("    sn_prediction — SpeciesNet per-box prediction")
    print("                    '<pred_common> [<match_level>/<verdict>]'")
    print()
    print("  Tagging instructions (FiftyOne sidebar → Tags):")
    print("    discard  → remove image + all annotations")
    print("    edit     → keep image; offending box(es) will be dropped")
    print("    (leave untagged to keep unchanged)")
    print()
    print("  After reviewing, export decisions with the snippet in the")
    print("  module docstring, then run:")
    print("    python scripts/dataset_quality/15-apply_contamination_decisions.py \\")
    print("      --decisions reports/multi_animal_contamination_decisions.json")
    print("=" * 60 + "\n")

    logger.info(
        "launching FiftyOne app on http://%s:%d", args.address, args.port
    )
    session = fo.launch_app(dataset, address=args.address, port=args.port)
    session.wait()


if __name__ == "__main__":
    main()
