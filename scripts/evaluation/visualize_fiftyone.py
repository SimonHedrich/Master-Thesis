"""Visualize YOLOv5s predictions vs. ground truth in FiftyOne.

Loads the GT subset JSON written by ``run_inference.py`` as a COCO detection
dataset, attaches the predictions JSON as a separate detections field, and
launches the FiftyOne app. Confidence is preserved on each detection so the
sidebar can filter low-conf boxes interactively.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import fiftyone as fo
import fiftyone.utils.coco as fouc

import scripts.training.yolov5s.constants as constants

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = constants.REPO_ROOT / "scripts" / "evaluation" / "outputs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", type=Path, default=DEFAULT_OUTPUT_DIR / "annotations_subset.json")
    p.add_argument("--predictions", type=Path, default=DEFAULT_OUTPUT_DIR / "predictions.json")
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
        help="Bind address. Default 0.0.0.0 so Tailscale peers can reach it; use 127.0.0.1 for local-only.",
    )
    return p.parse_args()


def load_categories(annotations_path: Path) -> list[dict]:
    """Return the COCO ``categories`` array (list of ``{id, name, supercategory}`` dicts)."""
    with annotations_path.open() as f:
        coco = json.load(f)
    return coco["categories"]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()

    logger.info("loading GT from %s (data root %s)", args.annotations, args.data_path)
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
    logger.info("attaching predictions from %s (%d classes)", args.predictions, len(categories))
    fouc.add_coco_labels(
        dataset,
        "predictions",
        str(args.predictions),
        categories,
        coco_id_field="ground_truth_coco_id",
    )

    logger.info("launching FiftyOne app on http://%s:%d", args.address, args.port)
    session = fo.launch_app(dataset, address=args.address, port=args.port)
    session.wait()


if __name__ == "__main__":
    main()
