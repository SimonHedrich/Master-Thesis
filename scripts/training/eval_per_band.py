#!/usr/bin/env python3
"""
Compute per-band mAP from YOLOv5 val.py --save-json output.

Joins COCO-format predictions with the dataset split summary to produce
a per-band (A/B/C/D) breakdown of mAP@0.5 and mAP@0.5:0.95.

Prerequisites:
    pip install pycocotools

Usage:
    python scripts/training/eval_per_band.py \\
        --predictions output/yolov5_wildlife/yolov5s_wildlife225_seed42_eval/predictions.json \\
        --split test

    # Or after val.py --save-json which writes to the run directory as *.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

REPO = Path("/home/debian/Master-Thesis")


def load_band_map() -> dict[str, str]:
    """Returns {class_name_lower: band} from dataset_split_summary.json."""
    summary_path = REPO / "reports" / "dataset_split_summary.json"
    summary = json.loads(summary_path.read_text())
    return {name.lower(): info["band"] for name, info in summary.items()}


def load_category_map(split: str) -> dict[int, str]:
    """Returns {coco_category_id: name} from the real split COCO file."""
    coco_path = REPO / "data" / "real" / f"annotations_{split}.json"
    coco = json.loads(coco_path.read_text())
    return {c["id"]: c["name"].lower() for c in coco["categories"]}


def run_coco_eval(gt_path: Path, pred_path: Path) -> dict:
    """Run pycocotools COCO eval, return per-category AP dict."""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        raise SystemExit(
            "pycocotools is required: pip install pycocotools  (or: pip install pycocotools-fix)"
        )

    coco_gt = COCO(str(gt_path))
    coco_dt = coco_gt.loadRes(str(pred_path))

    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    # Per-category AP@0.5 and AP@0.5:0.95
    per_cat: dict[int, dict] = {}
    cat_ids = coco_gt.getCatIds()
    for i, cat_id in enumerate(cat_ids):
        # evaluator.eval["precision"] shape: [TxRxKxAxM]
        # T=10 IoU thresholds, R=101 recall points, K=num_cats, A=4 area ranges, M=3 max dets
        ap50_95 = float(evaluator.eval["precision"][:, :, i, 0, 2].mean())
        ap50 = float(evaluator.eval["precision"][0, :, i, 0, 2].mean())
        per_cat[cat_id] = {"AP50": ap50, "AP50_95": ap50_95}

    return per_cat


def group_by_band(
    per_cat: dict[int, dict],
    cat_map: dict[int, str],
    band_map: dict[str, str],
) -> dict[str, dict]:
    """Aggregate per-category AP into per-band means."""
    bands: dict[str, list] = defaultdict(list)
    unmapped = []

    for cat_id, metrics in per_cat.items():
        name = cat_map.get(cat_id, "").lower()
        band = band_map.get(name)
        if band is None:
            unmapped.append(name)
            continue
        bands[band].append(metrics)

    if unmapped:
        print(f"[warn] {len(unmapped)} categories not found in band map: {unmapped[:5]}…")

    result = {}
    for band in sorted(bands):
        entries = bands[band]
        n = len(entries)
        mean_ap50 = sum(e["AP50"] for e in entries) / n if n else 0.0
        mean_ap50_95 = sum(e["AP50_95"] for e in entries) / n if n else 0.0
        result[band] = {
            "n_classes": n,
            "mAP@0.5": round(mean_ap50, 4),
            "mAP@0.5:0.95": round(mean_ap50_95, 4),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-band mAP from YOLOv5 --save-json output")
    parser.add_argument(
        "--predictions",
        required=True,
        type=Path,
        help="Path to COCO-format predictions JSON from val.py --save-json",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["val", "test"],
        help="Which GT split to evaluate against (default: test)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Optional path to write per-band JSON results",
    )
    args = parser.parse_args()

    gt_path = REPO / "data" / "real" / f"annotations_{args.split}.json"
    if not gt_path.exists():
        raise SystemExit(f"Ground-truth COCO file not found: {gt_path}")
    if not args.predictions.exists():
        raise SystemExit(f"Predictions file not found: {args.predictions}")

    print(f"Running COCO evaluation: {args.predictions} vs {gt_path}")
    per_cat = run_coco_eval(gt_path, args.predictions)

    cat_map = load_category_map(args.split)
    band_map = load_band_map()
    per_band = group_by_band(per_cat, cat_map, band_map)

    print(f"\n{'Band':<6} {'Classes':>7} {'mAP@0.5':>9} {'mAP@0.5:0.95':>13}")
    print("-" * 40)
    for band, metrics in sorted(per_band.items()):
        print(
            f"{band:<6} {metrics['n_classes']:>7} "
            f"{metrics['mAP@0.5']:>9.4f} {metrics['mAP@0.5:0.95']:>13.4f}"
        )

    if args.out:
        args.out.write_text(json.dumps(per_band, indent=2))
        print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
