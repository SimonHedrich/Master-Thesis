"""Run YOLOv5s inference on a sampled subset of the real test set.

Writes two JSON files to ``--output-dir``:

* ``annotations_subset.json`` — a COCO-format dataset JSON containing only
  the sampled images plus the full ``categories`` array. Consumed by
  ``visualize_fiftyone.py`` as ground truth.
* ``predictions.json`` — a flat COCO **results** array with one entry per
  predicted box: ``{image_id, category_id, bbox=[x,y,w,h], score}``. Bboxes
  are in pixel coordinates on the original image.

The 100-image sample is deterministic via the module-level ``SEED`` so that
re-running this script against a different checkpoint produces a directly
comparable view in FiftyOne.

Usage:
    uv run python -m scripts.evaluation.run_inference
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import cv2
import torch
from tqdm import tqdm
from yolov5.utils.general import non_max_suppression

import scripts.training.yolov5s.constants as constants
from scripts.training.yolov5s.transforms import letterbox, to_tensor
from scripts.training.yolov5s.yolov5s_model import yolov5s_model

logger = logging.getLogger(__name__)

SEED = 42

DEFAULT_OUTPUT_DIR = constants.REPO_ROOT / "scripts" / "evaluation" / "outputs"
_LATEST_RUN = constants.latest_run_dir()
DEFAULT_WEIGHTS = (_LATEST_RUN / "best.pt") if _LATEST_RUN is not None else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Checkpoint to run. Defaults to best.pt from --run-dir (or the "
        "latest training run under model_exports/).",
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Training-run directory whose best.pt to use (overrides the latest-run default).",
    )
    p.add_argument("--annotations", type=Path, default=constants.ANNOTATIONS_TEST)
    p.add_argument("--num-images", type=int, default=100)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--conf-thres", type=float, default=constants.EVAL_CONF_THRES)
    p.add_argument("--iou-thres", type=float, default=constants.EVAL_IOU_THRES)
    p.add_argument("--max-det", type=int, default=constants.EVAL_MAX_DET)
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return p.parse_args()


def build_subset(coco: dict, sampled_images: list[dict]) -> dict:
    """Filter the full COCO dict down to only the sampled images + their annotations."""
    sampled_ids = {img["id"] for img in sampled_images}
    annotations = [a for a in coco.get("annotations", []) if a["image_id"] in sampled_ids]
    return {
        "images": sampled_images,
        "annotations": annotations,
        "categories": coco["categories"],
    }


def unletterbox_xyxy(
    xyxy: list[float],
    r: float,
    dw: float,
    dh: float,
    w0: int,
    h0: int,
) -> tuple[float, float, float, float]:
    """Map a letterboxed-space xyxy box back to original-image pixel coords.

    Mirrors the math at scripts/training/yolov5s/evaluation.py:54-57. The
    ``dw, dh`` returned by ``transforms.letterbox`` are total padding
    (split across both sides), so we subtract half.
    """
    x1, y1, x2, y2 = xyxy
    x1 = max(0.0, min(w0, (x1 - dw / 2) / r))
    y1 = max(0.0, min(h0, (y1 - dh / 2) / r))
    x2 = max(0.0, min(w0, (x2 - dw / 2) / r))
    y2 = max(0.0, min(h0, (y2 - dh / 2) / r))
    return x1, y1, x2, y2


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()

    if args.run_dir is not None:
        args.weights = args.run_dir / "best.pt"
    if args.weights is None:
        raise SystemExit(
            f"no --weights given and no training run found under {constants.OUTPUT_DIR} "
            "— pass --weights or --run-dir."
        )
    if not args.weights.exists():
        raise SystemExit(f"weights not found: {args.weights}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    logger.info("device=%s weights=%s", device, args.weights)

    with args.annotations.open() as f:
        coco = json.load(f)
    logger.info(
        "loaded %s: %d images, %d annotations, %d categories",
        args.annotations,
        len(coco["images"]),
        len(coco.get("annotations", [])),
        len(coco["categories"]),
    )

    annotated_ids = {a["image_id"] for a in coco.get("annotations", [])}
    pool = [img for img in coco["images"] if img["id"] in annotated_ids]
    n = min(args.num_images, len(pool))
    sampled = random.Random(SEED).sample(pool, n)
    logger.info("sampled %d images from %d annotated (seed=%d)", n, len(pool), SEED)

    subset = build_subset(coco, sampled)
    subset_path = args.output_dir / "annotations_subset.json"
    with subset_path.open("w") as f:
        json.dump(subset, f)
    logger.info("wrote %s (%d annotations on sampled images)", subset_path, len(subset["annotations"]))

    model, _ = yolov5s_model(constants.NUM_CLASSES, args.weights, device)
    model.eval()

    results: list[dict] = []
    for img_meta in tqdm(sampled, desc="inference", unit="img", dynamic_ncols=True):
        img_path = constants.IMAGE_ROOT / img_meta["file_name"]
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            logger.warning("could not read %s — skipping image_id=%s", img_path, img_meta["id"])
            continue
        h0, w0 = img_bgr.shape[:2]

        img_lb, (r, _), (dw, dh) = letterbox(img_bgr, new_shape=constants.IMAGE_SIZE)
        tensor = to_tensor(img_lb).unsqueeze(0).to(device)

        with torch.no_grad():
            raw = model(tensor)
            inf_out = raw[0] if isinstance(raw, (tuple, list)) else raw
        dets = non_max_suppression(
            inf_out, args.conf_thres, args.iou_thres, max_det=args.max_det
        )[0]

        if dets is None or len(dets) == 0:
            continue

        for *xyxy, score, cls in dets.cpu().tolist():
            x1, y1, x2, y2 = unletterbox_xyxy(xyxy, r, dw, dh, w0, h0)
            results.append(
                {
                    "image_id": img_meta["id"],
                    "category_id": int(cls) + 1,  # model 0..224 -> COCO 1..225
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score),
                }
            )

    preds_path = args.output_dir / "predictions.json"
    with preds_path.open("w") as f:
        json.dump(results, f)

    mean_per_img = len(results) / max(1, n)
    logger.info(
        "wrote %s: %d predictions over %d images (mean=%.1f/img)",
        preds_path,
        len(results),
        n,
        mean_per_img,
    )


if __name__ == "__main__":
    main()
