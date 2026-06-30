"""Load a trained YOLOv5s checkpoint and cache COCO-format predictions JSON.

This module decouples expensive GPU inference from cheap scoring.  It writes a
predictions JSON whose schema is a frozen contract; the scoring module (built
separately) reads that file without ever touching the model.

Public API
----------
load_checkpoint(checkpoint_path, num_classes, device) -> nn.Module
run_inference(checkpoint_path, annotations_path, output_path, device,
              conf_thres, iou_thres, max_det, image_size, batch_size,
              num_workers, cache) -> dict
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from tqdm import tqdm
from yolov5.utils.general import non_max_suppression

import scripts.training.yolov5s.constants as constants
from scripts.training.yolov5s.dataset import CocoYoloDataset, Dataloader, collate_fn
from scripts.training.yolov5s.yolov5s_model import yolov5s_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------

def load_checkpoint(
    checkpoint_path: Path,
    num_classes: int,
    device: torch.device,
) -> nn.Module:
    """Build the YOLOv5s architecture and load weights from a training checkpoint.

    Handles two checkpoint formats:

    * **Training checkpoint dict** — a ``dict`` saved by the training pipeline
      containing a ``"model"`` key whose value is a plain ``state_dict`` (the
      canonical format written by ``training_pipeline.py``).  Any additional
      keys (``"epoch"``, ``"best_metric"``, …) are ignored.
    * **Raw state_dict** — a ``dict`` of ``{str: Tensor}`` with no ``"model"``
      key, as produced by ``torch.save(model.state_dict(), path)``.

    In both cases weights are loaded with ``strict=False`` so shape mismatches
    (e.g. a checkpoint from a different ``num_classes``) are skipped rather than
    raising.  The number of matched vs. total keys is logged at INFO level.

    The returned module is in ``eval()`` mode on ``device``.
    """
    checkpoint_path = Path(checkpoint_path)
    logger.info("loading checkpoint: %s", checkpoint_path)

    # Build fresh architecture (sets model.nc, model.hyp, etc.)
    model, _ = yolov5s_model(num_classes=num_classes, weights=None, device=device)

    raw = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Determine which loading path to take.
    if isinstance(raw, dict) and "model" in raw:
        # Training-checkpoint dict path — the value under "model" is already a
        # plain state_dict (training_pipeline.py calls
        # ``torch.save({"model": model.state_dict(), ...})``)
        candidate = raw["model"]
        # Guard: if somehow a full nn.Module was stored under "model", unwrap it.
        if isinstance(candidate, nn.Module):
            state_dict: dict[str, torch.Tensor] = candidate.state_dict()
            logger.info("checkpoint format: training dict (nn.Module under 'model' key)")
        else:
            state_dict = candidate
            logger.info("checkpoint format: training dict (state_dict under 'model' key)")
    else:
        # Raw state_dict path — the file IS the state_dict.
        if isinstance(raw, nn.Module):
            state_dict = raw.state_dict()
            logger.info("checkpoint format: raw nn.Module (state_dict extracted)")
        else:
            state_dict = raw
            logger.info("checkpoint format: raw state_dict")

    own = model.state_dict()
    matched = {k: v for k, v in state_dict.items() if k in own and own[k].shape == v.shape}
    skipped = len(own) - len(matched)
    model.load_state_dict(matched, strict=False)
    logger.info(
        "loaded %d/%d keys from checkpoint (skipped %d shape-mismatched / missing)",
        len(matched),
        len(own),
        skipped,
    )

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Inference runner
# ---------------------------------------------------------------------------

def _build_eval_header(
    checkpoint_path: Path,
    annotations_path: Path,
    conf_thres: float,
    iou_thres: float,
    max_det: int,
    image_size: int,
) -> dict[str, Any]:
    """Return the frozen header portion of the predictions JSON."""
    return {
        "checkpoint": str(checkpoint_path),
        "annotations": str(annotations_path),
        "eval": {
            "conf_thres": conf_thres,
            "iou_thres": iou_thres,
            "max_det": max_det,
            "image_size": image_size,
        },
    }


def _cache_hit(existing: dict, header: dict) -> bool:
    """Return True iff the cached file matches every field in *header*."""
    return (
        existing.get("checkpoint") == header["checkpoint"]
        and existing.get("annotations") == header["annotations"]
        and existing.get("eval") == header["eval"]
    )


def run_inference(
    checkpoint_path: Path,
    annotations_path: Path,
    output_path: Path,
    device: torch.device,
    conf_thres: float,
    iou_thres: float,
    max_det: int,
    image_size: int,
    batch_size: int,
    num_workers: int,
    cache: bool = True,
) -> dict:
    """Run the model over every image in *annotations_path* and write a
    predictions JSON to *output_path*.

    Cache behaviour
    ~~~~~~~~~~~~~~~
    When ``cache=True`` and *output_path* already exists, the file is loaded and
    its ``checkpoint``, ``annotations``, and ``eval`` header fields are compared
    to the requested configuration.  If all three match, the cached predictions
    are returned immediately without re-running inference.  If any field differs,
    inference is re-run and the file is overwritten.

    Output JSON schema (frozen contract)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    .. code-block:: json

        {
          "checkpoint": "<path str>",
          "annotations": "<path str>",
          "eval": {"conf_thres": 0.001, "iou_thres": 0.6, "max_det": 100, "image_size": 640},
          "num_images": <int>,
          "predictions": [
            {"image_id": <int>, "category_id": <int>, "bbox": [x, y, w, h], "score": <float>}
          ]
        }

    ``bbox`` is original-image-pixel **xywh** (COCO convention); ``category_id``
    is the COCO category id (1..225), not the YOLO class index (0..224).

    Returns
    -------
    The loaded predictions dict (either from cache or freshly computed).
    """
    checkpoint_path = Path(checkpoint_path)
    annotations_path = Path(annotations_path)
    output_path = Path(output_path)

    header = _build_eval_header(
        checkpoint_path, annotations_path, conf_thres, iou_thres, max_det, image_size
    )

    # ── Cache check ──────────────────────────────────────────────────────────
    if cache and output_path.exists():
        try:
            with open(output_path) as f:
                existing = json.load(f)
            if _cache_hit(existing, header):
                logger.info(
                    "cache HIT — reusing predictions from %s (%d predictions over %d images)",
                    output_path,
                    len(existing.get("predictions", [])),
                    existing.get("num_images", "?"),
                )
                return existing
            else:
                logger.info(
                    "cache MISS — header mismatch (checkpoint/annotations/eval changed); "
                    "recomputing and overwriting %s",
                    output_path,
                )
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("could not read cache file %s (%s); recomputing", output_path, exc)

    # ── Dataset + dataloader ─────────────────────────────────────────────────
    # augment=False, shuffle=False — deterministic ordering is required so that
    # every path maps unambiguously to its image_id.
    dataset = CocoYoloDataset(
        annotations_path=annotations_path,
        image_root=constants.IMAGE_ROOT,
        image_size=image_size,
        augment=False,
    )

    loader_wrapper = Dataloader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        worker_init_fn=None,
        generator=None,
    )
    data_loader = loader_wrapper.get_dataloader()

    # ── Build path → image_id lookup ─────────────────────────────────────────
    # dataset.images is the list of COCO image records already parsed from the
    # annotation JSON, so we do not need to re-open the file.
    # file_name values in the annotation file start with "data/..." and are
    # resolved relative to IMAGE_ROOT (== REPO_ROOT) — exactly how the dataset
    # constructs paths in __getitem__ (image_root / record["file_name"]).
    path_to_image_id: dict[str, int] = {
        str(constants.IMAGE_ROOT / rec["file_name"]): rec["id"]
        for rec in dataset.images
    }

    # ── Build YOLO-index → COCO category_id lookup ───────────────────────────
    # dataset.cat_id_to_yolo maps COCO cat_id (1..225) → YOLO index (0..224).
    # We need the inverse: YOLO index → COCO cat_id.
    yolo_to_cat: dict[int, int] = {v: k for k, v in dataset.cat_id_to_yolo.items()}

    # ── Load model ───────────────────────────────────────────────────────────
    model = load_checkpoint(checkpoint_path, num_classes=constants.NUM_CLASSES, device=device)
    model.eval()

    # ── Inference loop ───────────────────────────────────────────────────────
    predictions: list[dict[str, Any]] = []

    logger.info(
        "running inference: %d images, batch=%d, conf=%.4f, iou=%.2f, max_det=%d",
        len(dataset),
        batch_size,
        conf_thres,
        iou_thres,
        max_det,
    )

    with torch.no_grad():
        pbar = tqdm(data_loader, desc="predict", unit="batch", dynamic_ncols=True)
        for imgs, _targets, paths, shapes in pbar:
            imgs = imgs.to(device)

            # Forward pass — eval mode returns (inference_out, train_out); use
            # inference_out for NMS (mirrors evaluation.py exactly).
            raw = model(imgs)
            inference_out = raw[0] if isinstance(raw, (tuple, list)) else raw

            det_list = non_max_suppression(
                inference_out, conf_thres, iou_thres, max_det=max_det
            )

            for i, dets in enumerate(det_list):
                path_str = paths[i]
                image_id = path_to_image_id.get(path_str)
                if image_id is None:
                    logger.warning("no image_id found for path %s — skipping", path_str)
                    continue

                if dets is None or len(dets) == 0:
                    # No detections for this image — contribute no rows (correct).
                    continue

                # ── Un-letterbox: letterboxed xyxy → original-image xyxy ──────
                # This is the EXACT transform from evaluation.py, copied verbatim.
                (h0, w0), ((r, _), (dw, dh)) = shapes[i]

                boxes = dets[:, :4].clone()
                # Remove padding offset, then scale back to original coords.
                boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw / 2) / r
                boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh / 2) / r
                # Clamp to original image bounds.
                boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w0)
                boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h0)

                scores = dets[:, 4]
                yolo_labels = dets[:, 5].long()

                # ── Convert xyxy → xywh (COCO convention) and emit rows ───────
                for j in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[j].tolist()
                    w_box = x2 - x1
                    h_box = y2 - y1
                    score = float(scores[j].item())
                    yolo_idx = int(yolo_labels[j].item())
                    cat_id = yolo_to_cat.get(yolo_idx)
                    if cat_id is None:
                        logger.warning(
                            "YOLO index %d has no COCO category_id mapping — skipping",
                            yolo_idx,
                        )
                        continue
                    predictions.append(
                        {
                            "image_id": image_id,
                            "category_id": cat_id,
                            "bbox": [
                                round(x1, 3),
                                round(y1, 3),
                                round(w_box, 3),
                                round(h_box, 3),
                            ],
                            "score": round(score, 6),
                        }
                    )

    logger.info(
        "inference complete: %d predictions over %d images",
        len(predictions),
        len(dataset),
    )

    # ── Assemble and write output JSON ───────────────────────────────────────
    output_dict: dict[str, Any] = {
        **header,
        "num_images": len(dataset),
        "predictions": predictions,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_dict, f, indent=None, separators=(",", ":"))
    logger.info("predictions written to %s", output_path)

    return output_dict
