"""Anchor-fit audit: recomputes yolov5s anchors if they poorly fit the dataset.

Builds shapes/labels directly from COCO JSON metadata already loaded by
``CocoYoloDataset`` (each image record carries ``width``/``height`` — no
image decoding needed), then delegates to yolov5's own ``check_anchors``:
a best-possible-recall (BPR) diagnostic that only replaces anchors if the
current ones are a poor fit (BPR <= 0.98) *and* k-means finds a strictly
better set. Safe to always call — a no-op when the fit is already good.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from scripts.training.yolov5s.dataset import CocoYoloDataset

logger = logging.getLogger(__name__)


class _ShapesLabelsAdapter:
    """Minimal object satisfying ``yolov5.utils.autoanchor.check_anchors``'s
    dataset contract (``.shapes``: (N,2) width/height array; ``.labels``: list
    of per-image ``[cls, cx, cy, w, h]`` normalised-to-original-image arrays).
    """

    def __init__(self, shapes: np.ndarray, labels: list) -> None:
        self.shapes = shapes
        self.labels = labels


def _build_shapes_and_labels(dataset: "CocoYoloDataset") -> tuple:
    """(N,2) width/height array + per-image ``[cls, cx, cy, w, h]`` label arrays.

    Mirrors ``CocoYoloDataset._load_raw``'s bbox normalisation, but reads
    ``width``/``height`` straight from the COCO JSON instead of decoding
    every image (both fields are already present on every image record).
    """
    shapes = np.array(
        [[record["width"], record["height"]] for record in dataset.images],
        dtype=np.float64,
    )
    labels: list = []
    for record in dataset.images:
        w0, h0 = record["width"], record["height"]
        rows: list = []
        for ann in dataset.anns_by_image_id.get(record["id"], []):
            x, y, bw, bh = ann["bbox"]
            bw_n = bw / w0
            bh_n = bh / h0
            if bw_n <= 0 or bh_n <= 0:
                continue
            cx = (x + bw / 2) / w0
            cy = (y + bh / 2) / h0
            cls_idx = dataset.cat_id_to_yolo[ann["category_id"]]
            rows.append([float(cls_idx), cx, cy, bw_n, bh_n])
        labels.append(np.array(rows, dtype=np.float32) if rows else np.zeros((0, 5), dtype=np.float32))
    return shapes, labels


def _best_possible_recall(wh: np.ndarray, anchors_px: np.ndarray, thr: float) -> float:
    """Fraction of GT boxes with at least one anchor within the ``thr`` wh-ratio band."""
    if wh.shape[0] == 0:
        return 1.0
    r = wh[:, None, :] / anchors_px[None, :, :]
    x = np.minimum(r, 1 / r).min(axis=2)
    best = x.max(axis=1)
    return float((best > 1 / thr).mean())


def check_anchor_fit(
    model: torch.nn.Module,
    dataset: "CocoYoloDataset",
    thr: float,
    img_size: int,
) -> dict:
    """Audit + (if needed) recompute yolov5 anchors against the actual dataset.

    Returns a dict (``bpr``, ``anchors_changed``) for the caller to log
    (e.g. to MLflow) — ``check_anchors`` itself only logs to yolov5's own
    logger and has no return value.
    """
    from yolov5.utils.autoanchor import check_anchors

    shapes, labels = _build_shapes_and_labels(dataset)
    adapter = _ShapesLabelsAdapter(shapes, labels)

    detect = model.module.model[-1] if hasattr(model, "module") else model.model[-1]
    stride = detect.stride.to(detect.anchors.device).view(-1, 1, 1)
    anchors_px_before = (detect.anchors.clone() * stride).cpu().numpy().reshape(-1, 2)

    scale = img_size / shapes.max(1, keepdims=True)
    wh = np.concatenate([lbl[:, 3:5] * s for s, lbl in zip(shapes * scale, labels) if lbl.shape[0] > 0])
    bpr = _best_possible_recall(wh, anchors_px_before, thr)
    logger.info("autoanchor: BPR=%.4f against current anchors (thr=%g, imgsz=%d)", bpr, thr, img_size)

    check_anchors(dataset=adapter, model=model, thr=thr, imgsz=img_size)

    anchors_px_after = (detect.anchors.clone() * stride).cpu().numpy().reshape(-1, 2)
    changed = not np.allclose(anchors_px_before, anchors_px_after)
    if changed:
        logger.info("autoanchor: anchors were recomputed (poor fit detected)")
    return {"bpr": bpr, "anchors_changed": changed}
