"""Scoring engine for the model-evaluation suite.

Scores cached predictions against ground truth, applying label remaps
(granularity) and image filters (band / domain).  Does NOT run the model —
it consumes cached prediction dicts and COCO annotation JSON only.

Evaluation strategy: docs/plans/2026-06-10_model-evaluation-strategy.md
"""
from __future__ import annotations

import copy
import json
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torchmetrics.detection import MeanAveragePrecision

logger = logging.getLogger(__name__)

# Cap on images built into torchmetrics tensors per metric.update() call.
# score() can be asked to evaluate 60k+ images at once (full real+synthetic
# test set); building one Python list of per-image tensors for all of them
# and feeding it to MeanAveragePrecision in a single update() nearly OOM'd a
# 47GB host (RSS climbed past 30GB on the very first "headline" call before
# it was killed). Chunking avoids holding our own per-image tensor lists and
# a duplicate copy inside torchmetrics simultaneously at peak.
_SCORE_UPDATE_CHUNK_SIZE = 2000

# ---------------------------------------------------------------------------
# GT index builder
# ---------------------------------------------------------------------------

def build_gt_index(annotations_path: Path) -> dict:
    """Parse a COCO-format annotation JSON and return a lightweight index.

    Returns a dict with three sub-dicts:
        images:  {image_id: {"band": str, "width": int, "height": int, "file_name": str}}
        anns:    {image_id: [{"category_id": int, "bbox": [x,y,w,h], "area": float}]}
        cats:    {coco_id: name}

    The per-image ``band`` field (values 'A'/'B'/'C'/'D'/'negative') is read
    directly from each image record in the JSON.  Images without a ``band``
    key are stored with band=None (a warning is logged on the first occurrence).
    """
    annotations_path = Path(annotations_path)
    logger.info("loading GT index from %s", annotations_path)

    with annotations_path.open() as f:
        data = json.load(f)

    # ── categories ───────────────────────────────────────────────────────────
    cats: dict[int, str] = {c["id"]: c["name"] for c in data.get("categories", [])}

    # ── images ───────────────────────────────────────────────────────────────
    images: dict[int, dict] = {}
    warned_no_band = False
    for img in data.get("images", []):
        band = img.get("band")
        if band is None and not warned_no_band:
            logger.warning(
                "image record %d has no 'band' key — storing None (further "
                "warnings suppressed)",
                img["id"],
            )
            warned_no_band = True
        images[img["id"]] = {
            "band": band,
            "width": img.get("width", 0),
            "height": img.get("height", 0),
            "file_name": img.get("file_name", ""),
        }

    # ── annotations ──────────────────────────────────────────────────────────
    anns: dict[int, list[dict]] = {iid: [] for iid in images}
    for ann in data.get("annotations", []):
        iid = ann["image_id"]
        if iid not in anns:
            # annotation for an image not in the images list — keep it anyway
            anns[iid] = []
        anns[iid].append(
            {
                "category_id": ann["category_id"],
                "bbox": ann["bbox"],  # [x, y, w, h]
                "area": float(ann.get("area", ann["bbox"][2] * ann["bbox"][3])),
            }
        )

    logger.info(
        "GT index: %d images, %d categories, %d annotated images",
        len(images),
        len(cats),
        sum(1 for v in anns.values() if v),
    )
    return {"images": images, "anns": anns, "cats": cats}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_remap(category_id: int, remap: dict[int, int] | None) -> int:
    """Return remapped category id (identity if remap is None or key absent)."""
    if remap is None:
        return category_id
    return remap.get(category_id, category_id)


def _xywh_to_xyxy(bbox: list[float]) -> list[float]:
    """Convert [x, y, w, h] to [x1, y1, x2, y2]."""
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


# ---------------------------------------------------------------------------
# IoU helper (vectorised; prefers torchvision.ops.box_iou, numpy fallback)
# ---------------------------------------------------------------------------

try:
    from torchvision.ops import box_iou as _tv_box_iou  # type: ignore

    def _box_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
        """Return (N, M) IoU matrix.  boxes are xyxy float arrays."""
        ta = torch.from_numpy(boxes_a.astype(np.float32))
        tb = torch.from_numpy(boxes_b.astype(np.float32))
        return _tv_box_iou(ta, tb).numpy()

except ImportError:  # pragma: no cover
    logger.debug("torchvision not available — using numpy IoU fallback")

    def _box_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:  # type: ignore[misc]
        """Pure-numpy IoU, O(N*M) memory."""
        # boxes_a: (N,4), boxes_b: (M,4), xyxy
        x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
        y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
        x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
        y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
        inter_w = np.maximum(0.0, x2 - x1)
        inter_h = np.maximum(0.0, y2 - y1)
        inter = inter_w * inter_h
        area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
        area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
        union = area_a[:, None] + area_b[None, :] - inter
        return np.where(union > 0, inter / union, 0.0)


# ---------------------------------------------------------------------------
# Core scorer
# ---------------------------------------------------------------------------

def score(
    gt_index: dict,
    predictions: list[dict],
    image_ids: set[int] | None = None,
    remap: dict[int, int] | None = None,
    max_det: int = 100,
    class_metrics: bool = True,
) -> dict:
    """Score cached predictions against GT, returning the COCO-12 metric vector.

    Parameters
    ----------
    gt_index:
        Built by :func:`build_gt_index`.
    predictions:
        List of ``{"image_id": int, "category_id": int, "bbox": [x,y,w,h], "score": float}``.
    image_ids:
        Restrict evaluation to this set of image IDs.  ``None`` = all images in
        *gt_index*.
    remap:
        Label remap dict ``{old_id: new_id}`` applied to **both** predictions
        and GT category IDs.  ``None`` = identity (fine granularity).
    max_det:
        COCO max-detection cap.  Passed as the top element of
        ``max_detection_thresholds=[1, 10, max_det]``.
    class_metrics:
        If True, compute and return per-class AP.

    Returns
    -------
    Flat dict with keys:
        map, map_50, map_75, map_small, map_medium, map_large,
        mar_1, mar_10, mar_100, mar_small, mar_medium, mar_large,
        map_per_class (dict {label_id: ap_float}),
        n_images, n_dets, n_gt.
    All values are plain Python floats/ints.
    """
    all_images: dict = gt_index["images"]
    all_anns: dict = gt_index["anns"]

    # ── resolve image universe ────────────────────────────────────────────────
    if image_ids is None:
        image_ids = set(all_images.keys())
    else:
        unknown = image_ids - set(all_images.keys())
        if unknown:
            logger.warning(
                "score(): %d requested image_ids not found in gt_index (ignored)",
                len(unknown),
            )
        image_ids = image_ids & set(all_images.keys())

    if not image_ids:
        logger.warning("score(): no valid images — returning zero/NaN metrics")
        return _empty_result(class_metrics)

    # ── index predictions by image_id ─────────────────────────────────────────
    preds_by_image: dict[int, list[dict]] = {iid: [] for iid in image_ids}
    for p in predictions:
        iid = p["image_id"]
        if iid in preds_by_image:
            preds_by_image[iid].append(p)

    # ── count GT ─────────────────────────────────────────────────────────────
    n_gt = sum(len(all_anns.get(iid, [])) for iid in image_ids)
    n_dets = sum(len(preds_by_image[iid]) for iid in image_ids)

    if n_gt == 0:
        logger.warning(
            "score(): no GT annotations found for the %d requested images — "
            "returning zero/NaN metrics",
            len(image_ids),
        )
        return _empty_result(class_metrics)

    # ── cap predictions per image at max_det, across ALL classes combined ─────
    # (COCO semantics: the cap applies per-image, not per-class — a class's box
    # can be pushed out by higher-scoring boxes of *other* classes in the same
    # image, so this must happen before any per-class split below.)
    capped_preds_by_image: dict[int, list[dict]] = {
        iid: sorted(raw, key=lambda p: p["score"], reverse=True)[:max_det]
        for iid, raw in preds_by_image.items()
    }

    # ── remap once, and discover the class universe (GT ∪ predicted) ──────────
    # gt_by_image / pred_by_image hold (label, xyxy_box[, score]) tuples so each
    # per-class pass below can filter without touching the raw dicts again.
    gt_by_image: dict[int, list[tuple[int, list[float]]]] = {}
    pred_by_image: dict[int, list[tuple[int, list[float], float]]] = {}
    gt_classes: set[int] = set()
    pred_only_classes: set[int] = set()

    for iid in image_ids:
        gts = [
            (_apply_remap(g["category_id"], remap), _xywh_to_xyxy(g["bbox"]))
            for g in all_anns.get(iid, [])
        ]
        gt_by_image[iid] = gts
        gt_classes.update(lbl for lbl, _ in gts)

        preds = [
            (_apply_remap(p["category_id"], remap), _xywh_to_xyxy(p["bbox"]), p["score"])
            for p in capped_preds_by_image[iid]
        ]
        pred_by_image[iid] = preds
        pred_only_classes.update(lbl for lbl, _, _ in preds)

    pred_only_classes -= gt_classes
    all_classes = sorted(gt_classes | pred_only_classes)

    # ── per-class AP, one class at a time ──────────────────────────────────────
    # torchmetrics' MeanAveragePrecision keeps every image's boxes/labels/scores
    # resident until compute() — at full test-set scale (tens of thousands of
    # images × hundreds of classes) that single multi-class call is what drove
    # RSS past 30GB and OOM-killed the process. Scoring one class at a time,
    # restricted to only the images relevant to that class, keeps at most one
    # class's data resident at once. The final aggregate below reproduces
    # torchmetrics' own macro-average-over-classes-with-GT convention exactly
    # (validated against the prior single-call implementation).
    metric_keys = (
        "map", "map_50", "map_75", "map_small", "map_medium", "map_large",
        "mar_1", "mar_10", "mar_100", "mar_small", "mar_medium", "mar_large",
    )

    def _safe_float(v) -> float:
        v = v.item() if hasattr(v, "item") else float(v)
        return float(v) if not (math.isnan(v) or math.isinf(v)) else float("nan")

    map_per_class_dict: dict[int, float] = {}
    agg_sum = {k: 0.0 for k in metric_keys}
    agg_count = {k: 0 for k in metric_keys}

    for cid in all_classes:
        relevant_ids = [
            iid for iid in image_ids
            if any(lbl == cid for lbl, _ in gt_by_image[iid])
            or any(lbl == cid for lbl, _, _ in pred_by_image[iid])
        ]

        metric_c = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            class_metrics=False,
            max_detection_thresholds=[1, 10, max_det],
        )
        for chunk_start in range(0, len(relevant_ids), _SCORE_UPDATE_CHUNK_SIZE):
            chunk_ids = relevant_ids[chunk_start : chunk_start + _SCORE_UPDATE_CHUNK_SIZE]
            tm_preds: list[dict] = []
            tm_targets: list[dict] = []

            for iid in chunk_ids:
                gts_here = [box for lbl, box in gt_by_image[iid] if lbl == cid]
                if gts_here:
                    boxes_g = torch.tensor(gts_here, dtype=torch.float32)
                    labels_g = torch.zeros(len(gts_here), dtype=torch.long)
                else:
                    boxes_g = torch.zeros((0, 4), dtype=torch.float32)
                    labels_g = torch.zeros(0, dtype=torch.long)
                tm_targets.append({"boxes": boxes_g, "labels": labels_g})

                preds_here = [(box, sc) for lbl, box, sc in pred_by_image[iid] if lbl == cid]
                if preds_here:
                    boxes_p = torch.tensor([box for box, _ in preds_here], dtype=torch.float32)
                    scores_p = torch.tensor([sc for _, sc in preds_here], dtype=torch.float32)
                    labels_p = torch.zeros(len(preds_here), dtype=torch.long)
                else:
                    boxes_p = torch.zeros((0, 4), dtype=torch.float32)
                    scores_p = torch.zeros(0, dtype=torch.float32)
                    labels_p = torch.zeros(0, dtype=torch.long)
                tm_preds.append({"boxes": boxes_p, "scores": scores_p, "labels": labels_p})

            metric_c.update(tm_preds, tm_targets)

        result_c = metric_c.compute()
        has_gt = cid in gt_classes

        if class_metrics:
            map_per_class_dict[cid] = _safe_float(result_c["map"]) if has_gt else -1.0

        if not has_gt:
            continue  # predictions-only class: excluded from the macro mean (matches
                       # torchmetrics' own -1-for-no-GT convention)
        for k in metric_keys:
            v = _safe_float(result_c[k])
            if v == -1.0:
                continue  # this class has no qualifying GT/pred for this specific
                          # submetric (e.g. no medium-sized instances) — exclude
                          # only from that submetric's mean, same as torchmetrics
            agg_sum[k] += v
            agg_count[k] += 1

    logger.debug(
        "score(): n_images=%d n_gt=%d n_dets=%d n_classes=%d  mAP=%.4f mAP50=%.4f",
        len(image_ids), n_gt, n_dets, len(all_classes),
        agg_sum["map"] / agg_count["map"] if agg_count["map"] else -1.0,
        agg_sum["map_50"] / agg_count["map_50"] if agg_count["map_50"] else -1.0,
    )

    # ── assemble flat output dict ─────────────────────────────────────────────
    out: dict = {
        k: (agg_sum[k] / agg_count[k] if agg_count[k] else -1.0) for k in metric_keys
    }
    out["map_per_class"] = map_per_class_dict
    out["n_images"] = len(image_ids)
    out["n_dets"] = n_dets
    out["n_gt"] = n_gt
    return out


def _empty_result(class_metrics: bool) -> dict:
    """Return a zero/NaN result dict for degenerate cases."""
    out: dict = {
        "map": float("nan"),
        "map_50": float("nan"),
        "map_75": float("nan"),
        "map_small": float("nan"),
        "map_medium": float("nan"),
        "map_large": float("nan"),
        "mar_1": float("nan"),
        "mar_10": float("nan"),
        "mar_100": float("nan"),
        "mar_small": float("nan"),
        "mar_medium": float("nan"),
        "mar_large": float("nan"),
        "map_per_class": {},
        "n_images": 0,
        "n_dets": 0,
        "n_gt": 0,
    }
    return out


# ---------------------------------------------------------------------------
# Domain merging
# ---------------------------------------------------------------------------

def merge_domains(
    real_gt: dict,
    real_preds: list[dict],
    synth_gt: dict,
    synth_preds: list[dict],
) -> tuple[dict, list[dict]]:
    """Build the 'mixed' GT index + prediction list.

    Synthetic image IDs are offset past ``max(real image_id)`` to avoid
    collisions.  Category IDs already align and are not modified.
    Does not mutate inputs (deep copies where needed).

    Parameters
    ----------
    real_gt, synth_gt:
        GT indices as returned by :func:`build_gt_index`.
    real_preds, synth_preds:
        Prediction lists in the canonical format.

    Returns
    -------
    ``(mixed_gt_index, mixed_predictions)``
    """
    real_image_ids = set(real_gt["images"].keys())
    if not real_image_ids:
        offset = 0
    else:
        offset = max(real_image_ids)

    logger.info(
        "merge_domains: %d real images + %d synth images (synth offset=%d)",
        len(real_image_ids),
        len(synth_gt["images"]),
        offset,
    )

    # ── offset synthetic GT ───────────────────────────────────────────────────
    new_synth_images: dict[int, dict] = {}
    for sid, sinfo in synth_gt["images"].items():
        new_synth_images[sid + offset] = copy.copy(sinfo)

    new_synth_anns: dict[int, list[dict]] = {}
    for sid, ann_list in synth_gt["anns"].items():
        new_synth_anns[sid + offset] = [copy.copy(a) for a in ann_list]

    # ── merge GT index ────────────────────────────────────────────────────────
    # Start from shallow copies of real; add offsetted synth entries
    merged_images = dict(real_gt["images"])
    merged_images.update(new_synth_images)

    merged_anns = dict(real_gt["anns"])
    merged_anns.update(new_synth_anns)

    merged_cats = dict(real_gt["cats"])  # cats assumed identical; real wins on collision
    merged_cats.update(synth_gt["cats"])
    merged_cats.update(real_gt["cats"])  # real wins

    mixed_gt: dict = {"images": merged_images, "anns": merged_anns, "cats": merged_cats}

    # ── offset synthetic predictions ──────────────────────────────────────────
    mixed_preds: list[dict] = list(real_preds)  # shallow copy of list
    for p in synth_preds:
        new_p = dict(p)
        new_p["image_id"] = p["image_id"] + offset
        mixed_preds.append(new_p)

    return mixed_gt, mixed_preds


# ---------------------------------------------------------------------------
# Image-ID filter by band
# ---------------------------------------------------------------------------

def filter_image_ids_by_band(gt_index: dict, bands: set[str]) -> set[int]:
    """Return image IDs whose ``band`` field is in ``bands``."""
    return {
        iid
        for iid, info in gt_index["images"].items()
        if info.get("band") in bands
    }


# ---------------------------------------------------------------------------
# Domain-shift delta (strategy §6b)
# ---------------------------------------------------------------------------

def domain_shift_delta(
    real_gt: dict,
    real_preds: list[dict],
    synth_gt: dict,
    synth_preds: list[dict],
    remap: dict[int, int] | None = None,
    max_det: int = 100,
    class_to_band: dict[int, str] | None = None,
) -> dict:
    """Per-class paired delta mAP_real − mAP_synth (strategy §6b).

    Scores real and synthetic domains separately with ``class_metrics=True``,
    aligns per-class AP by (remapped) label ID, and returns the paired delta.

    Parameters
    ----------
    class_to_band:
        Optional ``{label_id: band}`` mapping (using *remapped* label IDs).
        When provided, ``by_band`` in the return dict gives per-band mean delta.

    Returns
    -------
    dict with keys:
        per_class:    {label_id: {"real": float, "synth": float, "delta": float}}
        mean_delta:   float (mean over classes present in both domains)
        by_band:      {band: mean_delta}  (empty if class_to_band is None)
    """
    logger.info("computing domain_shift_delta (remap=%s)", "identity" if remap is None else "custom")

    real_result = score(real_gt, real_preds, remap=remap, max_det=max_det, class_metrics=True)
    synth_result = score(synth_gt, synth_preds, remap=remap, max_det=max_det, class_metrics=True)

    real_per_class: dict[int, float] = real_result.get("map_per_class", {})
    synth_per_class: dict[int, float] = synth_result.get("map_per_class", {})

    # Align on the intersection of classes present in BOTH domains
    common_labels = set(real_per_class.keys()) & set(synth_per_class.keys())
    logger.info(
        "domain_shift_delta: %d real classes, %d synth classes, %d common",
        len(real_per_class),
        len(synth_per_class),
        len(common_labels),
    )

    per_class: dict[int, dict] = {}
    deltas: list[float] = []
    for lid in sorted(common_labels):
        r_ap = real_per_class[lid]
        s_ap = synth_per_class[lid]
        delta = r_ap - s_ap
        per_class[lid] = {"real": r_ap, "synth": s_ap, "delta": delta}
        deltas.append(delta)

    mean_delta = float(np.mean(deltas)) if deltas else float("nan")

    # ── by_band breakdown ─────────────────────────────────────────────────────
    by_band: dict[str, float] = {}
    if class_to_band is not None:
        band_deltas: dict[str, list[float]] = {}
        for lid, entry in per_class.items():
            band = class_to_band.get(lid)
            if band is not None:
                band_deltas.setdefault(band, []).append(entry["delta"])
        by_band = {
            band: float(np.mean(vals))
            for band, vals in band_deltas.items()
            if vals
        }

    return {"per_class": per_class, "mean_delta": mean_delta, "by_band": by_band}


# ---------------------------------------------------------------------------
# Within-group confusion (strategy §4.2)
# ---------------------------------------------------------------------------

def within_group_confusion(
    gt_index: dict,
    predictions: list[dict],
    coarse_remap: dict[int, int],
    lookalike_group_ids: list[int],
    image_ids: set[int] | None = None,
    iou_thr: float = 0.5,
) -> dict:
    """Compute within-look-alike-group fine-species confusion rate (§4.2).

    For each look-alike group: of detections that are (a) correctly localised
    (IoU ≥ ``iou_thr`` vs. a GT box) AND (b) land in the right coarse group,
    what fraction get the wrong fine species?

    Uses a greedy IoU matcher per image: sort detections by score descending,
    match each to the best available (unmatched) GT box with IoU ≥ ``iou_thr``.

    Parameters
    ----------
    gt_index:
        GT index from :func:`build_gt_index`.
    predictions:
        Canonical prediction list (fine category IDs, un-remapped).
    coarse_remap:
        Maps fine ``category_id → coarse group_id``.  Must cover all fine IDs
        present in the data.
    lookalike_group_ids:
        The subset of coarse group IDs that are look-alike groups (others are
        ignored in the confusion count, but still used for localisation matching).
    image_ids:
        Restrict to these image IDs.  ``None`` = all in gt_index.
    iou_thr:
        IoU threshold for considering a detection "correctly localised".

    Returns
    -------
    dict with keys:
        by_group:                {group_id: {"matched": int, "confused": int, "confusion_rate": float}}
        overall_confusion_rate:  float
        pairs:                   {(true_fine_id, pred_fine_id): count}  — within-group only
    """
    all_images = gt_index["images"]
    all_anns = gt_index["anns"]
    lookalike_set = set(lookalike_group_ids)

    if image_ids is None:
        image_ids = set(all_images.keys())
    else:
        image_ids = image_ids & set(all_images.keys())

    # Index predictions by image_id
    preds_by_image: dict[int, list[dict]] = {iid: [] for iid in image_ids}
    for p in predictions:
        iid = p["image_id"]
        if iid in preds_by_image:
            preds_by_image[iid].append(p)

    # Accumulators
    group_matched: dict[int, int] = {g: 0 for g in lookalike_set}
    group_confused: dict[int, int] = {g: 0 for g in lookalike_set}
    confusion_pairs: dict[tuple[int, int], int] = {}
    total_matched = 0
    total_confused = 0

    for iid in sorted(image_ids):
        raw_preds = preds_by_image.get(iid, [])
        raw_gts = all_anns.get(iid, [])

        if not raw_preds or not raw_gts:
            continue

        # Sort dets by score descending
        sorted_preds = sorted(raw_preds, key=lambda p: p["score"], reverse=True)

        # Convert to xyxy numpy arrays
        gt_boxes = np.array([_xywh_to_xyxy(g["bbox"]) for g in raw_gts], dtype=np.float32)
        det_boxes = np.array([_xywh_to_xyxy(p["bbox"]) for p in sorted_preds], dtype=np.float32)

        iou_mat = _box_iou_matrix(det_boxes, gt_boxes)  # (N_det, N_gt)

        matched_gt: set[int] = set()  # indices of GT boxes already consumed

        for det_idx, det in enumerate(sorted_preds):
            det_fine = det["category_id"]
            det_coarse = coarse_remap.get(det_fine)
            if det_coarse is None:
                continue  # unknown fine label — skip

            # Find best unmatched GT with IoU >= threshold
            ious = iou_mat[det_idx]
            best_gt_idx = -1
            best_iou = iou_thr - 1e-9  # must strictly beat the threshold
            for gt_idx, iou_val in enumerate(ious):
                if gt_idx in matched_gt:
                    continue
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_gt_idx = gt_idx

            if best_gt_idx < 0:
                continue  # no sufficiently overlapping GT found

            # Check coarse group match
            gt_fine = raw_gts[best_gt_idx]["category_id"]
            gt_coarse = coarse_remap.get(gt_fine)
            if gt_coarse is None:
                continue  # GT has unknown coarse group — skip

            if det_coarse != gt_coarse:
                # Coarse group mismatch — not a within-group error
                matched_gt.add(best_gt_idx)
                continue

            # This detection is correctly localised AND in the right group
            matched_gt.add(best_gt_idx)

            if det_coarse not in lookalike_set:
                continue  # not a look-alike group — don't count for confusion

            is_confused = det_fine != gt_fine

            group_matched[det_coarse] = group_matched.get(det_coarse, 0) + 1
            total_matched += 1
            if is_confused:
                group_confused[det_coarse] = group_confused.get(det_coarse, 0) + 1
                total_confused += 1
                pair = (gt_fine, det_fine)
                confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1

    # ── assemble results ──────────────────────────────────────────────────────
    by_group: dict[int, dict] = {}
    for g in lookalike_set:
        matched = group_matched.get(g, 0)
        confused = group_confused.get(g, 0)
        rate = confused / matched if matched > 0 else float("nan")
        by_group[g] = {"matched": matched, "confused": confused, "confusion_rate": rate}

    overall_rate = total_confused / total_matched if total_matched > 0 else float("nan")

    return {
        "by_group": by_group,
        "overall_confusion_rate": overall_rate,
        "pairs": {k: v for k, v in confusion_pairs.items()},
    }


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(
    gt_index: dict,
    predictions: list[dict],
    image_ids: set[int],
    remap: dict[int, int] | None,
    max_det: int,
    n_boot: int = 1000,
    metric_key: str = "map",
    seed: int = 42,
) -> dict:
    """Image-level bootstrap CI for a scalar headline metric.

    Resamples ``image_ids`` WITH replacement ``n_boot`` times, calls
    :func:`score` each time, and returns a 95 % percentile CI.

    Parameters
    ----------
    n_boot:
        Number of bootstrap replicates.
    metric_key:
        Which key from :func:`score`'s output to bootstrap (e.g. ``'map'``,
        ``'map_50'``).
    seed:
        Seed for ``numpy.random.default_rng``.

    Returns
    -------
    dict with keys: ``mean``, ``lo`` (2.5th pct), ``hi`` (97.5th pct), ``std``.
    """
    image_id_list = sorted(image_ids)
    if not image_id_list:
        logger.warning("bootstrap_ci: no image_ids — returning NaN CI")
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "std": float("nan")}

    rng = np.random.default_rng(seed)
    n = len(image_id_list)
    id_array = np.array(image_id_list, dtype=np.int64)

    boot_values: list[float] = []
    logger.info("bootstrap_ci: %d replicates, n_images=%d, metric=%s", n_boot, n, metric_key)

    for _ in range(n_boot):
        sample_ids = set(id_array[rng.integers(0, n, size=n)].tolist())
        result = score(
            gt_index,
            predictions,
            image_ids=sample_ids,
            remap=remap,
            max_det=max_det,
            class_metrics=False,  # speed: skip per-class in bootstrap
        )
        val = result.get(metric_key, float("nan"))
        if not math.isnan(val):
            boot_values.append(val)

    if not boot_values:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "std": float("nan")}

    arr = np.array(boot_values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "lo": float(np.percentile(arr, 2.5)),
        "hi": float(np.percentile(arr, 97.5)),
        "std": float(arr.std(ddof=0)),
    }


# ---------------------------------------------------------------------------
# Ephemeral smoke test (run with:  python -m scripts.training.yolov5s.eval_suite.scoring)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # ── Build two fake COCO-format annotation files ───────────────────────────
    def _make_coco_json(images, annotations, categories):
        return {"images": images, "annotations": annotations, "categories": categories}

    cats = [{"id": 1, "name": "lion"}, {"id": 2, "name": "tiger"}, {"id": 3, "name": "leopard"}]

    real_data = _make_coco_json(
        images=[
            {"id": 1, "width": 640, "height": 480, "file_name": "img1.jpg", "band": "C"},
            {"id": 2, "width": 640, "height": 480, "file_name": "img2.jpg", "band": "D"},
        ],
        annotations=[
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 80, 60], "area": 4800},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [200, 200, 100, 80], "area": 8000},
            {"id": 3, "image_id": 2, "category_id": 3, "bbox": [50, 50, 120, 90], "area": 10800},
        ],
        categories=cats,
    )

    synth_data = _make_coco_json(
        images=[
            {"id": 1, "width": 640, "height": 480, "file_name": "s1.jpg", "band": "A"},
        ],
        annotations=[
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [15, 15, 80, 60], "area": 4800},
        ],
        categories=cats,
    )

    # Write to temp files
    with tempfile.TemporaryDirectory() as tmp:
        real_path = Path(tmp) / "real_test.json"
        synth_path = Path(tmp) / "synth_test.json"
        real_path.write_text(json.dumps(real_data))
        synth_path.write_text(json.dumps(synth_data))

        # ── build GT indices ──────────────────────────────────────────────────
        real_gt = build_gt_index(real_path)
        synth_gt = build_gt_index(synth_path)

        print(f"\nreal GT images: {list(real_gt['images'].keys())}")
        print(f"synth GT images: {list(synth_gt['images'].keys())}")

        # ── fake predictions (mostly correct for image 1, wrong for image 2) ──
        real_preds = [
            {"image_id": 1, "category_id": 1, "bbox": [10, 10, 80, 60], "score": 0.95},
            {"image_id": 1, "category_id": 2, "bbox": [200, 200, 100, 80], "score": 0.88},
            {"image_id": 2, "category_id": 1, "bbox": [50, 50, 120, 90], "score": 0.70},  # wrong class
        ]
        synth_preds = [
            {"image_id": 1, "category_id": 1, "bbox": [15, 15, 80, 60], "score": 0.92},
        ]

        # ── score() — fine granularity ────────────────────────────────────────
        result_fine = score(real_gt, real_preds, class_metrics=True)
        assert isinstance(result_fine["map"], float), "map should be float"
        assert isinstance(result_fine["map_per_class"], dict), "map_per_class should be dict"
        assert result_fine["n_gt"] == 3
        assert result_fine["n_images"] == 2
        print(f"\nFine score: map={result_fine['map']:.4f}  map_50={result_fine['map_50']:.4f}")
        print(f"  per_class: {result_fine['map_per_class']}")

        # ── score() — detect granularity (all → class 1) ─────────────────────
        detect_remap = {1: 1, 2: 1, 3: 1}
        result_detect = score(real_gt, real_preds, remap=detect_remap, class_metrics=True)
        assert result_detect["n_gt"] == 3
        print(f"\nDetect score: map={result_detect['map']:.4f}  map_50={result_detect['map_50']:.4f}")
        print(f"  per_class (should be single key=1): {result_detect['map_per_class']}")
        assert set(result_detect["map_per_class"].keys()) == {1}, \
            f"detect remap should collapse to class 1, got {set(result_detect['map_per_class'].keys())}"

        # ── filter_image_ids_by_band ──────────────────────────────────────────
        band_c_ids = filter_image_ids_by_band(real_gt, {"C"})
        assert band_c_ids == {1}, f"expected {{1}}, got {band_c_ids}"
        print(f"\nBand C image IDs: {band_c_ids}")

        # ── merge_domains ─────────────────────────────────────────────────────
        mixed_gt, mixed_preds = merge_domains(real_gt, real_preds, synth_gt, synth_preds)
        assert len(mixed_gt["images"]) == 3  # 2 real + 1 synth
        synth_offset_id = 2 + 1  # max real id=2, synth id=1 → 3
        assert synth_offset_id in mixed_gt["images"]
        print(f"\nMixed GT image IDs: {sorted(mixed_gt['images'].keys())}")

        result_mixed = score(mixed_gt, mixed_preds, class_metrics=True)
        print(f"Mixed score: map={result_mixed['map']:.4f}  map_50={result_mixed['map_50']:.4f}")

        # ── within_group_confusion ────────────────────────────────────────────
        # Groups: big cats (1,2,3) → group 10
        coarse_remap = {1: 10, 2: 10, 3: 10}
        confusion = within_group_confusion(
            real_gt,
            real_preds,
            coarse_remap=coarse_remap,
            lookalike_group_ids=[10],
        )
        print(f"\nWithin-group confusion: {confusion}")
        assert "by_group" in confusion
        assert 10 in confusion["by_group"]
        assert isinstance(confusion["overall_confusion_rate"], float)

        # ── domain_shift_delta ────────────────────────────────────────────────
        delta = domain_shift_delta(real_gt, real_preds, synth_gt, synth_preds)
        print(f"\nDomain shift delta: mean_delta={delta['mean_delta']:.4f}")
        assert "per_class" in delta
        assert isinstance(delta["mean_delta"], float)

        # ── bootstrap_ci (small n_boot for speed) ────────────────────────────
        ci = bootstrap_ci(real_gt, real_preds, image_ids={1, 2}, remap=None, max_det=100, n_boot=50, seed=42)
        print(f"\nBootstrap CI (50 replicates): mean={ci['mean']:.4f} [{ci['lo']:.4f}, {ci['hi']:.4f}] std={ci['std']:.4f}")
        assert all(isinstance(v, float) for v in ci.values())

    print("\n[OK] All smoke tests passed.")
