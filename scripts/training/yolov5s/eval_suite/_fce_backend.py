"""Internal per-class scoring engine backed by ``faster_coco_eval``.

Not part of the public API — ``scoring.score()`` is the sole caller. Kept as
a separate module so ``score()``'s per-class loop only has to swap which
function computes the COCO-12 metric vector for one class; the loop
structure itself (restrict to *relevant_ids*, one class at a time) is
unchanged from the prior ``torchmetrics``-based implementation, since that
structure — not the per-class engine — is what bounds peak memory (see
``docs/plans/2026-07-22_eval-suite-scoring-performance-investigation.md``
and the benchmark recorded there: a single all-classes ``faster_coco_eval``
call is ~11-15x faster than the per-class loop but peaks at 23-27GB RSS at
real/mixed test-set scale, versus ~2.5GB keeping the per-class restriction —
so the per-class loop is kept and only its inner engine is swapped, trading
some of that speedup for the same memory safety the loop already had).
"""
from __future__ import annotations

import math

import faster_coco_eval as fce

METRIC_KEYS = (
    "map", "map_50", "map_75", "map_small", "map_medium", "map_large",
    "mar_1", "mar_10", "mar_100", "mar_small", "mar_medium", "mar_large",
)


def _safe_float(v) -> float:
    v = float(v)
    return v if not (math.isnan(v) or math.isinf(v)) else float("nan")


def score_class(
    cid: int,
    relevant_ids: list[int],
    gt_by_image: dict[int, list[tuple[int, list[float]]]],
    pred_by_image: dict[int, list[tuple[int, list[float], float]]],
    image_wh: dict[int, tuple[int, int]],
    max_det: int,
) -> dict[str, float]:
    """Compute the COCO-12 metric vector for one class, restricted to
    *relevant_ids* (mirrors the caller's own memory-bounding restriction).

    ``gt_by_image``/``pred_by_image`` hold ``(label, xyxy_box[, score])``
    tuples exactly as built by ``scoring.score()``. Returns a flat dict over
    :data:`METRIC_KEYS`, using ``-1.0`` for any submetric with no qualifying
    GT/pred — matching the COCO/pycocotools convention ``scoring.score()``'s
    aggregation step already expects.
    """
    images = [
        {"id": iid, "width": image_wh.get(iid, (0, 0))[0], "height": image_wh.get(iid, (0, 0))[1],
         "file_name": ""}
        for iid in relevant_ids
    ]
    categories = [{"id": cid, "name": str(cid)}]

    gt_annotations = []
    next_ann_id = 1
    for iid in relevant_ids:
        for lbl, box in gt_by_image[iid]:
            if lbl != cid:
                continue
            x1, y1, x2, y2 = box
            gt_annotations.append({
                "id": next_ann_id, "image_id": iid, "category_id": cid,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": (x2 - x1) * (y2 - y1), "iscrowd": 0,
            })
            next_ann_id += 1

    dt_list = [
        {"image_id": iid, "category_id": cid, "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
         "score": sc}
        for iid in relevant_ids
        for lbl, box, sc in pred_by_image[iid]
        if lbl == cid
    ]

    coco_gt = fce.COCO({"images": images, "annotations": gt_annotations, "categories": categories})
    # loadRes() indexes anns[0] internally and chokes on an empty list — build
    # an empty-detections COCO object directly in that case instead.
    if dt_list:
        coco_dt = coco_gt.loadRes(dt_list)
    else:
        coco_dt = fce.COCO({"images": images, "annotations": [], "categories": categories})

    ev = fce.COCOeval_faster(coco_gt, coco_dt, iouType="bbox")
    ev.params.maxDets = [1, 10, max_det]
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    return {k: _safe_float(v) for k, v in zip(METRIC_KEYS, ev.stats[:12])}
