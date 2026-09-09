"""Parity test for the ``scoring.py`` scoring-engine swap (TODO.md 2.1):
per-class ``torchmetrics.MeanAveragePrecision`` → per-class
``faster_coco_eval`` (see ``_fce_backend.py`` and
``docs/plans/2026-07-22_eval-suite-scoring-performance-investigation.md``).

Run command
-----------
    uv run python -m scripts.training.yolov5s.eval_suite.tests.test_scoring_engine_parity
    uv run pytest scripts/training/yolov5s/eval_suite/tests/

``tests/fixtures/scoring_parity_reference.json`` is a frozen snapshot of the
*old* (pre-rewrite) implementation's output on :func:`make_synthetic_case`,
generated once before the rewrite landed — this test never re-runs the old
implementation, only compares the current one against that snapshot.
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

from scripts.training.yolov5s.eval_suite import scoring

MAX_DET = 100  # matches production (constants.EVAL_MAX_DET) — torchmetrics names its
# mar_* keys after the actual threshold values, so any other max_det here would
# silently rename "mar_100" and break both engines' shared metric_keys assumption
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "scoring_parity_reference.json"
TOL = 1e-4

# Class-ID groups exercising distinct, deliberately non-degenerate behaviors:
PERFECT_CLASSES = list(range(1, 6))      # clean single GT + perfect-IoU pred per instance
ALLMISS_CLASSES = list(range(6, 11))     # GT present, zero predictions at all
PARTIAL_CLASSES = list(range(11, 16))    # half instances perfect, half far-off (IoU ~ 0)
OVERFLOW_CLASSES = list(range(16, 19))   # share one image with >MAX_DET total predictions
PREDONLY_CLASSES = list(range(19, 21))   # predictions with no GT anywhere (exercises -1.0 sentinel)


def make_synthetic_case(seed: int = 42, n_images: int = 500, n_classes: int = 20):
    """Build a deterministic synthetic COCO GT dict + predictions list with
    deliberately varied per-class hit rates (see the module-level class-ID
    groups above). Returns ``(gt_json_dict, predictions_list)``."""
    del seed, n_classes  # kept for signature stability; groups above are fixed
    images = [
        {"id": i, "width": 640, "height": 480, "file_name": f"img{i}.jpg", "band": "C"}
        for i in range(1, n_images + 1)
    ]
    categories = [{"id": c, "name": f"cls{c}"} for c in range(1, 21)]
    annotations: list[dict] = []
    predictions: list[dict] = []
    ann_id = 1

    # ── perfect: one clean GT + perfect-IoU prediction per instance ─────────
    for idx, cid in enumerate(PERFECT_CLASSES):
        size = 20 + idx * 40  # varies small→large for area-range bin coverage
        for img_id in range(1 + idx, n_images, 50):
            box = [10, 10, size, size]
            annotations.append({
                "id": ann_id, "image_id": img_id, "category_id": cid,
                "bbox": box, "area": size * size,
            })
            ann_id += 1
            predictions.append({"image_id": img_id, "category_id": cid, "bbox": box, "score": 0.9})

    # ── all-miss: GT present, no predictions for this class anywhere ───────
    for idx, cid in enumerate(ALLMISS_CLASSES):
        for img_id in range(2 + idx, n_images, 60):
            box = [30, 30, 40, 40]
            annotations.append({
                "id": ann_id, "image_id": img_id, "category_id": cid,
                "bbox": box, "area": 1600,
            })
            ann_id += 1

    # ── partial: alternating perfect-IoU / far-off (~0 IoU) predictions ────
    for idx, cid in enumerate(PARTIAL_CLASSES):
        instance_imgs = list(range(3 + idx, n_images, 40))
        for j, img_id in enumerate(instance_imgs):
            box = [50, 50, 30, 30]
            annotations.append({
                "id": ann_id, "image_id": img_id, "category_id": cid,
                "bbox": box, "area": 900,
            })
            ann_id += 1
            if j % 2 == 0:
                predictions.append({"image_id": img_id, "category_id": cid, "bbox": box, "score": 0.85})
            else:
                predictions.append({
                    "image_id": img_id, "category_id": cid,
                    "bbox": [500, 400, 30, 30], "score": 0.85,
                })

    # ── overflow: one shared image gets far more than MAX_DET predictions,
    # split across these classes, so the per-image score-based cap (applied
    # across all classes combined) has to drop some. Each class gets one
    # high-score correct prediction plus enough low-score noise that the
    # three classes together exceed MAX_DET=100 on this single image ────────
    overflow_img = n_images
    for idx, cid in enumerate(OVERFLOW_CLASSES):
        box = [5, 5, 15, 15]
        annotations.append({
            "id": ann_id, "image_id": overflow_img, "category_id": cid,
            "bbox": box, "area": 225,
        })
        ann_id += 1
        predictions.append({"image_id": overflow_img, "category_id": cid, "bbox": box, "score": 0.99})
        for k in range(40):
            predictions.append({
                "image_id": overflow_img, "category_id": cid,
                "bbox": [100 + k * 5, 100, 10, 10], "score": 0.001 + k * 0.001,
            })

    # ── predictions-only: no GT anywhere for these classes ──────────────────
    for idx, cid in enumerate(PREDONLY_CLASSES):
        for img_id in range(4 + idx, n_images, 80):
            predictions.append({
                "image_id": img_id, "category_id": cid, "bbox": [200, 200, 20, 20], "score": 0.7,
            })

    gt_json = {"images": images, "annotations": annotations, "categories": categories}
    return gt_json, predictions


def _score_synthetic_case() -> dict:
    gt_json, predictions = make_synthetic_case()
    with tempfile.TemporaryDirectory() as tmp:
        ann_path = Path(tmp) / "synthetic_gt.json"
        ann_path.write_text(json.dumps(gt_json))
        gt_index = scoring.build_gt_index(ann_path)
    return scoring.score(gt_index, predictions, remap=None, max_det=MAX_DET, class_metrics=True)


def _isclose(a: float, b: float, tol: float = TOL) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    return math.isclose(a, b, abs_tol=tol)


def test_hand_verified_anchors():
    """A few per-class results independently reasoned about, not just
    compared engine-to-engine — an anchor against both engines agreeing on
    the wrong number."""
    result = _score_synthetic_case()
    per_class = result["map_per_class"]

    # class 1: clean, unambiguous perfect matches, no capping interference
    # (its images — 1, 51, 101, ... — never coincide with the overflow image).
    assert _isclose(per_class[1], 1.0), f"expected class 1 AP == 1.0, got {per_class[1]}"

    # class 6: GT present, zero predictions anywhere -> AP must be exactly 0.
    assert _isclose(per_class[6], 0.0), f"expected class 6 AP == 0.0, got {per_class[6]}"

    # class 19: predictions-only, no GT anywhere -> excluded via -1.0 sentinel.
    assert per_class[19] == -1.0, f"expected class 19 sentinel -1.0, got {per_class[19]}"
    assert 19 not in [k for k in per_class if per_class[k] != -1.0 and k == 19]


def test_matches_pre_rewrite_reference():
    """Current (faster_coco_eval-backed) score() must match the frozen
    pre-rewrite (torchmetrics-backed) snapshot within float tolerance."""
    assert FIXTURE_PATH.exists(), f"reference fixture missing: {FIXTURE_PATH}"
    with FIXTURE_PATH.open() as f:
        reference = json.load(f)

    result = _score_synthetic_case()

    for key in (
        "map", "map_50", "map_75", "map_small", "map_medium", "map_large",
        "mar_1", "mar_10", "mar_100", "mar_small", "mar_medium", "mar_large",
    ):
        assert _isclose(result[key], reference[key]), (
            f"aggregate mismatch on {key}: new={result[key]} reference={reference[key]}"
        )

    ref_per_class = {int(k): v for k, v in reference["map_per_class"].items()}
    new_per_class = result["map_per_class"]
    assert set(new_per_class.keys()) == set(ref_per_class.keys())
    for cid, ref_ap in ref_per_class.items():
        assert _isclose(new_per_class[cid], ref_ap), (
            f"class {cid} AP mismatch: new={new_per_class[cid]} reference={ref_ap}"
        )

    assert result["n_images"] == reference["n_images"]
    assert result["n_dets"] == reference["n_dets"]
    assert result["n_gt"] == reference["n_gt"]


if __name__ == "__main__":
    test_hand_verified_anchors()
    print("[OK] hand-verified anchors")
    test_matches_pre_rewrite_reference()
    print("[OK] matches pre-rewrite reference")
    print("\n[OK] All parity tests passed.")
