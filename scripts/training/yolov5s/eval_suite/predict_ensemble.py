"""Generate COCO-format predictions from the MegaDetector v5 + SpeciesNet ensemble.

This script is the inference counterpart of ``predict.py`` for the teacher-baseline
pipeline: MegaDetector v5 detects animals; SpeciesNet classifies each crop; the
SpeciesNet probabilities are projected onto the 225-class evaluation set;
the joint ``md_conf × sn_score`` detection confidence is stored in the COCO
predictions JSON consumed by ``run_evaluation.py`` via ``--real-predictions``/
``--synth-predictions``.

Run inside the default training container (make run).

Usage
-----
    # off-the-shelf (pretrained) ensemble — output-dir defaults to
    # scripts/training/megadet_speciesnet_ensemble/model_exports/pretrained/:
    uv run python -m scripts.training.yolov5s.eval_suite.predict_ensemble

    # smoke test (100 images per domain):
    uv run python -m scripts.training.yolov5s.eval_suite.predict_ensemble --limit 100

    # fine-tuned SpeciesNet classifier — output-dir defaults to
    # scripts/training/megadet_speciesnet_ensemble/model_exports/finetuned-<run_name>/
    # (derived from the checkpoint's parent directory name):
    uv run python -m scripts.training.yolov5s.eval_suite.predict_ensemble \\
        --checkpoint scripts/training/teacher_finetune/model_exports/<run_name>/best.pt

Output JSON schema (frozen contract, same as predict.py)
---------------------------------------------------------
::

    {
      "checkpoint": "<megadet_speciesnet_ensemble-pretrained | path to fine-tuned .pt>",
      "annotations": "<path str>",
      "eval": {"conf_thres": <md_conf>, "iou_thres": 0.6, "max_det": 300, "image_size": null},
      "num_images": <int>,
      "predictions": [
        {"image_id": <int>, "category_id": <int>, "bbox": [x, y, w, h], "score": <float>}
      ]
    }

``bbox`` is original-image-pixel **xywh** (COCO convention).
``category_id`` is the COCO category id (1..225).
``score`` is ``md_conf × top-225-class-score`` (joint detection × classification).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[4]
CLASSES_225_PATH = REPO_ROOT / "reports" / "classes_225.csv"
ENSEMBLE_MODEL_EXPORTS = REPO_ROOT / "scripts" / "training" / "megadet_speciesnet_ensemble" / "model_exports"

# Identifier stored in the predictions JSON header when no --checkpoint is given.
PRETRAINED_CHECKPOINT_ID = "megadet_speciesnet_ensemble-pretrained"

_DEFAULT_REAL_ANN = REPO_ROOT / "data" / "real" / "annotations_test.json"
_DEFAULT_SYNTH_ANN = REPO_ROOT / "data" / "synthetic" / "annotations_test.json"


# ── Environment check ─────────────────────────────────────────────────────────

def _check_environment() -> None:
    try:
        import speciesnet  # noqa: F401
    except ImportError:
        print(
            "ERROR: 'speciesnet' is not installed.\n"
            "Rebuild the training image (make build && make run), or inside\n"
            "the container run: uv sync --frozen --no-dev",
            file=sys.stderr,
        )
        sys.exit(1)


# ── 225-class taxonomy maps ───────────────────────────────────────────────────

def _load_taxonomy_maps() -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """Parse classes_225.csv → three taxonomy-keyed dicts for 225-class projection.

    Returns (genus_species_to_225, genus_to_225, family_to_225) where each maps
    a lowercased taxonomy key to a 0-based row index into classes_225.csv
    (== COCO category_id - 1).
    """
    genus_species_to_225: dict[str, int] = {}
    genus_to_225: dict[str, int] = {}
    family_to_225: dict[str, int] = {}

    with open(CLASSES_225_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx_225, row in enumerate(reader):
            sci = row["scientific_name"].strip().lower()
            level = row["level"].strip()
            parts = sci.split()
            if level == "species" and len(parts) >= 2:
                genus_species_to_225[f"{parts[0]} {' '.join(parts[1:])}"] = idx_225
            elif level == "genus" and parts:
                genus_to_225[parts[0]] = idx_225
            elif level == "family" and parts:
                family_to_225[parts[0]] = idx_225

    logger.info(
        "taxonomy maps: %d species-level, %d genus-level, %d family-level entries",
        len(genus_species_to_225), len(genus_to_225), len(family_to_225),
    )
    return genus_species_to_225, genus_to_225, family_to_225


def _compute_probs_225(
    sn_classes: list[str],
    sn_scores: list[float],
    genus_species_to_225: dict[str, int],
    genus_to_225: dict[str, int],
    family_to_225: dict[str, int],
) -> list[float]:
    """Project SpeciesNet top-K class labels onto the 225-class probability vector.

    Each SpeciesNet label has the form ``uuid;class;order;family;genus;species;common``.
    Lookup priority per label: species (genus+species match) → genus → family.
    Probabilities are accumulated, so if two SpeciesNet classes map to the same
    225-class entry the scores sum.

    Returns a 225-element list; all values are 0.0 if no label maps to any
    225-class entry (out-of-distribution image).
    """
    probs = [0.0] * 225
    for label, score in zip(sn_classes, sn_scores):
        parts = label.split(";")
        if len(parts) < 6:
            continue
        family = parts[3].lower().strip()
        genus = parts[4].lower().strip()
        species = parts[5].lower().strip()

        cls225_idx = genus_species_to_225.get(f"{genus} {species}")
        if cls225_idx is None:
            cls225_idx = genus_to_225.get(genus)
        if cls225_idx is None:
            cls225_idx = family_to_225.get(family)
        if cls225_idx is not None:
            probs[cls225_idx] += score

    return probs


# ── MegaDetector ──────────────────────────────────────────────────────────────

def _load_megadetector(device: str):
    try:
        from PytorchWildlife.models import detection as pw_detection
    except ImportError as exc:
        print(f"ERROR: missing PytorchWildlife — {exc}", file=sys.stderr)
        sys.exit(1)

    logger.info("Loading MegaDetector v5 ...")
    model = pw_detection.MegaDetectorV5(device=device, pretrained=True)
    model.model.eval()
    return model


def _detect_animals(
    md_model,
    img: Image.Image,
    device: str,
    md_conf_thres: float,
) -> list[dict[str, Any]]:
    """Run MegaDetector on a PIL image. Returns animal detections (abs-pixel xyxy, conf)."""
    import torch
    from yolov5.utils.general import non_max_suppression, scale_boxes

    W, H = img.size
    img_size = md_model.IMAGE_SIZE  # 1280

    img_tensor = md_model.transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        raw = md_model.model(img_tensor)[0].float().cpu()

    preds = non_max_suppression(raw, conf_thres=md_conf_thres)
    pred = preds[0]

    if pred is None or len(pred) == 0:
        return []

    pred_np = pred.numpy().copy()
    pred_np[:, :4] = scale_boxes([img_size] * 2, pred_np[:, :4], (H, W)).round()

    return sorted(
        [
            {
                "x1": max(0, int(row[0])), "y1": max(0, int(row[1])),
                "x2": min(W, int(row[2])), "y2": min(H, int(row[3])),
                "conf": float(row[4]),
            }
            for row in pred_np
            if int(row[5]) == 0  # class 0 = animal
        ],
        key=lambda d: d["conf"],
        reverse=True,
    )


# ── SpeciesNet classifier ─────────────────────────────────────────────────────

def _load_speciesnet(checkpoint_path: Path | None = None):
    """Load the stock SpeciesNet classifier, optionally overwriting its weights.

    If *checkpoint_path* is given, it must be a ``teacher_finetune``-style
    checkpoint (a dict with a ``"model"`` key holding a plain ``state_dict()``
    for ``SpeciesNet(...).classifier.model`` — see
    ``scripts/training/teacher_finetune/training_pipeline.py``'s
    ``_save_checkpoint`` and ``cache_soft_labels.py``'s reload of the same
    checkpoint format). MegaDetector is untouched either way — fine-tuning is
    currently scoped to the SpeciesNet classifier only.
    """
    from speciesnet import SpeciesNet, DEFAULT_MODEL
    logger.info("Loading SpeciesNet classifier ...")
    sn = SpeciesNet(DEFAULT_MODEL, components="classifier", geofence=False)

    if checkpoint_path is not None:
        import torch
        logger.info("Loading fine-tuned classifier weights from %s", checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location=sn.classifier.device, weights_only=False)
        sn.classifier.model.load_state_dict(ckpt["model"])
        sn.classifier.model.eval()

    return sn


def _classify_crops(
    sn,
    crops: list[Image.Image],
) -> list[tuple[list[str], list[float]] | None]:
    """Classify PIL crops in-memory via the classifier's preprocess/batch_predict.

    Bypasses ``SpeciesNet.classify()``'s file-based orchestration (temp-file
    save/read/delete + internal thread-pool queues), which is unnecessary
    overhead here since the crops already live in memory and the whole batch
    can go through the classifier model in a single stacked forward pass
    (``SpeciesNetClassifier.batch_predict``). Equivalent to the previous
    ``sn.classify(filepaths=...)`` call: no bboxes are passed either way, since
    MegaDetector's crop has already been applied.

    Returns one (classes, scores) tuple per crop (None on preprocessing failure).
    """
    if not crops:
        return []

    fake_paths = [f"crop_{i:04d}" for i in range(len(crops))]
    imgs = []
    for i, crop in enumerate(crops):
        try:
            imgs.append(sn.classifier.preprocess(crop, bboxes=None, resize=True))
        except Exception as exc:
            logger.warning("failed to preprocess crop %d: %s", i, exc)
            imgs.append(None)

    try:
        results = sn.classifier.batch_predict(fake_paths, imgs)
    except Exception as exc:
        logger.warning("SpeciesNet batch_predict failed: %s", exc)
        return [None] * len(crops)

    out: list[tuple[list[str], list[float]] | None] = []
    for r in results:
        cls_data = r.get("classifications")
        if cls_data:
            out.append((cls_data.get("classes", []), cls_data.get("scores", [])))
        else:
            out.append(None)
    return out


# ── Caching helpers ───────────────────────────────────────────────────────────

def _build_header(annotations_path: Path, md_conf: float, checkpoint_path: Path | None) -> dict[str, Any]:
    return {
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else PRETRAINED_CHECKPOINT_ID,
        "annotations": str(annotations_path),
        "eval": {
            "conf_thres": md_conf,
            "iou_thres": 0.6,
            "max_det": 300,
            "image_size": None,
        },
    }


def _cache_hit(existing: dict, header: dict) -> bool:
    return (
        existing.get("checkpoint") == header["checkpoint"]
        and existing.get("annotations") == header["annotations"]
        and existing.get("eval") == header["eval"]
    )


# ── Main inference entry point ────────────────────────────────────────────────

def run_inference(
    annotations_path: Path,
    output_path: Path,
    device: str,
    md_conf: float,
    sn_batch_size: int,
    cache: bool,
    limit: int | None,
    checkpoint_path: Path | None = None,
) -> dict:
    """Run MegaDetector→SpeciesNet on all images in annotations_path.

    Parameters
    ----------
    annotations_path:
        COCO annotations JSON (real or synthetic test set).
    output_path:
        Destination for the predictions JSON (same schema as predict.py output).
    device:
        ``'cuda'`` or ``'cpu'``.
    md_conf:
        MegaDetector confidence threshold for animal detections.
    sn_batch_size:
        Number of crops to classify per SpeciesNet call.
    cache:
        Return cached predictions if output_path exists with a matching header.
    limit:
        If not None, subsample this many images (fixed seed, smoke-test mode).
    checkpoint_path:
        If given, a fine-tuned SpeciesNet classifier checkpoint (see
        ``_load_speciesnet``) used instead of the stock pretrained weights.
        Recorded in the cache header so pretrained and fine-tuned runs are
        never mistaken for cache hits of one another.

    Returns
    -------
    The predictions dict (from cache or freshly computed).
    """
    annotations_path = Path(annotations_path)
    output_path = Path(output_path)

    header = _build_header(annotations_path, md_conf, checkpoint_path)

    if cache and output_path.exists():
        try:
            with open(output_path) as f:
                existing = json.load(f)
            if _cache_hit(existing, header):
                logger.info(
                    "cache HIT — reusing %s (%d predictions over %d images)",
                    output_path,
                    len(existing.get("predictions", [])),
                    existing.get("num_images", "?"),
                )
                return existing
            else:
                logger.info("cache MISS — header mismatch; recomputing %s", output_path)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("cannot read cache %s (%s); recomputing", output_path, exc)

    with open(annotations_path) as f:
        ann_data = json.load(f)

    image_records = ann_data["images"]
    if limit is not None:
        import random
        rng = random.Random(42)
        image_records = rng.sample(image_records, min(limit, len(image_records)))
        logger.info("subsampled to %d images (--limit %d)", len(image_records), limit)

    logger.info(
        "=== ensemble inference: %d images, md_conf=%.2f, sn_batch=%d, device=%s ===",
        len(image_records), md_conf, sn_batch_size, device,
    )

    gs_to_225, g_to_225, f_to_225 = _load_taxonomy_maps()
    md_model = _load_megadetector(device)
    sn = _load_speciesnet(checkpoint_path)

    predictions: list[dict[str, Any]] = []

    # Rolling crop buffer: flush to SpeciesNet every sn_batch_size crops.
    crop_buffer: list[dict[str, Any]] = []

    def _flush() -> None:
        if not crop_buffer:
            return
        crops_pil = [item["crop"] for item in crop_buffer]
        sn_results = _classify_crops(sn, crops_pil)

        for item, sn_res in zip(crop_buffer, sn_results):
            if sn_res is None:
                continue
            sn_classes, sn_scores = sn_res
            if not sn_classes:
                continue

            probs_225 = _compute_probs_225(
                sn_classes, sn_scores, gs_to_225, g_to_225, f_to_225
            )
            best_idx = max(range(225), key=lambda i: probs_225[i])
            best_score = probs_225[best_idx]

            if best_score == 0.0:
                continue  # no 225-class match → no prediction

            cat_id = best_idx + 1  # 0-based idx_225 → 1-based COCO category_id
            joint_score = item["md_conf"] * min(best_score, 1.0)

            predictions.append({
                "image_id": item["image_id"],
                "category_id": cat_id,
                "bbox": [
                    round(item["x1"], 3),
                    round(item["y1"], 3),
                    round(item["w"], 3),
                    round(item["h"], 3),
                ],
                "score": round(joint_score, 6),
            })

        crop_buffer.clear()

    for rec in tqdm(image_records, desc="ensemble-predict", unit="img", dynamic_ncols=True):
        image_path = REPO_ROOT / rec["file_name"]
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as exc:
            logger.warning("cannot open %s: %s", image_path, exc)
            continue

        W, H = img.size

        try:
            detections = _detect_animals(md_model, img, device, md_conf)
        except Exception as exc:
            logger.warning("MegaDetector failed on %s: %s", image_path, exc)
            continue

        for det in detections:
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            w_box = x2 - x1
            h_box = y2 - y1
            if w_box < 2 or h_box < 2:
                continue
            crop = img.crop((x1, y1, x2, y2))
            crop_buffer.append({
                "image_id": rec["id"],
                "x1": float(x1), "y1": float(y1),
                "w": float(w_box), "h": float(h_box),
                "md_conf": det["conf"],
                "crop": crop,
            })

            if len(crop_buffer) >= sn_batch_size:
                _flush()

    _flush()  # remaining crops

    logger.info(
        "ensemble inference complete: %d predictions over %d images",
        len(predictions), len(image_records),
    )

    output_dict: dict[str, Any] = {
        **header,
        "num_images": len(image_records),
        "predictions": predictions,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_dict, f, indent=None, separators=(",", ":"))
    logger.info("predictions written to %s", output_path)

    return output_dict


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--real-ann", type=Path, default=_DEFAULT_REAL_ANN,
        help="Real test annotations COCO JSON.",
    )
    p.add_argument(
        "--synth-ann", type=Path, default=_DEFAULT_SYNTH_ANN,
        help="Synthetic test annotations; pass 'none' to skip.",
    )
    p.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Fine-tuned SpeciesNet classifier checkpoint (a teacher_finetune-style "
        "best.pt/last.pt). Omit to use the stock pretrained SpeciesNet classifier.",
    )
    p.add_argument(
        "--output-dir", type=Path, default=None,
        help="Write predictions_real.json / predictions_synth.json here. Defaults to "
        "megadet_speciesnet_ensemble/model_exports/pretrained/ (no --checkpoint) or "
        ".../model_exports/finetuned-<checkpoint's run dir name>/ (with --checkpoint).",
    )
    p.add_argument("--device", default="auto", help="'auto' | 'cuda' | 'cpu'")
    p.add_argument("--md-conf", type=float, default=0.1,
                   help="MegaDetector animal-detection confidence threshold (default: 0.1).")
    p.add_argument("--batch-size", type=int, default=32,
                   help="SpeciesNet crops per classify() call (default: 32).")
    p.add_argument("--limit", type=int, default=None,
                   help="Subsample N images per domain (smoke test).")
    p.add_argument("--no-cache", action="store_true", help="Ignore cached predictions.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    _check_environment()

    import torch
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    logger.info("device: %s", device)

    if args.output_dir is None:
        if args.checkpoint is not None:
            run_label = f"finetuned-{args.checkpoint.resolve().parent.name}"
        else:
            run_label = "pretrained"
        args.output_dir = ENSEMBLE_MODEL_EXPORTS / run_label
        logger.info("--output-dir not given; defaulting to %s", args.output_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_inference(
        annotations_path=args.real_ann,
        output_path=args.output_dir / "predictions_real.json",
        device=device,
        md_conf=args.md_conf,
        sn_batch_size=args.batch_size,
        cache=not args.no_cache,
        limit=args.limit,
        checkpoint_path=args.checkpoint,
    )

    synth_ann = None if str(args.synth_ann).lower() == "none" else args.synth_ann
    if synth_ann is not None:
        if synth_ann.exists():
            run_inference(
                annotations_path=synth_ann,
                output_path=args.output_dir / "predictions_synth.json",
                device=device,
                md_conf=args.md_conf,
                sn_batch_size=args.batch_size,
                cache=not args.no_cache,
                limit=args.limit,
                checkpoint_path=args.checkpoint,
            )
        else:
            logger.warning("synthetic annotations not found at %s — skipping", synth_ann)
    else:
        logger.info("synthetic domain skipped (--synth-ann none).")


if __name__ == "__main__":
    main()
