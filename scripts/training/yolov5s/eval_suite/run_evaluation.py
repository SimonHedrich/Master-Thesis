"""Standalone evaluation entrypoint for a trained YOLOv5s checkpoint.

Runs the full evaluation strategy
(``docs/plans/2026-06-10_model-evaluation-strategy.md``) against a saved
checkpoint: inference is run once per component domain (real, synthetic), cached
to a predictions JSON, then scored many ways (granularity × band × domain) and
written to a Markdown report + CSV/JSON artifacts.

Usage
-----
    # evaluate best.pt from the latest training run (default):
    uv run python -m scripts.training.yolov5s.eval_suite.run_evaluation

    # a specific run, or an explicit checkpoint:
    uv run python -m scripts.training.yolov5s.eval_suite.run_evaluation \
        --run-dir scripts/training/yolov5s/model_exports/yolov5s-20260602-233434
    uv run python -m scripts.training.yolov5s.eval_suite.run_evaluation \
        --checkpoint .../yolov5s-20260602-233434/best.pt

    # smoke test on 40 images per domain, CPU:
    uv run python -m scripts.training.yolov5s.eval_suite.run_evaluation --limit 40 --device cpu

    # evaluate pre-computed predictions (e.g. MegaDetector+SpeciesNet ensemble):
    uv run python -m scripts.training.yolov5s.eval_suite.run_evaluation \
        --real-predictions .../megadet_speciesnet_ensemble/predictions_real.json \
        --synth-predictions .../megadet_speciesnet_ensemble/predictions_synth.json \
        --output-dir .../megadet_speciesnet_ensemble/eval/

This module is *independent* of training — it only needs a checkpoint and the
test annotation files. It can also be invoked programmatically via
:func:`evaluate_checkpoint` (used by the optional post-training hook) or via
:func:`evaluate_from_predictions` for pre-computed prediction files.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

import scripts.training.yolov5s.constants as constants
from scripts.training.yolov5s.eval_suite import grouping, predict, report, scoring

logger = logging.getLogger(__name__)

DEFAULT_REAL_ANN = constants.REPO_ROOT / "data" / "real" / "annotations_test.json"
DEFAULT_SYNTH_ANN = constants.REPO_ROOT / "data" / "synthetic" / "annotations_test.json"


def _subsample_annotations(src: Path, n: int, dst: Path, seed: int = constants.SEED) -> Path:
    """Write a reduced COCO JSON with a *representative random* sample of *n* images
    (+ their anns) to *dst*.

    Used only for smoke testing (``--limit``). A fixed-seed random sample is used
    rather than the first *n* records, because the annotation files are ordered by
    class — the first N would all be the same one or two (band-A) classes and give
    a misleadingly degenerate metric. Categories are preserved in full so the label
    space is unchanged.
    """
    import random

    with src.open() as f:
        data = json.load(f)
    rng = random.Random(seed)
    keep_images = rng.sample(data["images"], min(n, len(data["images"])))
    keep_ids = {img["id"] for img in keep_images}
    keep_anns = [a for a in data["annotations"] if a["image_id"] in keep_ids]
    reduced = {
        "info": data.get("info", {}),
        "licenses": data.get("licenses", []),
        "categories": data["categories"],
        "images": keep_images,
        "annotations": keep_anns,
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(reduced))
    logger.info("subsampled %s → %s (%d images, %d anns)", src.name, dst, len(keep_images), len(keep_anns))
    return dst


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def evaluate_checkpoint(
    checkpoint: Path,
    *,
    real_ann: Path = DEFAULT_REAL_ANN,
    synth_ann: Path | None = DEFAULT_SYNTH_ANN,
    output_dir: Path | None = None,
    device: torch.device | None = None,
    max_det: int = constants.EVAL_MAX_DET,
    conf_thres: float = constants.EVAL_CONF_THRES,
    iou_thres: float = constants.EVAL_IOU_THRES,
    image_size: int = constants.IMAGE_SIZE,
    batch_size: int = constants.BATCH_SIZE,
    num_workers: int = constants.NUM_WORKERS,
    bootstrap_n: int = 0,
    cache: bool = True,
    log_mlflow: bool = False,
    limit: int | None = None,
) -> dict:
    """Run the full evaluation and write artifacts. Returns the report dict."""
    checkpoint = Path(checkpoint)
    if device is None:
        device = _resolve_device("auto")
    if output_dir is None:
        output_dir = checkpoint.parent / f"eval_{checkpoint.stem}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    real_ann = Path(real_ann)
    synth_ann = Path(synth_ann) if synth_ann is not None else None
    has_synth = synth_ann is not None and synth_ann.exists()
    if synth_ann is not None and not has_synth:
        logger.warning("synthetic annotations %s not found — evaluating real-only", synth_ann)
        synth_ann = None

    if limit is not None:
        sub_dir = output_dir / "_subsets"
        real_ann = _subsample_annotations(real_ann, limit, sub_dir / "real.json")
        if synth_ann is not None:
            synth_ann = _subsample_annotations(synth_ann, limit, sub_dir / "synth.json")

    logger.info("=== evaluation ===")
    logger.info("checkpoint=%s device=%s output_dir=%s", checkpoint, device, output_dir)
    logger.info("real_ann=%s synth_ann=%s max_det=%d", real_ann, synth_ann, max_det)

    # ── 1. Inference (once per domain, cached) ─────────────────────────────────
    real_pred = predict.run_inference(
        checkpoint_path=checkpoint, annotations_path=real_ann,
        output_path=output_dir / "predictions_real.json", device=device,
        conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det,
        image_size=image_size, batch_size=batch_size, num_workers=num_workers, cache=cache,
    )
    synth_pred = None
    if synth_ann is not None:
        synth_pred = predict.run_inference(
            checkpoint_path=checkpoint, annotations_path=synth_ann,
            output_path=output_dir / "predictions_synth.json", device=device,
            conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det,
            image_size=image_size, batch_size=batch_size, num_workers=num_workers, cache=cache,
        )

    # ── 2. GT indices + mixed merge ────────────────────────────────────────────
    real_gt = scoring.build_gt_index(real_ann)
    real_preds = real_pred["predictions"]
    if synth_ann is not None:
        synth_gt = scoring.build_gt_index(synth_ann)
        synth_preds = synth_pred["predictions"]
        mixed_gt, mixed_preds = scoring.merge_domains(real_gt, real_preds, synth_gt, synth_preds)
    else:
        synth_gt = synth_preds = None
        mixed_gt, mixed_preds = real_gt, real_preds

    # ── 3. Label remaps + band/group metadata ─────────────────────────────────
    cat_id_to_name = real_gt["cats"]
    cat_ids = sorted(cat_id_to_name.keys())
    remaps = {
        "fine": grouping.identity_remap(cat_ids),
        "coarse": grouping.load_coarse_remap(),
        "detect": grouping.load_detect_remap(cat_ids),
    }
    band_info = grouping.load_class_to_band(cat_id_to_name=cat_id_to_name)
    band_by_id = band_info["by_id"]
    lookalike_gids = grouping.lookalike_group_ids()
    group_labels = grouping.load_group_labels()

    # ── 4. Assemble + emit ─────────────────────────────────────────────────────
    rep = report.build_full_report(
        real_gt=real_gt, real_preds=real_preds,
        synth_gt=synth_gt, synth_preds=synth_preds,
        mixed_gt=mixed_gt, mixed_preds=mixed_preds,
        remaps=remaps, band_by_id=band_by_id,
        lookalike_gids=lookalike_gids, group_labels=group_labels,
        cat_id_to_name=cat_id_to_name, max_det=max_det,
        bootstrap_n=bootstrap_n, checkpoint=str(checkpoint),
    )
    paths = report.emit_all(rep, output_dir, log_mlflow=log_mlflow)
    logger.info("evaluation complete — report at %s", paths["markdown"])
    return rep


def evaluate_from_predictions(
    real_pred_path: Path,
    *,
    synth_pred_path: Path | None = None,
    real_ann: Path = DEFAULT_REAL_ANN,
    synth_ann: Path | None = DEFAULT_SYNTH_ANN,
    output_dir: Path,
    max_det: int = constants.EVAL_MAX_DET,
    bootstrap_n: int = 0,
) -> dict:
    """Run the full evaluation using pre-computed predictions JSON files.

    Skips the inference phase entirely — reads predictions from disk.  The
    predictions JSON must match the frozen schema from ``predict.py`` (or
    ``predict_ensemble.py``): a ``"predictions"`` list of
    ``{image_id, category_id, bbox, score}`` dicts.

    Parameters
    ----------
    real_pred_path:
        Path to the real-domain predictions JSON.
    synth_pred_path:
        Optional path to the synthetic-domain predictions JSON.
    real_ann:
        Real test annotations COCO JSON (for GT indexing).
    synth_ann:
        Synthetic test annotations COCO JSON.  Ignored if *synth_pred_path*
        is ``None`` or the file doesn't exist.
    output_dir:
        Directory for all report artifacts (created if absent).
    max_det:
        Maximum detections per image passed to the scorer.
    bootstrap_n:
        Number of bootstrap replicates for headline CI (0 = disabled).

    Returns
    -------
    The full report dict (same structure as :func:`evaluate_checkpoint`).
    """
    real_pred_path = Path(real_pred_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== evaluate_from_predictions ===")
    logger.info("real_pred=%s  synth_pred=%s", real_pred_path, synth_pred_path)
    logger.info("real_ann=%s  output_dir=%s", real_ann, output_dir)

    with open(real_pred_path) as f:
        real_pred = json.load(f)
    checkpoint_id = real_pred.get("checkpoint", str(real_pred_path))

    synth_pred = None
    if synth_pred_path is not None:
        synth_pred_path = Path(synth_pred_path)
        if synth_pred_path.exists():
            with open(synth_pred_path) as f:
                synth_pred = json.load(f)
        else:
            logger.warning("synth_pred_path %s not found — evaluating real-only", synth_pred_path)

    real_ann = Path(real_ann)
    synth_ann_path = Path(synth_ann) if synth_ann is not None else None
    has_synth = synth_ann_path is not None and synth_ann_path.exists() and synth_pred is not None

    # ── GT indices + mixed merge ────────────────────────────────────────────
    real_gt = scoring.build_gt_index(real_ann)
    real_preds_list = real_pred["predictions"]

    if has_synth:
        synth_gt = scoring.build_gt_index(synth_ann_path)
        synth_preds_list = synth_pred["predictions"]
        mixed_gt, mixed_preds = scoring.merge_domains(real_gt, real_preds_list, synth_gt, synth_preds_list)
    else:
        synth_gt = synth_preds_list = None
        mixed_gt, mixed_preds = real_gt, real_preds_list

    # ── Label remaps + band/group metadata ─────────────────────────────────
    cat_id_to_name = real_gt["cats"]
    cat_ids = sorted(cat_id_to_name.keys())
    remaps = {
        "fine": grouping.identity_remap(cat_ids),
        "coarse": grouping.load_coarse_remap(),
        "detect": grouping.load_detect_remap(cat_ids),
    }
    band_info = grouping.load_class_to_band(cat_id_to_name=cat_id_to_name)
    band_by_id = band_info["by_id"]
    lookalike_gids = grouping.lookalike_group_ids()
    group_labels = grouping.load_group_labels()

    # ── Assemble + emit ─────────────────────────────────────────────────────
    rep = report.build_full_report(
        real_gt=real_gt, real_preds=real_preds_list,
        synth_gt=synth_gt, synth_preds=synth_preds_list,
        mixed_gt=mixed_gt, mixed_preds=mixed_preds,
        remaps=remaps, band_by_id=band_by_id,
        lookalike_gids=lookalike_gids, group_labels=group_labels,
        cat_id_to_name=cat_id_to_name, max_det=max_det,
        bootstrap_n=bootstrap_n, checkpoint=checkpoint_id,
    )
    paths = report.emit_all(rep, output_dir, log_mlflow=False)
    logger.info("evaluation complete — report at %s", paths["markdown"])
    return rep


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a YOLOv5s checkpoint (strategy doc §9).")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to best.pt / last.pt. If omitted, uses best.pt from --run-dir "
        "(or the latest training run under model_exports/).",
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Training-run directory to evaluate (defaults to the latest run). "
        "Ignored if --checkpoint is an existing path.",
    )
    p.add_argument("--real-ann", type=Path, default=DEFAULT_REAL_ANN)
    p.add_argument("--synth-ann", type=Path, default=DEFAULT_SYNTH_ANN,
                   help="Synthetic test annotations; pass 'none' to skip (real-only).")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--device", default="auto", help="'auto' | 'cuda' | 'cpu'")
    p.add_argument("--max-det", type=int, default=constants.EVAL_MAX_DET)
    p.add_argument("--bootstrap", type=int, default=0, help="bootstrap replicates for headline CI (0=off)")
    p.add_argument("--batch-size", type=int, default=constants.BATCH_SIZE)
    p.add_argument("--num-workers", type=int, default=constants.NUM_WORKERS)
    p.add_argument("--limit", type=int, default=None, help="subsample N images/domain (smoke test)")
    p.add_argument("--no-cache", action="store_true", help="ignore cached predictions")
    p.add_argument("--mlflow", action="store_true", help="log scalars + artifacts to MLflow")
    # Pre-computed predictions mode (e.g. MegaDetector+SpeciesNet ensemble)
    p.add_argument(
        "--real-predictions", type=Path, default=None,
        help="Pre-computed COCO predictions JSON for the real domain; skips YOLOv5s "
        "inference. Requires --output-dir. Use with predict_ensemble.py output.",
    )
    p.add_argument(
        "--synth-predictions", type=Path, default=None,
        help="Pre-computed COCO predictions JSON for the synthetic domain "
        "(used together with --real-predictions).",
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    synth_ann = None if str(args.synth_ann).lower() == "none" else args.synth_ann

    # ── Pre-computed predictions mode ─────────────────────────────────────
    if args.real_predictions is not None:
        if args.output_dir is None:
            p.error("--output-dir is required when --real-predictions is used")
        if not args.real_predictions.exists():
            p.error(f"--real-predictions file not found: {args.real_predictions}")
        if args.limit is not None:
            logger.warning("--limit is ignored when --real-predictions is provided")
        evaluate_from_predictions(
            real_pred_path=args.real_predictions,
            synth_pred_path=args.synth_predictions,
            real_ann=args.real_ann,
            synth_ann=synth_ann,
            output_dir=args.output_dir,
            max_det=args.max_det,
            bootstrap_n=args.bootstrap,
        )
        return

    # ── Checkpoint mode ───────────────────────────────────────────────────
    # Resolve the run dir: explicit --run-dir, else the latest training run.
    run_dir = args.run_dir or constants.latest_run_dir()

    checkpoint = args.checkpoint
    if checkpoint is not None:
        # resolve a bare filename (e.g. "best.pt") against the chosen / latest run dir
        if not checkpoint.is_absolute() and not checkpoint.exists() and run_dir is not None:
            candidate = run_dir / checkpoint
            if candidate.exists():
                checkpoint = candidate
    else:
        if run_dir is None:
            p.error(
                "no --checkpoint given and no training run found under "
                f"{constants.OUTPUT_DIR} — pass --checkpoint or --run-dir."
            )
        checkpoint = run_dir / "best.pt"

    if not checkpoint.exists():
        p.error(f"checkpoint not found: {checkpoint}")

    logger.info("evaluating checkpoint: %s", checkpoint)

    evaluate_checkpoint(
        checkpoint=checkpoint, real_ann=args.real_ann, synth_ann=synth_ann,
        output_dir=args.output_dir, device=_resolve_device(args.device),
        max_det=args.max_det, bootstrap_n=args.bootstrap,
        batch_size=args.batch_size, num_workers=args.num_workers,
        limit=args.limit, cache=not args.no_cache, log_mlflow=args.mlflow,
    )


if __name__ == "__main__":
    main()
