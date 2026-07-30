"""Standalone evaluation entrypoint for a trained cell's YOLO26n checkpoint.

Adapted from scripts/training/yolo26n/eval_suite/run_evaluation.py for the
synthetic-generator comparison experiment, and deliberately simpler than the
production version in two ways:

1. ``DEFAULT_REAL_ANN`` points at this experiment's fixed real test set
   (data/synthetic_model_comparison/test/annotations_test.json) instead of
   the 225-class production test set, and the synthetic/"mixed"-domain path
   is dropped entirely — this experiment's test set is real-only by design
   (docs/synthetic-model-comparison/01_experiment-design.md §5 point 3:
   synthetic images are train-only), so there is no synthetic
   test-annotations file to evaluate against here.
2. The report itself only covers this experiment's actual Axis C ask
   (docs/synthetic-model-comparison/06_evaluation-methodology.md §4):
   headline real-test mAP, per-class AP (rare-species readout), and the
   zebra-style within-look-alike-group confusion matrix. The production
   suite's granularity-gap, band×granularity grid, detect-analog, and
   domain-shift tiers are not computed here — see eval_suite/report.py's
   module docstring for why each was dropped.

grouping/scoring/report are this package's own eval_suite (originally copied
from scripts/training/yolov5s/eval_suite/{grouping,scoring,report}.py, then
trimmed to the above scope). grouping's default reports/lookalike_groups_v2.csv
and reports/dataset_split_summary.json are still valid here — this
experiment's 12 classes keep their original 225-class taxonomy ids (e.g. the
zebra look-alike group), so no regeneration is needed (see
docs/synthetic-model-comparison/11_detector-architecture-selection.md).

Usage
-----
    # evaluate best.pt from the latest training run (default):
    python -m scripts.synthetic_model_comparison.training.eval_suite.run_evaluation

    # a specific run, or an explicit checkpoint:
    python -m scripts.synthetic_model_comparison.training.eval_suite.run_evaluation \\
        --run-dir scripts/synthetic_model_comparison/training/model_exports/<run_name>
    python -m scripts.synthetic_model_comparison.training.eval_suite.run_evaluation \\
        --checkpoint .../<run_name>/best.pt

    # smoke test on 40 images, CPU:
    python -m scripts.synthetic_model_comparison.training.eval_suite.run_evaluation --limit 40 --device cpu

This module is *independent* of training — it only needs a checkpoint and the
real test annotations file. It can also be invoked programmatically via
:func:`evaluate_checkpoint` (used by the optional post-training --full-eval hook).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

import scripts.synthetic_model_comparison.training.constants as constants
from scripts.synthetic_model_comparison.training.eval_suite import grouping, predict, report, scoring

logger = logging.getLogger(__name__)

DEFAULT_REAL_ANN = constants.ANNOTATIONS_TEST


def _subsample_annotations(src: Path, n: int, dst: Path, seed: int = constants.SEED) -> Path:
    """Write a reduced COCO JSON with a *representative random* sample of *n* images
    (+ their anns) to *dst*.

    Used only for smoke testing (``--limit``). A fixed-seed random sample is used
    rather than the first *n* records, because the annotation files are ordered by
    class — the first N would all be the same one or two classes and give a
    misleadingly degenerate metric. Categories are preserved in full so the label
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
    """Run the (real-only) evaluation and write artifacts. Returns the report dict."""
    checkpoint = Path(checkpoint)
    if device is None:
        device = _resolve_device("auto")
    if output_dir is None:
        output_dir = checkpoint.parent / f"eval_{checkpoint.stem}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    real_ann = Path(real_ann)

    if limit is not None:
        sub_dir = output_dir / "_subsets"
        real_ann = _subsample_annotations(real_ann, limit, sub_dir / "real.json")

    logger.info("=== evaluation (real-only) ===")
    logger.info("checkpoint=%s device=%s output_dir=%s", checkpoint, device, output_dir)
    logger.info("real_ann=%s max_det=%d", real_ann, max_det)

    # ── 1. Inference (cached) ───────────────────────────────────────────────
    real_pred = predict.run_inference(
        checkpoint_path=checkpoint, annotations_path=real_ann,
        output_path=output_dir / "predictions_real.json", device=device,
        conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det,
        image_size=image_size, batch_size=batch_size, num_workers=num_workers, cache=cache,
    )

    # ── 2. GT index ──────────────────────────────────────────────────────────
    real_gt = scoring.build_gt_index(real_ann)
    real_preds = real_pred["predictions"]

    # ── 3. Label remaps + look-alike group metadata ─────────────────────────
    cat_id_to_name = real_gt["cats"]
    cat_ids = sorted(cat_id_to_name.keys())
    fine_remap = grouping.identity_remap(cat_ids)
    coarse_remap = grouping.load_coarse_remap()
    lookalike_gids = grouping.lookalike_group_ids()
    group_labels = grouping.load_group_labels()

    # Band (A/B/C/D) is a static per-species property in
    # reports/dataset_split_summary.json, independent of this experiment's
    # expanded real-test counts (02_class-selection.md §4a only changes how
    # many test images a class gets, not which band it's in) — so the
    # production band lookup is reused unchanged.
    band_info = grouping.load_class_to_band(cat_id_to_name=cat_id_to_name)
    band_by_id = band_info["by_id"]

    # ── 4. Assemble + emit ─────────────────────────────────────────────────
    rep = report.build_report(
        gt=real_gt, preds=real_preds,
        fine_remap=fine_remap, coarse_remap=coarse_remap,
        band_by_id=band_by_id,
        lookalike_gids=lookalike_gids, group_labels=group_labels,
        cat_id_to_name=cat_id_to_name, max_det=max_det,
        bootstrap_n=bootstrap_n, checkpoint=str(checkpoint),
    )
    paths = report.emit_all(rep, output_dir, log_mlflow=log_mlflow)
    logger.info("evaluation complete — report at %s", paths["markdown"])
    return rep


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a synthetic-model-comparison YOLO26n checkpoint.")
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
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--device", default="auto", help="'auto' | 'cuda' | 'cpu'")
    p.add_argument("--max-det", type=int, default=constants.EVAL_MAX_DET)
    p.add_argument("--bootstrap", type=int, default=0, help="bootstrap replicates for headline CI (0=off)")
    p.add_argument("--batch-size", type=int, default=constants.BATCH_SIZE)
    p.add_argument("--num-workers", type=int, default=constants.NUM_WORKERS)
    p.add_argument("--limit", type=int, default=None, help="subsample N images (smoke test)")
    p.add_argument("--no-cache", action="store_true", help="ignore cached predictions")
    p.add_argument("--mlflow", action="store_true", help="log scalars + artifacts to MLflow")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    run_dir = args.run_dir or constants.latest_run_dir()

    checkpoint = args.checkpoint
    if checkpoint is not None:
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
        checkpoint=checkpoint, real_ann=args.real_ann,
        output_dir=args.output_dir, device=_resolve_device(args.device),
        max_det=args.max_det, bootstrap_n=args.bootstrap,
        batch_size=args.batch_size, num_workers=args.num_workers,
        limit=args.limit, cache=not args.no_cache, log_mlflow=args.mlflow,
    )


if __name__ == "__main__":
    main()
