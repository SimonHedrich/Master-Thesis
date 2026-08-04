"""Entry point: wires MLflow, builds all components, runs TrainingPipeline.

Usage:
    uv run python -m scripts.training.yolov5s.run_training_pipeline

Note: Run inside the default training container (make run), from /app (repo root).
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import subprocess
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import torch
from dotenv import load_dotenv

import scripts.training.yolov5s.constants as constants
from scripts.training.yolov5s.autoanchor import check_anchor_fit
from scripts.training.yolov5s.dataset import CocoYoloDataset, Dataloader, collate_fn, make_worker_init_fn
from scripts.training.yolov5s.logging_setup import setup_logging
from scripts.training.yolov5s.loss import YoloLoss
from scripts.training.yolov5s.training_pipeline import TrainingPipeline
from scripts.training.yolov5s.yolov5s_model import model_optimizer, model_scheduler, yolov5s_model

logger = logging.getLogger(__name__)


def set_seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _run_full_evaluation(run_dir: Path, smoke: bool, device: torch.device) -> None:
    """Optional post-training hook: run the comprehensive evaluation suite on best.pt.

    Independent of the lightweight per-epoch eval — it produces the full
    granularity × band × domain report (strategy doc §9) and logs it to the
    active MLflow run. Kept best-effort: a failure here must not fail the run.
    """
    from scripts.training.yolov5s.eval_suite.run_evaluation import evaluate_checkpoint

    best_path = run_dir / "best.pt"
    if not best_path.exists():
        logger.warning("full-eval requested but %s not found — skipping", best_path)
        return
    # On a smoke run, subsample so the hook is quick; otherwise score the full sets.
    limit = 200 if smoke else None
    synth_ann = constants.DATA_ROOT / "synthetic" / "annotations_test.json"
    logger.info("=== post-training full evaluation (limit=%s) ===", limit)
    evaluate_checkpoint(
        checkpoint=best_path,
        real_ann=constants.ANNOTATIONS_TEST,
        synth_ann=synth_ann if synth_ann.exists() else None,
        output_dir=run_dir / "evaluation",
        device=device,
        max_det=constants.EVAL_MAX_DET,
        batch_size=constants.BATCH_SIZE,
        # 0, not constants.NUM_WORKERS: this DataLoader is freshly constructed
        # here, deep into a process that has already been driving CUDA for
        # the whole training run — forking new worker processes at this point
        # (rather than near process start, like dl_train/dl_val/dl_test) hung
        # indefinitely at 0% GPU/CPU utilization in practice.
        num_workers=0,
        log_mlflow=True,
        limit=limit,
    )


def training_run(
    smoke: bool,
    run_dir: Path,
    log_file: Path | None,
    full_eval: bool = False,
    resume_from: Path | None = None,
) -> dict:
    set_seed(constants.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = 1 if smoke else constants.EPOCH_COUNT
    logger.info("seed=%d device=%s epochs=%d smoke=%s", constants.SEED, device, epochs, smoke)

    ds_train = CocoYoloDataset(
        constants.ANNOTATIONS_VAL if smoke else constants.ANNOTATIONS_TRAIN,
        constants.IMAGE_ROOT,
        constants.IMAGE_SIZE,
        augment=True,
    )
    ds_val = CocoYoloDataset(
        constants.ANNOTATIONS_VAL, constants.IMAGE_ROOT, constants.IMAGE_SIZE, augment=False
    )
    ds_test = CocoYoloDataset(
        constants.ANNOTATIONS_VAL if smoke else constants.ANNOTATIONS_TEST,
        constants.IMAGE_ROOT,
        constants.IMAGE_SIZE,
        augment=False,
    )

    num_workers = 0 if smoke else constants.NUM_WORKERS

    # Seed discipline: deterministic per-worker RNG seeding for the train loader.
    train_generator = torch.Generator().manual_seed(constants.SEED)
    train_worker_init = make_worker_init_fn(constants.SEED)

    dl_train = Dataloader(
        ds_train,
        constants.BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        worker_init_fn=train_worker_init,
        generator=train_generator,
    ).get_dataloader()
    dl_val = Dataloader(ds_val, constants.BATCH_SIZE, shuffle=False, num_workers=num_workers, collate_fn=collate_fn).get_dataloader()
    dl_test = Dataloader(ds_test, constants.BATCH_SIZE, shuffle=False, num_workers=num_workers, collate_fn=collate_fn).get_dataloader()

    model, _preprocess = yolov5s_model(constants.NUM_CLASSES, constants.PRETRAINED_WEIGHTS, device)
    model.names = ds_train.class_names

    autoanchor_result = {"bpr": None, "anchors_changed": None}
    if resume_from is None:
        # Skip on resume: the checkpoint's anchors already reflect whatever a
        # prior run (or this same check) set them to.
        autoanchor_result = check_anchor_fit(
            model, ds_train, thr=constants.HYP_ANCHOR_T, img_size=constants.IMAGE_SIZE
        )

    optimizer = model_optimizer(model)
    scheduler = model_scheduler(optimizer, steps_per_epoch=len(dl_train), epochs=epochs)
    loss_fn = YoloLoss(model)

    git_sha = ""
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(constants.REPO_ROOT),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        pass

    params = constants.as_dict()
    params["dataset_size_train"] = len(ds_train)
    params["dataset_size_val"] = len(ds_val)
    params["dataset_size_test"] = len(ds_test)
    params["git_sha"] = git_sha
    params["device"] = str(device)
    params["smoke"] = smoke
    params["epochs_actual"] = epochs
    params["resume_from"] = str(resume_from) if resume_from is not None else ""
    # Effective (nc/nl/imgsz-autoscaled) loss gains actually used by ComputeLoss —
    # distinct from the raw HYP_BOX/CLS/OBJ constants above (see yolov5s_model._hyp_dict).
    params["hyp_box_effective"] = model.hyp["box"]
    params["hyp_cls_effective"] = model.hyp["cls"]
    params["hyp_obj_effective"] = model.hyp["obj"]
    params["autoanchor_bpr"] = autoanchor_result["bpr"]
    params["autoanchor_anchors_changed"] = autoanchor_result["anchors_changed"]
    # mlflow.log_params accepts str values only
    mlflow.log_params({k: str(v) for k, v in params.items()})

    logger.info("run config:")
    for k in sorted(params):
        logger.info("  %s = %s", k, params[k])

    pipeline = TrainingPipeline(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        dl_train=dl_train,
        dl_val=dl_val,
        dl_test=dl_test,
        device=device,
        epochs=epochs,
        output_dir=run_dir,
        log_every_n_steps=constants.MLFLOW_LOG_EVERY_N_STEPS,
        eval_conf_thres=constants.EVAL_CONF_THRES,
        eval_iou_thres=constants.EVAL_IOU_THRES,
        eval_max_det=constants.EVAL_MAX_DET,
        warmup_epochs=constants.WARMUP_EPOCHS,
        selection_metric=constants.SELECTION_METRIC,
        early_stop=constants.EARLY_STOP,
        early_stop_patience=constants.EARLY_STOP_PATIENCE,
        early_stop_min_delta=constants.EARLY_STOP_MIN_DELTA,
        use_ema=constants.USE_EMA,
        use_amp=constants.USE_AMP,
        resume_from=resume_from,
    )

    try:
        result = pipeline.run_pipeline()
    except Exception as e:
        mlflow.log_param("error", str(e))
        logger.exception("run failed: %s", e)
        raise

    if full_eval:
        try:
            _run_full_evaluation(run_dir, smoke, device)
        except Exception as e:  # best-effort: never fail the run on eval issues
            logger.exception("post-training full evaluation failed: %s", e)

    if log_file is not None and log_file.exists():
        mlflow.log_artifact(str(log_file))
        logger.info("uploaded log artifact: %s", log_file)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Quick wiring test: 1 epoch on val set")
    parser.add_argument(
        "--full-eval",
        action="store_true",
        help="After training, run the comprehensive evaluation suite on best.pt "
        "(granularity × band × domain report) and log it to MLflow.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Path to a best.pt/last.pt checkpoint from a previous run. Restores "
        "model/optimizer/scheduler/EMA/AMP state and continues at the next epoch, "
        "under a NEW run dir and a NEW MLflow run.",
    )
    args = parser.parse_args()
    smoke = args.smoke
    if args.resume_from is not None and not args.resume_from.is_file():
        parser.error(f"--resume-from checkpoint not found: {args.resume_from}")

    load_dotenv(Path(__file__).parent / ".env")

    run_name = f"yolov5s-{'smoke-' if smoke else ''}{datetime.now():%Y%m%d-%H%M%S}"
    # Each run gets its own timestamped sub-directory; checkpoints, the log, and
    # any eval reports for this run land inside it.
    run_dir = constants.OUTPUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / f"{run_name}.log"
    setup_logging(log_file)

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or ""
    mlflow.set_tracking_uri(tracking_uri)
    experiment = os.environ.get("MLFLOW_EXPERIMENT_NAME", constants.MLFLOW_EXPERIMENT_DEFAULT)
    mlflow.set_experiment(experiment)

    tags = {
        "model": "yolov5s",
        "dataset": "wildlife225",
        "seed": str(constants.SEED),
        "smoke": str(smoke),
    }

    logger.info("=== yolov5s training run ===")
    logger.info("run_name=%s experiment=%s", run_name, experiment)
    if args.resume_from is not None:
        logger.info("resuming from checkpoint: %s", args.resume_from)
    logger.info("mlflow tracking_uri=%s", tracking_uri or "(unset)")
    logger.info("run dir: %s", run_dir)
    logger.info("log file: %s", log_file)

    try:
        with mlflow.start_run(run_name=run_name, tags=tags):
            training_run(
                smoke, run_dir, log_file, full_eval=args.full_eval, resume_from=args.resume_from
            )
    finally:
        logging.shutdown()
