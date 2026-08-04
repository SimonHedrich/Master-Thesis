"""Entry point: wires MLflow, builds all components, runs TrainingPipeline."""
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

import scripts.training.yolo26n.constants as constants
from scripts.training.yolo26n.evaluation import eval_log_mlflow, evaluate
from scripts.training.yolo26n.kd_dataset import KDCocoYoloDataset, kd_collate_fn
from scripts.training.yolo26n.kd_loss import KDYolo26Loss
from scripts.training.yolo26n.loss import Yolo26Loss
from scripts.training.yolo26n.yolo26n_model import model_optimizer, model_scheduler, yolo26n_model
from scripts.training.yolov5s.dataset import CocoYoloDataset, Dataloader, collate_fn, make_worker_init_fn
from scripts.training.yolov5s.logging_setup import setup_logging
from scripts.training.yolov5s.training_pipeline import TrainingPipeline

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
    from scripts.training.yolo26n.eval_suite.run_evaluation import evaluate_checkpoint

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


def _resolve_teacher_cache(smoke: bool, teacher_cache: Path | None) -> Path:
    if teacher_cache is not None:
        return teacher_cache
    split = "val" if smoke else "train"
    return constants.DATA_ROOT / "real" / f"teacher_soft_labels_{split}.jsonl"


def training_run(
    smoke: bool,
    run_dir: Path,
    log_file: Path | None,
    full_eval: bool = False,
    kd: bool = False,
    teacher_cache: Path | None = None,
    init_from: str = "coco",
    resume_from: Path | None = None,
) -> dict:
    set_seed(constants.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = 1 if smoke else constants.EPOCH_COUNT
    logger.info(
        "seed=%d device=%s epochs=%d smoke=%s kd=%s init_from=%s",
        constants.SEED, device, epochs, smoke, kd, init_from if kd else "coco",
    )

    # Weight init: KD must default to COCO-pretrained (isolates the KD signal
    # from Phase 1's fine-tuned weights, per the plan doc §3.1) — "phase1" is
    # an explicit, logged opt-in, never the silent default.
    weights_path = constants.PRETRAINED_WEIGHTS
    if kd and init_from == "phase1":
        phase1_dir = constants.latest_run_dir()
        if phase1_dir is None:
            raise SystemExit(
                "--init-from phase1 requires an existing Goal A run under "
                f"{constants.OUTPUT_DIR}, but none was found"
            )
        weights_path = phase1_dir / "best.pt"
        logger.info("--init-from phase1: loading weights from %s", weights_path)

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

    base_train = CocoYoloDataset(
        constants.ANNOTATIONS_VAL if smoke else constants.ANNOTATIONS_TRAIN,
        constants.IMAGE_ROOT,
        constants.IMAGE_SIZE,
        augment=True,
    )
    if kd:
        cache_path = _resolve_teacher_cache(smoke, teacher_cache)
        if not cache_path.exists():
            raise SystemExit(
                f"--kd requires a teacher soft-label cache at {cache_path} — run "
                "scripts/training/teacher_finetune/cache_soft_labels.py first "
                f"(--split {'val' if smoke else 'train'})"
            )
        ds_train = KDCocoYoloDataset(base_train, cache_path, constants.NUM_CLASSES)
        train_collate_fn = kd_collate_fn
    else:
        ds_train = base_train
        train_collate_fn = collate_fn

    dl_train = Dataloader(
        ds_train,
        constants.BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=train_collate_fn,
        worker_init_fn=train_worker_init,
        generator=train_generator,
    ).get_dataloader()
    dl_val = Dataloader(ds_val, constants.BATCH_SIZE, shuffle=False, num_workers=num_workers, collate_fn=collate_fn).get_dataloader()
    dl_test = Dataloader(ds_test, constants.BATCH_SIZE, shuffle=False, num_workers=num_workers, collate_fn=collate_fn).get_dataloader()

    model, _preprocess = yolo26n_model(constants.NUM_CLASSES, weights_path, device)
    model.names = ds_train.class_names

    optimizer = model_optimizer(model)
    scheduler = model_scheduler(optimizer, steps_per_epoch=len(dl_train), epochs=epochs)
    if kd:
        loss_fn = KDYolo26Loss(
            model,
            kd_alpha=constants.KD_ALPHA,
            kd_temperature=constants.KD_TEMPERATURE,
            apply_to=constants.KD_APPLY_TO,
        )
    else:
        loss_fn = Yolo26Loss(model)

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
    params["kd"] = kd
    params["init_from"] = init_from if kd else "coco"
    params["weights_path"] = str(weights_path)
    params["resume_from"] = str(resume_from) if resume_from is not None else ""
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
        evaluate_fn=evaluate,
        eval_log_mlflow_fn=eval_log_mlflow,
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
        "--kd",
        action="store_true",
        help="Enable KD training mode: distill the frozen teacher_finetune soft-label "
        "cache into the student instead of pure hard-label supervision.",
    )
    parser.add_argument(
        "--teacher-cache",
        type=Path,
        default=None,
        help="Path to teacher_soft_labels_{split}.jsonl. Only used with --kd. Defaults to "
        "data/real/teacher_soft_labels_val.jsonl on --smoke (matches the val-split smoke "
        "training set) or teacher_soft_labels_train.jsonl otherwise.",
    )
    parser.add_argument(
        "--init-from",
        choices=["coco", "phase1"],
        default="coco",
        help="Only meaningful with --kd. 'coco' (default): initialize from the "
        "COCO-pretrained checkpoint, isolating the KD signal from Phase 1's fine-tuned "
        "weights. 'phase1': initialize from the latest Goal A run's best.pt instead "
        "(explicit opt-in, not the default).",
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

    run_name = (
        f"yolo26n-{'kd-' if args.kd else ''}{'smoke-' if smoke else ''}"
        f"{datetime.now():%Y%m%d-%H%M%S}"
    )
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
        "model": "yolo26n",
        "dataset": "wildlife225",
        "seed": str(constants.SEED),
        "smoke": str(smoke),
        "mode": "kd" if args.kd else "direct-ft",
    }

    logger.info("=== yolo26n training run ===")
    logger.info("run_name=%s experiment=%s", run_name, experiment)
    if args.resume_from is not None:
        logger.info("resuming from checkpoint: %s", args.resume_from)
    logger.info("mlflow tracking_uri=%s", tracking_uri or "(unset)")
    logger.info("run dir: %s", run_dir)
    logger.info("log file: %s", log_file)

    try:
        with mlflow.start_run(run_name=run_name, tags=tags):
            training_run(
                smoke,
                run_dir,
                log_file,
                full_eval=args.full_eval,
                kd=args.kd,
                teacher_cache=args.teacher_cache,
                init_from=args.init_from,
                resume_from=args.resume_from,
            )
    finally:
        logging.shutdown()
