"""Entry point: wires MLflow, builds all components, runs TrainingPipeline.

Mirrors `scripts/training/yolov5s/run_training_pipeline.py`'s structure
(argparse `--smoke`, `.env` loading, MLflow experiment/run wiring, seed
discipline, `constants.as_dict()` param logging). Does **not** import
`scripts.training.yolov5s.dataset`/`.transforms` for the `DataLoader`
wrapper — those pull in `yolov5.utils.augmentations`, which requires the
`yolov5` PyPI package this package's Docker image (`Dockerfile.speciesnet`)
deliberately does not install (keeps the classifier and detector environments
cleanly separated). `logging_setup.py` has no such dependency and is imported
directly.
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
from torch.utils.data import DataLoader

import scripts.training.teacher_finetune.constants as constants
from scripts.training.teacher_finetune.dataset import SpeciesNetCropDataset, collate_fn
from scripts.training.teacher_finetune.loss import GroupedCrossEntropyLoss
from scripts.training.teacher_finetune.taxonomy import build_group_table, projection_tables
from scripts.training.teacher_finetune.teacher_model import (
    model_optimizer,
    model_scheduler,
    speciesnet_model,
)
from scripts.training.teacher_finetune.training_pipeline import TrainingPipeline
from scripts.training.yolov5s.logging_setup import setup_logging

logger = logging.getLogger(__name__)


def set_seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_worker_init_fn(seed: int):
    def worker_init_fn(worker_id: int) -> None:
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return worker_init_fn


def training_run(
    smoke: bool,
    run_dir: Path,
    log_file: Path | None,
    resume_from: Path | None = None,
) -> dict:
    set_seed(constants.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = 1 if smoke else constants.EPOCH_COUNT
    logger.info("seed=%d device=%s epochs=%d smoke=%s", constants.SEED, device, epochs, smoke)

    model, preprocess_fn, _labels = speciesnet_model(device)

    ds_train = SpeciesNetCropDataset(
        constants.ANNOTATIONS_VAL if smoke else constants.ANNOTATIONS_TRAIN,
        constants.IMAGE_ROOT,
        preprocess_fn,
    )
    ds_val = SpeciesNetCropDataset(constants.ANNOTATIONS_VAL, constants.IMAGE_ROOT, preprocess_fn)
    ds_test = SpeciesNetCropDataset(
        constants.ANNOTATIONS_VAL if smoke else constants.ANNOTATIONS_TEST,
        constants.IMAGE_ROOT,
        preprocess_fn,
    )

    num_workers = 0 if smoke else constants.NUM_WORKERS
    train_generator = torch.Generator().manual_seed(constants.SEED)
    train_worker_init = make_worker_init_fn(constants.SEED)

    dl_train = DataLoader(
        ds_train,
        batch_size=constants.BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=True,
        worker_init_fn=train_worker_init,
        generator=train_generator,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=constants.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    dl_test = DataLoader(
        ds_test,
        batch_size=constants.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    optimizer = model_optimizer(model)
    scheduler = model_scheduler(optimizer, steps_per_epoch=len(dl_train), epochs=epochs)

    group_table, _levels = build_group_table()
    loss_fn = GroupedCrossEntropyLoss(group_table)
    idx_to_label, genus_species_to_225, genus_to_225, family_to_225 = projection_tables()

    git_sha = ""
    try:
        git_sha = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(constants.REPO_ROOT),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
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
        warmup_epochs=constants.WARMUP_EPOCHS,
        selection_metric=constants.SELECTION_METRIC,
        early_stop=constants.EARLY_STOP,
        early_stop_patience=constants.EARLY_STOP_PATIENCE,
        early_stop_min_delta=constants.EARLY_STOP_MIN_DELTA,
        use_ema=constants.USE_EMA,
        use_amp=constants.USE_AMP,
        idx_to_label=idx_to_label,
        genus_species_to_225=genus_species_to_225,
        genus_to_225=genus_to_225,
        family_to_225=family_to_225,
        resume_from=resume_from,
    )

    try:
        result = pipeline.run_pipeline()
    except Exception as e:
        mlflow.log_param("error", str(e))
        logger.exception("run failed: %s", e)
        raise

    if log_file is not None and log_file.exists():
        mlflow.log_artifact(str(log_file))
        logger.info("uploaded log artifact: %s", log_file)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Quick wiring test: 1 epoch on val set")
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

    run_name = f"teacher-finetune-{'smoke-' if smoke else ''}{datetime.now():%Y%m%d-%H%M%S}"
    run_dir = constants.OUTPUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / f"{run_name}.log"
    setup_logging(log_file)

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or ""
    mlflow.set_tracking_uri(tracking_uri)
    experiment = os.environ.get("MLFLOW_EXPERIMENT_NAME", constants.MLFLOW_EXPERIMENT_DEFAULT)
    mlflow.set_experiment(experiment)

    tags = {
        "model": "speciesnet-classifier",
        "dataset": "wildlife225",
        "seed": str(constants.SEED),
        "smoke": str(smoke),
    }

    logger.info("=== teacher_finetune (SpeciesNet classifier) training run ===")
    logger.info("run_name=%s experiment=%s", run_name, experiment)
    if args.resume_from is not None:
        logger.info("resuming from checkpoint: %s", args.resume_from)
    logger.info("mlflow tracking_uri=%s", tracking_uri or "(unset)")
    logger.info("run dir: %s", run_dir)
    logger.info("log file: %s", log_file)

    try:
        with mlflow.start_run(run_name=run_name, tags=tags):
            training_run(smoke, run_dir, log_file, resume_from=args.resume_from)
    finally:
        logging.shutdown()
