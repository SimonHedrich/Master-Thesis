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

import scripts.training.yolov5s.constants as constants
from scripts.training.yolov5s.dataset import CocoYoloDataset, Dataloader, collate_fn
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


def training_run(smoke: bool, log_file: Path | None) -> dict:
    set_seed(constants.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = 1 if smoke else constants.EPOCH_COUNT
    logger.info("seed=%d device=%s epochs=%d smoke=%s", constants.SEED, device, epochs, smoke)

    ds_train = CocoYoloDataset(
        constants.ANNOTATIONS_VAL if smoke else constants.ANNOTATIONS_TRAIN,
        constants.IMAGE_ROOT,
        constants.IMAGE_SIZE,
    )
    ds_val = CocoYoloDataset(constants.ANNOTATIONS_VAL, constants.IMAGE_ROOT, constants.IMAGE_SIZE)
    ds_test = CocoYoloDataset(
        constants.ANNOTATIONS_VAL if smoke else constants.ANNOTATIONS_TEST,
        constants.IMAGE_ROOT,
        constants.IMAGE_SIZE,
    )

    num_workers = 0 if smoke else constants.NUM_WORKERS

    dl_train = Dataloader(ds_train, constants.BATCH_SIZE, shuffle=True, num_workers=num_workers, collate_fn=collate_fn).get_dataloader()
    dl_val = Dataloader(ds_val, constants.BATCH_SIZE, shuffle=False, num_workers=num_workers, collate_fn=collate_fn).get_dataloader()
    dl_test = Dataloader(ds_test, constants.BATCH_SIZE, shuffle=False, num_workers=num_workers, collate_fn=collate_fn).get_dataloader()

    model, _preprocess = yolov5s_model(constants.NUM_CLASSES, constants.PRETRAINED_WEIGHTS, device)
    model.names = ds_train.class_names

    optimizer = model_optimizer(model)
    scheduler = model_scheduler(optimizer, epochs)
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
        output_dir=constants.OUTPUT_DIR,
        log_every_n_steps=constants.MLFLOW_LOG_EVERY_N_STEPS,
        eval_conf_thres=constants.EVAL_CONF_THRES,
        eval_iou_thres=constants.EVAL_IOU_THRES,
        eval_max_det=constants.EVAL_MAX_DET,
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
    args = parser.parse_args()
    smoke = args.smoke

    load_dotenv(Path(__file__).parent / ".env")

    run_name = f"yolov5s-{'smoke-' if smoke else ''}{datetime.now():%Y%m%d-%H%M%S}"
    log_file = constants.OUTPUT_DIR / f"{run_name}.log"
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
    logger.info("mlflow tracking_uri=%s", tracking_uri or "(unset)")
    logger.info("log file: %s", log_file)

    try:
        with mlflow.start_run(run_name=run_name, tags=tags):
            training_run(smoke, log_file)
    finally:
        logging.shutdown()
