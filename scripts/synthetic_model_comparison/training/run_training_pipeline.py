"""Entry point: wires MLflow, builds all components, runs TrainingPipeline.

Adapted from scripts/training/yolo26n/run_training_pipeline.py for the
synthetic-generator comparison experiment
(docs/synthetic-model-comparison/11_detector-architecture-selection.md):

- Required --generator/--prompt-regime select which cell to train on
  (data/synthetic_model_comparison/train/<generator>/<prompt-regime>/), since
  this experiment trains one model per cell rather than one fixed dataset.
- The cell's exported annotations.json (scripts/synthetic_model_comparison/5-export_coco.py)
  is auto-split into an internal train/val pair (split_dataset.py) if the
  split files are missing or stale — dl_train/dl_val come from that split and
  drive training/early-stopping every epoch.
- dl_test is always the experiment's FIXED REAL test set
  (constants.ANNOTATIONS_TEST), evaluated exactly once at the end via
  TrainingPipeline's existing final-eval codepath — never used for
  per-epoch model selection.
- --seed overrides constants.SEED so the >=3-seeds-per-cell recommendation
  (docs/synthetic-model-comparison/06_evaluation-methodology.md) doesn't
  require editing constants between runs; the train/val split itself stays
  fixed across seeds (constants.SPLIT_SEED, independent of --seed).
- No --kd/--teacher-cache/--init-from: this experiment compares generators
  via direct fine-tuning, not distillation.

Usage:
    # 1-epoch wiring check on the cell's val split:
    PYTHONPATH=. uv run -m scripts.synthetic_model_comparison.training.run_training_pipeline \
        --generator gemini-3.1-flash-image-preview --prompt-regime full --smoke

    # full run (repeat with --seed 43, 44, ... for the >=3-seeds recommendation):
    PYTHONPATH=. uv run -m scripts.synthetic_model_comparison.training.run_training_pipeline \
        --generator gemini-3.1-flash-image-preview --prompt-regime full --seed 42

    # full run + the eval_suite report (headline mAP / per-class AP / confusion) on best.pt:
    PYTHONPATH=. uv run -m scripts.synthetic_model_comparison.training.run_training_pipeline \
        --generator gemini-3.1-flash-image-preview --prompt-regime full --seed 42 --full-eval

    # resume a crashed run (new run dir + new MLflow run):
    PYTHONPATH=. uv run -m scripts.synthetic_model_comparison.training.run_training_pipeline \
        --generator gemini-3.1-flash-image-preview --prompt-regime full \
        --resume-from scripts/synthetic_model_comparison/training/model_exports/<run_name>/last.pt
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

import scripts.synthetic_model_comparison.training.constants as constants
from scripts.synthetic_model_comparison.training.dataset import CocoYoloDataset, Dataloader, collate_fn, make_worker_init_fn
from scripts.synthetic_model_comparison.training.evaluation import eval_log_mlflow, evaluate
from scripts.synthetic_model_comparison.training.loss import Yolo26Loss
from scripts.synthetic_model_comparison.training.logging_setup import setup_logging
from scripts.synthetic_model_comparison.training.optim import model_optimizer, model_scheduler
from scripts.synthetic_model_comparison.training.split_dataset import split_cell
from scripts.synthetic_model_comparison.training.training_pipeline import TrainingPipeline
from scripts.synthetic_model_comparison.training.yolo26n_model import yolo26n_model

logger = logging.getLogger(__name__)


def set_seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def slugify(name: str) -> str:
    return name.replace(".", "-").replace("_", "-")


def _run_full_evaluation(run_dir: Path, generator: str, prompt_regime: str, smoke: bool, device: torch.device) -> None:
    """Optional post-training hook: run the eval_suite report on best.pt.

    Independent of the lightweight per-epoch eval — produces the headline
    mAP / per-class AP / within-group confusion report and logs it to the
    active MLflow run. Kept best-effort: a failure here must not fail the run.
    """
    from scripts.synthetic_model_comparison.training.eval_suite.run_evaluation import evaluate_checkpoint

    best_path = run_dir / "best.pt"
    if not best_path.exists():
        logger.warning("full-eval requested but %s not found — skipping", best_path)
        return
    limit = 200 if smoke else None
    logger.info("=== post-training full evaluation (limit=%s) ===", limit)
    evaluate_checkpoint(
        checkpoint=best_path,
        real_ann=constants.ANNOTATIONS_TEST,
        output_dir=run_dir / "evaluation",
        device=device,
        max_det=constants.EVAL_MAX_DET,
        batch_size=constants.BATCH_SIZE,
        num_workers=constants.NUM_WORKERS,
        log_mlflow=True,
        limit=limit,
    )


def training_run(
    generator: str,
    prompt_regime: str,
    seed: int,
    smoke: bool,
    run_dir: Path,
    log_file: Path | None,
    full_eval: bool = False,
    resume_from: Path | None = None,
) -> dict:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = 1 if smoke else constants.EPOCH_COUNT
    logger.info(
        "generator=%s prompt_regime=%s seed=%d device=%s epochs=%d smoke=%s",
        generator, prompt_regime, seed, device, epochs, smoke,
    )

    cell = constants.cell_dir(generator, prompt_regime)
    train_split, val_split = split_cell(cell)

    weights_path = constants.PRETRAINED_WEIGHTS

    # dl_test is ALWAYS the fixed real test set — never the cell's own val
    # split — per the "reserve the real test set for the final Axis C report
    # only" design (11_detector-architecture-selection.md §7). On --smoke,
    # both val and test point at the (tiny) internal val split so the wiring
    # check stays cheap.
    ds_val = CocoYoloDataset(val_split, constants.IMAGE_ROOT, constants.IMAGE_SIZE, augment=False)
    ds_test = CocoYoloDataset(
        val_split if smoke else constants.ANNOTATIONS_TEST,
        constants.IMAGE_ROOT,
        constants.IMAGE_SIZE,
        augment=False,
    )

    num_workers = 0 if smoke else constants.NUM_WORKERS

    train_generator = torch.Generator().manual_seed(seed)
    train_worker_init = make_worker_init_fn(seed)

    ds_train = CocoYoloDataset(
        val_split if smoke else train_split,
        constants.IMAGE_ROOT,
        constants.IMAGE_SIZE,
        augment=True,
    )

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

    model, _preprocess = yolo26n_model(constants.NUM_CLASSES, weights_path, device)
    model.names = ds_train.class_names

    optimizer = model_optimizer(model)
    scheduler = model_scheduler(optimizer, steps_per_epoch=len(dl_train), epochs=epochs)
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
    params["generator"] = generator
    params["prompt_regime"] = prompt_regime
    params["run_seed"] = seed
    params["dataset_size_train"] = len(ds_train)
    params["dataset_size_val"] = len(ds_val)
    params["dataset_size_test"] = len(ds_test)
    params["git_sha"] = git_sha
    params["device"] = str(device)
    params["smoke"] = smoke
    params["epochs_actual"] = epochs
    params["weights_path"] = str(weights_path)
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
            _run_full_evaluation(run_dir, generator, prompt_regime, smoke, device)
        except Exception as e:  # best-effort: never fail the run on eval issues
            logger.exception("post-training full evaluation failed: %s", e)

    if log_file is not None and log_file.exists():
        mlflow.log_artifact(str(log_file))
        logger.info("uploaded log artifact: %s", log_file)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", required=True, metavar="NAME",
                        help="e.g. gemini-3.1-flash-image-preview")
    parser.add_argument("--prompt-regime", required=True, choices=["full", "compressed"])
    parser.add_argument("--seed", type=int, default=None,
                         help="Training-run seed override (default: constants.SEED). "
                              "The internal train/val split stays fixed across seeds "
                              "(constants.SPLIT_SEED) — only model init/dataloader "
                              "shuffling varies.")
    parser.add_argument("--smoke", action="store_true", help="Quick wiring test: 1 epoch on the val split")
    parser.add_argument(
        "--full-eval",
        action="store_true",
        help="After training, run the eval_suite report on best.pt "
        "(headline mAP / per-class AP / within-group confusion) and log it to MLflow.",
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
    seed = args.seed if args.seed is not None else constants.SEED
    if args.resume_from is not None and not args.resume_from.is_file():
        parser.error(f"--resume-from checkpoint not found: {args.resume_from}")

    load_dotenv(Path(__file__).parent / ".env")

    gen_slug = slugify(args.generator)
    run_name = (
        f"yolo26n-{gen_slug}-{args.prompt_regime}-seed{seed}-"
        f"{'smoke-' if smoke else ''}{datetime.now():%Y%m%d-%H%M%S}"
    )
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
        "dataset": "synthetic-model-comparison",
        "generator": args.generator,
        "prompt_regime": args.prompt_regime,
        "seed": str(seed),
        "smoke": str(smoke),
        "mode": "direct-ft",
    }

    logger.info("=== yolo26n synthetic-model-comparison training run ===")
    logger.info("run_name=%s experiment=%s", run_name, experiment)
    if args.resume_from is not None:
        logger.info("resuming from checkpoint: %s", args.resume_from)
    logger.info("mlflow tracking_uri=%s", tracking_uri or "(unset)")
    logger.info("run dir: %s", run_dir)
    logger.info("log file: %s", log_file)

    try:
        with mlflow.start_run(run_name=run_name, tags=tags):
            training_run(
                args.generator,
                args.prompt_regime,
                seed,
                smoke,
                run_dir,
                log_file,
                full_eval=args.full_eval,
                resume_from=args.resume_from,
            )
    finally:
        logging.shutdown()
