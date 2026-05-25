"""
MLflow logging adapter for YOLOv5's training callback system.

YOLOv5@5cdad89 exposes a Callbacks class in utils/callbacks.py. This module
provides a callback object that logs metrics to MLflow and a helper to wire it
into a YOLOv5 model at the start of training.

Usage (inside train.py or a wrapper script):
    from mlflow_yolov5_callback import register_mlflow_callbacks
    register_mlflow_callbacks(
        model,
        run_name="yolov5s_wildlife225_seed42",
        tags={"model": "yolov5s", "seed": "42", "dataset": "wildlife225"},
    )

If the YOLOv5 version at commit 5cdad89 does not expose model.add_callback,
patch train.py to instantiate MLflowYOLOv5Callback directly and call its
methods from the relevant locations in the training loop.
"""

from __future__ import annotations

import os
from pathlib import Path

import mlflow


class MLflowYOLOv5Callback:
    """Adapts YOLOv5 fit/train callbacks to MLflow metric logging."""

    # Keys emitted by YOLOv5's on_fit_epoch_end in order
    EPOCH_KEYS = (
        "train/box_loss",
        "train/obj_loss",
        "train/cls_loss",
        "metrics/precision",
        "metrics/recall",
        "metrics/mAP_0.5",
        "metrics/mAP_0.5:0.95",
        "val/box_loss",
        "val/obj_loss",
        "val/cls_loss",
    )

    def __init__(
        self,
        run_name: str,
        tags: dict | None = None,
        tracking_uri: str | None = None,
        experiment_name: str = "yolov5-wildlife",
    ) -> None:
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name=run_name, tags=tags or {})
        self._run_id = mlflow.active_run().info.run_id

    def on_fit_epoch_end(self, vals: list, epoch: int, best_fitness: float, fi: float) -> None:
        metrics = {}
        for key, val in zip(self.EPOCH_KEYS, vals):
            try:
                metrics[key] = float(val)
            except (TypeError, ValueError):
                pass
        if metrics:
            mlflow.log_metrics(metrics, step=epoch)

    def on_train_end(self, last: Path, best: Path, epoch: int, results: list) -> None:
        if best and Path(best).exists():
            mlflow.log_artifact(str(best), artifact_path="weights")
        if last and Path(last).exists():
            mlflow.log_artifact(str(last), artifact_path="weights")

        final_keys = ("precision", "recall", "mAP_0.5", "mAP_0.5:0.95")
        if results and len(results) >= 4:
            for key, val in zip(final_keys, results[-4:]):
                try:
                    mlflow.log_metric(f"final/{key}", float(val))
                except (TypeError, ValueError):
                    pass

        mlflow.end_run()


def register_mlflow_callbacks(model, run_name: str, tags: dict | None = None) -> MLflowYOLOv5Callback:
    """
    Register MLflow callbacks on a YOLOv5 model that exposes add_callback().
    Returns the callback instance for manual use if needed.
    """
    cb = MLflowYOLOv5Callback(run_name=run_name, tags=tags)
    model.add_callback("on_fit_epoch_end", cb.on_fit_epoch_end)
    model.add_callback("on_train_end", cb.on_train_end)
    return cb
