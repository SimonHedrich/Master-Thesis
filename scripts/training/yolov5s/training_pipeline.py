"""Training loop for YOLOv5s fine-tuning.

LR/auto-stop design (see
docs/plans/2026-06-10_training-hyperparameters-autostop-lr-schedule.md):
linear warmup → ReduceLROnPlateau, with early stopping. A single validation
metric (``selection_metric``) drives best-checkpoint selection, the plateau LR
drops, and the early-stop decision. EMA (smoothed weights) and AMP (mixed
precision) are optional bolt-ons that do not change the control flow.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import mlflow
import torch
from tqdm import tqdm
from yolov5.utils.torch_utils import ModelEMA

from scripts.training.yolov5s.evaluation import eval_log_mlflow, evaluate

logger = logging.getLogger(__name__)


class TrainingPipeline:
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        dl_train,
        dl_val,
        dl_test,
        device: torch.device,
        epochs: int,
        output_dir: Path,
        log_every_n_steps: int,
        eval_conf_thres: float,
        eval_iou_thres: float,
        eval_max_det: int,
        warmup_epochs: int,
        learning_rate: float,
        selection_metric: str,
        early_stop: bool,
        early_stop_patience: int,
        early_stop_min_delta: float,
        use_ema: bool,
        use_amp: bool,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dl_train = dl_train
        self.dl_val = dl_val
        self.dl_test = dl_test
        self.device = device
        self.epochs = epochs
        self.output_dir = output_dir
        self.log_every_n_steps = log_every_n_steps
        self.eval_conf_thres = eval_conf_thres
        self.eval_iou_thres = eval_iou_thres
        self.eval_max_det = eval_max_det

        # Scheduler / auto-stop config.
        self.warmup_epochs = warmup_epochs
        self.learning_rate = learning_rate
        self.selection_metric = selection_metric
        self.early_stop = early_stop
        self.early_stop_patience = early_stop_patience
        self.early_stop_min_delta = early_stop_min_delta

        # Training-quality add-ons. AMP is CUDA-only; it degrades to a no-op on CPU
        # (e.g. smoke runs on a machine without a GPU) so the code path stays single.
        self.use_amp = use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.use_amp)
        self.ema = ModelEMA(self.model) if use_ema else None

        output_dir.mkdir(parents=True, exist_ok=True)
        self.best_metric = -1.0
        self.epochs_since_improve = 0
        self.global_step = 0
        self.current_epoch = 0
        logger.info(
            "output dir: %s | selection_metric=%s early_stop=%s(patience=%d, min_delta=%g) "
            "ema=%s amp=%s",
            output_dir,
            self.selection_metric,
            self.early_stop,
            self.early_stop_patience,
            self.early_stop_min_delta,
            self.ema is not None,
            self.use_amp,
        )

    def _eval_model(self) -> torch.nn.Module:
        """The weights to evaluate / checkpoint: EMA copy if enabled, else raw."""
        return self.ema.ema if self.ema is not None else self.model

    def _train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        sums: dict[str, float] = {"loss_box": 0.0, "loss_obj": 0.0, "loss_cls": 0.0, "loss_total": 0.0}
        count = 0

        lr_start = self.optimizer.param_groups[0]["lr"]
        logger.info("epoch %d/%d — train start (lr=%g, batches=%d)", epoch + 1, self.epochs, lr_start, len(self.dl_train))
        t0 = time.time()

        pbar = tqdm(
            self.dl_train,
            desc=f"epoch {epoch + 1}/{self.epochs} train",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        )
        for imgs, targets, _, _ in pbar:
            imgs = imgs.to(self.device)
            targets = targets.to(self.device)

            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                preds = self.model(imgs)
                total, parts = self.loss_fn(preds, targets)

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(total).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.ema is not None:
                self.ema.update(self.model)

            for k, v in parts.items():
                sums[k] += v
            count += 1

            pbar.set_postfix(
                box=f"{parts['loss_box']:.3f}",
                obj=f"{parts['loss_obj']:.3f}",
                cls=f"{parts['loss_cls']:.3f}",
                total=f"{parts['loss_total']:.3f}",
            )

            if self.global_step % self.log_every_n_steps == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                for k, v in parts.items():
                    mlflow.log_metric(f"train/step/{k}", v, step=self.global_step)
                mlflow.log_metric("train/lr", lr, step=self.global_step)
                if self.global_step > 0:
                    logger.info(
                        "step %d | box=%.4f obj=%.4f cls=%.4f total=%.4f lr=%g",
                        self.global_step,
                        parts["loss_box"],
                        parts["loss_obj"],
                        parts["loss_cls"],
                        parts["loss_total"],
                        lr,
                    )

            self.global_step += 1

        avgs = {k: v / max(count, 1) for k, v in sums.items()}
        for k, v in avgs.items():
            mlflow.log_metric(f"train/epoch_{k}", v, step=epoch)

        logger.info(
            "epoch %d/%d — train done in %.1fs | avg box=%.4f obj=%.4f cls=%.4f total=%.4f",
            epoch + 1,
            self.epochs,
            time.time() - t0,
            avgs["loss_box"],
            avgs["loss_obj"],
            avgs["loss_cls"],
            avgs["loss_total"],
        )
        return avgs

    def _save_checkpoint(self, name: str) -> None:
        """Save the deployable (EMA if enabled) weights plus resume/provenance state.

        ``model`` always holds the weights downstream loaders should deploy — the
        EMA copy when EMA is on — so ``best.pt`` reflects the published-quality
        model, not the last noisy step.
        """
        path = self.output_dir / name
        ckpt: dict[str, object] = {
            "model": self._eval_model().state_dict(),
            "epoch": self.current_epoch,
            "best_metric": self.best_metric,
            "selection_metric": self.selection_metric,
            "epochs_since_improve": self.epochs_since_improve,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        if self.ema is not None:
            ckpt["ema_updates"] = self.ema.updates
        if self.use_amp:
            ckpt["scaler"] = self.scaler.state_dict()
        torch.save(ckpt, path)
        logger.info("saved checkpoint: %s", path)

    def run_pipeline(self) -> dict:
        logger.info(
            "starting training: ceiling=%d epochs | train_batches=%d val_batches=%d test_batches=%d | device=%s",
            self.epochs,
            len(self.dl_train),
            len(self.dl_val),
            len(self.dl_test),
            self.device,
        )

        for epoch in range(self.epochs):
            self.current_epoch = epoch

            # Linear warmup: set lr directly for the first WARMUP_EPOCHS epochs.
            # After that the plateau scheduler owns the lr (do NOT touch it here).
            if epoch < self.warmup_epochs:
                warmup_lr = self.learning_rate * (epoch + 1) / self.warmup_epochs
                for g in self.optimizer.param_groups:
                    g["lr"] = warmup_lr

            # Notify the training dataset of the current epoch so the
            # close-mosaic tail (AUG_CLOSE_MOSAIC) can suppress compositing
            # for the final N epochs.  Guarded with getattr so this is a no-op
            # when the dataset does not expose set_epoch (e.g. in unit tests).
            _ds = getattr(self.dl_train, "dataset", None)
            if _ds is not None and hasattr(_ds, "set_epoch"):
                _ds.set_epoch(epoch, self.epochs)

            self._train_one_epoch(epoch)

            # Evaluate the EMA weights (smoother, published-quality) when EMA is on.
            val_result = evaluate(
                self._eval_model(),
                self.dl_val,
                self.device,
                self.eval_conf_thres,
                self.eval_iou_thres,
                self.eval_max_det,
            )
            metric = val_result[self.selection_metric]
            logger.info(
                "epoch %d/%d — val mAP50=%.4f mAP50_95=%.4f (selection %s=%.4f)",
                epoch + 1,
                self.epochs,
                val_result["mAP50"],
                val_result["mAP50_95"],
                self.selection_metric,
                metric,
            )
            eval_log_mlflow(val_result, "val", step=epoch)

            # Checkpoint + early-stop bookkeeping, both keyed off the single
            # selection metric. Warmup epochs are excluded from patience counting
            # (the metric is too noisy to trust early), but a warmup improvement
            # still updates best.pt.
            if metric > self.best_metric + self.early_stop_min_delta:
                self.best_metric = metric
                self._save_checkpoint("best.pt")
                logger.info("new best %s=%.4f at epoch %d", self.selection_metric, self.best_metric, epoch + 1)
                if epoch >= self.warmup_epochs:
                    self.epochs_since_improve = 0
            elif epoch >= self.warmup_epochs:
                self.epochs_since_improve += 1

            # LR schedule: warmup handled above; plateau drives it afterwards,
            # watching the same metric as the early stop.
            if epoch >= self.warmup_epochs:
                self.scheduler.step(metric)

            current_lr = self.optimizer.param_groups[0]["lr"]
            mlflow.log_metric("train/epoch_lr", current_lr, step=epoch)
            logger.info(
                "epoch %d/%d done — next lr=%g, epochs_since_improve=%d/%d",
                epoch + 1,
                self.epochs,
                current_lr,
                self.epochs_since_improve,
                self.early_stop_patience,
            )

            if self.early_stop and self.epochs_since_improve >= self.early_stop_patience:
                logger.info(
                    "early stopping at epoch %d — no %s improvement (>%g) for %d epochs",
                    epoch + 1,
                    self.selection_metric,
                    self.early_stop_min_delta,
                    self.epochs_since_improve,
                )
                break

        self._save_checkpoint("last.pt")

        # Final test eval on the BEST checkpoint, not the last (possibly-overfit)
        # weights. best.pt is written at least once (epoch-0 metric beats the
        # -1.0 sentinel); the guard is defensive.
        logger.info("training finished — loading best.pt for final test evaluation")
        eval_model = self._eval_model()
        best_path = self.output_dir / "best.pt"
        if best_path.exists():
            best_ckpt = torch.load(best_path, map_location=self.device, weights_only=False)
            eval_model.load_state_dict(best_ckpt["model"])
            logger.info(
                "loaded best.pt (%s=%.4f @ epoch %s)",
                self.selection_metric,
                best_ckpt.get("best_metric", float("nan")),
                best_ckpt.get("epoch"),
            )
        else:
            logger.warning("best.pt not found — running final test eval on current weights")

        test_result = evaluate(
            eval_model,
            self.dl_test,
            self.device,
            self.eval_conf_thres,
            self.eval_iou_thres,
            self.eval_max_det,
        )
        logger.info(
            "test mAP50=%.4f mAP50_95=%.4f",
            test_result["mAP50"],
            test_result["mAP50_95"],
        )
        eval_log_mlflow(test_result, "test")

        mlflow.log_artifact(str(self.output_dir / "best.pt"))
        mlflow.log_artifact(str(self.output_dir / "last.pt"))
        logger.info("uploaded checkpoints to MLflow")

        return test_result
