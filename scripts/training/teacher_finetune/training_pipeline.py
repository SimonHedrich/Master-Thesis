"""Training loop for the SpeciesNet classifier fine-tune.

New `TrainingPipeline` class (not an import from `yolov5s/training_pipeline.py`
— unlike Goal A's detector-to-detector reuse, this loop has no bbox decode, no
letterbox, no mAP metric, and a different validation contract). It mirrors the
same engineering conventions as the detector pipelines exactly: EMA, AMP,
`OneCycleLR` (warmup → peak → cosine anneal, stepped every batch),
absolute-min-delta early-stop, per-run checkpoint dir, and MLflow logging
cadence — per the implementation plan's explicit instruction that this file
share "the same engineering conventions... but not hyperparameter-identical."
"""
from __future__ import annotations

import logging
import shutil
import time
from copy import deepcopy
from pathlib import Path

import mlflow
import torch
from tqdm import tqdm

from scripts.training.teacher_finetune.evaluate import eval_log_mlflow, evaluate

logger = logging.getLogger(__name__)


class ModelEMA:
    """Exponential moving average of model weights.

    A small, self-contained reimplementation of `yolov5.utils.torch_utils.ModelEMA`
    (same decay-ramp formula) rather than importing the `yolov5` package into
    the SpeciesNet Docker image — that package is detector-specific and pulls
    in dependencies this classifier-only environment has no other use for.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999, tau: int = 2000) -> None:
        import math

        self.ema = deepcopy(model).eval()
        self.updates = 0
        self._decay_fn = lambda x: decay * (1 - math.exp(-x / tau))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model: torch.nn.Module) -> None:
        self.updates += 1
        d = self._decay_fn(self.updates)
        msd = model.state_dict()
        with torch.no_grad():
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()


class TrainingPipeline:
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.OneCycleLR,
        dl_train,
        dl_val,
        dl_test,
        device: torch.device,
        epochs: int,
        output_dir: Path,
        log_every_n_steps: int,
        warmup_epochs: int,
        selection_metric: str,
        early_stop: bool,
        early_stop_patience: int,
        early_stop_min_delta: float,
        use_ema: bool,
        use_amp: bool,
        idx_to_label: dict[int, str],
        genus_species_to_225: dict[str, int],
        genus_to_225: dict[str, int],
        family_to_225: dict[str, int],
        resume_from: Path | None = None,
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

        self.warmup_epochs = warmup_epochs
        self.selection_metric = selection_metric
        self.early_stop = early_stop
        self.early_stop_patience = early_stop_patience
        self.early_stop_min_delta = early_stop_min_delta

        # Projection tables, threaded through to evaluate() at val/test time.
        self.idx_to_label = idx_to_label
        self.genus_species_to_225 = genus_species_to_225
        self.genus_to_225 = genus_to_225
        self.family_to_225 = family_to_225

        self.use_amp = use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.use_amp)
        self.ema = ModelEMA(self.model) if use_ema else None

        output_dir.mkdir(parents=True, exist_ok=True)
        self.best_metric = -1.0
        self.epochs_since_improve = 0
        self.global_step = 0
        self.current_epoch = 0
        self.start_epoch = 0
        if resume_from is not None:
            self._load_resume_checkpoint(resume_from)
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
        return self.ema.ema if self.ema is not None else self.model

    def _load_resume_checkpoint(self, path: Path) -> None:
        """Restore full training state from a ``best.pt``/``last.pt`` checkpoint.

        Checkpoints store the deployable (EMA when enabled) weights, so on
        resume both the raw model and the EMA copy start from those — the raw
        pre-EMA weights are not recoverable, which is a standard, benign
        approximation. Optimizer/scheduler/scaler state and the best-metric /
        patience bookkeeping continue exactly where the crashed run left off.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        if self.ema is not None:
            self.ema.ema.load_state_dict(ckpt["model"])
            self.ema.updates = ckpt.get("ema_updates", self.ema.updates)
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        if self.use_amp and "scaler" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler"])
        self.best_metric = ckpt.get("best_metric", -1.0)
        self.epochs_since_improve = ckpt.get("epochs_since_improve", 0)
        self.start_epoch = ckpt["epoch"] + 1
        self.current_epoch = ckpt["epoch"]
        # global_step is not stored; reconstruct it so step-metric curves in the
        # new MLflow run line up with where the crashed run stopped.
        self.global_step = self.start_epoch * len(self.dl_train)

        # Carry the previous best.pt into the new run dir so the final
        # test eval (which loads best.pt) still works even if this resumed
        # run never beats the inherited best_metric.
        prev_best = path.parent / "best.pt"
        new_best = self.output_dir / "best.pt"
        if prev_best.exists() and prev_best.resolve() != new_best.resolve():
            shutil.copy2(prev_best, new_best)
            logger.info("copied %s -> %s", prev_best, new_best)

        logger.info(
            "resumed from %s: continuing at epoch %d (best %s=%.4f, epochs_since_improve=%d)",
            path,
            self.start_epoch + 1,
            ckpt.get("selection_metric", self.selection_metric),
            self.best_metric,
            self.epochs_since_improve,
        )

    def _evaluate(self, data_loader) -> dict:
        return evaluate(
            self._eval_model(),
            data_loader,
            self.device,
            self.idx_to_label,
            self.genus_species_to_225,
            self.genus_to_225,
            self.family_to_225,
        )

    def _train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        sums: dict[str, float] = {}
        count = 0

        lr_start = self.optimizer.param_groups[0]["lr"]
        logger.info(
            "epoch %d/%d — train start (lr=%g, batches=%d)",
            epoch + 1,
            self.epochs,
            lr_start,
            len(self.dl_train),
        )
        t0 = time.time()

        pbar = tqdm(
            self.dl_train,
            desc=f"epoch {epoch + 1}/{self.epochs} train",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        )
        for arrs, labels in pbar:
            arrs = arrs.to(self.device)
            labels = labels.to(self.device)

            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(arrs)
                loss = self.loss_fn(logits, labels)

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            if self.ema is not None:
                self.ema.update(self.model)

            parts = {"loss_grouped_ce": loss.item()}
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v
            count += 1

            pbar.set_postfix(**{k: f"{v:.3f}" for k, v in parts.items()})

            if self.global_step % self.log_every_n_steps == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                for k, v in parts.items():
                    mlflow.log_metric(f"train/step/{k}", v, step=self.global_step)
                mlflow.log_metric("train/lr", lr, step=self.global_step)
                if self.global_step > 0:
                    parts_str = " ".join(f"{k}={v:.4f}" for k, v in parts.items())
                    logger.info("step %d | %s lr=%g", self.global_step, parts_str, lr)

            self.global_step += 1

        avgs = {k: v / max(count, 1) for k, v in sums.items()}
        for k, v in avgs.items():
            mlflow.log_metric(f"train/epoch_{k}", v, step=epoch)

        avgs_str = " ".join(f"{k}={v:.4f}" for k, v in avgs.items())
        logger.info(
            "epoch %d/%d — train done in %.1fs | avg %s",
            epoch + 1,
            self.epochs,
            time.time() - t0,
            avgs_str,
        )
        return avgs

    def _save_checkpoint(self, name: str) -> None:
        """Save the deployable (EMA if enabled) weights plus resume/provenance
        state. `model` is a plain `state_dict()` — per §2.4 of the implementation
        plan, this is exactly the artifact Goal B needs.
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

        if self.start_epoch >= self.epochs:
            logger.warning(
                "resume checkpoint is already at epoch %d >= ceiling %d — skipping training, running final eval only",
                self.start_epoch,
                self.epochs,
            )

        for epoch in range(self.start_epoch, self.epochs):
            self.current_epoch = epoch

            self._train_one_epoch(epoch)

            val_result = self._evaluate(self.dl_val)
            metric = val_result[self.selection_metric]
            logger.info(
                "epoch %d/%d — val accuracy_top1=%.4f f1_macro=%.4f f1_micro=%.4f (selection %s=%.4f)",
                epoch + 1,
                self.epochs,
                val_result["accuracy_top1"],
                val_result["f1_macro"],
                val_result["f1_micro"],
                self.selection_metric,
                metric,
            )
            eval_log_mlflow(val_result, "val", step=epoch)

            if metric > self.best_metric + self.early_stop_min_delta:
                self.best_metric = metric
                self._save_checkpoint("best.pt")
                logger.info("new best %s=%.4f at epoch %d", self.selection_metric, self.best_metric, epoch + 1)
                if epoch >= self.warmup_epochs:
                    self.epochs_since_improve = 0
            elif epoch >= self.warmup_epochs:
                self.epochs_since_improve += 1

            # Refresh last.pt every epoch (not only at run end) so a crash at
            # any point leaves a current resume checkpoint — the yolov5s
            # log_table crash lost last.pt entirely because it was only written
            # after the loop (see
            # docs/progress_notes/2026-07-13_mlflow-log-table-crash-and-resume.md).
            self._save_checkpoint("last.pt")

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

        test_result = self._evaluate(self.dl_test)
        logger.info(
            "test accuracy_top1=%.4f f1_macro=%.4f f1_micro=%.4f",
            test_result["accuracy_top1"],
            test_result["f1_macro"],
            test_result["f1_micro"],
        )
        eval_log_mlflow(test_result, "test")

        mlflow.log_artifact(str(self.output_dir / "best.pt"))
        mlflow.log_artifact(str(self.output_dir / "last.pt"))
        logger.info("uploaded checkpoints to MLflow")

        return test_result
