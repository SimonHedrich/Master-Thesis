"""Training loop for YOLOv5s fine-tuning."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import mlflow
import torch
from tqdm import tqdm

from scripts.training.yolov5s.evaluation import eval_log_mlflow, evaluate

logger = logging.getLogger(__name__)


class TrainingPipeline:
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
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

        output_dir.mkdir(parents=True, exist_ok=True)
        self.best_map = -1.0
        self.global_step = 0
        self.current_epoch = 0
        logger.info("output dir: %s", output_dir)

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

            preds = self.model(imgs)
            total, parts = self.loss_fn(preds, targets)

            self.optimizer.zero_grad(set_to_none=True)
            total.backward()
            self.optimizer.step()

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
        path = self.output_dir / name
        torch.save(
            {"model": self.model.state_dict(), "epoch": self.current_epoch},
            path,
        )
        logger.info("saved checkpoint: %s", path)

    def run_pipeline(self) -> dict:
        logger.info(
            "starting training: %d epochs | train_batches=%d val_batches=%d test_batches=%d | device=%s",
            self.epochs,
            len(self.dl_train),
            len(self.dl_val),
            len(self.dl_test),
            self.device,
        )

        for epoch in range(self.epochs):
            self.current_epoch = epoch
            self._train_one_epoch(epoch)

            val_result = evaluate(
                self.model,
                self.dl_val,
                self.device,
                self.eval_conf_thres,
                self.eval_iou_thres,
                self.eval_max_det,
            )
            logger.info(
                "epoch %d/%d — val mAP50=%.4f mAP50_95=%.4f",
                epoch + 1,
                self.epochs,
                val_result["mAP50"],
                val_result["mAP50_95"],
            )
            eval_log_mlflow(val_result, "val", step=epoch)

            self.scheduler.step()
            new_lr = self.optimizer.param_groups[0]["lr"]
            mlflow.log_metric("train/lr", new_lr, step=epoch)
            logger.info("scheduler stepped — next epoch lr=%g", new_lr)

            if val_result["mAP50"] > self.best_map:
                self.best_map = val_result["mAP50"]
                self._save_checkpoint("best.pt")
                logger.info("new best mAP50=%.4f at epoch %d", self.best_map, epoch + 1)

        self._save_checkpoint("last.pt")

        logger.info("training finished — running final test evaluation")
        test_result = evaluate(
            self.model,
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
