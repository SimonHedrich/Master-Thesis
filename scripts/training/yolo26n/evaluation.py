"""Evaluation loop and MLflow logging helper for the YOLO26n pipeline."""
from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import mlflow
import torch
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm

import scripts.training.yolo26n.constants as constants

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    conf_thres: float,
    iou_thres: float,
    max_det: int,
) -> dict:
    # conf_thres/iou_thres are accepted for call-site parity with yolov5s'
    # evaluate() (training_pipeline.py passes them positionally regardless of
    # which package's evaluation.py is in scope) but are UNUSED here: yolo26n's
    # Detect head (end2end=True) is NMS-free — postprocess() does score-ranked
    # top-k filtering only, governed by max_det (already wired onto the Detect
    # module instance in yolo26n_model()), with no separate confidence/IoU
    # threshold knob exposed at this layer.
    del conf_thres, iou_thres
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True)
    metric.to(device)

    logger.info("evaluating on %d batches", len(data_loader))
    pbar = tqdm(data_loader, desc="eval", unit="batch", leave=False, dynamic_ncols=True)
    for imgs, targets, _paths, shapes in pbar:
        imgs = imgs.to(device)
        targets = targets.to(device)

        raw = model(imgs)
        # yolo26n's Detect head (end2end=True) is NMS-free: postprocess()
        # already does score-ranked top-k filtering (Detect.max_det, set in
        # yolo26n_model()). model(imgs) in eval mode returns (y, preds); y is
        # (batch, min(max_det, num_anchors), 6) = [x1,y1,x2,y2,score,cls_idx]
        # in the SAME letterboxed-canvas pixel space as yolov5's NMS output,
        # so every line below this point is unchanged from yolov5s/evaluation.py.
        y = raw[0] if isinstance(raw, (tuple, list)) else raw
        det_list = [y[i] for i in range(y.shape[0])]

        preds_list: list[dict] = []
        targets_list: list[dict] = []

        for i, dets in enumerate(det_list):
            (h0, w0), ((r, _), (dw, dh)) = shapes[i]

            if dets is not None and len(dets):
                boxes = dets[:, :4].clone()
                # un-letterbox: remove padding offset then scale back to original coords
                boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw / 2) / r
                boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh / 2) / r
                boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w0)
                boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h0)
                preds_list.append(
                    {
                        "boxes": boxes,
                        "scores": dets[:, 4],
                        "labels": dets[:, 5].long(),
                    }
                )
            else:
                preds_list.append(
                    {
                        "boxes": torch.zeros((0, 4), device=device),
                        "scores": torch.zeros(0, device=device),
                        "labels": torch.zeros(0, dtype=torch.long, device=device),
                    }
                )

            img_targets = targets[targets[:, 0] == i]
            if img_targets.shape[0] > 0:
                # columns: [batch_idx, cls, cx, cy, w_norm, h_norm] in normalized coords
                cls = img_targets[:, 1].long()
                cx = img_targets[:, 2] * constants.IMAGE_SIZE
                cy = img_targets[:, 3] * constants.IMAGE_SIZE
                bw = img_targets[:, 4] * constants.IMAGE_SIZE
                bh = img_targets[:, 5] * constants.IMAGE_SIZE
                # convert from letterboxed normalized to original-image xyxy
                x1 = ((cx - bw / 2) - dw / 2) / r
                y1 = ((cy - bh / 2) - dh / 2) / r
                x2 = ((cx + bw / 2) - dw / 2) / r
                y2 = ((cy + bh / 2) - dh / 2) / r
                gt_boxes = torch.stack([x1, y1, x2, y2], dim=1)
                gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]].clamp(0, w0)
                gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]].clamp(0, h0)
                targets_list.append({"boxes": gt_boxes, "labels": cls})
            else:
                targets_list.append(
                    {
                        "boxes": torch.zeros((0, 4), device=device),
                        "labels": torch.zeros(0, dtype=torch.long, device=device),
                    }
                )

        metric.update(preds_list, targets_list)

    result = metric.compute()
    logger.debug("metric.compute keys: %s", list(result.keys()))

    per_class = result.get("map_per_class")
    per_class_list = None
    if per_class is not None and hasattr(per_class, "ndim") and per_class.ndim == 1:
        per_class_list = per_class.tolist()

    return {
        "mAP50": result["map_50"].item(),
        "mAP50_95": result["map"].item(),
        "per_class_AP": per_class_list,
        "class_names": getattr(data_loader.dataset, "class_names", None),
    }


def eval_log_mlflow(result: dict, prefix: str, step: int | None = None) -> None:
    mlflow.log_metric(f"{prefix}/mAP50", result["mAP50"], step=step)
    mlflow.log_metric(f"{prefix}/mAP50_95", result["mAP50_95"], step=step)
    logger.info(
        "mlflow: logged %s/mAP50=%.4f %s/mAP50_95=%.4f (step=%s)",
        prefix,
        result["mAP50"],
        prefix,
        result["mAP50_95"],
        step,
    )

    per_class = result.get("per_class_AP")
    class_names = result.get("class_names")
    if per_class is not None and class_names is not None:
        rows = [
            {"class_idx": i, "class_name": class_names[i] if i < len(class_names) else str(i), "AP50_95": ap}
            for i, ap in enumerate(per_class)
        ]
        # Deliberately log_artifact, NOT mlflow.log_table: log_table appends an
        # entry per file to the run's `mlflow.loggedArtifacts` tag, which is
        # capped at 8000 chars server-side — one table per epoch overflows and
        # silently corrupts the tag after ~137 epochs, killing the run (see
        # docs/progress_notes/2026-07-13_mlflow-log-table-crash-and-resume.md).
        # Best-effort: a logging failure must never abort a multi-day training run.
        try:
            with tempfile.TemporaryDirectory() as tmp:
                name = f"{prefix}_per_class_ap_step{step if step is not None else 'final'}.json"
                table_path = Path(tmp) / name
                table_path.write_text(json.dumps(rows, indent=1))
                mlflow.log_artifact(str(table_path), artifact_path="per_class_ap")
            logger.info("mlflow: logged %s per-class AP table (%d classes)", prefix, len(rows))
        except Exception:
            logger.exception("mlflow: failed to log %s per-class AP table — continuing", prefix)
