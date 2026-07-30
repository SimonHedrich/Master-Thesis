"""Factory functions for model, optimizer, and scheduler (YOLO26n)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import IterableSimpleNamespace

import scripts.synthetic_model_comparison.training.constants as constants
from scripts.synthetic_model_comparison.training import transforms
from scripts.synthetic_model_comparison.training.optim import model_optimizer, model_scheduler

logger = logging.getLogger(__name__)

__all__ = ["yolo26n_model", "model_optimizer", "model_scheduler"]


def yolo26n_model(
    num_classes: int,
    weights: Path | None,
    device: torch.device,
) -> tuple[nn.Module, Callable]:
    model = DetectionModel(cfg=constants.MODEL_CONFIG, ch=3, nc=num_classes, verbose=False)

    if weights is not None and weights.exists():
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        raw = ckpt.get("model", ckpt)
        state = raw.state_dict() if hasattr(raw, "state_dict") else raw
        own = model.state_dict()
        filtered = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(filtered, strict=False)
        logger.info(
            "loaded %d/%d keys from %s (skipped %d shape-mismatched)",
            len(filtered),
            len(own),
            weights,
            len(own) - len(filtered),
        )
    else:
        logger.info("no pretrained weights — training from random initialization")

    # v8DetectionLoss / E2ELoss read model.args via ATTRIBUTE access (h =
    # model.args; h.box / h.cls / h.dfl; E2ELoss.decay reads
    # self.one2one.hyp.epochs). This pipeline bypasses Ultralytics' high-level
    # Model/Trainer API entirely (same bypass pattern as yolov5s_model.py
    # constructing yolov5.models.yolo.Model directly), so model.args is NEVER
    # auto-populated — it must be set here as an IterableSimpleNamespace
    # (attribute access), NOT a plain dict, or v8DetectionLoss.__init__ /
    # E2ELoss.decay raise AttributeError.
    model.args = IterableSimpleNamespace(
        box=constants.HYP_BOX,
        cls=constants.HYP_CLS,
        dfl=constants.HYP_DFL,
        epochs=constants.EPOCH_COUNT,  # E2ELoss.decay's o2m/o2o anneal horizon (safety ceiling)
    )
    model.names = [str(i) for i in range(num_classes)]

    # Person-class downweighting (project convention, strategy doc §3) —
    # v8DetectionLoss already reads getattr(model, "class_weights", None) and
    # folds it into the BCE cls loss, so this would be a one-line set:
    #   model.class_weights = torch.ones(num_classes)
    #   model.class_weights[PERSON_YOLO_IDX] = 0.3
    # Left off for now — yolov5s does not currently set this either, so
    # omitting it here keeps the two pipelines comparable until both adopt it
    # together.

    # Cap eval-time detections to EVAL_MAX_DET (Detect.max_det defaults to
    # 300; postprocess() reads it from the Detect module instance, not from a
    # config object, so it must be set on the instance).
    model.model[-1].max_det = constants.EVAL_MAX_DET

    model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info("model: yolo26n, nc=%d, params=%.2fM, device=%s", num_classes, num_params / 1e6, device)

    def preprocess(img: np.ndarray) -> torch.Tensor:
        img_lb, _, _ = transforms.letterbox(img, new_shape=constants.IMAGE_SIZE)
        return transforms.to_tensor(img_lb)

    return model, preprocess
