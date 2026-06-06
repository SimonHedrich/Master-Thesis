"""Factory functions for model, optimizer, and scheduler."""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

import scripts.training.yolov5s.constants as constants
from scripts.training.yolov5s import transforms

logger = logging.getLogger(__name__)


def _hyp_dict() -> dict:
    return {
        "box": constants.HYP_BOX,
        "cls": constants.HYP_CLS,
        "cls_pw": constants.HYP_CLS_PW,
        "obj": constants.HYP_OBJ,
        "obj_pw": constants.HYP_OBJ_PW,
        "iou_t": constants.HYP_IOU_T,
        "anchor_t": constants.HYP_ANCHOR_T,
        "fl_gamma": constants.HYP_FL_GAMMA,
        "label_smoothing": constants.HYP_LABEL_SMOOTHING,
    }


def yolov5s_model(
    num_classes: int,
    weights: Path | None,
    device: torch.device,
) -> tuple[nn.Module, Callable]:
    import yolov5 as _yv5_pkg
    from yolov5.models.yolo import Model

    cfg = str(Path(_yv5_pkg.__file__).parent / "models" / "yolov5s.yaml")
    model = Model(cfg=cfg, ch=3, nc=num_classes)

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

    model.nc = num_classes
    model.hyp = _hyp_dict()
    model.gr = 1.0  # objectness gain for IoU; ComputeLoss reads this
    model.names = [str(i) for i in range(num_classes)]
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info("model: yolov5s, nc=%d, params=%.2fM, device=%s", num_classes, num_params / 1e6, device)

    def preprocess(img: np.ndarray) -> torch.Tensor:
        img_lb, _, _ = transforms.letterbox(img, new_shape=constants.IMAGE_SIZE)
        return transforms.to_tensor(img_lb)

    return model, preprocess


def model_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    g0: list[nn.Parameter] = []  # BN weights — no decay
    g1: list[nn.Parameter] = []  # conv weights — with decay
    g2: list[nn.Parameter] = []  # biases — no decay

    for v in model.modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):
            g0.append(v.weight)
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            g1.append(v.weight)

    if constants.OPTIMIZER == "SGD":
        optimizer: torch.optim.Optimizer = torch.optim.SGD(
            g0,
            lr=constants.LEARNING_RATE,
            momentum=constants.MOMENTUM,
            nesterov=constants.NESTEROV,
        )
    else:
        optimizer = torch.optim.AdamW(
            g0,
            lr=constants.LEARNING_RATE,
            betas=(constants.MOMENTUM, 0.999),
        )

    optimizer.add_param_group({"params": g1, "weight_decay": constants.WEIGHT_DECAY})
    optimizer.add_param_group({"params": g2, "weight_decay": 0.0})

    logger.info(
        "optimizer=%s lr=%g | param groups: g0(BN,no-decay)=%d g1(conv,decay=%g)=%d g2(bias,no-decay)=%d",
        constants.OPTIMIZER,
        constants.LEARNING_RATE,
        len(g0),
        constants.WEIGHT_DECAY,
        len(g1),
        len(g2),
    )
    return optimizer


def model_scheduler(
    optimizer: torch.optim.Optimizer, epochs: int
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup = constants.WARMUP_EPOCHS
    lrf = constants.LRF

    def lf(epoch: int) -> float:
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, epochs - warmup)
        return (1 - lrf) * 0.5 * (1 + math.cos(math.pi * progress)) + lrf

    logger.info(
        "scheduler: warmup %d epochs → cosine to lr0*%g over %d epochs",
        warmup,
        lrf,
        epochs,
    )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
