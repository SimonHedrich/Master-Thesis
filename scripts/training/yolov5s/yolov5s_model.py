"""Factory functions for model, optimizer, and scheduler."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

import scripts.training.yolov5s.constants as constants
from scripts.training.yolov5s import transforms

logger = logging.getLogger(__name__)


def _hyp_dict(num_classes: int, nl: int) -> dict:
    """Loss-gain hyp dict, autoscaled per YOLOv5's own ``train.py`` convention.

    ``constants.HYP_BOX/CLS/OBJ`` are COCO (nc=80) reference values; YOLOv5
    scales them for the actual class count / image size / detection-layer
    count (``box *= 3/nl``, ``cls *= nc/80 * 3/nl``, ``obj *= (imgsz/640)**2 *
    3/nl`` — see ``yolov5/train.py``). Skipping this (as the raw constants
    would) under-weights cls loss by ``80/nc`` for nc > 80, biasing the
    box/cls balance toward pure localization on many-class datasets.
    """
    return {
        "box": constants.HYP_BOX * 3 / nl,
        "cls": constants.HYP_CLS * num_classes / 80 * 3 / nl,
        "cls_pw": constants.HYP_CLS_PW,
        "obj": constants.HYP_OBJ * (constants.IMAGE_SIZE / 640) ** 2 * 3 / nl,
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

    nl = model.model[-1].nl  # number of detection layers, for hyp autoscaling
    model.nc = num_classes
    model.hyp = _hyp_dict(num_classes, nl)
    model.gr = 1.0  # objectness gain for IoU; ComputeLoss reads this
    model.names = [str(i) for i in range(num_classes)]
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info("model: yolov5s, nc=%d, params=%.2fM, device=%s", num_classes, num_params / 1e6, device)
    logger.info(
        "hyp (autoscaled for nc=%d, nl=%d): box=%.5g cls=%.5g obj=%.5g",
        num_classes,
        nl,
        model.hyp["box"],
        model.hyp["cls"],
        model.hyp["obj"],
    )

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
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
    epochs: int,
) -> torch.optim.lr_scheduler.OneCycleLR:
    """OneCycleLR: warmup → peak LR → cosine annealing, stepped every batch.

    ``total_steps = steps_per_epoch * epochs`` covers the safety ceiling; early
    stopping will fire before the cycle completes if the model converges early,
    which is fine — ``best.pt`` captures the peak regardless. Call
    ``scheduler.step()`` once per batch (not per epoch), with no metric argument.
    """
    total_steps = steps_per_epoch * epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=constants.ONE_CYCLE_MAX_LR,
        total_steps=total_steps,
        pct_start=constants.ONE_CYCLE_PCT_START,
        anneal_strategy="cos",
        div_factor=constants.ONE_CYCLE_DIV_FACTOR,
        final_div_factor=constants.ONE_CYCLE_FINAL_DIV_FACTOR,
    )
    logger.info(
        "scheduler: OneCycleLR(max_lr=%g, total_steps=%d, pct_start=%g, "
        "div_factor=%g, final_div_factor=%g) — stepped per batch",
        constants.ONE_CYCLE_MAX_LR,
        total_steps,
        constants.ONE_CYCLE_PCT_START,
        constants.ONE_CYCLE_DIV_FACTOR,
        constants.ONE_CYCLE_FINAL_DIV_FACTOR,
    )
    return scheduler
