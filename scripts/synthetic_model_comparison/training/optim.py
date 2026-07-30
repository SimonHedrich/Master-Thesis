"""Optimizer and scheduler factories.

Extracted from scripts/training/yolov5s/yolov5s_model.py's model_optimizer/
model_scheduler (architecture-generic — they only walk model.modules() /
model.parameters(), nothing yolov5-specific), rather than copying that whole
file, since this package doesn't need yolov5s' model-construction code.
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn

import scripts.synthetic_model_comparison.training.constants as constants

logger = logging.getLogger(__name__)


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
    """OneCycleLR: warmup → peak LR → cosine annealing, stepped every batch."""
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
