"""Factory functions for the SpeciesNet classifier model, optimizer, and scheduler.

Same `(model, preprocess)`-tuple factory shape as `yolov5s_model.py`'s
`yolov5s_model()`, adapted for a classifier instead of a detector.
"""
from __future__ import annotations

import logging
import sys
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

import scripts.training.teacher_finetune.constants as constants

logger = logging.getLogger(__name__)


def _check_environment() -> None:
    """Same guard as `scripts/dataset_quality/{6,7}-*speciesnet*.py` — this
    package only runs inside `Dockerfile.speciesnet` (Python 3.11)."""
    try:
        import speciesnet  # noqa: F401
    except ImportError:
        print(
            "ERROR: 'speciesnet' is not installed.\n"
            "This script must run inside Dockerfile.speciesnet (Python 3.11).\n"
            "  make speciesnet-build\n"
            "  make speciesnet-start",
            file=sys.stderr,
        )
        sys.exit(1)


def _freeze_backbone(model: nn.Module, fraction: float) -> None:
    """Freeze the first ``fraction`` of the model's parameter tensors, by
    ``named_parameters()`` iteration order (earliest/lowest-level layers
    first). The remaining tensors — later blocks + the classification head —
    stay trainable.

    Iterating flat parameter-tensor order rather than hardcoding block names
    (e.g. timm-style ``blocks[i]`` vs. torchvision-style ``features[i]``) keeps
    this robust regardless of which module-naming convention SpeciesNet's
    EfficientNetV2-M implementation actually uses — flagged in the
    implementation plan as something to double check once `speciesnet` is
    installed and the real module structure can be inspected, since a
    fraction-of-tensors split may not align with a fraction-of-depth split if
    parameter tensors are unevenly distributed across blocks.
    """
    named_params = list(model.named_parameters())
    n_freeze = int(len(named_params) * fraction)
    for i, (name, p) in enumerate(named_params):
        p.requires_grad = i >= n_freeze
    n_trainable = sum(p.requires_grad for _, p in named_params)
    logger.info(
        "froze %d/%d parameter tensors (fraction=%.2f), %d trainable",
        n_freeze,
        len(named_params),
        fraction,
        n_trainable,
    )


def speciesnet_model(
    device: torch.device,
    freeze_fraction: float = constants.FREEZE_PARAM_FRACTION,
) -> tuple[nn.Module, Callable, list[str]]:
    """Load SpeciesNet's classifier, apply the freeze split, return
    ``(model, preprocess_fn, labels)``.

    ``preprocess_fn(img, bbox_norm) -> np.ndarray | None`` replicates
    `SpeciesNetClassifier.preprocess_crop()`
    (`scripts/dataset_quality/6-classify_speciesnet.py`) verbatim so train-time
    and inference-time preprocessing stay identical (parent strategy doc §2.1).
    """
    _check_environment()
    from speciesnet import DEFAULT_MODEL, SpeciesNet
    from speciesnet.utils import BBox

    pipeline = SpeciesNet(DEFAULT_MODEL, components="classifier", geofence=False)
    clf = pipeline.classifier
    model: nn.Module = clf.model
    labels: list[str] = list(clf.labels)

    _freeze_backbone(model, freeze_fraction)
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "model: speciesnet-classifier, nc=%d, params=%.2fM (%.2fM trainable), device=%s",
        constants.NUM_CLASSES_LEAF,
        num_params / 1e6,
        num_trainable / 1e6,
        device,
    )

    def preprocess_fn(img: Image.Image, bbox_norm: list) -> "np.ndarray | None":
        bbox = BBox(*bbox_norm)
        preprocessed = clf.preprocess(img, bboxes=[bbox])
        if preprocessed is None:
            return None
        return (preprocessed.arr / 255).astype(np.float32)

    return model, preprocess_fn, labels


def model_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    """Single-group AdamW over trainable (unfrozen) parameters only.

    Unlike `yolov5s_model.py`'s 3-param-group BN/conv/bias split (a YOLOv5
    convention for training a detector from scratch), a straightforward single
    AdamW group with uniform weight decay is the standard recipe for
    fine-tuning a pretrained classifier at a low learning rate.
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=constants.LEARNING_RATE,
        weight_decay=constants.WEIGHT_DECAY,
    )
    logger.info(
        "optimizer=%s lr=%g weight_decay=%g | trainable params=%d",
        constants.OPTIMIZER,
        constants.LEARNING_RATE,
        constants.WEIGHT_DECAY,
        len(trainable),
    )
    return optimizer


def model_scheduler(
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
    epochs: int,
) -> torch.optim.lr_scheduler.OneCycleLR:
    """OneCycleLR: warmup → peak LR → cosine annealing, stepped every batch.

    Same construction as `yolov5s_model.py`'s `model_scheduler()`. ``total_steps
    = steps_per_epoch * epochs`` covers the safety ceiling; early stopping will
    fire before the cycle completes if the model converges early, which is
    fine — ``best.pt`` captures the peak regardless. Call ``scheduler.step()``
    once per batch (not per epoch), with no metric argument.
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
