"""Thin wrapper around yolov5.utils.loss.ComputeLoss."""
from __future__ import annotations

import torch
from yolov5.utils.loss import ComputeLoss


class YoloLoss:
    def __init__(self, model: torch.nn.Module) -> None:
        self.compute = ComputeLoss(model)

    def __call__(
        self, preds: list, targets: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        total, parts = self.compute(preds, targets)
        return total, {
            "loss_box": parts[0].item(),
            "loss_obj": parts[1].item(),
            "loss_cls": parts[2].item(),
            "loss_total": total.item(),
        }
