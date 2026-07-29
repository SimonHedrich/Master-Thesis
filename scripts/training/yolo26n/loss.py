"""Thin wrapper around Ultralytics' E2ELoss (yolo26n is end2end: True)."""
from __future__ import annotations

import torch


class Yolo26Loss:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        # Built once and cached here (not via model.loss()'s lazy
        # self.criterion) so the loss-wrapper owns its own state, not the
        # model — mirrors YoloLoss's pattern of holding
        # self.compute = ComputeLoss(model) rather than relying on model state.
        self.criterion = model.init_criterion()  # E2ELoss, since yolo26n sets end2end=True

    def __call__(
        self, preds: dict, targets: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # preds is the model's raw training-mode forward output:
        #   {"one2many": {"boxes":..., "scores":..., "feats":...}, "one2one": {...}}
        # targets is the existing flat [N,6] tensor from CocoYoloDataset/collate_fn:
        #   columns = [batch_idx, cls, cx, cy, w, h], all normalized [0,1].
        batch = {
            "batch_idx": targets[:, 0],
            "cls": targets[:, 1],
            "bboxes": targets[:, 2:6],
        }
        # E2ELoss.__call__ (mirroring Ultralytics' own Trainer, which does
        # `loss.sum()` before backward — ultralytics/engine/trainer.py:451)
        # returns a 3-element [box, cls, dfl] tensor, not a scalar: it is
        # `loss_one2many[0]*o2m + loss_one2one[0]*o2o`, and v8DetectionLoss.loss()
        # returns the gain-multiplied-but-unsummed 3-vector times batch_size.
        raw_total, parts = self.criterion(preds, batch)
        total = raw_total.sum()
        return total, {
            "loss_box": parts[0].item(),
            "loss_cls": parts[1].item(),
            "loss_dfl": parts[2].item(),
            "loss_total": total.item(),
        }

    def update(self) -> None:
        """Call once per epoch: anneals E2ELoss's one2many/one2one weight mix (o2m: 0.8→0.1)."""
        upd = getattr(self.criterion, "update", None)
        if upd is not None:
            upd()
