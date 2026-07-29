"""Knowledge-distillation loss adapter for yolo26n (Goal B).

Blends `TaskAlignedAssigner`'s soft `target_scores` toward a per-image,
cached SpeciesNet teacher distribution before the classification BCE loss —
see docs/plans/2026-06-30_yolo26-kd-and-teacher-finetune-implementation-plan.md
§3.4 for the design rationale.

`KDv8DetectionLoss.get_assigned_targets_and_loss()` is a **version-pinned
copy** of `ultralytics.utils.loss.v8DetectionLoss.get_assigned_targets_and_loss`
(ultralytics==8.4.60, ultralytics/utils/loss.py:398-460) with one insertion
between the assigner call and the BCE call — Ultralytics exposes no extension
point there, so subclassing-with-override isn't possible for just that
section. Re-diff this method against the installed ultralytics version after
any version bump.
"""
from __future__ import annotations

import torch
from ultralytics.utils.loss import E2ELoss, v8DetectionLoss
from ultralytics.utils.tal import make_anchors

import scripts.training.yolo26n.constants as constants
from scripts.training.yolo26n.loss import Yolo26Loss


def _temperature_scale(probs: torch.Tensor, temperature: float, eps: float = 1e-8) -> torch.Tensor:
    """Approximate a temperature-scaled redistribution of an already-softmax'd
    probability vector by re-normalizing on the simplex: `p_i^(1/T)`,
    renormalized to sum to 1.

    This is an approximation of true logit-temperature scaling — the teacher
    cache stores post-softmax probabilities, not logits (plan doc §3.4's
    documented fallback). Exact (no-op) at T=1.
    """
    if temperature == 1.0:
        return probs
    scaled = (probs + eps).pow(1.0 / temperature)
    return scaled / scaled.sum(dim=-1, keepdim=True)


class KDv8DetectionLoss(v8DetectionLoss):
    """`v8DetectionLoss` with the assigned `target_scores` blended toward a
    per-image teacher soft-label distribution before the BCE cls loss.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tal_topk: int = 10,
        tal_topk2: int | None = None,
        kd_alpha: float = 0.5,
        kd_temperature: float = 4.0,
    ) -> None:
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        self.kd_alpha = kd_alpha
        self.kd_temperature = kd_temperature

    def get_assigned_targets_and_loss(self, preds: dict, batch: dict) -> tuple:
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = (
            torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype)
            * self.stride[0]
        )

        # Targets
        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1
        )
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        # ── KD blend (the only deviation from v8DetectionLoss) ─────────────
        teacher_probs = batch.get("teacher_probs")  # [bs, nc] or None
        if teacher_probs is not None:
            teacher_probs = teacher_probs.to(device=self.device, dtype=target_scores.dtype)
            has_teacher = teacher_probs.sum(dim=-1) > 0  # [bs] — zero vector = uncached image
            if has_teacher.any():
                teacher_scaled = _temperature_scale(teacher_probs, self.kd_temperature)
                blend_mask = fg_mask & has_teacher.unsqueeze(1)  # [bs, num_anchors]
                teacher_bc = teacher_scaled.unsqueeze(1).expand(-1, target_scores.shape[1], -1)
                target_scores = torch.where(
                    blend_mask.unsqueeze(-1),
                    (1 - self.kd_alpha) * target_scores + self.kd_alpha * teacher_bc,
                    target_scores,
                )
        # ── end KD blend ────────────────────────────────────────────────────

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss with optional class weighting
        bce_loss = self.bce(pred_scores, target_scores.to(dtype))  # (bs, num_anchors, nc)
        if self.class_weights is not None:
            bce_loss *= self.class_weights
        loss[1] = bce_loss.sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            loss.detach(),
        )  # loss(box, cls, dfl)


class KDE2ELoss(E2ELoss):
    """`E2ELoss` with per-head selection of `KDv8DetectionLoss` vs.
    `v8DetectionLoss`, controlled by `constants.KD_APPLY_TO`.

    Only `__init__` is overridden — `__call__`, `update()`, `decay()` are
    inherited unchanged from `E2ELoss` since they are polymorphic over
    whatever `.loss()`-exposing object is stored in `self.one2many`/
    `self.one2one`.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        kd_alpha: float,
        kd_temperature: float,
        apply_to: str = "one2one",
    ) -> None:
        if apply_to not in ("one2one", "one2many", "both"):
            raise ValueError(f"apply_to must be one2one|one2many|both, got {apply_to!r}")

        kd_kwargs = {"kd_alpha": kd_alpha, "kd_temperature": kd_temperature}
        o2m_is_kd = apply_to in ("one2many", "both")
        o2o_is_kd = apply_to in ("one2one", "both")

        self.one2many = (
            KDv8DetectionLoss(model, tal_topk=10, **kd_kwargs)
            if o2m_is_kd
            else v8DetectionLoss(model, tal_topk=10)
        )
        self.one2one = (
            KDv8DetectionLoss(model, tal_topk=7, tal_topk2=1, **kd_kwargs)
            if o2o_is_kd
            else v8DetectionLoss(model, tal_topk=7, tal_topk2=1)
        )

        # Loss-weight annealing state — verbatim from E2ELoss.__init__.
        self.updates = 0
        self.total = 1.0
        self.o2m = 0.8
        self.o2o = 0.2
        self.o2m_copy = 0.8
        self.final_o2m = 0.1


class KDYolo26Loss(Yolo26Loss):
    """`Yolo26Loss` variant that threads a `teacher_probs` tensor through to
    `KDE2ELoss`. Only `__init__` and `__call__` are overridden — `update()`
    is inherited unchanged from `Yolo26Loss`.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        kd_alpha: float = constants.KD_ALPHA,
        kd_temperature: float = constants.KD_TEMPERATURE,
        apply_to: str = constants.KD_APPLY_TO,
    ) -> None:
        self.model = model
        self.criterion = KDE2ELoss(
            model, kd_alpha=kd_alpha, kd_temperature=kd_temperature, apply_to=apply_to
        )

    def __call__(
        self, preds: dict, targets: torch.Tensor, teacher_probs: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        batch = {
            "batch_idx": targets[:, 0],
            "cls": targets[:, 1],
            "bboxes": targets[:, 2:6],
            "teacher_probs": teacher_probs,
        }
        raw_total, parts = self.criterion(preds, batch)
        total = raw_total.sum()
        return total, {
            "loss_box": parts[0].item(),
            "loss_cls": parts[1].item(),
            "loss_dfl": parts[2].item(),
            "loss_total": total.item(),
        }
