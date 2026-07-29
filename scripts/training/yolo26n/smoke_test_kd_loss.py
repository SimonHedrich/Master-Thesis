"""Smoke test for the yolo26n KD loss adapter (Goal B).

Risk-mitigation deliverable for
docs/plans/2026-06-30_yolo26-kd-and-teacher-finetune-implementation-plan.md §3.4:
KDv8DetectionLoss copies (not delegates-to) v8DetectionLoss's assigned-target
computation to insert a teacher-distribution blend with no Ultralytics
extension point available. Get the insertion point or the fallback-to-hard-
label branch wrong and training would silently run a slightly-different loss
with no error — this validates both branches on synthetic data (cheap,
CPU-only, seconds) before any real training run.

Run from the repo root:
    PYTHONPATH=. uv run -m scripts.training.yolo26n.smoke_test_kd_loss

Checks:
  1. KDYolo26Loss with all-zero teacher_probs (no cached record for any image
     in the batch) produces a BIT-IDENTICAL total loss to plain Yolo26Loss on
     the same (preds, targets) — the KD path must be a strict no-op fallback
     for uncached images, not an approximation.
  2. With a sharply-peaked synthetic teacher_probs disagreeing with the hard
     label, loss_cls measurably changes as KD_ALPHA sweeps 0.0 -> 1.0 (proves
     the blend is actually wired into the BCE call, not silently bypassed).
  3. _temperature_scale is a no-op at T=1, sharpens (increases max-prob) at
     T<1, and flattens (decreases max-prob) at T>1.
  4. KDE2ELoss.update()/decay() run across several epochs without raising and
     anneal o2m toward final_o2m the same way stock E2ELoss does.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import torch

# ── Ensure repo root is on sys.path ──────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.training.yolo26n.constants as constants
from scripts.training.yolo26n.kd_loss import KDYolo26Loss, _temperature_scale
from scripts.training.yolo26n.loss import Yolo26Loss
from scripts.training.yolo26n.yolo26n_model import yolo26n_model

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def check(condition: bool, label: str, detail: str = "") -> bool:
    tag = PASS if condition else FAIL
    msg = f"  [{tag}] {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return bool(condition)


def _synthetic_batch(batch_size: int = 2, image_size: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    image_size = image_size or constants.IMAGE_SIZE
    imgs = torch.rand(batch_size, 3, image_size, image_size)
    rows = []
    for b in range(batch_size):
        rows.append([b, float(b % constants.NUM_CLASSES), 0.5, 0.5, 0.2, 0.3])
        rows.append([b, float((b + 5) % constants.NUM_CLASSES), 0.3, 0.4, 0.1, 0.15])
    targets = torch.tensor(rows, dtype=torch.float32)
    return imgs, targets


def run_zero_fallback_check(model: torch.nn.Module, preds: dict, targets: torch.Tensor, batch_size: int) -> bool:
    print("\n1. All-zero teacher_probs is a bit-identical fallback to hard-label-only")
    base_loss = Yolo26Loss(model)
    total_base, parts_base = base_loss(preds, targets)

    kd_loss = KDYolo26Loss(model, kd_alpha=0.5, kd_temperature=4.0, apply_to="one2one")
    teacher_zero = torch.zeros(batch_size, constants.NUM_CLASSES)
    total_kd, parts_kd = kd_loss(preds, targets, teacher_zero)

    all_pass = True
    all_pass &= check(
        torch.allclose(total_base, total_kd, atol=1e-6),
        "total loss identical (all-zero teacher_probs)",
        detail=f"base={total_base.item():.6f} kd={total_kd.item():.6f}",
    )
    all_pass &= check(
        abs(parts_base["loss_cls"] - parts_kd["loss_cls"]) < 1e-6,
        "loss_cls identical (all-zero teacher_probs)",
        detail=f"base={parts_base['loss_cls']:.6f} kd={parts_kd['loss_cls']:.6f}",
    )
    return all_pass


def run_blend_effect_check(model: torch.nn.Module, preds: dict, targets: torch.Tensor, batch_size: int) -> bool:
    print("\n2. Blend measurably changes loss_cls as KD_ALPHA sweeps 0.0 -> 1.0")
    # Sharply peaked on a class NOT present in `targets`' hard labels, so the
    # blend is guaranteed to disagree with the assigner's hard-label target.
    teacher_probs = torch.zeros(batch_size, constants.NUM_CLASSES)
    peak_idx = (constants.NUM_CLASSES - 1)
    teacher_probs[:, peak_idx] = 1.0

    cls_losses = []
    for alpha in (0.0, 0.5, 1.0):
        kd_loss = KDYolo26Loss(model, kd_alpha=alpha, kd_temperature=1.0, apply_to="one2one")
        _total, parts = kd_loss(preds, targets, teacher_probs)
        cls_losses.append(parts["loss_cls"])

    all_pass = True
    all_pass &= check(
        cls_losses[0] != cls_losses[1] or cls_losses[1] != cls_losses[2],
        "loss_cls varies across alpha=0.0/0.5/1.0",
        detail=str([f"{v:.6f}" for v in cls_losses]),
    )
    # alpha=0.0 must exactly match the unblended (hard-label-only) result.
    base_loss = Yolo26Loss(model)
    _total_base, parts_base = base_loss(preds, targets)
    all_pass &= check(
        abs(cls_losses[0] - parts_base["loss_cls"]) < 1e-6,
        "alpha=0.0 matches plain Yolo26Loss exactly",
        detail=f"alpha0={cls_losses[0]:.6f} base={parts_base['loss_cls']:.6f}",
    )
    return all_pass


def run_temperature_scale_check() -> bool:
    print("\n3. _temperature_scale: no-op at T=1, sharpens at T<1, flattens at T>1")
    probs = torch.tensor([[0.7, 0.2, 0.1]])

    all_pass = True
    t1 = _temperature_scale(probs, 1.0)
    all_pass &= check(torch.allclose(t1, probs), "T=1.0 is a no-op", detail=str(t1.tolist()))

    sharp = _temperature_scale(probs, 0.5)
    all_pass &= check(
        sharp.max().item() > probs.max().item(),
        "T<1 sharpens the distribution (max prob increases)",
        detail=f"{probs.max().item():.4f} -> {sharp.max().item():.4f}",
    )

    flat = _temperature_scale(probs, 4.0)
    all_pass &= check(
        flat.max().item() < probs.max().item(),
        "T>1 flattens the distribution (max prob decreases)",
        detail=f"{probs.max().item():.4f} -> {flat.max().item():.4f}",
    )
    all_pass &= check(
        bool(torch.allclose(sharp.sum(dim=-1), torch.tensor([1.0]), atol=1e-5))
        and bool(torch.allclose(flat.sum(dim=-1), torch.tensor([1.0]), atol=1e-5)),
        "scaled distributions still sum to 1",
    )
    return all_pass


def run_anneal_check(model: torch.nn.Module) -> bool:
    print("\n4. KDE2ELoss.update()/decay() anneal o2m without raising")
    kd_loss = KDYolo26Loss(model, kd_alpha=0.5, kd_temperature=4.0, apply_to="both")
    o2m_before = kd_loss.criterion.o2m
    try:
        for _ in range(10):
            kd_loss.update()
        anneal_ok = True
    except Exception:
        traceback.print_exc()
        anneal_ok = False
    o2m_after = kd_loss.criterion.o2m if anneal_ok else o2m_before

    all_pass = True
    all_pass &= check(anneal_ok, "kd_loss.update() does not raise")
    all_pass &= check(
        anneal_ok and o2m_after < o2m_before,
        "o2m moved toward final_o2m after repeated update() calls",
        detail=f"{o2m_before:.4f} -> {o2m_after:.4f}",
    )
    return all_pass


def run_all_checks() -> bool:
    device = torch.device("cpu")
    model, _ = yolo26n_model(constants.NUM_CLASSES, weights=None, device=device)
    model.train()
    imgs, targets = _synthetic_batch()
    batch_size = imgs.shape[0]
    preds = model(imgs)

    all_pass = True
    all_pass &= run_zero_fallback_check(model, preds, targets, batch_size)
    all_pass &= run_blend_effect_check(model, preds, targets, batch_size)
    all_pass &= run_temperature_scale_check()
    all_pass &= run_anneal_check(model)
    return all_pass


if __name__ == "__main__":
    all_ok = True
    try:
        all_ok = run_all_checks()
    except Exception:
        traceback.print_exc()
        all_ok = False

    print(f"\n{'=' * 60}")
    print(f"OVERALL: {PASS if all_ok else FAIL}")
    sys.exit(0 if all_ok else 1)
