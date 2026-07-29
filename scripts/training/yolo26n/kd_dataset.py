"""KD dataset wrapper: attaches a per-image teacher soft-label vector to
CocoYoloDataset via composition (base augmentation/letterbox path untouched).

Goal B is scoped to single-label KD (one teacher distribution per image, not
per animal instance) — see
docs/plans/2026-06-30_yolo26-kd-and-teacher-finetune-implementation-plan.md §3.
Multi-animal pseudo-GT is a documented follow-on, not built here.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from scripts.training.yolov5s.dataset import CocoYoloDataset
from scripts.training.yolov5s.dataset import collate_fn as base_collate_fn

logger = logging.getLogger(__name__)


def _load_teacher_cache(path: Path, num_classes: int) -> dict[str, torch.Tensor]:
    """Load a `teacher_soft_labels_*.jsonl` cache produced by
    `scripts/training/teacher_finetune/cache_soft_labels.py`.

    The cache is per-annotation (`{"filepath", "detection_idx", "probs_225",
    "prob_225_sum"}`) — an image with 2 annotated animals produces 2 records
    at detection_idx 0 and 1. Only detection_idx == 0 (the primary/first
    annotation for that image) is kept, matching this dataset's one-vector-
    per-image contract.
    """
    cache: dict[str, torch.Tensor] = {}
    n_secondary = 0
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if rec["detection_idx"] != 0:
                n_secondary += 1
                continue
            probs = rec["probs_225"]
            if len(probs) != num_classes:
                raise ValueError(
                    f"{path}: probs_225 length {len(probs)} != num_classes {num_classes}"
                )
            cache[rec["filepath"]] = torch.tensor(probs, dtype=torch.float32)
    logger.info(
        "loaded teacher soft-label cache %s: %d images (%d secondary/non-primary "
        "detections discarded — single-label KD scope)",
        path,
        len(cache),
        n_secondary,
    )
    return cache


class KDCocoYoloDataset:
    """Wraps a `CocoYoloDataset` instance by composition — delegates every
    image-level concern (augmentation, letterbox, close-mosaic epoch state,
    class names) to the base instance, and adds a `teacher_probs` tensor
    (`[num_classes]`, zeros if the image has no cached teacher record) to
    each item.
    """

    def __init__(
        self, base: CocoYoloDataset, teacher_cache_path: Path, num_classes: int
    ) -> None:
        self.base = base
        self.num_classes = num_classes
        self._teacher_probs = _load_teacher_cache(teacher_cache_path, num_classes)

        n_missing = sum(
            1 for img in base.images if img["file_name"] not in self._teacher_probs
        )
        logger.info(
            "%d/%d training images have NO cached teacher record — teacher_probs "
            "will be all-zero for those (hard-label-only fallback)",
            n_missing,
            len(base.images),
        )

    def __len__(self) -> int:
        return len(self.base)

    def set_epoch(self, *args, **kwargs) -> None:
        set_epoch = getattr(self.base, "set_epoch", None)
        if set_epoch is not None:
            set_epoch(*args, **kwargs)

    @property
    def class_names(self) -> list[str]:
        return self.base.class_names

    def __getitem__(self, idx: int):
        img, targets, path, shapes = self.base[idx]
        file_name = self.base.images[idx]["file_name"]
        teacher_probs = self._teacher_probs.get(
            file_name, torch.zeros(self.num_classes, dtype=torch.float32)
        )
        return img, targets, teacher_probs, path, shapes


def kd_collate_fn(batch: list):
    imgs, targets_list, teacher_probs_list, paths, shapes = zip(*batch)
    imgs_t, targets_t, paths_t, shapes_t = base_collate_fn(
        list(zip(imgs, targets_list, paths, shapes))
    )
    teacher_probs_t = torch.stack(teacher_probs_list, dim=0)
    return imgs_t, targets_t, teacher_probs_t, paths_t, shapes_t
