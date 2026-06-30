"""COCO JSON → YOLOv5-format dataset and dataloader wrapper."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import scripts.training.yolov5s.constants as constants
from scripts.training.yolov5s import transforms

logger = logging.getLogger(__name__)

# Emitted at most once per process for the copy-paste no-op warning.
_COPY_PASTE_WARNED = False


class CocoYoloDataset(Dataset):
    def __init__(
        self,
        annotations_path: Path,
        image_root: Path,
        image_size: int,
        augment: bool = False,
    ) -> None:
        with open(annotations_path) as f:
            coco = json.load(f)

        self.image_root = image_root
        self.image_size = image_size
        self.augment = augment

        self.images: list[dict] = coco["images"]
        self.anns_by_image_id: dict[int, list[dict]] = {}
        for ann in coco["annotations"]:
            self.anns_by_image_id.setdefault(ann["image_id"], []).append(ann)

        # COCO ids 1..225 → YOLO indices 0..224
        sorted_cats = sorted(coco["categories"], key=lambda c: c["id"])
        self.cat_id_to_yolo: dict[int, int] = {c["id"]: i for i, c in enumerate(sorted_cats)}
        self.class_names: list[str] = [c["name"] for c in sorted_cats]

        # Close-mosaic epoch tracking (set_epoch updates these each epoch)
        self._current_epoch: int = 0
        self._total_epochs: int = 0

        logger.info(
            "dataset %s: %d images, %d annotations, %d classes",
            annotations_path.name,
            len(self.images),
            len(coco["annotations"]),
            len(self.class_names),
        )

    def set_epoch(self, epoch: int, total_epochs: int) -> None:
        """Inform the dataset of the current training epoch.

        Used by the close-mosaic tail: when
        ``epoch >= total_epochs - AUG_CLOSE_MOSAIC``, mosaic and mixup are
        suppressed so the model adapts to single-image statistics in the
        final epochs.  Safe to call even when compositing is disabled
        (AUG_MOSAIC == 0).
        """
        self._current_epoch = epoch
        self._total_epochs = total_epochs

    def _compositing_active(self) -> bool:
        """Return True when mosaic/mixup should be used this epoch."""
        close = int(constants.AUG_CLOSE_MOSAIC)
        if close > 0 and self._total_epochs > 0:
            if self._current_epoch >= self._total_epochs - close:
                return False
        return True

    def _load_raw(self, idx: int) -> tuple[np.ndarray, np.ndarray, int, int]:
        """Load a raw (non-letterboxed) image and its normalised labels.

        Returns
        -------
        img    : HWC BGR uint8
        labels : (N, 5) float32 [cls, cx, cy, w, h] normalised to [0,1]
        h0, w0 : original image dimensions (for shapes metadata)
        """
        record = self.images[idx]
        path = self.image_root / record["file_name"]
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"cv2.imread returned None for: {path}")
        h0, w0 = img.shape[:2]

        anns = self.anns_by_image_id.get(record["id"], [])
        raw_labels: list[list[float]] = []
        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            cx = (x + bw / 2) / w0
            cy = (y + bh / 2) / h0
            bw_n = bw / w0
            bh_n = bh / h0
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            bw_n = max(0.0, min(1.0, bw_n))
            bh_n = max(0.0, min(1.0, bh_n))
            if bw_n <= 0 or bh_n <= 0:
                continue
            cls_idx = self.cat_id_to_yolo[ann["category_id"]]
            raw_labels.append([float(cls_idx), cx, cy, bw_n, bh_n])

        labels = (
            np.array(raw_labels, dtype=np.float32)
            if raw_labels
            else np.zeros((0, 5), dtype=np.float32)
        )
        return img, labels, h0, w0

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str, tuple]:
        global _COPY_PASTE_WARNED

        record = self.images[idx]
        path = self.image_root / record["file_name"]

        # Deterministic per-sample RNG (same seed discipline as augment_basic)
        rng = np.random.default_rng(abs(hash((constants.SEED, idx))) % (2**32))

        # ── Decide whether to run mosaic this sample ───────────────────────────
        use_mosaic = (
            self.augment
            and constants.AUG_MOSAIC > 0.0
            and self._compositing_active()
            and rng.random() < constants.AUG_MOSAIC
        )

        # ── Copy-paste no-op warning (once per process) ───────────────────────
        if constants.AUG_COPY_PASTE > 0.0 and not _COPY_PASTE_WARNED:
            _COPY_PASTE_WARNED = True
            logger.warning(
                "AUG_COPY_PASTE=%.3f is set but copy-paste is a NO-OP: "
                "it requires instance masks, and our GT is box-only. "
                "No data transformation will occur.",
                constants.AUG_COPY_PASTE,
            )

        if use_mosaic:
            # ── Setup-C path: mosaic ──────────────────────────────────────────
            # Pick 3 additional random indices (may repeat; mirrors yolov5).
            n = len(self.images)
            extra_ids = [int(rng.integers(0, n)) for _ in range(3)]
            mosaic_indices = [idx] + extra_ids

            # Load all 4 raw images; stash primary image's original dimensions
            # for the shapes metadata tuple returned by __getitem__.
            raw0 = self._load_raw(mosaic_indices[0])
            h0, w0 = raw0[2], raw0[3]
            samples = [raw0[:2]] + [self._load_raw(i)[:2] for i in mosaic_indices[1:]]

            img, labels_arr = transforms.build_mosaic(
                samples,
                self.image_size,
                rng,
                degrees=constants.AUG_DEGREES,
                translate=constants.AUG_TRANSLATE,
                scale=constants.AUG_SCALE,
                shear=constants.AUG_SHEAR,
                perspective=constants.AUG_PERSPECTIVE,
            )
            # img is now (IMAGE_SIZE, IMAGE_SIZE, 3); labels_arr is normalised xywh.
            # Mosaic's random_perspective already handled scale/translate → only
            # apply HSV jitter + hflip from the basic set (no scale/translate).
            img, labels_arr = transforms.augment_basic(
                img,
                labels_arr,
                hflip=constants.AUG_HFLIP,
                hsv=constants.AUG_HSV,
                hsv_h=constants.AUG_HSV_H,
                hsv_s=constants.AUG_HSV_S,
                hsv_v=constants.AUG_HSV_V,
                scale=0.0,        # already applied inside build_mosaic
                translate=0.0,    # already applied inside build_mosaic
                degrees=0.0,
                shear=0.0,
                perspective=0.0,
                rng=rng,
            )

            # ── Optional MixUp on top of mosaic ──────────────────────────────
            use_mixup = (
                constants.AUG_MIXUP > 0.0
                and self._compositing_active()
                and rng.random() < constants.AUG_MIXUP
            )
            if use_mixup:
                # Build a second mosaic to blend with
                extra2_ids = [int(rng.integers(0, n)) for _ in range(4)]
                samples2 = [self._load_raw(i)[:2] for i in extra2_ids]
                # Use arithmetic seed derivation (no Python hash, no PYTHONHASHSEED
                # dependency) so two separate processes with the same SEED produce
                # byte-identical augmentation (plan §5.2).
                # The constant 1_000_003 is a large prime; multiplier 2 separates
                # the mixup-seed space from the primary-rng seed space.
                rng2 = np.random.default_rng((constants.SEED * 1_000_003 + idx * 2 + 1) % (2**32))
                img2, labels2 = transforms.build_mosaic(
                    samples2,
                    self.image_size,
                    rng2,
                    degrees=constants.AUG_DEGREES,
                    translate=constants.AUG_TRANSLATE,
                    scale=constants.AUG_SCALE,
                    shear=constants.AUG_SHEAR,
                    perspective=constants.AUG_PERSPECTIVE,
                )
                img, labels_arr = transforms.mixup(img, labels_arr, img2, labels2, rng)

            # h0, w0 captured above from raw0; mosaic has no single scale factor
            r = 1.0
            dw = 0.0
            dh = 0.0

        else:
            # ── Setup A/B path (or mosaic probability miss): single-image ─────
            img = cv2.imread(str(path))
            if img is None:
                raise FileNotFoundError(f"cv2.imread returned None for: {path}")

            h0, w0 = img.shape[:2]
            img, (r, _), (dw, dh) = transforms.letterbox(img, new_shape=self.image_size)

            anns = self.anns_by_image_id.get(record["id"], [])
            raw_labels: list[list[float]] = []  # [cls, cx, cy, w_norm, h_norm]
            for ann in anns:
                x, y, w, h = ann["bbox"]
                x_new = x * r + dw / 2
                y_new = y * r + dh / 2
                w_new = w * r
                h_new = h * r
                cx = (x_new + w_new / 2) / self.image_size
                cy = (y_new + h_new / 2) / self.image_size
                w_norm = w_new / self.image_size
                h_norm = h_new / self.image_size
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                w_norm = max(0.0, min(1.0, w_norm))
                h_norm = max(0.0, min(1.0, h_norm))
                if w_norm <= 0 or h_norm <= 0:
                    continue
                cls_idx = self.cat_id_to_yolo[ann["category_id"]]
                raw_labels.append([float(cls_idx), cx, cy, w_norm, h_norm])

            labels_arr = (
                np.array(raw_labels, dtype=np.float32)
                if raw_labels
                else np.zeros((0, 5), dtype=np.float32)
            )

            if self.augment:
                # Seed deterministically per sample; augment_hsv will still use
                # global np.random (seeded by worker_init_fn for reproducibility).
                img, labels_arr = transforms.augment_basic(
                    img,
                    labels_arr,
                    hflip=constants.AUG_HFLIP,
                    hsv=constants.AUG_HSV,
                    hsv_h=constants.AUG_HSV_H,
                    hsv_s=constants.AUG_HSV_S,
                    hsv_v=constants.AUG_HSV_V,
                    scale=constants.AUG_SCALE,
                    translate=constants.AUG_TRANSLATE,
                    degrees=constants.AUG_DEGREES,
                    shear=constants.AUG_SHEAR,
                    perspective=constants.AUG_PERSPECTIVE,
                    rng=rng,
                )

        # ── Finalise labels → target tensor ───────────────────────────────────
        # `use_mosaic` implies `self.augment`, so both cases that went through
        # any augmentation share the clip+degenerate-box guard below.
        # The plain `else` (no augmentation at all) skips it for speed.
        rows: list[list[float]] = []
        if self.augment:
            for label in labels_arr:
                cls_i, cx, cy, w_n, h_n = label
                cx = float(np.clip(cx, 0.0, 1.0))
                cy = float(np.clip(cy, 0.0, 1.0))
                w_n = float(np.clip(w_n, 0.0, 1.0))
                h_n = float(np.clip(h_n, 0.0, 1.0))
                if w_n <= 0 or h_n <= 0:
                    continue
                rows.append([0.0, float(cls_i), cx, cy, w_n, h_n])
        else:
            # No augmentation — pass raw_labels through directly
            rows = [[0.0] + lbl for lbl in labels_arr.tolist() if lbl[3] > 0 and lbl[4] > 0]

        if rows:
            targets = torch.tensor(rows, dtype=torch.float32)
        else:
            targets = torch.zeros((0, 6), dtype=torch.float32)

        image_tensor = transforms.to_tensor(img)
        shapes = ((h0, w0), ((r, r), (dw, dh)))
        return image_tensor, targets, str(path), shapes


def collate_fn(batch: list) -> tuple[torch.Tensor, torch.Tensor, list, list]:
    imgs = torch.stack([b[0] for b in batch])
    target_parts: list[torch.Tensor] = []
    for i, b in enumerate(batch):
        t = b[1]
        if t.shape[0] > 0:
            t = t.clone()
            t[:, 0] = i
        target_parts.append(t)
    targets = torch.cat(target_parts, dim=0)
    paths = [b[2] for b in batch]
    shapes = [b[3] for b in batch]
    return imgs, targets, paths, shapes


def make_worker_init_fn(seed: int) -> Callable[[int], None]:
    """Return a worker_init_fn that seeds numpy/random/torch per worker."""

    def worker_init_fn(worker_id: int) -> None:
        import random as py_random

        worker_seed = seed + worker_id
        py_random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return worker_init_fn


class Dataloader:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        collate_fn: Callable = collate_fn,
        worker_init_fn: Callable[[int], None] | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        self._loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            drop_last=shuffle,
            worker_init_fn=worker_init_fn,
            generator=generator,
        )

    def get_dataloader(self) -> DataLoader:
        return self._loader
