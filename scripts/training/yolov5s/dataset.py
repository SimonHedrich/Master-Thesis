"""COCO JSON → YOLOv5-format dataset and dataloader wrapper."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader, Dataset

from scripts.training.yolov5s import transforms

logger = logging.getLogger(__name__)


class CocoYoloDataset(Dataset):
    def __init__(self, annotations_path: Path, image_root: Path, image_size: int) -> None:
        with open(annotations_path) as f:
            coco = json.load(f)

        self.image_root = image_root
        self.image_size = image_size

        self.images: list[dict] = coco["images"]
        self.anns_by_image_id: dict[int, list[dict]] = {}
        for ann in coco["annotations"]:
            self.anns_by_image_id.setdefault(ann["image_id"], []).append(ann)

        # COCO ids 1..225 → YOLO indices 0..224
        sorted_cats = sorted(coco["categories"], key=lambda c: c["id"])
        self.cat_id_to_yolo: dict[int, int] = {c["id"]: i for i, c in enumerate(sorted_cats)}
        self.class_names: list[str] = [c["name"] for c in sorted_cats]

        logger.info(
            "dataset %s: %d images, %d annotations, %d classes",
            annotations_path.name,
            len(self.images),
            len(coco["annotations"]),
            len(self.class_names),
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str, tuple]:
        record = self.images[idx]
        file_name = record["file_name"]
        path = self.image_root / file_name

        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"cv2.imread returned None for: {path}")

        h0, w0 = img.shape[:2]
        img, (r, _), (dw, dh) = transforms.letterbox(img, new_shape=self.image_size)

        anns = self.anns_by_image_id.get(record["id"], [])
        rows: list[list[float]] = []
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
            # column 0 is a batch-index placeholder; collate_fn fills it
            rows.append([0.0, float(cls_idx), cx, cy, w_norm, h_norm])

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


class Dataloader:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        collate_fn=collate_fn,
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
        )

    def get_dataloader(self) -> DataLoader:
        return self._loader
