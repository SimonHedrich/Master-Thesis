"""COCO JSON → SpeciesNet-crop dataset for classifier fine-tuning.

Reads `data/real/annotations_{split}.json` directly — **not**
`filter_results.jsonl` (see README.md's "Deviations from the detector
pipelines" section for the rationale: `annotations_*.json` is downstream of
the contamination-review pipeline and is the exact file every other model in
the comparison matrix trains against, so reusing it here guarantees identical
`(image, bbox, label)` triples, not just an identical split).

Each annotation's absolute-pixel COCO bbox `[x, y, w, h]` is normalized by its
image's `width`/`height` (already present in the `images` list) before being
handed to `preprocess_fn` — ultimately
`SpeciesNetClassifier.preprocess_crop()`'s `BBox(*bbox_norm)` + `clf.preprocess()`
call, injected from `teacher_model.py` so this module has no direct
`speciesnet` import and stays unit-testable without the package installed.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class SpeciesNetCropDataset(Dataset):
    def __init__(
        self,
        annotations_path: Path,
        image_root: Path,
        preprocess_fn: Callable[[Image.Image, list], "np.ndarray | None"],
    ) -> None:
        with open(annotations_path) as f:
            coco = json.load(f)

        self.image_root = image_root
        self.preprocess_fn = preprocess_fn

        images_by_id = {img["id"]: img for img in coco["images"]}
        # COCO category ids are 1-based, in classes_225.csv row order — same
        # convention `category_id - 1 == idx_225` used by the detector pipelines.
        self.class_names: list[str] = [
            c["name"] for c in sorted(coco["categories"], key=lambda c: c["id"])
        ]

        self.samples: list[tuple[str, list, int, int, int]] = []
        # Parallel array (same index as `samples`) — used only by evaluate.py's
        # per-source accuracy breakdown, so training's __getitem__ doesn't pay
        # for it. Relies on eval dataloaders using shuffle=False (the existing
        # convention), so batch order matches this list's order.
        self.sources: list[str] = []
        skipped = 0
        for ann in coco["annotations"]:
            image = images_by_id.get(ann["image_id"])
            if image is None:
                skipped += 1
                continue
            self.samples.append(
                (
                    image["file_name"],
                    ann["bbox"],  # absolute pixel COCO [x, y, w, h]
                    ann["category_id"],
                    image["width"],
                    image["height"],
                )
            )
            self.sources.append(image.get("source", "unknown"))
        if skipped:
            logger.warning(
                "dataset %s: skipped %d annotations with no matching image",
                annotations_path.name,
                skipped,
            )

        logger.info(
            "dataset %s: %d crops, %d classes",
            annotations_path.name,
            len(self.samples),
            len(self.class_names),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        file_name, bbox_px, category_id, width, height = self.samples[idx]
        path = self.image_root / file_name

        x, y, w, h = bbox_px
        bbox_norm = [x / width, y / height, w / width, h / height]

        with Image.open(path) as raw:
            img = raw.convert("RGB")
            arr = self.preprocess_fn(img, bbox_norm)
            if arr is None:
                # SpeciesNet's preprocess() considers the crop invalid (e.g. a
                # degenerate bbox). Fall back to a full-image crop rather than
                # silently dropping the sample and shrinking the batch mid-epoch.
                arr = self.preprocess_fn(img, [0.0, 0.0, 1.0, 1.0])
                if arr is None:
                    raise RuntimeError(
                        f"preprocess_fn returned None even for a full-image crop: {path}"
                    )

        idx_225 = category_id - 1
        return torch.from_numpy(arr), idx_225


def collate_fn(batch: list) -> tuple[torch.Tensor, torch.Tensor]:
    arrs = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return arrs, labels
