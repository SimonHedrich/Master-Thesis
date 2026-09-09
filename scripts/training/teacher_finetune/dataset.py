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

# The 4 known data bands (see data/real/annotations_*.json's image["band"]
# field) — every non-smoke, full train/val dataset must cover all four.
KNOWN_BANDS = {"A", "B", "C", "D"}


class SpeciesNetCropDataset(Dataset):
    def __init__(
        self,
        annotations_path: Path | list[Path],
        image_root: Path,
        preprocess_fn: Callable[[Image.Image, list], "np.ndarray | None"],
        check_coverage: bool = False,
    ) -> None:
        paths = [annotations_path] if isinstance(annotations_path, Path) else list(annotations_path)

        self.image_root = image_root
        self.preprocess_fn = preprocess_fn

        self.class_names: list[str] | None = None
        self.samples: list[tuple[str, list, int, int, int]] = []
        # Parallel array (same index as `samples`) — used only by evaluate.py's
        # per-source accuracy breakdown, so training's __getitem__ doesn't pay
        # for it. Relies on eval dataloaders using shuffle=False (the existing
        # convention), so batch order matches this list's order.
        self.sources: list[str] = []

        # Coverage bookkeeping (only populated when check_coverage=True) —
        # guards against a repeat of the Band-A bug: a whole class/band
        # silently getting zero images because a source file wasn't merged in.
        cat_ids_with_images: set[int] = set()
        bands_present: set[str] = set()

        # Each source file is resolved (image_id -> image record) independently
        # rather than merged into one global dict, so per-file id numbering
        # (e.g. real vs. synthetic COCO JSONs both starting at id=1) can never
        # collide — `samples`/`sources` are flat, position-based lists that
        # don't retain any notion of image id past this loop.
        for path in paths:
            with open(path) as f:
                coco = json.load(f)

            # COCO category ids are 1-based, in classes_225.csv row order —
            # same convention `category_id - 1 == idx_225` used by the
            # detector pipelines. Every source must share this taxonomy.
            cats = [c["name"] for c in sorted(coco["categories"], key=lambda c: c["id"])]
            if self.class_names is None:
                self.class_names = cats
            elif cats != self.class_names:
                raise ValueError(f"{path}: category table does not match the first source file")

            images_by_id = {img["id"]: img for img in coco["images"]}
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
                if check_coverage:
                    cat_ids_with_images.add(ann["category_id"])
                    if image.get("band") in KNOWN_BANDS:
                        bands_present.add(image["band"])
            if skipped:
                logger.warning(
                    "dataset %s: skipped %d annotations with no matching image",
                    path.name,
                    skipped,
                )

        logger.info(
            "dataset [%s]: %d crops, %d classes (merged)",
            ", ".join(p.name for p in paths),
            len(self.samples),
            len(self.class_names),
        )

        if check_coverage:
            source_desc = ", ".join(p.name for p in paths)
            all_cat_ids = set(range(1, len(self.class_names) + 1))
            missing_classes = sorted(all_cat_ids - cat_ids_with_images)
            if missing_classes:
                raise ValueError(
                    f"coverage check failed for [{source_desc}]: {len(missing_classes)} "
                    f"class(es) have zero crops after merging — category ids {missing_classes}"
                )
            missing_bands = KNOWN_BANDS - bands_present
            if missing_bands:
                raise ValueError(
                    f"coverage check failed for [{source_desc}]: data band(s) "
                    f"{sorted(missing_bands)} entirely absent from {len(self.samples)} "
                    "merged crops — a source annotation file is likely missing from the merge"
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
