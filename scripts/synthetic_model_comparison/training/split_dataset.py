"""Carve a stratified internal train/val split out of one cell's annotations.json.

This experiment has no dedicated validation set — each generator x
prompt-regime cell exports a single train-only annotations.json (see
scripts/synthetic_model_comparison/5-export_coco.py). Per-epoch model
selection / early stopping still needs *some* held-out data, but it must not
be the fixed real test set (data/synthetic_model_comparison/test/annotations_test.json) —
see docs/synthetic-model-comparison/11_detector-architecture-selection.md §7
for why: evaluating the ~9,742-image real test set every epoch would dominate
wall-clock for no benefit, and it would mean model selection peeks at the
same set used for the headline metric.

Instead, this module splits each cell's own synthetic images into an
internal train/val pair, stratified per class, with a fixed SPLIT_SEED
independent of the training-run SEED — so the split stays identical across
the >=3 training seeds recommended per cell
(docs/synthetic-model-comparison/06_evaluation-methodology.md), and only the
model's random init / dataloader shuffling varies between seeds.

Usage (also invoked automatically by run_training_pipeline.py if the split
files are missing or stale):
    uv run python -m scripts.synthetic_model_comparison.training.split_dataset \\
        --generator gemini-3.1-flash-image-preview --prompt-regime full
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import scripts.synthetic_model_comparison.training.constants as constants

logger = logging.getLogger(__name__)


def _stratify_key(image: dict, anns_by_image_id: dict[int, list[dict]]) -> int | str:
    """Group images by their (single) category id; images with zero
    annotations — e.g. classes with no images yet in a partially-populated
    cell — are stratified into their own bucket so they never crash the
    split, they just never get selected into either half."""
    anns = anns_by_image_id.get(image["id"], [])
    if not anns:
        return "unlabeled"
    return anns[0]["category_id"]


def split_coco(coco: dict, val_fraction: float, seed: int) -> tuple[dict, dict]:
    """Return (train_coco, val_coco) — same categories, disjoint images/annotations."""
    anns_by_image_id: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        anns_by_image_id.setdefault(ann["image_id"], []).append(ann)

    groups: dict[int | str, list[dict]] = {}
    for img in coco["images"]:
        key = _stratify_key(img, anns_by_image_id)
        groups.setdefault(key, []).append(img)

    rng = random.Random(seed)
    train_images: list[dict] = []
    val_images: list[dict] = []
    for key, imgs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        imgs_sorted = sorted(imgs, key=lambda im: im["id"])  # deterministic order pre-shuffle
        shuffled = imgs_sorted[:]
        rng.shuffle(shuffled)
        n_val = round(len(shuffled) * val_fraction)
        val_images.extend(shuffled[:n_val])
        train_images.extend(shuffled[n_val:])

    def _build(images: list[dict]) -> dict:
        image_ids = {img["id"] for img in images}
        annotations = [a for a in coco["annotations"] if a["image_id"] in image_ids]
        return {
            "info": coco.get("info", {}),
            "licenses": coco.get("licenses", []),
            "categories": coco["categories"],
            "images": images,
            "annotations": annotations,
        }

    return _build(train_images), _build(val_images)


def split_cell(cell_dir: Path, force: bool = False) -> tuple[Path, Path]:
    src = cell_dir / "annotations.json"
    train_out = cell_dir / "annotations_train_split.json"
    val_out = cell_dir / "annotations_val_split.json"

    if not src.exists():
        raise SystemExit(
            f"ERROR: {src} not found — run "
            "scripts/synthetic_model_comparison/5-export_coco.py for this cell first"
        )

    if (
        not force
        and train_out.exists()
        and val_out.exists()
        and train_out.stat().st_mtime >= src.stat().st_mtime
        and val_out.stat().st_mtime >= src.stat().st_mtime
    ):
        logger.info("split already up to date: %s / %s", train_out.name, val_out.name)
        return train_out, val_out

    with open(src, encoding="utf-8") as f:
        coco = json.load(f)

    train_coco, val_coco = split_coco(coco, constants.VAL_FRACTION, constants.SPLIT_SEED)

    with open(train_out, "w", encoding="utf-8") as f:
        json.dump(train_coco, f, indent=2)
    with open(val_out, "w", encoding="utf-8") as f:
        json.dump(val_coco, f, indent=2)

    logger.info(
        "split %s: %d images -> %d train / %d val (val_fraction=%.2f, seed=%d)",
        src.name,
        len(coco["images"]),
        len(train_coco["images"]),
        len(val_coco["images"]),
        constants.VAL_FRACTION,
        constants.SPLIT_SEED,
    )

    per_class_train: dict[str, int] = {}
    per_class_val: dict[str, int] = {}
    cat_names = {c["id"]: c["name"] for c in coco["categories"]}
    for ann in train_coco["annotations"]:
        name = cat_names.get(ann["category_id"], str(ann["category_id"]))
        per_class_train[name] = per_class_train.get(name, 0) + 1
    for ann in val_coco["annotations"]:
        name = cat_names.get(ann["category_id"], str(ann["category_id"]))
        per_class_val[name] = per_class_val.get(name, 0) + 1
    for name in sorted(cat_names.values()):
        logger.info(
            "  %-25s train=%3d  val=%3d", name, per_class_train.get(name, 0), per_class_val.get(name, 0)
        )

    return train_out, val_out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", required=True)
    parser.add_argument("--prompt-regime", required=True, choices=["full", "compressed"])
    parser.add_argument("--force", action="store_true", help="Rebuild the split even if up to date")
    args = parser.parse_args()

    cell = constants.cell_dir(args.generator, args.prompt_regime)
    split_cell(cell, force=args.force)


if __name__ == "__main__":
    main()
