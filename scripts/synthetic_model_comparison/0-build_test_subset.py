#!/usr/bin/env python3
"""
Stage 0 — Build the 12-class model-comparison real test subset

Materializes the real test set defined in
docs/synthetic-model-comparison/02_class-selection.md §4a for the
synthetic-generator comparison experiment: for Band A classes, the
unmodified base `test` split; for Band B/D classes, the expanded
train+val+test pool (this experiment never trains on real images, so those
reserved images are otherwise idle and safe to fold into test). Copies the
matching image files into a new self-contained directory and writes a
matching COCO-format annotation file, plus the frozen class-list CSV that
§7 of that doc recommends.

Output lives under a `test/` subdir of data/synthetic_model_comparison/ so
that per-generator/per-prompt-regime synthetic train sets can later be added
as sibling `train/<generator>/<regime>/` subdirs of the same parent.

Usage:
    uv run python scripts/synthetic_model_comparison/0-build_test_subset.py
    uv run python scripts/synthetic_model_comparison/0-build_test_subset.py --dry-run

Outputs:
    data/synthetic_model_comparison/test/images/<class_slug>/*
    data/synthetic_model_comparison/test/annotations_test.json
    reports/model_comparison_classes.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_DIR = REPO_ROOT / "data" / "real"
OUT_DIR = REPO_ROOT / "data" / "synthetic_model_comparison" / "test"
OUT_IMAGES_DIR = OUT_DIR / "images"
OUT_ANNOTATIONS_PATH = OUT_DIR / "annotations_test.json"
CLASSES_CSV_PATH = REPO_ROOT / "reports" / "model_comparison_classes.csv"

# The 12 classes frozen in docs/synthetic-model-comparison/02_class-selection.md §4,
# with their band (determines which splits get folded into this experiment's test set).
CLASS_BANDS = {
    "plains zebra": "D",
    "grevy's zebra": "B",
    "mountain zebra": "D",
    "red fox": "D",
    "american black bear": "D",
    "lion": "D",
    "kinkajou": "A",
    "water deer": "A",
    "ringtail": "A",
    "saiga": "A",
    "aye-aye": "A",
    "pangolin family": "A",
}

# Band A: unmodified base test split only. Band B/D: expanded train+val+test pool.
BAND_A_SPLITS = ["test"]
EXPANDED_SPLITS = ["train", "val", "test"]


def slugify(class_name: str) -> str:
    return class_name.replace(" ", "_")


def splits_for_band(band: str) -> list[str]:
    return BAND_A_SPLITS if band == "A" else EXPANDED_SPLITS


def load_split(split: str) -> dict:
    with open(REAL_DIR / f"annotations_{split}.json", encoding="utf-8") as f:
        return json.load(f)


def collect_per_class_records(splits: dict[str, dict]) -> dict[str, list[tuple[str, dict, list[dict]]]]:
    """For each class, gather (split_name, image_dict, [annotation_dicts]) tuples
    from every split that class needs, using the source data's own category ids."""
    category_name_to_id = {c["name"]: c["id"] for c in next(iter(splits.values()))["categories"]}
    for split_name, coco in splits.items():
        this_split_ids = {c["name"]: c["id"] for c in coco["categories"]}
        assert this_split_ids == category_name_to_id, (
            f"category id/name mismatch between splits (found in {split_name})"
        )

    records: dict[str, list[tuple[str, dict, list[dict]]]] = {name: [] for name in CLASS_BANDS}

    for split_name, coco in splits.items():
        images_by_id = {im["id"]: im for im in coco["images"]}
        anns_by_image_by_category: dict[int, dict[int, list[dict]]] = {}
        for ann in coco["annotations"]:
            per_category = anns_by_image_by_category.setdefault(ann["category_id"], {})
            per_category.setdefault(ann["image_id"], []).append(ann)

        for class_name, band in CLASS_BANDS.items():
            if split_name not in splits_for_band(band):
                continue
            category_id = category_name_to_id[class_name]
            for image_id, anns in anns_by_image_by_category.get(category_id, {}).items():
                records[class_name].append((split_name, images_by_id[image_id], anns))

    return records


def build_subset(
    records: dict[str, list[tuple[str, dict, list[dict]]]],
    categories: list[dict],
    dry_run: bool,
) -> tuple[dict, list[tuple[str, int, int]]]:
    """Copy images and assign fresh sequential ids. Returns (coco_dict, per-class summary rows)."""
    category_name_to_dict = {c["name"]: c for c in categories}

    out_categories = [category_name_to_dict[name] for name in CLASS_BANDS]
    out_images: list[dict] = []
    out_annotations: list[dict] = []
    summary_rows: list[tuple[str, int, int]] = []

    next_image_id = 1
    next_ann_id = 1

    for class_name in CLASS_BANDS:
        class_slug = slugify(class_name)
        dest_dir = OUT_IMAGES_DIR / class_slug
        if not dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)

        seen_basenames: dict[str, str] = {}  # basename -> source file_name, to catch collisions
        class_image_count = 0
        class_byte_count = 0

        for split_name, image, anns in tqdm(
            records[class_name], desc=f"{class_name} ({class_slug})", leave=False
        ):
            src_path = REPO_ROOT / image["file_name"]
            basename = src_path.name
            if basename in seen_basenames and seen_basenames[basename] != image["file_name"]:
                raise RuntimeError(
                    f"Filename collision for class '{class_name}': "
                    f"'{basename}' comes from both '{seen_basenames[basename]}' and "
                    f"'{image['file_name']}'"
                )
            seen_basenames[basename] = image["file_name"]

            dest_path = dest_dir / basename
            if not dry_run:
                shutil.copy2(src_path, dest_path)
            class_byte_count += src_path.stat().st_size
            class_image_count += 1

            new_image = dict(image)
            new_image["id"] = next_image_id
            new_image["file_name"] = str(dest_path.relative_to(REPO_ROOT))
            out_images.append(new_image)

            for ann in anns:
                new_ann = dict(ann)
                new_ann["id"] = next_ann_id
                new_ann["image_id"] = next_image_id
                out_annotations.append(new_ann)
                next_ann_id += 1

            next_image_id += 1

        summary_rows.append((class_name, class_image_count, class_byte_count))

    coco = {
        "info": {
            "description": (
                "12-class model-comparison expanded real test subset "
                "(docs/synthetic-model-comparison/02_class-selection.md §4a)"
            ),
            "date_created": "2026-07-16",
            "version": "1.0",
        },
        "licenses": [],
        "categories": out_categories,
        "images": out_images,
        "annotations": out_annotations,
    }
    return coco, summary_rows


def write_classes_csv(summary_rows: list[tuple[str, int, int]]) -> None:
    with open(CLASSES_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "band", "exp_test_count"])
        for class_name, count, _ in summary_rows:
            writer.writerow([class_name, CLASS_BANDS[class_name], count])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print the plan without copying files or writing outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    splits = {split: load_split(split) for split in EXPANDED_SPLITS}
    categories = splits["test"]["categories"]

    records = collect_per_class_records(splits)
    coco, summary_rows = build_subset(records, categories, dry_run=args.dry_run)

    print(f"\n{'class':22s} {'band':4s} {'images':>8s} {'size (MB)':>10s}")
    total_images = 0
    total_bytes = 0
    for class_name, count, byte_count in summary_rows:
        print(f"{class_name:22s} {CLASS_BANDS[class_name]:4s} {count:8d} {byte_count / 1e6:10.1f}")
        total_images += count
        total_bytes += byte_count
    print(f"{'TOTAL':22s} {'':4s} {total_images:8d} {total_bytes / 1e6:10.1f}")

    if args.dry_run:
        print("\n--dry-run: no files copied, no JSON/CSV written.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_ANNOTATIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)
    write_classes_csv(summary_rows)

    print(f"\nWrote {OUT_ANNOTATIONS_PATH.relative_to(REPO_ROOT)}")
    print(f"Wrote {CLASSES_CSV_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
