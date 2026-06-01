#!/usr/bin/env python3
"""
Prepares the YOLOv5 training dataset from real + synthetic COCO annotations.

Reads data/real/annotations_{split}.json and data/synthetic/annotations_{split}.json
for each split, converts COCO [x,y,w,h] pixel bboxes to YOLO normalized [cx,cy,w,h],
optionally caps training images per class, and writes YOLO TXT label files +
image symlinks into data/training/yolov5/.

Note: the real split has already been capped at 1,500/class by the split assignment
script, so MAX_TRAIN_PER_CLASS is a safety guard that currently has no effect.

Outputs:
    data/training/yolov5/images/{train,val,test}/<16-char hex>.{ext}  (symlinks)
    data/training/yolov5/labels/{train,val,test}/<16-char hex>.txt
    data/training/wildlife225_yolov5.yaml
    data/training/yolov5/prep_stats.json
"""

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

REPO = Path("/home/debian/Master-Thesis")
OUT_DIR = REPO / "data" / "training" / "yolov5"
YAML_PATH = REPO / "data" / "training" / "wildlife225_yolov5.yaml"

SPLITS = ("train", "val", "test")
MAX_TRAIN_PER_CLASS = 1500
SEED = 42


def load_coco(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def coco_to_yolo(bbox: list, img_w: int, img_h: int) -> tuple:
    """COCO [x,y,w,h] pixels → YOLO normalized [cx,cy,w,h], clamped to [0,1]."""
    x, y, w, h = bbox
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    nw = w / img_w
    nh = h / img_h
    return (
        max(0.0, min(1.0, cx)),
        max(0.0, min(1.0, cy)),
        max(0.0, min(1.0, nw)),
        max(0.0, min(1.0, nh)),
    )


def make_stem(source: str, file_name: str) -> str:
    """Deterministic 16-char hex stem from (source, file_name)."""
    return hashlib.sha256(f"{source}:{file_name}".encode()).hexdigest()[:16]


def build_split_data(
    split: str,
    cat_id_to_yolo: dict,
    cap: int | None,
    rng: random.Random,
) -> tuple[dict, dict, list]:
    """
    Merge real + synthetic COCO for one split.

    Returns:
        img_meta  — stem → {abs_path, ext, file_name, source}
        img_labels — stem → list of YOLO annotation lines
        selected_stems — stems to actually write (after optional cap)
    """
    img_meta: dict = {}
    img_labels: dict = defaultdict(list)
    img_primary_class: dict = {}

    for source, base_dir in [("real", REPO / "data" / "real"),
                              ("synth", REPO / "data" / "synthetic")]:
        coco_path = base_dir / f"annotations_{split}.json"
        if not coco_path.exists():
            print(f"  [{split}/{source}] skip — {coco_path} not found")
            continue

        coco = load_coco(coco_path)

        anns_by_img: dict = defaultdict(list)
        for ann in coco["annotations"]:
            anns_by_img[ann["image_id"]].append(ann)

        dupes = 0
        for img in coco["images"]:
            file_name = img["file_name"]
            stem = make_stem(source, file_name)
            ext = Path(file_name).suffix.lower()
            abs_path = REPO / file_name

            if stem in img_meta:
                dupes += 1
            img_meta[stem] = {
                "abs_path": abs_path,
                "ext": ext,
                "file_name": file_name,
                "source": source,
            }

            for ann in anns_by_img.get(img["id"], []):
                yolo_cls = cat_id_to_yolo.get(ann["category_id"])
                if yolo_cls is None:
                    continue
                cx, cy, w, h = coco_to_yolo(
                    ann["bbox"], img["width"], img["height"]
                )
                img_labels[stem].append(
                    f"{yolo_cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                )
                if stem not in img_primary_class:
                    img_primary_class[stem] = yolo_cls

        if dupes:
            print(f"  [{split}/{source}] {dupes} duplicate file_names deduplicated (labels merged)")

    all_stems = list(img_meta.keys())

    if cap is None:
        return img_meta, dict(img_labels), all_stems

    # Per-class hard cap for the train split
    class_buckets: dict = defaultdict(list)
    background_stems = []
    for stem in all_stems:
        cls = img_primary_class.get(stem)
        if cls is None:
            background_stems.append(stem)
        else:
            class_buckets[cls].append(stem)

    selected = list(background_stems)
    for cls, stems in class_buckets.items():
        take = min(cap, len(stems))
        selected.extend(rng.sample(stems, take))

    return img_meta, dict(img_labels), selected


def write_split(split: str, img_meta: dict, img_labels: dict, stems: list) -> dict:
    img_dir = OUT_DIR / "images" / split
    lbl_dir = OUT_DIR / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    missing = 0
    for stem in stems:
        meta = img_meta[stem]
        abs_path = meta["abs_path"]
        ext = meta["ext"]

        link = img_dir / f"{stem}{ext}"
        if not link.exists():
            if abs_path.exists():
                link.symlink_to(abs_path)
            else:
                print(f"  [warn] source image missing: {abs_path}")
                missing += 1
                continue

        lines = img_labels.get(stem, [])
        lbl_file = lbl_dir / f"{stem}.txt"
        lbl_file.write_text("\n".join(lines) + ("\n" if lines else ""))

    return {
        "images": len(stems),
        "with_labels": sum(1 for s in stems if img_labels.get(s)),
        "missing_source": missing,
    }


def write_yaml(categories: list) -> None:
    names_yaml = "\n".join(f"  - {c['name']}" for c in categories)
    yaml = f"""\
# YOLOv5 dataset descriptor — wildlife {len(categories)}-class detection
# Auto-generated by scripts/training/1-prepare_yolov5_dataset.py
# Train/val use absolute paths for compatibility with YOLOv5@5cdad89

train: {OUT_DIR / 'images' / 'train'}
val:   {OUT_DIR / 'images' / 'val'}
test:  {OUT_DIR / 'images' / 'test'}

nc: {len(categories)}
names:
{names_yaml}
"""
    YAML_PATH.write_text(yaml)


def main() -> None:
    print("Loading categories from real train COCO…")
    real_train = load_coco(REPO / "data" / "real" / "annotations_train.json")
    categories = sorted(real_train["categories"], key=lambda c: c["id"])
    cat_id_to_yolo = {c["id"]: i for i, c in enumerate(categories)}
    print(f"  {len(categories)} categories (IDs {categories[0]['id']}–{categories[-1]['id']})")

    rng = random.Random(SEED)
    stats = {}

    for split in SPLITS:
        print(f"\nProcessing split: {split}")
        cap = MAX_TRAIN_PER_CLASS if split == "train" else None
        img_meta, img_labels, stems = build_split_data(split, cat_id_to_yolo, cap, rng)
        split_stats = write_split(split, img_meta, img_labels, stems)
        stats[split] = split_stats
        print(f"  {split_stats['images']} images, "
              f"{split_stats['with_labels']} with annotations, "
              f"{split_stats['missing_source']} missing source files")

    write_yaml(categories)
    print(f"\nDataset YAML written to {YAML_PATH}")

    stats_path = OUT_DIR / "prep_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"Stats written to {stats_path}")

    # Quick sanity check: each split should have the right categories
    total = sum(s["images"] for s in stats.values())
    print(f"\nTotal images across all splits: {total}")
    if stats["train"]["images"] == 0:
        print("  [ERROR] train split is empty!")
    if stats["val"]["images"] == 0:
        print("  [ERROR] val split is empty!")


if __name__ == "__main__":
    main()
