"""Stage 7: Load synthetic COCO annotations into FiftyOne for visual verification.

Edit SPLITS to select which dataset splits to include in the app.
"""

from pathlib import Path
from collections import Counter

import fiftyone as fo

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data" / "synthetic"

SPLITS = ["train", "val", "test"]  # edit to load a subset

ANNOTATION_FILES = {
    "train": DATA_DIR / "annotations_train.json",
    "val":   DATA_DIR / "annotations_val.json",
    "test":  DATA_DIR / "annotations_test.json",
}

DATASET_NAME = "synthetic_wildlife"


def main():
    if fo.dataset_exists(DATASET_NAME):
        fo.delete_dataset(DATASET_NAME)
    dataset = fo.Dataset(name=DATASET_NAME)

    for split in SPLITS:
        ann_path = ANNOTATION_FILES[split]
        if not ann_path.exists():
            print(f"[warn] {ann_path} not found, skipping {split}")
            continue

        tmp_name = f"{DATASET_NAME}_{split}_tmp"
        if fo.dataset_exists(tmp_name):
            fo.delete_dataset(tmp_name)

        print(f"Loading {split} …")
        split_ds = fo.Dataset.from_dir(
            dataset_type=fo.types.COCODetectionDataset,
            data_path=str(REPO_ROOT),
            labels_path=str(ann_path),
            name=tmp_name,
        )
        split_ds.tag_samples(split)
        dataset.merge_samples(split_ds)
        fo.delete_dataset(tmp_name)

    print(f"\nLoaded {len(dataset)} samples across splits: {SPLITS}")
    _print_class_counts(dataset)

    session = fo.launch_app(dataset, address="0.0.0.0")
    session.wait()


def _print_class_counts(dataset):
    counts = Counter()
    for sample in dataset.iter_samples():
        dets = sample.get_field("detections")
        if dets and dets.detections:
            # All detections in a synthetic image share the same species; count image once
            counts[dets.detections[0].label] += 1
    print(f"\nImages per class ({len(counts)} classes):")
    for label, n in sorted(counts.items()):
        print(f"  {label:<40} {n:>5}")
    print(f"  {'TOTAL':<40} {sum(counts.values()):>5}")


if __name__ == "__main__":
    main()
