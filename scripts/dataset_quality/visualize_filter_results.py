"""
Visualize filter_results.jsonl quality-filter output using FiftyOne.

Each sample shows:
  - Bounding box from MegaDetector (or ground-truth for LILA BC / Open Images)
  - Tag: "passed" or "rejected"
  - Fields: stage_failed, reason, bbox_conf, stages_done, vlm_caption (Wikimedia only)

Usage:
  python scripts/dataset_quality/visualize_filter_results.py [--source wikimedia]

Sources: wikimedia (default), gbif, inaturalist, openimages, lila_bc
"""

import argparse
import json
from pathlib import Path

import fiftyone as fo

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

RESULTS_PATHS = {
    "wikimedia":  "data/wikimedia/filter_results.jsonl",
    "gbif":       "data/gbif/filter_results.jsonl",
    "inaturalist":"data/inaturalist/filter_results.jsonl",
    "openimages": "data/supplementary_openimages/filter_results.jsonl",
    "lila_bc":    "data/lila_bc/filter_results.jsonl",
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def yolo_to_fo(bbox: list) -> list:
    """Convert YOLO [xc, yc, w, h] (normalized) to FiftyOne [x, y, w, h] (top-left, normalized)."""
    xc, yc, w, h = bbox
    return [xc - w / 2, yc - h / 2, w, h]


def load_jsonl(path: Path) -> list:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


# ── Dataset builder ───────────────────────────────────────────────────────────
def build_dataset(source: str) -> fo.Dataset:
    results_path = REPO_ROOT / RESULTS_PATHS[source]
    if not results_path.exists():
        raise FileNotFoundError(f"No filter_results.jsonl found at {results_path}")

    entries = load_jsonl(results_path)
    dataset_name = f"filter-results-{source}"

    if fo.dataset_exists(dataset_name):
        fo.delete_dataset(dataset_name)

    dataset = fo.Dataset(dataset_name)
    samples = []
    skipped = 0

    for entry in entries:
        abs_path = REPO_ROOT / entry["filepath"]
        if not abs_path.is_file():
            skipped += 1
            continue

        # Species label from parent directory name
        species = Path(entry["filepath"]).parent.name

        # Bbox detection field
        bbox_raw = entry.get("bbox")
        bbox_conf = entry.get("bbox_conf")
        if bbox_raw is not None:
            det = fo.Detection(
                label=species,
                bounding_box=yolo_to_fo(bbox_raw),
                confidence=bbox_conf,
            )
            detections = fo.Detections(detections=[det])
        else:
            detections = fo.Detections(detections=[])

        sample = fo.Sample(filepath=str(abs_path))

        # Pass/reject tag
        sample.tags.append("passed" if entry.get("passed") else "rejected")

        # Quality-filter metadata fields
        sample["ground_truth"] = detections
        sample["stage_failed"] = entry.get("stage_failed")
        sample["reason"] = entry.get("reason")
        sample["bbox_conf"] = bbox_conf
        sample["stages_done"] = entry.get("stages_done", [])

        # Wikimedia-only VLM fields
        if "vlm_caption" in entry:
            sample["vlm_caption"] = entry["vlm_caption"]
        if "vlm_error" in entry:
            sample["vlm_error"] = entry["vlm_error"]

        samples.append(sample)

    dataset.add_samples(samples)
    dataset.save()

    total = len(entries)
    loaded = len(samples)
    passed = sum(1 for e in entries if e.get("passed"))
    print(
        f"Source: {source} | total: {total} | loaded: {loaded} "
        f"(skipped {skipped} missing files) | passed: {passed} | rejected: {total - passed}"
    )
    return dataset


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Visualize filter_results.jsonl in FiftyOne."
    )
    parser.add_argument(
        "--source",
        choices=list(RESULTS_PATHS.keys()),
        default="wikimedia",
        help="Dataset source to visualize (default: wikimedia)",
    )
    args = parser.parse_args()

    dataset = build_dataset(args.source)
    session = fo.launch_app(dataset)
    session.wait()


if __name__ == "__main__":
    main()
