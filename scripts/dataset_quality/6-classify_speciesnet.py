"""SpeciesNet classification on MegaDetector crops for all passed dataset images.

Runs SpeciesNet's EfficientNetV2-M classifier on every animal detection stored in
filter_results.jsonl for images that have completed the caption_eval stage. Writes
one speciesnet_results.jsonl per source dataset and a single class manifest.

This script is **pure data capture** — no filtering decisions, no 225-class mapping.
Script 7 (7-filter_speciesnet.py) handles those steps and can be re-run cheaply
without touching this output.

Design notes
------------
- All entries in filter_results.jsonl["detections"] are already animal-only (category
  "1"). Person and vehicle detections were dropped by the earlier filter pipeline and
  cannot be recovered without re-running MegaDetector. Fields such as n_person_detections
  and has_human from the strategy document are therefore not populated.
- The full 2498-class probability vector is stored per detection as speciesnet_scores
  (a float list indexed by position in data/speciesnet_classes.json). The class strings
  themselves are saved once to that manifest to avoid per-detection redundancy.
- The classifier is accessed via clf.preprocess() + clf.model() directly rather than
  the high-level sn.classify() method, which only returns top-5.
- MegaDetector bboxes (bbox field, COCO format [xmin, ymin, width, height] normalised)
  are passed directly to SpeciesNet's preprocess() to crop the image before classification.
- Each image is opened once (PIL) and reused across all its detections.

Usage:
    python scripts/dataset_quality/6-classify_speciesnet.py --source gbif
    python scripts/dataset_quality/6-classify_speciesnet.py --source all
    python scripts/dataset_quality/6-classify_speciesnet.py --source inaturalist --force

Output:
    data/{source}/speciesnet_results.jsonl   — one record per image
    data/speciesnet_classes.json             — SpeciesNet label list (written once)

Must run inside Dockerfile.speciesnet (Python 3.11, speciesnet package).

Recommended workflow:

    # 1. Build the image once
    make speciesnet-build

    # 2. Start the container — drops you into a bash shell inside it
    make speciesnet-start

    # Inside the container: launch with nohup, then follow the log
    nohup python /app/scripts/dataset_quality/6-classify_speciesnet.py \\
        --source all \\
        > /app/output/speciesnet_classify_all.log 2>&1 &

    tail -f /app/output/speciesnet_classify_all.log

    # 3. Exit the shell and stop the container when done
    # exit
    make speciesnet-stop
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

# ── Constants ─────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[2]

RESULTS_PATHS = {
    "gbif":        REPO_ROOT / "data" / "gbif"        / "filter_results.jsonl",
    "inaturalist": REPO_ROOT / "data" / "inaturalist" / "filter_results.jsonl",
    "wikimedia":   REPO_ROOT / "data" / "wikimedia"   / "filter_results.jsonl",
    "openimages":  REPO_ROOT / "data" / "openimages"  / "filter_results.jsonl",
    "images_cv":   REPO_ROOT / "data" / "images_cv"   / "filter_results.jsonl",
}

OUTPUT_PATHS = {
    source: REPO_ROOT / "data" / source / "speciesnet_results.jsonl"
    for source in RESULTS_PATHS
}

SPECIESNET_CLASSES_PATH = REPO_ROOT / "data" / "speciesnet_classes.json"

MIN_CROP_PX = 32   # detections whose crop is smaller than this in either dim are skipped
FLUSH_EVERY = 100  # checkpoint flush interval (images processed, not detections)


# ── Environment guard ─────────────────────────────────────────────────────────

def _check_environment() -> None:
    try:
        import speciesnet  # noqa: F401
    except ImportError:
        print(
            "ERROR: 'speciesnet' is not installed.\n"
            "This script must run inside Dockerfile.speciesnet (Python 3.11).\n"
            "  make speciesnet-build\n"
            "  make speciesnet-start",
            file=sys.stderr,
        )
        sys.exit(1)


# ── SpeciesNet wrapper ────────────────────────────────────────────────────────

class SpeciesNetClassifier:
    """Wraps SpeciesNet for per-detection crop classification with full probability output.

    Uses the low-level clf.preprocess() + clf.model() path to obtain the complete
    2498-class probability vector. The high-level sn.classify() method only returns
    top-5 and is not used here.
    """

    def __init__(self) -> None:
        import torch
        import numpy as np
        from speciesnet import SpeciesNet, DEFAULT_MODEL
        from speciesnet.utils import BBox

        self._torch = torch
        self._np = np
        self._BBox = BBox

        pipeline = SpeciesNet(DEFAULT_MODEL, components="classifier", geofence=False)
        self._clf = pipeline.classifier
        self.labels: list[str] = list(self._clf.labels)

    def classify(
        self,
        img,  # PIL.Image already opened by caller
        bbox_norm: list[float],
    ) -> dict:
        """Classify one detection crop and return the full probability vector.

        bbox_norm is [xmin, ymin, width, height] normalised (COCO format, same as
        stored in filter_results.jsonl).
        """
        torch = self._torch
        np = self._np

        bbox = self._BBox(*bbox_norm)

        t0 = time.perf_counter()
        preprocessed = self._clf.preprocess(img, bboxes=[bbox])
        if preprocessed is None:
            raise RuntimeError("preprocess() returned None — image may be invalid")

        batch_arr = np.stack([preprocessed.arr / 255], axis=0, dtype=np.float32)
        batch_tensor = torch.from_numpy(batch_arr).to(self._clf.device)

        with torch.no_grad():
            logits = self._clf.model(batch_tensor).cpu()

        scores = torch.softmax(logits, dim=-1)[0]
        inference_ms = (time.perf_counter() - t0) * 1000

        top1_score, top1_idx = scores.max(dim=-1)
        top1_idx = int(top1_idx.item())

        return {
            "scores": scores.tolist(),
            "top1_idx": top1_idx,
            "top1_label": self.labels[top1_idx],
            "top1_score": round(float(top1_score.item()), 6),
            "inference_ms": round(inference_ms, 1),
        }


# ── JSONL helpers ─────────────────────────────────────────────────────────────

def load_filter_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def load_existing_output(path: Path) -> tuple[list[dict], set[str]]:
    if not path.exists():
        return [], set()
    records: list[dict] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                records.append(rec)
                seen.add(rec["filepath"])
    return records, seen


def save_output(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def save_class_manifest(path: Path, labels: list[str]) -> None:
    """Write the SpeciesNet label list once. Skipped if the file already exists."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(labels, f)
    print(f"Wrote SpeciesNet class manifest ({len(labels)} classes) → {path}")


# ── Per-source processing ─────────────────────────────────────────────────────

def process_source(
    source: str,
    classifier: SpeciesNetClassifier,
    force: bool,
    min_crop: int,
    md_conf_floor: float,
) -> None:
    from PIL import Image

    filter_path = RESULTS_PATHS[source]
    output_path = OUTPUT_PATHS[source]

    if not filter_path.exists():
        print(f"[{source}] filter_results.jsonl not found — skipping.")
        return

    entries = load_filter_results(filter_path)

    if force and output_path.exists():
        output_path.unlink()
        print(f"[{source}] --force: cleared existing {output_path.name}")

    existing_records, seen = load_existing_output(output_path)

    pending = [
        e for e in entries
        if e.get("passed")
        and "caption_eval" in e.get("stages_done", [])
        and e["filepath"] not in seen
    ]

    if not pending:
        print(f"[{source}] nothing to do "
              f"({len(existing_records):,} already classified, {len(entries):,} total entries).")
        return

    print(f"[{source}] {len(pending):,} images to classify "
          f"({len(existing_records):,} already done, {len(entries):,} total).")

    records = list(existing_records)
    flush_count = 0

    for entry in tqdm(pending, desc=f"classifying {source}", unit="img"):
        fp_rel = entry["filepath"]
        fp_abs = REPO_ROOT / fp_rel
        expected_common = Path(fp_rel).parent.name

        t_total = time.perf_counter()

        # Open image once; reused across all detections for this entry.
        try:
            img = Image.open(fp_abs).convert("RGB")
            img_w, img_h = img.size
        except Exception as exc:
            records.append({
                "filepath": fp_rel,
                "expected_common": expected_common,
                "error": str(exc),
                "speciesnet_detections": [],
                "n_animal_detections": 0,
                "inference_total_ms": 0.0,
            })
            flush_count += 1
            if flush_count % FLUSH_EVERY == 0:
                save_output(output_path, records)
            continue

        detections_raw = entry.get("detections") or []
        sn_detections: list[dict] = []

        for det_idx, det in enumerate(detections_raw):
            # bbox is [xmin, ymin, width, height] normalised (COCO format)
            bbox = det["bbox"]
            conf = float(det["conf"])

            w_px = int(bbox[2] * img_w)
            h_px = int(bbox[3] * img_h)

            if conf < md_conf_floor:
                sn_detections.append({
                    "detection_idx": det_idx,
                    "bbox_norm": bbox,
                    "megadetector_conf": round(conf, 6),
                    "speciesnet_skipped": True,
                    "skip_reason": "low_megadetector_conf",
                    "crop_size_px": [w_px, h_px],
                })
                continue

            if w_px < min_crop or h_px < min_crop:
                sn_detections.append({
                    "detection_idx": det_idx,
                    "bbox_norm": bbox,
                    "megadetector_conf": round(conf, 6),
                    "speciesnet_skipped": True,
                    "skip_reason": "crop_too_small",
                    "crop_size_px": [w_px, h_px],
                })
                continue

            try:
                result = classifier.classify(img, bbox)
            except Exception as exc:
                sn_detections.append({
                    "detection_idx": det_idx,
                    "bbox_norm": bbox,
                    "megadetector_conf": round(conf, 6),
                    "speciesnet_skipped": True,
                    "skip_reason": f"inference_error: {exc}",
                    "crop_size_px": [w_px, h_px],
                })
                continue

            sn_detections.append({
                "detection_idx": det_idx,
                "bbox_norm": bbox,
                "megadetector_conf": round(conf, 6),
                "speciesnet_scores": result["scores"],
                "speciesnet_top1_idx": result["top1_idx"],
                "speciesnet_top1": result["top1_label"],
                "speciesnet_top1_score": result["top1_score"],
                "crop_size_px": [w_px, h_px],
                "speciesnet_skipped": False,
                "skip_reason": None,
                "inference_ms": result["inference_ms"],
            })

        img.close()
        inference_total_ms = round((time.perf_counter() - t_total) * 1000, 1)

        records.append({
            "filepath": fp_rel,
            "expected_common": expected_common,
            "speciesnet_detections": sn_detections,
            "n_animal_detections": len(detections_raw),
            "inference_total_ms": inference_total_ms,
        })

        flush_count += 1
        if flush_count % FLUSH_EVERY == 0:
            save_output(output_path, records)

    save_output(output_path, records)
    classified = sum(
        1 for r in records
        if r.get("speciesnet_detections") and not r.get("error")
    )
    print(f"[{source}] done — {classified:,} images classified, "
          f"results in {output_path.relative_to(REPO_ROOT)}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    _check_environment()

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        required=True,
        choices=list(RESULTS_PATHS.keys()) + ["all"],
        help="Dataset source to classify, or 'all' for every source in sequence.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Clear existing speciesnet_results.jsonl and re-run from scratch.",
    )
    parser.add_argument(
        "--min-crop",
        type=int,
        default=MIN_CROP_PX,
        metavar="PX",
        help=f"Skip detections whose crop is smaller than PX in either dimension "
             f"(default: {MIN_CROP_PX}).",
    )
    parser.add_argument(
        "--md-conf",
        type=float,
        default=0.1,
        metavar="CONF",
        help="Skip detections below this MegaDetector confidence (default: 0.1).",
    )
    args = parser.parse_args()

    sources = list(RESULTS_PATHS.keys()) if args.source == "all" else [args.source]

    print("Loading SpeciesNet EfficientNetV2-M …")
    classifier = SpeciesNetClassifier()
    print(f"Model ready — {len(classifier.labels)} SpeciesNet classes.\n")

    save_class_manifest(SPECIESNET_CLASSES_PATH, classifier.labels)

    for source in sources:
        process_source(
            source,
            classifier,
            force=args.force,
            min_crop=args.min_crop,
            md_conf_floor=args.md_conf,
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()
