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
- speciesnet_scores is stored as a sparse dict {str(class_idx): score} containing only
  entries with probability >= --min-score (default 0.01). At this threshold ~4 classes
  are kept per detection on average (out of 2498), giving a ~600x reduction in file size
  vs storing the full probability vector. The top-1 result is always preserved separately
  regardless of threshold. Use --migrate-scores to re-encode existing full-vector files.
- The classifier is accessed via clf.preprocess() + clf.model() directly rather than
  the high-level sn.classify() method, which only returns top-5.
- MegaDetector bboxes (bbox field, COCO format [xmin, ymin, width, height] normalised)
  are passed directly to SpeciesNet's preprocess() to crop the image before classification.
- Image loading AND crop preprocessing run in background threads (ThreadPoolExecutor,
  --workers N). SpeciesNet preprocess() uses PIL resize + numpy ops that release the GIL,
  enabling true CPU parallelism. The main thread only dispatches GPU forward passes.
- GPU batching happens on the main thread once the worker pool delivers preprocessed
  arrays; during torch CUDA ops the GIL is released so workers continue preprocessing
  the next chunk.
- Images are processed in chunks of FLUSH_EVERY. Each chunk appends only its new records
  to the output file (append mode, O(1) per flush regardless of total file size).

Usage:
    uv run python scripts/dataset_quality/6-classify_speciesnet.py --source gbif
    uv run python scripts/dataset_quality/6-classify_speciesnet.py --source all
    uv run python scripts/dataset_quality/6-classify_speciesnet.py --source inaturalist --force
    uv run python scripts/dataset_quality/6-classify_speciesnet.py --source gbif --workers 8

    # Re-encode existing full-vector output files to sparse format (no GPU needed):
    uv run python scripts/dataset_quality/6-classify_speciesnet.py --migrate-scores gbif
    uv run python scripts/dataset_quality/6-classify_speciesnet.py --migrate-scores all

Output:
    data/{source}/speciesnet_results.jsonl   — one record per image
    data/speciesnet_classes.json             — SpeciesNet label list (written once)

Run inside the default training container (`make run`).

Recommended workflow:

    # 1. Build the image once, then start the container and exec a bash shell
    make build
    make run

    # Inside the container: launch with nohup, then follow the log
    nohup uv run python scripts/dataset_quality/6-classify_speciesnet.py \\
        --source all \\
        > /app/output/speciesnet_classify_all.log 2>&1 &

    tail -f /app/output/speciesnet_classify_all.log

    # 2. Exit the shell and stop the container when done
    # exit
    make stop
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
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
            "Run inside the default training container:\n"
            "  make build\n"
            "  make run",
            file=sys.stderr,
        )
        sys.exit(1)


# ── SpeciesNet wrapper ────────────────────────────────────────────────────────

class SpeciesNetClassifier:
    """Wraps SpeciesNet for batched crop classification with sparse probability output.

    Uses the low-level clf.preprocess() + clf.model() path to obtain the complete
    2498-class probability vector, then stores only entries above a threshold to keep
    output file sizes manageable. The high-level sn.classify() method only returns
    top-5 and is not used here.

    Preprocessing (CPU) and inference (GPU) are split into separate methods so that
    crops from multiple images can be preprocessed in parallel worker threads and
    then batched together before a single GPU forward pass.
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

    def preprocess_crop(
        self,
        img,  # PIL.Image already opened by caller
        bbox_norm: list[float],
    ):
        """Crop and preprocess one detection into a float32 numpy array (CPU only).

        Thread-safe: SpeciesNet.preprocess() is a stateless image transform (PIL +
        numpy) that releases the GIL, so multiple workers may call this simultaneously.
        Returns None if SpeciesNet's preprocess() considers the crop invalid.
        """
        bbox = self._BBox(*bbox_norm)
        preprocessed = self._clf.preprocess(img, bboxes=[bbox])
        if preprocessed is None:
            return None
        return (preprocessed.arr / 255).astype(self._np.float32)

    def classify_batch(self, arrs: list, min_score: float = 0.01) -> list[dict]:
        """One GPU forward pass for N preprocessed crop arrays.

        arrs     — list of float32 numpy arrays returned by preprocess_crop().
        min_score — only keep class probabilities >= this threshold in the output dict.

        Returns a list of N result dicts with keys:
          scores      — sparse dict {str(class_idx): probability} for prob >= min_score
          top1_idx    — index of the highest-probability class (always present)
          top1_label  — label string for top1_idx (always present)
          top1_score  — probability of the top-1 class (always present)
        """
        torch = self._torch
        np = self._np
        batch = np.stack(arrs, axis=0)
        tensor = torch.from_numpy(batch).to(self._clf.device)
        with torch.no_grad():
            logits = self._clf.model(tensor).cpu()
        scores_all = torch.softmax(logits, dim=-1)
        results = []
        for scores in scores_all:
            top1_score, top1_idx = scores.max(dim=-1)
            top1_idx = int(top1_idx.item())
            scores_list = scores.tolist()
            results.append({
                "scores": {
                    str(i): round(s, 6)
                    for i, s in enumerate(scores_list)
                    if s >= min_score
                },
                "top1_idx": top1_idx,
                "top1_label": self.labels[top1_idx],
                "top1_score": round(float(top1_score.item()), 6),
            })
        return results


# ── Image loading + preprocessing (worker thread) ─────────────────────────────

def _load_and_preprocess(args: tuple) -> dict:
    """Load one image and preprocess all its eligible detection crops. Runs in a thread.

    PIL JPEG decode and SpeciesNet preprocess() both release the GIL, enabling true
    CPU parallelism across multiple workers while the main thread runs GPU inference.

    Returns a dict with one of two shapes:
      {"load_error": str}                              — on image open failure
      {"eligible": list[dict], "skipped": list[dict]}  — on success
        eligible dicts: det_idx, bbox, conf, w_px, h_px, arr (numpy float32)
        skipped dicts:  ready-to-output detection records with speciesnet_skipped=True
    """
    entry, repo_root, classifier, md_conf_floor, min_crop = args
    from PIL import Image

    fp_abs = repo_root / entry["filepath"]
    try:
        img = Image.open(fp_abs).convert("RGB")
    except Exception as exc:
        return {"load_error": str(exc)}

    img_w, img_h = img.width, img.height
    eligible: list[dict] = []
    skipped: list[dict] = []

    for det_idx, det in enumerate(entry.get("detections") or []):
        bbox = det["bbox"]
        conf = float(det["conf"])
        w_px = int(bbox[2] * img_w)
        h_px = int(bbox[3] * img_h)

        if conf < md_conf_floor:
            skipped.append({
                "detection_idx": det_idx,
                "bbox_norm": bbox,
                "megadetector_conf": round(conf, 6),
                "speciesnet_skipped": True,
                "skip_reason": "low_megadetector_conf",
                "crop_size_px": [w_px, h_px],
            })
            continue

        if w_px < min_crop or h_px < min_crop:
            skipped.append({
                "detection_idx": det_idx,
                "bbox_norm": bbox,
                "megadetector_conf": round(conf, 6),
                "speciesnet_skipped": True,
                "skip_reason": "crop_too_small",
                "crop_size_px": [w_px, h_px],
            })
            continue

        try:
            arr = classifier.preprocess_crop(img, bbox)
        except Exception as exc:
            skipped.append({
                "detection_idx": det_idx,
                "bbox_norm": bbox,
                "megadetector_conf": round(conf, 6),
                "speciesnet_skipped": True,
                "skip_reason": f"inference_error: {exc}",
                "crop_size_px": [w_px, h_px],
            })
            continue

        if arr is None:
            skipped.append({
                "detection_idx": det_idx,
                "bbox_norm": bbox,
                "megadetector_conf": round(conf, 6),
                "speciesnet_skipped": True,
                "skip_reason": "preprocess_returned_none",
                "crop_size_px": [w_px, h_px],
            })
            continue

        eligible.append({
            "det_idx": det_idx,
            "bbox": bbox,
            "conf": conf,
            "w_px": w_px,
            "h_px": h_px,
            "arr": arr,
        })

    img.close()
    return {"eligible": eligible, "skipped": skipped}


# ── Batch flush helper ────────────────────────────────────────────────────────

def _flush_det_buffer(
    det_buffer: list[dict],
    batch_results: dict,
    classifier: SpeciesNetClassifier,
    min_score: float,
) -> None:
    """Run one GPU forward pass on all crops in det_buffer and store results.

    Each entry in det_buffer must have keys: arr, local_idx, det_idx.
    Results are stored in batch_results keyed by (local_idx, det_idx).
    inference_ms is the batch-averaged per-detection GPU time.
    """
    t0 = time.perf_counter()
    arrs = [item["arr"] for item in det_buffer]
    results = classifier.classify_batch(arrs, min_score=min_score)
    ms_per = round((time.perf_counter() - t0) * 1000 / len(arrs), 1)
    for item, result in zip(det_buffer, results):
        result["inference_ms"] = ms_per
        batch_results[(item["local_idx"], item["det_idx"])] = result


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


def load_existing_output(path: Path) -> tuple[int, set[str]]:
    """Stream-read the existing output file; return (n_classified, seen_filepaths).

    Does not hold records in memory — only the filepath set and a count are retained.
    n_classified counts records that have non-empty speciesnet_detections and no error.
    """
    if not path.exists():
        return 0, set()
    n_classified = 0
    seen: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            seen.add(rec["filepath"])
            if rec.get("speciesnet_detections") and not rec.get("error"):
                n_classified += 1
    return n_classified, seen


def append_output(path: Path, new_records: list[dict]) -> None:
    """Append new records to the output file. O(len(new_records)), not O(total)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for rec in new_records:
            f.write(json.dumps(rec) + "\n")


def save_class_manifest(path: Path, labels: list[str]) -> None:
    """Write the SpeciesNet label list once. Skipped if the file already exists."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(labels, f)
    print(f"Wrote SpeciesNet class manifest ({len(labels)} classes) → {path}")


# ── Score migration ───────────────────────────────────────────────────────────

def migrate_scores(source: str, min_score: float) -> None:
    """Re-encode speciesnet_scores from full list → sparse dict in an existing output file.

    No GPU required — pure JSON transform. Reads the file and atomically rewrites it.
    Idempotent: records already in sparse-dict format are left unchanged.
    """
    output_path = OUTPUT_PATHS[source]
    if not output_path.exists():
        print(f"[{source}] {output_path.name} not found — nothing to migrate.")
        return

    size_before = output_path.stat().st_size
    print(f"[{source}] migrating scores (min_score={min_score}) — "
          f"input: {size_before / 1e9:.2f} GB")

    tmp_path = output_path.with_suffix(".jsonl.migrating")
    n_records = 0
    n_already_sparse = 0

    with open(output_path, encoding="utf-8") as f_in, \
         open(tmp_path, "w", encoding="utf-8") as f_out:
        for line in tqdm(f_in, desc=f"migrating {source}", unit="rec"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for det in rec.get("speciesnet_detections") or []:
                scores = det.get("speciesnet_scores")
                if scores is None:
                    continue
                if isinstance(scores, dict):
                    n_already_sparse += 1
                    continue
                det["speciesnet_scores"] = {
                    str(i): round(s, 6)
                    for i, s in enumerate(scores)
                    if s >= min_score
                }
            f_out.write(json.dumps(rec) + "\n")
            n_records += 1

    tmp_path.replace(output_path)
    size_after = output_path.stat().st_size
    ratio = size_before / max(size_after, 1)
    print(f"[{source}] done — {n_records:,} records, "
          f"{size_before / 1e9:.2f} GB → {size_after / 1e9:.2f} GB "
          f"({ratio:.1f}× smaller)")


# ── Per-source processing ─────────────────────────────────────────────────────

def process_source(
    source: str,
    classifier: SpeciesNetClassifier,
    force: bool,
    min_crop: int,
    md_conf_floor: float,
    batch_size: int,
    workers: int,
    min_score: float,
) -> None:
    filter_path = RESULTS_PATHS[source]
    output_path = OUTPUT_PATHS[source]

    if not filter_path.exists():
        print(f"[{source}] filter_results.jsonl not found — skipping.")
        return

    entries = load_filter_results(filter_path)

    if force and output_path.exists():
        output_path.unlink()
        print(f"[{source}] --force: cleared existing {output_path.name}")

    n_existing, seen = load_existing_output(output_path)

    pending = [
        e for e in entries
        if e.get("passed")
        and "caption_eval" in e.get("stages_done", [])
        and e["filepath"] not in seen
    ]

    if not pending:
        print(f"[{source}] nothing to do "
              f"({n_existing:,} already classified, {len(entries):,} total entries).")
        return

    print(f"[{source}] {len(pending):,} images to classify "
          f"({n_existing:,} already done, {len(entries):,} total).")

    classified_count = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        with tqdm(total=len(pending), desc=f"classifying {source}", unit="img") as pbar:
            offset = 0
            while offset < len(pending):
                chunk = pending[offset:offset + FLUSH_EVERY]

                # Submit image load + crop preprocess for this chunk to the thread pool.
                # Workers handle PIL open, GIL-releasing JPEG decode, and stateless
                # SpeciesNet preprocess() calls. The main thread only batches GPU calls.
                load_args = [
                    (entry, REPO_ROOT, classifier, md_conf_floor, min_crop)
                    for entry in chunk
                ]

                # Phase A ── parallel I/O + CPU preprocess (workers) + GPU batching (main)
                det_buffer: list[dict] = []
                batch_results: dict[tuple, dict] = {}
                chunk_skip_dets: list = [None] * len(chunk)
                chunk_eligible_meta: list = [None] * len(chunk)
                chunk_errors: list = [None] * len(chunk)

                for local_idx, (entry, result) in enumerate(
                    zip(chunk, executor.map(_load_and_preprocess, load_args))
                ):
                    if result.get("load_error"):
                        chunk_errors[local_idx] = result["load_error"]
                        chunk_skip_dets[local_idx] = []
                        chunk_eligible_meta[local_idx] = []
                        pbar.update(1)
                        continue

                    chunk_skip_dets[local_idx] = result["skipped"]
                    chunk_eligible_meta[local_idx] = [
                        (e["det_idx"], e["bbox"], e["conf"], e["w_px"], e["h_px"])
                        for e in result["eligible"]
                    ]

                    for e in result["eligible"]:
                        det_buffer.append({
                            "arr": e["arr"],
                            "local_idx": local_idx,
                            "det_idx": e["det_idx"],
                        })

                    # Flush GPU batch. During classify_batch(), PyTorch releases the GIL
                    # for CUDA ops, so background threads continue preprocessing images.
                    if len(det_buffer) >= batch_size:
                        _flush_det_buffer(det_buffer, batch_results, classifier, min_score)
                        det_buffer.clear()

                    pbar.update(1)

                if det_buffer:
                    _flush_det_buffer(det_buffer, batch_results, classifier, min_score)
                    det_buffer.clear()

                # Phase B ── record assembly (pure dict work, very fast)
                new_records: list[dict] = []
                for local_idx, entry in enumerate(chunk):
                    fp_rel = entry["filepath"]
                    expected_common = Path(fp_rel).parent.name
                    detections_raw = entry.get("detections") or []

                    if chunk_errors[local_idx]:
                        new_records.append({
                            "filepath": fp_rel,
                            "expected_common": expected_common,
                            "error": chunk_errors[local_idx],
                            "speciesnet_detections": [],
                            "n_animal_detections": 0,
                            "inference_total_ms": 0.0,
                        })
                        continue

                    sn_detections: list[dict] = list(chunk_skip_dets[local_idx])
                    inference_total_ms = 0.0

                    for det_idx, bbox, conf, w_px, h_px in chunk_eligible_meta[local_idx]:
                        result = batch_results.get((local_idx, det_idx))
                        if result is None:
                            sn_detections.append({
                                "detection_idx": det_idx,
                                "bbox_norm": bbox,
                                "megadetector_conf": round(conf, 6),
                                "speciesnet_skipped": True,
                                "skip_reason": "batch_result_missing",
                                "crop_size_px": [w_px, h_px],
                            })
                        else:
                            inference_total_ms += result["inference_ms"]
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

                    sn_detections.sort(key=lambda x: x["detection_idx"])

                    rec = {
                        "filepath": fp_rel,
                        "expected_common": expected_common,
                        "speciesnet_detections": sn_detections,
                        "n_animal_detections": len(detections_raw),
                        "inference_total_ms": round(inference_total_ms, 1),
                    }
                    new_records.append(rec)
                    if rec.get("speciesnet_detections") and not rec.get("error"):
                        classified_count += 1

                append_output(output_path, new_records)
                offset += FLUSH_EVERY

    print(f"[{source}] done — {classified_count:,} new images classified, "
          f"results in {output_path.relative_to(REPO_ROOT)}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--source",
        choices=list(RESULTS_PATHS.keys()) + ["all"],
        help="Dataset source to classify, or 'all' for every source in sequence.",
    )
    mode.add_argument(
        "--migrate-scores",
        dest="migrate_scores",
        metavar="SOURCE",
        choices=list(RESULTS_PATHS.keys()) + ["all"],
        help="Re-encode speciesnet_scores from full float list to sparse dict in an "
             "existing output file. No GPU required. Combine with --min-score to set "
             "the retention threshold.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Clear existing speciesnet_results.jsonl and re-run from scratch "
             "(ignored when --migrate-scores is used).",
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        metavar="N",
        help="Detection crops per GPU forward pass (default: 128). "
             "Reduce if VRAM is insufficient.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        metavar="N",
        help="Background threads for parallel image loading and crop preprocessing "
             "(default: 8). Each worker loads, JPEG-decodes, and preprocesses one "
             "image concurrently with GPU inference on the main thread.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.01,
        metavar="PROB",
        help="Minimum SpeciesNet probability to retain in speciesnet_scores dict "
             "(default: 0.01). At this threshold ~4 classes are kept per detection "
             "on average, giving ~600x smaller score storage than a full vector. "
             "Also used by --migrate-scores.",
    )
    args = parser.parse_args()

    if args.migrate_scores:
        sources = (
            list(RESULTS_PATHS.keys()) if args.migrate_scores == "all"
            else [args.migrate_scores]
        )
        for source in sources:
            migrate_scores(source, args.min_score)
        print("\nAll done.")
        return

    _check_environment()

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
            batch_size=args.batch_size,
            workers=args.workers,
            min_score=args.min_score,
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()
