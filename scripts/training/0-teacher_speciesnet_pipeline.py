"""
SpeciesNet two-stage teacher pipeline.

Stage 1 — MegaDetector v5a (YOLOv5x6, 141.8M params, ~270MB)
    Detects animal / person / vehicle bounding boxes in the full image.
    Only 'animal' detections (class 1) above conf_threshold are forwarded.

Stage 2 — SpeciesNet EfficientNetV2-M (54M params, ~87MB)
    Classifies each cropped animal detection over ~3,537 species labels.
    Outputs the full probability distribution — used as soft labels for KD.

This pipeline is:
  - The primary teacher for knowledge distillation (Path A)
  - The baseline inference system to beat (two-stage vs. one-shot comparison)

Runs inside Dockerfile.speciesnet (Python 3.11, speciesnet package).
On the host (Python 3.13) it will print a clear error and exit.

Usage:
    # Verify pipeline loads and runs (no real images needed)
    python scripts/training/0-teacher_speciesnet_pipeline.py --smoke-test

    # Single image demo with visualised detections
    python scripts/training/0-teacher_speciesnet_pipeline.py \\
        --image path/to/image.jpg

    # Batch mode: process a directory, write JSONL results
    python scripts/training/0-teacher_speciesnet_pipeline.py \\
        --dir data/training/ \\
        --output output/teacher_labels.jsonl

    # Soft-label mode: filter to the 225 student classes, write KD targets
    python scripts/training/0-teacher_speciesnet_pipeline.py \\
        --dir data/training/ \\
        --output output/kd_soft_labels.jsonl \\
        --classes resources/2026-03-19_student_model_labels.txt \\
        --soft-labels

    # Label validation: check each image's SpeciesNet prediction against its
    # expected class label (derived from the parent directory name).
    # Reads cached MegaDetector bboxes from filter_results.jsonl to skip the
    # MegaDetector stage — roughly halves total runtime vs. a fresh two-stage run.
    python scripts/training/0-teacher_speciesnet_pipeline.py \\
        --validate-labels \\
        --classes resources/2026-03-19_student_model_labels.txt \\
        --output output/species_validation.jsonl

Makefile shortcuts:
    make speciesnet-smoke
    make speciesnet-demo    IMAGE=path/to/image.jpg
    make speciesnet-labels  DIR=data/training/ OUT=output/kd_soft_labels.jsonl

Output JSONL format (one JSON object per line, one line per image):
    {
      "image": "relative/path/to/image.jpg",
      "detections": [
        {
          "bbox_norm": [x1, y1, x2, y2],   # normalised 0-1 coords
          "conf": 0.97,                      # MegaDetector confidence
          "predictions": [                   # top-5 SpeciesNet predictions
            {"label": "Odocoileus virginianus", "score": 0.85},
            ...
          ],
          "soft_label": {                    # only with --soft-labels
            "label_index": 42,               # index in the student 225-class list
            "probs_225": [0.001, ..., 0.85, ...]  # renormalised 225-dim vector
          }
        }
      ],
      "inference_ms": 145.2
    }

Validation output JSONL (one object per image, --validate-labels):
    {
      "image": "data/gbif/images/red fox/img.jpg",
      "expected_common": "red fox",
      "expected_scientific": "Vulpes vulpes",
      "top_prediction": "Vulpes vulpes",
      "top_score": 0.91,
      "match": true,
      "top5": [{"label": "Vulpes vulpes", "score": 0.91}, ...],
      "inference_ms": 52.1
    }
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Standard filter_results.jsonl paths produced by 1-filter_dataset_quality.py
FILTER_RESULTS_PATHS = {
    "gbif":        REPO_ROOT / "data" / "gbif" / "filter_results.jsonl",
    "inaturalist": REPO_ROOT / "data" / "inaturalist" / "filter_results.jsonl",
    "wikimedia":   REPO_ROOT / "data" / "wikimedia" / "filter_results.jsonl",
    "openimages":  REPO_ROOT / "data" / "openimages" / "filter_results.jsonl",
    "images_cv":   REPO_ROOT / "data" / "images_cv" / "filter_results.jsonl",
}

# ── Environment check ─────────────────────────────────────────────────────────

def _check_environment() -> None:
    """Exit with a helpful message if not running in the SpeciesNet Docker image."""
    try:
        import speciesnet  # noqa: F401
    except ImportError:
        print(
            "ERROR: 'speciesnet' is not installed.\n"
            "This script must run inside Dockerfile.speciesnet (Python 3.11).\n"
            "  make speciesnet-build && make speciesnet-smoke\n"
            "  make speciesnet-demo IMAGE=path/to/image.jpg",
            file=sys.stderr,
        )
        sys.exit(1)


# ── MegaDetector (Stage 1) ────────────────────────────────────────────────────

class MegaDetectorStage:
    """Thin wrapper around PytorchWildlife MegaDetectorV5."""

    ANIMAL_CLASS = 1  # MegaDetector class IDs: 1=animal, 2=person, 3=vehicle

    def __init__(self, conf_threshold: float = 0.1, device: str = "auto") -> None:
        from PytorchWildlife.models import MegaDetectorV5

        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.conf_threshold = conf_threshold
        self.device = device
        self._model = MegaDetectorV5(device=device, pretrained=True)

    def detect(self, image_path: str | Path) -> list[dict]:
        """
        Run MegaDetector on a single image.
        Returns a list of animal detections, each with normalised bbox coords.
        """
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        w, h = img.size

        # PytorchWildlife returns results as a list with per-image dicts
        results = self._model.single_image_detection(img, image_path=str(image_path))

        detections = []
        for det in results.get("detections", []):
            if det.get("category") != str(self.ANIMAL_CLASS):
                continue
            if det.get("conf", 0) < self.conf_threshold:
                continue
            bbox = det["bbox"]  # [x, y, width, height] normalised
            x, y, bw, bh = bbox
            detections.append({
                "bbox_norm": [x, y, x + bw, y + bh],
                "conf": float(det["conf"]),
            })

        return detections


# ── SpeciesNet classifier (Stage 2) ──────────────────────────────────────────

class SpeciesNetStage:
    """Wrapper around the speciesnet package's EfficientNetV2-M classifier."""

    def __init__(self) -> None:
        from speciesnet import SpeciesNet
        # geofence=False: skip geographic post-processing (we want raw probabilities)
        # country=None:   no geo-prior applied
        self._pipeline = SpeciesNet(geofence=False)

        # Extract label list from the SpeciesNet package for index mapping
        self._labels: list[str] = self._extract_labels()

    def _extract_labels(self) -> list[str]:
        """Extract the full SpeciesNet label list (3537+ classes)."""
        try:
            # speciesnet exposes labels through the classifier attribute
            clf = self._pipeline.classifier
            if hasattr(clf, "class_names"):
                return list(clf.class_names)
            if hasattr(clf, "labels"):
                return list(clf.labels)
        except AttributeError:
            pass
        return []

    def classify_image(self, image_path: str | Path, top_k: int = 10) -> list[dict]:
        """
        Run SpeciesNet on a full image (the pipeline handles cropping internally).
        Returns top-k predictions for the image.
        """
        instances = [{"filepath": str(image_path)}]
        results = self._pipeline.predict(instances=instances)

        predictions_out = []
        for pred in results.get("predictions", []):
            for inst_preds in pred.get("classifications", []):
                label = inst_preds.get("label") or inst_preds.get("name", "")
                score = float(inst_preds.get("score", 0))
                predictions_out.append({"label": label, "score": score})
        return predictions_out[:top_k]

    def classify_image_with_detection(
        self,
        image_path: str | Path,
        bbox_yolo: list[float],
        conf: float,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Run SpeciesNet with a pre-computed MegaDetector bbox to skip its internal
        detector stage.  bbox_yolo is [xc, yc, w, h] normalised — the same format
        stored in filter_results.jsonl — so no conversion is needed.

        Falls back to classify_image() if the API does not accept cached detections.
        """
        instance = {
            "filepath": str(image_path),
            "detections": [{"category": "1", "conf": conf, "bbox": bbox_yolo}],
        }
        try:
            results = self._pipeline.predict(instances=[instance])
        except (TypeError, KeyError):
            # speciesnet version does not support cached detections; run full pipeline
            return self.classify_image(image_path, top_k=top_k)

        predictions_out = []
        for pred in results.get("predictions", []):
            for inst_preds in pred.get("classifications", []):
                label = inst_preds.get("label") or inst_preds.get("name", "")
                score = float(inst_preds.get("score", 0))
                predictions_out.append({"label": label, "score": score})
        return predictions_out[:top_k]

    def classify_image_full_probs(
        self, image_path: str | Path
    ) -> tuple[list[dict], np.ndarray | None]:
        """
        Run SpeciesNet and return (top_predictions, full_probability_vector).
        The probability vector covers all ~3537 SpeciesNet classes.
        """
        instances = [{"filepath": str(image_path)}]
        results = self._pipeline.predict(instances=instances)

        top_preds = []
        full_probs = None

        for pred in results.get("predictions", []):
            classifications = pred.get("classifications", [])
            top_preds = [
                {"label": c.get("label", c.get("name", "")), "score": float(c.get("score", 0))}
                for c in classifications
            ]
            # Some speciesnet versions expose raw scores
            if "all_scores" in pred:
                full_probs = np.array(pred["all_scores"], dtype=np.float32)
            elif "logits" in pred:
                import torch
                logits = torch.tensor(pred["logits"])
                full_probs = torch.softmax(logits, dim=-1).numpy()

        return top_preds, full_probs


# ── Soft-label mapping ────────────────────────────────────────────────────────

def load_student_classes(class_file: str | Path) -> list[str]:
    """Load the 225 student model class names from the label file."""
    return [
        line.strip()
        for line in Path(class_file).read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def map_to_student_classes(
    full_probs: np.ndarray,
    speciesnet_labels: list[str],
    student_classes: list[str],
) -> np.ndarray:
    """
    Project a full SpeciesNet probability vector onto the 225 student classes.
    Classes not in the student label set are discarded; the result is renormalised.

    Returns a 225-dim float32 probability vector.
    """
    student_probs = np.zeros(len(student_classes), dtype=np.float32)

    label_to_idx = {label: i for i, label in enumerate(speciesnet_labels)}

    for student_idx, student_label in enumerate(student_classes):
        sn_idx = label_to_idx.get(student_label)
        if sn_idx is not None and sn_idx < len(full_probs):
            student_probs[student_idx] = full_probs[sn_idx]

    total = student_probs.sum()
    if total > 0:
        student_probs /= total

    return student_probs


# ── Filter-results helpers ────────────────────────────────────────────────────

def load_filter_results_index(
    paths: list[Path] | None = None,
) -> dict[str, dict]:
    """
    Read all filter_results.jsonl files and return an index keyed by filepath.
    Only entries where passed=True are included.

    bbox is stored as [xc, yc, w, h] normalised (YOLO format), sorted by conf desc.
    Returns: {filepath_str: {"bbox": [...], "conf": float}}
    """
    if paths is None:
        paths = [p for p in FILTER_RESULTS_PATHS.values() if p.exists()]

    index: dict[str, dict] = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if not entry.get("passed", False):
                    continue
                fp = entry["filepath"]
                detections = entry.get("detections") or []
                if detections:
                    best = detections[0]  # already sorted by conf desc
                    index[fp] = {"bbox": best["bbox"], "conf": best["conf"]}
                else:
                    # passed but no bbox (e.g. openimages with pre-annotated boxes)
                    index[fp] = {"bbox": None, "conf": None}
    return index


def load_label_mapping(class_file: Path) -> dict[str, str]:
    """
    Parse the student labels CSV to build common_name → scientific_name mapping.
    CSV format: UUID;class;order;family;genus;species;common_name
    Returns: {common_name_lower: "Genus species"}
    """
    mapping: dict[str, str] = {}
    with open(class_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(";")
            if len(parts) < 7:
                continue
            genus = parts[4].strip()
            species_ep = parts[5].strip()
            common = parts[6].strip()
            if genus and species_ep and common:
                sci = f"{genus.capitalize()} {species_ep}"
                mapping[common.lower()] = sci
    return mapping


# ── Label validation pipeline ─────────────────────────────────────────────────

def run_validation(
    class_file: Path,
    output_path: Path | None,
    verbose: bool,
) -> None:
    """
    For every passed image in filter_results.jsonl:
      1. Derive expected species from the parent directory name.
      2. Run SpeciesNet with the cached MegaDetector bbox (skips internal detector).
      3. Compare top prediction against the expected scientific name.
      4. Write per-image JSONL and a per-class summary CSV.

    Expected directory structure: data/{source}/images/{common_name}/{file}.jpg
    """
    import csv as _csv
    from collections import defaultdict
    from tqdm import tqdm

    if not class_file.exists():
        print(f"ERROR: class file not found: {class_file}", file=sys.stderr)
        sys.exit(1)

    label_map = load_label_mapping(class_file)   # common_name_lower → "Genus species"
    bbox_index = load_filter_results_index()

    if not bbox_index:
        print("ERROR: no passed images found in filter_results.jsonl files.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(bbox_index):,} passed images from filter_results.jsonl")
    print(f"Loaded {len(label_map)} class mappings from {class_file.name}")

    classifier = SpeciesNetStage()

    writer = None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = open(output_path, "w", encoding="utf-8")

    per_class: dict[str, dict] = defaultdict(lambda: {"total": 0, "match": 0, "no_pred": 0})

    try:
        items = list(bbox_index.items())
        for filepath, det in tqdm(items, desc="SpeciesNet validation", disable=not verbose):
            t0 = time.perf_counter()

            common_name = Path(filepath).parent.name.lower()
            expected_sci = label_map.get(common_name)

            try:
                if det["bbox"] is not None:
                    preds = classifier.classify_image_with_detection(
                        REPO_ROOT / filepath,
                        bbox_yolo=det["bbox"],
                        conf=det["conf"],
                        top_k=5,
                    )
                else:
                    preds = classifier.classify_image(REPO_ROOT / filepath, top_k=5)

                elapsed_ms = (time.perf_counter() - t0) * 1000

                top_pred = preds[0]["label"] if preds else None
                top_score = preds[0]["score"] if preds else 0.0
                match = (
                    expected_sci is not None
                    and top_pred is not None
                    and top_pred.lower() == expected_sci.lower()
                )

                record = {
                    "image": filepath,
                    "expected_common": common_name,
                    "expected_scientific": expected_sci,
                    "top_prediction": top_pred,
                    "top_score": round(top_score, 4),
                    "match": match,
                    "top5": preds,
                    "inference_ms": round(elapsed_ms, 1),
                }

                stats = per_class[common_name]
                stats["total"] += 1
                if not preds:
                    stats["no_pred"] += 1
                elif match:
                    stats["match"] += 1

            except Exception as exc:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                record = {
                    "image": filepath,
                    "expected_common": common_name,
                    "expected_scientific": expected_sci,
                    "error": str(exc),
                    "inference_ms": round(elapsed_ms, 1),
                }

            if writer:
                writer.write(json.dumps(record) + "\n")

    finally:
        if writer:
            writer.close()

    # Write per-class summary CSV
    summary_path = (output_path.with_suffix(".summary.csv") if output_path
                    else Path("output") / "species_validation_summary.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=["common_name", "expected_scientific",
                                           "total", "match", "match_rate", "no_pred"])
        w.writeheader()
        for common, stats in sorted(per_class.items()):
            total = stats["total"]
            match = stats["match"]
            w.writerow({
                "common_name": common,
                "expected_scientific": label_map.get(common, ""),
                "total": total,
                "match": match,
                "match_rate": f"{match / total:.3f}" if total else "0.000",
                "no_pred": stats["no_pred"],
            })

    total_imgs = sum(s["total"] for s in per_class.values())
    total_match = sum(s["match"] for s in per_class.values())
    print(f"\nValidation complete: {total_match:,}/{total_imgs:,} images matched "
          f"({total_match/total_imgs*100:.1f}%)" if total_imgs else "\nNo images processed.")
    print(f"Per-class summary: {summary_path}")
    if output_path:
        print(f"Per-image results: {output_path}")


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    image_paths: list[Path],
    output_path: Path | None,
    student_classes: list[str] | None,
    soft_labels: bool,
    conf_threshold: float,
    verbose: bool,
) -> list[dict]:
    """
    Run the two-stage SpeciesNet pipeline on a list of images.

    Stage 1: MegaDetector detects animal bounding boxes.
    Stage 2: SpeciesNet classifies the full image (it handles cropping internally).

    When soft_labels=True and student_classes is provided, the output includes
    a renormalised 225-dim probability vector for use as KD targets.
    """
    from tqdm import tqdm

    detector = MegaDetectorStage(conf_threshold=conf_threshold)
    classifier = SpeciesNetStage()

    all_results = []
    writer = None

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = open(output_path, "w", encoding="utf-8")

    try:
        for image_path in tqdm(image_paths, desc="SpeciesNet pipeline", disable=not verbose):
            t0 = time.perf_counter()

            try:
                detections = detector.detect(image_path)

                if soft_labels and student_classes:
                    top_preds, full_probs = classifier.classify_image_full_probs(image_path)
                else:
                    top_preds = classifier.classify_image(image_path)
                    full_probs = None

                elapsed_ms = (time.perf_counter() - t0) * 1000

                record: dict = {
                    "image": str(image_path),
                    "detections": [],
                    "predictions": top_preds,
                    "inference_ms": round(elapsed_ms, 1),
                }

                # Attach per-detection soft labels if requested
                if soft_labels and student_classes and full_probs is not None:
                    probs_225 = map_to_student_classes(
                        full_probs, classifier._labels, student_classes
                    )
                    top_student_idx = int(np.argmax(probs_225))
                    for det in detections:
                        det["soft_label"] = {
                            "label_index": top_student_idx,
                            "label_name": student_classes[top_student_idx],
                            "probs_225": probs_225.tolist(),
                        }

                record["detections"] = detections

            except Exception as exc:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                record = {
                    "image": str(image_path),
                    "error": str(exc),
                    "inference_ms": round(elapsed_ms, 1),
                }

            all_results.append(record)

            if writer:
                writer.write(json.dumps(record) + "\n")
                writer.flush()

    finally:
        if writer:
            writer.close()

    return all_results


# ── Smoke test ────────────────────────────────────────────────────────────────

def smoke_test() -> None:
    """
    Verify the full pipeline loads and runs on a synthetic dummy image.
    No real image is required — creates a temporary noise image.
    """
    import tempfile
    import torch
    from PIL import Image

    sep = "─" * 60
    print(f"\n{sep}")
    print("  SpeciesNet Pipeline Smoke Test")
    print(sep)

    # System info
    try:
        import speciesnet
        print(f"  speciesnet : {speciesnet.__version__}")
    except AttributeError:
        print("  speciesnet : installed (version unknown)")
    print(f"  torch      : {torch.__version__}")
    print(f"  device     : {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print(f"  GPU        : {torch.cuda.get_device_name(0)}")
    print(sep)

    # Create a 640×640 synthetic RGB image
    rng = np.random.default_rng(0)
    img_array = rng.integers(0, 256, (640, 640, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        tmp_path = Path(f.name)
        img.save(tmp_path, "JPEG")

    results = []

    # Stage 1: MegaDetector
    print("\n[1/2] Loading MegaDetector v5...")
    t0 = time.perf_counter()
    try:
        detector = MegaDetectorStage()
        detections = detector.detect(tmp_path)
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"      PASS  {elapsed:.0f}ms  {len(detections)} animal detection(s) on noise image")
        results.append(("MegaDetector v5a", "PASS", elapsed))
    except Exception as e:
        print(f"      FAIL  {e}")
        results.append(("MegaDetector v5a", "FAIL", None))

    # Stage 2: SpeciesNet classifier
    print("[2/2] Loading SpeciesNet EfficientNetV2-M...")
    t0 = time.perf_counter()
    try:
        classifier = SpeciesNetStage()
        top_preds = classifier.classify_image(tmp_path, top_k=3)
        elapsed = (time.perf_counter() - t0) * 1000
        label_count = len(classifier._labels)
        print(f"      PASS  {elapsed:.0f}ms  {label_count} labels  top: {top_preds[0]['label'] if top_preds else 'n/a'}")
        results.append(("SpeciesNet EfficientNetV2-M", "PASS", elapsed))
    except Exception as e:
        print(f"      FAIL  {e}")
        results.append(("SpeciesNet EfficientNetV2-M", "FAIL", None))
    finally:
        tmp_path.unlink(missing_ok=True)

    print()
    n_pass = sum(1 for _, s, _ in results if s == "PASS")
    n_fail = sum(1 for _, s, _ in results if s == "FAIL")
    print(f"  PASS: {n_pass}  FAIL: {n_fail}\n")

    if n_fail:
        sys.exit(1)


# ── CLI ───────────────────────────────────────────────────────────────────────

def collect_images(path: Path, extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png")) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in extensions)


def main() -> None:
    _check_environment()

    parser = argparse.ArgumentParser(
        description="SpeciesNet two-stage teacher pipeline (MegaDetector + EfficientNetV2-M)"
    )
    parser.add_argument("--smoke-test", action="store_true",
                        help="Verify pipeline loads and runs (no real images needed)")
    parser.add_argument("--validate-labels", action="store_true",
                        help="Check each image's SpeciesNet prediction against its expected "
                             "class label (from parent directory name). Reads cached "
                             "MegaDetector bboxes from filter_results.jsonl to skip the "
                             "MegaDetector stage. Requires --classes.")
    parser.add_argument("--image", type=Path,
                        help="Single image to process")
    parser.add_argument("--dir", type=Path,
                        help="Directory of images to process in batch")
    parser.add_argument("--output", type=Path,
                        help="JSONL output file path (required for --dir / --validate-labels)")
    parser.add_argument("--classes", type=Path,
                        default=Path("resources/2026-03-19_student_model_labels.txt"),
                        help="Student model class list (225 labels) for soft-label mapping "
                             "and label validation")
    parser.add_argument("--soft-labels", action="store_true",
                        help="Include renormalised 225-class probability vectors in output")
    parser.add_argument("--conf", type=float, default=0.1,
                        help="MegaDetector confidence threshold (default: 0.1)")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Show progress bar")

    args = parser.parse_args()

    if args.smoke_test:
        smoke_test()
        return

    if args.validate_labels:
        run_validation(
            class_file=args.classes,
            output_path=args.output or Path("output") / "species_validation.jsonl",
            verbose=args.verbose,
        )
        return

    if args.image is None and args.dir is None:
        parser.print_help()
        sys.exit(1)

    student_classes = None
    if args.soft_labels:
        if not args.classes.exists():
            print(f"ERROR: class file not found: {args.classes}", file=sys.stderr)
            sys.exit(1)
        student_classes = load_student_classes(args.classes)
        print(f"Loaded {len(student_classes)} student class labels from {args.classes}")

    if args.image:
        image_paths = [args.image]
    else:
        image_paths = collect_images(args.dir)
        print(f"Found {len(image_paths)} images in {args.dir}")

    output_path = args.output
    if args.dir and output_path is None:
        output_path = Path("output") / "teacher_labels.jsonl"
        print(f"No --output specified; writing to {output_path}")

    results = run_pipeline(
        image_paths=image_paths,
        output_path=output_path,
        student_classes=student_classes,
        soft_labels=args.soft_labels,
        conf_threshold=args.conf,
        verbose=args.verbose,
    )

    if args.image:
        print(json.dumps(results[0], indent=2))

    if output_path:
        print(f"\nWrote {len(results)} records to {output_path}")


if __name__ == "__main__":
    main()
