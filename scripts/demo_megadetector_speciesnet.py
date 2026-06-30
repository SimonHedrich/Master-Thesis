#!/usr/bin/env python3
"""Run MegaDetector v5 + SpeciesNet on a single camera-trap image.

Steps:
  1. MegaDetector v5 → find all animal detections, print confidences
  2. Crop the top-confidence bbox from the image and save it
  3. SpeciesNet classifier → print top-5 species predictions on the crop

Usage:
    python scripts/demo_megadetector_speciesnet.py
    python scripts/demo_megadetector_speciesnet.py --image path/to/image.jpg
"""

import argparse
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = REPO_ROOT / "resources" / "orinoquia_camera_trap.jpg"


def _parse_sn_label(label: str) -> tuple[str, str]:
    """Return (common_name, scientific_name) from a SpeciesNet label string.

    Label format: UUID;class;order;family;genus;species;common name
    """
    parts = label.split(";")
    common = parts[-1] if parts else label
    if len(parts) >= 6 and parts[4] and parts[5]:
        sci = f"{parts[4]} {parts[5]}"
    else:
        sci = ""
    return common, sci


def run_megadetector(image_path: Path, device: str):
    """Run MegaDetector v5 on image_path. Returns (PIL image, sorted animal detections)."""
    try:
        import torch
        from PytorchWildlife.models import detection as pw_detection
        from yolov5.utils.general import non_max_suppression, scale_boxes
    except ImportError as exc:
        print(f"ERROR: missing dependency — {exc}", file=sys.stderr)
        sys.exit(1)

    print("Loading MegaDetector v5 …")
    model = pw_detection.MegaDetectorV5(device=device, pretrained=True)
    model.model.eval()

    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    img_size = model.IMAGE_SIZE  # 1280

    img_tensor = model.transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        raw = model.model(img_tensor)[0].float().cpu()

    preds = non_max_suppression(raw, conf_thres=0.2)
    pred = preds[0]

    if pred is None or len(pred) == 0:
        return img, []

    pred_np = pred.numpy().copy()
    pred_np[:, :4] = scale_boxes([img_size] * 2, pred_np[:, :4], (H, W)).round()

    animal_dets = sorted(
        [
            {
                "x1": int(x1), "y1": int(y1),
                "x2": int(x2), "y2": int(y2),
                "conf": float(c),
            }
            for x1, y1, x2, y2, c, cls in pred_np
            if int(cls) == 0
        ],
        key=lambda d: d["conf"],
        reverse=True,
    )
    return img, animal_dets


def run_speciesnet(crop_path: Path):
    """Run SpeciesNet classifier-only on a pre-cropped image. Returns predictions list."""
    try:
        from speciesnet import SpeciesNet, DEFAULT_MODEL
    except ImportError as exc:
        print(f"ERROR: missing dependency — {exc}", file=sys.stderr)
        sys.exit(1)

    print("Loading SpeciesNet classifier …")
    sn = SpeciesNet(DEFAULT_MODEL, components="classifier", geofence=False)
    result = sn.classify(filepaths=[str(crop_path)])
    if not result or "predictions" not in result:
        return None
    return result["predictions"][0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE,
                        metavar="PATH", help="Input image (default: resources/orinoquia_camera_trap.jpg)")
    args = parser.parse_args()

    image_path: Path = args.image.resolve()
    if not image_path.exists():
        print(f"ERROR: image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Image:  {image_path}\n")

    # ── MegaDetector ──────────────────────────────────────────────────────────
    img, animal_dets = run_megadetector(image_path, device)

    print(f"\nMegaDetector — {len(animal_dets)} animal detection(s) found")
    for i, d in enumerate(animal_dets, 1):
        print(f"  #{i}  conf={d['conf']:.3f}  bbox=[{d['x1']}, {d['y1']}, {d['x2']}, {d['y2']}]")

    if not animal_dets:
        print("No animals detected — stopping.")
        sys.exit(0)

    # ── Crop ──────────────────────────────────────────────────────────────────
    top = animal_dets[0]
    crop = img.crop((top["x1"], top["y1"], top["x2"], top["y2"]))
    crop_path = image_path.parent / (image_path.stem + "_crop.jpg")
    crop.save(crop_path)
    print(f"\nTop detection crop saved → {crop_path}  ({crop.width}×{crop.height} px)")

    # ── SpeciesNet ────────────────────────────────────────────────────────────
    print()
    pred_entry = run_speciesnet(crop_path)

    if pred_entry is None:
        print("SpeciesNet returned no result.")
        sys.exit(0)

    classifications = pred_entry.get("classifications")
    if not classifications:
        failures = pred_entry.get("failures", [])
        print(f"SpeciesNet — no classifications (failures: {failures})")
        sys.exit(0)

    classes = classifications.get("classes", [])
    scores = classifications.get("scores", [])

    print("SpeciesNet — top-5 classifications")
    for rank, (label, score) in enumerate(zip(classes, scores), 1):
        common, sci = _parse_sn_label(label)
        sci_str = f"({sci})" if sci else ""
        print(f"  {rank}. {common:<35} {sci_str:<30}  {score:.4f}")


if __name__ == "__main__":
    main()
