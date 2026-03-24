"""
Visualize SpeciesNet predictions on GBIF images using FiftyOne.

Images:  resources/GBIFImages/images/
Labels:  resources/SNPredictions_all.json

Label schema (per sample):
  - classifications: top-5 species predictions with scores
  - detections:      animal bounding boxes [x, y, w, h] (relative, top-left origin)
  - prediction:      top-1 species (semicolon-delimited taxonomy string)
  - prediction_score: confidence of top-1 prediction
"""

import json
import os
import fiftyone as fo

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(REPO_ROOT, "resources", "GBIFImages", "images")
LABELS_PATH = os.path.join(REPO_ROOT, "resources", "SNPredictions_all.json")

DATASET_NAME = "gbif-sn-predictions"


def parse_taxonomy(label_str: str) -> str:
    """Return the common name (last field) from a SpeciesNet taxonomy string.

    Format: uuid;class;order;family;genus;species;common_name
    Falls back to the full string if it cannot be parsed.
    """
    parts = label_str.split(";")
    if len(parts) >= 7 and parts[-1]:
        return parts[-1]
    # No common name — use genus + species if available
    if len(parts) >= 6 and parts[4] and parts[5]:
        return f"{parts[4]} {parts[5]}"
    return label_str


def build_dataset(labels_path: str, images_dir: str) -> fo.Dataset:
    with open(labels_path) as f:
        data = json.load(f)

    # Delete existing dataset with the same name so re-runs start clean
    if fo.dataset_exists(DATASET_NAME):
        fo.delete_dataset(DATASET_NAME)

    dataset = fo.Dataset(DATASET_NAME)
    samples = []

    for pred in data["predictions"]:
        image_path = os.path.join(images_dir, pred["filepath"])
        if not os.path.isfile(image_path):
            continue

        # ── Top-1 classification ──────────────────────────────────────────
        top_label = parse_taxonomy(pred["prediction"])
        top_classification = fo.Classification(
            label=top_label,
            confidence=pred.get("prediction_score"),
            prediction_source=pred.get("prediction_source"),
            taxonomy=pred["prediction"],
        )

        # ── Top-5 classifications ─────────────────────────────────────────
        clf_data = pred.get("classifications", {})
        top5 = fo.Classifications(
            classifications=[
                fo.Classification(
                    label=parse_taxonomy(cls),
                    confidence=score,
                    taxonomy=cls,
                )
                for cls, score in zip(
                    clf_data.get("classes", []),
                    clf_data.get("scores", []),
                )
            ]
        )

        # ── Detections ────────────────────────────────────────────────────
        det_list = []
        for det in pred.get("detections", []):
            x, y, w, h = det["bbox"]
            det_list.append(
                fo.Detection(
                    label=det.get("label", "animal"),
                    bounding_box=[x, y, w, h],
                    confidence=det.get("conf"),
                )
            )
        detections = fo.Detections(detections=det_list)

        sample = fo.Sample(filepath=image_path)
        sample["prediction"] = top_classification
        sample["top5"] = top5
        sample["detections"] = detections
        samples.append(sample)

    dataset.add_samples(samples)
    dataset.save()
    print(f"Loaded {len(dataset)} samples into dataset '{DATASET_NAME}'")
    return dataset


if __name__ == "__main__":
    dataset = build_dataset(LABELS_PATH, IMAGES_DIR)
    session = fo.launch_app(dataset)
    session.wait()
