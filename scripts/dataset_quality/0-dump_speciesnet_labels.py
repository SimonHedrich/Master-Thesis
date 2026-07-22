"""Dump the SpeciesNet classifier's idx → label map to data/speciesnet_labels.json.

Step 0 of docs/plans/2026-06-09_flag-cross-species-contamination-multi-box.md.

`data/speciesnet_classes.json` stores only integer indices ([0,1,2,...]) which are
useless for taxonomy. The full label strings
("uuid;class;order;family;genus;species;common") live only inside the SpeciesNet
classifier. This one-time dump resolves idx → label so the flagging script
(14-flag_multi_animal_contamination.py) needs no GPU/Docker and is cheap to re-run.

Reuses the exact label-extraction logic from 7-filter_speciesnet.py
(load_speciesnet_labels) so the index ordering is guaranteed identical to the one
used to write speciesnet_results.jsonl.

Requires the `speciesnet` package and downloads the classifier weights on first run
(~214 MB from Kaggle). `speciesnet` is part of the default training container
(`make run`) and also installs cleanly on the host Python 3.13 environment:

    uv run python scripts/dataset_quality/0-dump_speciesnet_labels.py
    # (re)install the pinned yolov5 afterwards if speciesnet downgraded it:
    pip install 'yolov5==7.0.13'

Output: data/speciesnet_labels.json  — {"<idx>": "uuid;class;order;family;genus;species;common", ...}
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = REPO_ROOT / "data" / "speciesnet_labels.json"


def load_speciesnet_labels() -> dict[int, str]:
    """Load SpeciesNet classifier labels as {int_idx: label_string}.

    Identical extraction logic to 7-filter_speciesnet.py:load_speciesnet_labels so
    the resulting index ordering matches speciesnet_results.jsonl exactly.
    """
    from speciesnet import SpeciesNet, DEFAULT_MODEL

    print("Loading SpeciesNet EfficientNetV2-M for label lookup …")
    clf = SpeciesNet(DEFAULT_MODEL, components="classifier", geofence=False).classifier
    print("  Classifier loaded. Extracting labels …")

    for attr in ("class_names", "labels"):
        if not hasattr(clf, attr):
            continue
        raw = getattr(clf, attr)

        # Dict-like: {int_idx: label_string}
        if hasattr(raw, "items"):
            result: dict[int, str] = {int(k): str(v) for k, v in raw.items()}
            sample = next(iter(result.values()), "")
            if isinstance(sample, str) and ";" in sample:
                print(f"  Labels from clf.{attr} ({len(result)} classes). "
                      f"[0] = {sample[:70]}")
                return result

        # Sequence
        labels_list = list(raw)
        if labels_list and isinstance(labels_list[0], str) and ";" in labels_list[0]:
            result = {i: v for i, v in enumerate(labels_list)}
            print(f"  Labels from clf.{attr} ({len(result)} classes). "
                  f"[0] = {labels_list[0][:70]}")
            return result

    raise RuntimeError(
        "Could not extract string labels from SpeciesNet classifier.\n"
        "Tried clf.class_names and clf.labels — neither returned 'uuid;...' strings."
    )


def main() -> None:
    idx_to_label = load_speciesnet_labels()
    out = {str(k): v for k, v in sorted(idx_to_label.items())}
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"Wrote {len(out):,} labels → {OUTPUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
