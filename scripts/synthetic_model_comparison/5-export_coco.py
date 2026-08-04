#!/usr/bin/env python3
"""
Stage 5 — COCO export (per generator-comparison cell)

Adapted from scripts/synthetic/6-export_coco.py for this experiment's
per-cell layout. Two real (non-cosmetic) differences from the production
script:

1. Categories are the FIXED 12-class ids/names from
   data/synthetic_model_comparison/test/annotations_test.json — NOT freshly
   assigned 1..N alphabetically like production's script does. This keeps
   category ids identical across every generator cell and compatible with
   this experiment's real test set and reports/lookalike_groups_v2.csv (which
   are keyed by the original 225-class taxonomy ids, unrenumbered).
2. Field names match this experiment's index.jsonl schema (written by
   1-select_train_subset_incumbent.py): "source_split" (not "split"),
   "pose"/"environment" (not "behavior"). There is no train/val distinction
   at the index level here — every image in a cell is train-only, per
   docs/synthetic-model-comparison/01_experiment-design.md §5 point 3 — so
   unlike production's script this always exports a single annotations.json
   per cell (no split filtering, no test-index/test-detections handling).

Merges index.jsonl + md_detections.jsonl + single_detect_flags.jsonl +
manual_labels.jsonl for one cell into:
    data/synthetic_model_comparison/train/<generator>/<regime>/annotations.json

Usage:
    uv run python scripts/synthetic_model_comparison/5-export_coco.py \\
        --generator gemini-3.1-flash-image-preview --prompt-regime full
    uv run python scripts/synthetic_model_comparison/5-export_coco.py --all
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_ROOT = REPO_ROOT / "data" / "synthetic_model_comparison" / "train"
TEST_ANNOTATIONS = REPO_ROOT / "data" / "synthetic_model_comparison" / "test" / "annotations_test.json"

CONF_PASS = 0.5


# ── Helpers ────────────────────────────────────────────────────────────────────

def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def load_flags(path: Path) -> dict[str, str]:
    """Load single_detect_flags.jsonl, replaying undo entries."""
    flags: dict[str, str] = {}
    for entry in read_jsonl(path):
        fp = entry.get("filepath", "")
        dec = entry.get("decision", "")
        if dec == "undo":
            flags.pop(fp, None)
        elif dec in ("single", "multi"):
            flags[fp] = dec
    return flags


def load_last_per_key(path: Path, key: str) -> dict[str, dict]:
    """Read a JSONL file; last entry per key value wins."""
    result: dict[str, dict] = {}
    for entry in read_jsonl(path):
        k = entry.get(key, "")
        if k:
            result[k] = entry
    return result


def normalize_class(name: str) -> str:
    return name.strip().lower().replace("_", " ")


def load_fixed_categories() -> tuple[list[dict], dict[str, int]]:
    """Load the frozen 12-class categories from the experiment's real test set."""
    with open(TEST_ANNOTATIONS, encoding="utf-8") as f:
        test_coco = json.load(f)
    categories = sorted(test_coco["categories"], key=lambda c: c["id"])
    cat_id_map = {normalize_class(c["name"]): c["id"] for c in categories}
    return categories, cat_id_map


def yolo_to_coco(bbox: list, img_w: int, img_h: int) -> list[int]:
    xc, yc, w, h = bbox
    x1 = round((xc - w / 2) * img_w)
    y1 = round((yc - h / 2) * img_h)
    w_px = round(w * img_w)
    h_px = round(h * img_h)
    x1 = max(0, x1)
    y1 = max(0, y1)
    w_px = min(w_px, img_w - x1)
    h_px = min(h_px, img_h - y1)
    return [x1, y1, w_px, h_px]


def get_bboxes(
    fp: str,
    md_entry: dict,
    flags: dict[str, str],
    manual: dict[str, dict],
) -> tuple[list[dict] | None, str]:
    """
    Returns (bboxes_or_None, warning_str) per merge logic.
    bboxes_or_None == None means skip the image entirely (no annotation).
    Each bbox dict: {"bbox": [xc,yc,w,h], "source": str, "conf": float|None}
    """
    n_sig = md_entry.get("n_significant", 0)
    manual_entry = manual.get(fp)
    flag = flags.get(fp)

    if manual_entry and manual_entry.get("skipped"):
        return None, ""

    if n_sig == 0:
        if manual_entry:
            return manual_entry["bboxes"], ""
        return None, f"SKIP  n_sig=0  no manual label: {fp}"

    if n_sig == 1:
        if flag == "multi":
            if manual_entry:
                return manual_entry["bboxes"], ""
            return None, f"SKIP  n_sig=1 flagged multi  no manual label: {fp}"
        if flag == "single":
            dets = md_entry.get("detections", [])
            sig = [d for d in dets if d["conf"] >= CONF_PASS] or dets[:1]
            if sig:
                d = sig[0]
                return [{"bbox": d["bbox"], "source": "megadetector", "conf": d["conf"]}], ""
            return None, f"SKIP  n_sig=1  no detections stored: {fp}"
        # no flag yet — proceed with MD's best-effort box, warn (doesn't block export)
        dets = md_entry.get("detections", [])
        sig = [d for d in dets if d["conf"] >= CONF_PASS] or dets[:1]
        if sig:
            d = sig[0]
            return (
                [{"bbox": d["bbox"], "source": "megadetector", "conf": d["conf"]}],
                f"WARN  n_sig=1  no flag, using MD best-effort: {fp}",
            )
        return None, f"SKIP  n_sig=1  no flag and no dets: {fp}"

    # n_sig >= 2
    if manual_entry:
        return manual_entry["bboxes"], ""
    dets = [d for d in md_entry.get("detections", []) if d["conf"] >= CONF_PASS]
    warn = f"WARN  n_sig={n_sig}  no manual label, using MD best-effort: {fp}"
    return (
        [{"bbox": d["bbox"], "source": "megadetector", "conf": d["conf"]} for d in dets],
        warn,
    )


def discover_cells() -> list[tuple[str, str, Path]]:
    cells = []
    for index_path in sorted(TRAIN_ROOT.glob("*/*/index.jsonl")):
        cell_dir = index_path.parent
        cells.append((cell_dir.parent.name, cell_dir.name, cell_dir))
    return cells


# ── Export ─────────────────────────────────────────────────────────────────────

def export_cell(
    generator: str,
    regime: str,
    cell_dir: Path,
    categories: list[dict],
    cat_id_map: dict[str, int],
) -> None:
    index_by_name = {e["filename"]: e for e in read_jsonl(cell_dir / "index.jsonl")}
    md_by_fp = {e["filepath"]: e for e in read_jsonl(cell_dir / "md_detections.jsonl")}
    flags = load_flags(cell_dir / "single_detect_flags.jsonl")
    manual = load_last_per_key(cell_dir / "manual_labels.jsonl", "filepath")

    print(f"\n[{generator}/{regime}] {len(index_by_name)} indexed images, "
          f"{len(md_by_fp)} MD detections, {len(flags)} flags, {len(manual)} manual labels")

    images: list[dict] = []
    annotations: list[dict] = []
    img_id = 1
    ann_id = 1
    warns: list[str] = []

    stats = {"total": 0, "annotated": 0, "skipped": 0, "ann_per_img": []}
    class_ann_counts: dict[str, int] = defaultdict(int)

    for fp in sorted(md_by_fp):
        md = md_by_fp[fp]
        fname = Path(fp).name
        idx = index_by_name.get(fname)
        if not idx:
            continue

        stats["total"] += 1

        bboxes, warn = get_bboxes(fp, md, flags, manual)
        if warn:
            warns.append(warn)

        img_entry: dict = {
            "id": img_id,
            "file_name": fp,
            "width": md["width"],
            "height": md["height"],
            "band": idx.get("band", ""),
            "source_split": idx.get("source_split", ""),
            "shot_type": idx.get("shot_type", ""),
            "distance": idx.get("distance", ""),
            "lighting": idx.get("lighting", ""),
            "occlusion": idx.get("occlusion", ""),
            "pose": idx.get("pose", ""),
            "environment": idx.get("environment", ""),
        }
        images.append(img_entry)

        if bboxes is None:
            stats["skipped"] += 1
        else:
            stats["annotated"] += 1
            cls_norm = normalize_class(md.get("class", ""))
            if cls_norm not in cat_id_map:
                warns.append(f"WARN  class '{cls_norm}' not in the frozen 12-class list: {fp}")
                img_id += 1
                continue
            cat_id = cat_id_map[cls_norm]
            count = 0
            for b in bboxes:
                coco_bbox = yolo_to_coco(b["bbox"], md["width"], md["height"])
                w_px, h_px = coco_bbox[2], coco_bbox[3]
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox": coco_bbox,
                    "area": w_px * h_px,
                    "iscrowd": 0,
                    "source": b.get("source", ""),
                    "conf": b.get("conf"),
                })
                ann_id += 1
                count += 1
                class_ann_counts[cls_norm] += 1
            stats["ann_per_img"].append(count)

        img_id += 1

    if warns:
        print(f"  Warnings ({len(warns)}):")
        for w in warns[:50]:
            print(f"    {w}")
        if len(warns) > 50:
            print(f"    ... and {len(warns) - 50} more")

    coco = {
        "info": {
            "description": f"Synthetic model comparison train images — {generator}/{regime}",
            "date_created": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "version": "1.0",
        },
        "licenses": [],
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }

    out_path = cell_dir / "annotations.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)

    n_tot, n_ann, n_skp = stats["total"], stats["annotated"], stats["skipped"]
    print(f"  wrote {out_path}")
    print(f"    images : {n_tot} total  |  {n_ann} annotated  |  {n_skp} skipped")
    print(f"    boxes  : {len(annotations)} total")
    if stats["ann_per_img"]:
        mean = statistics.mean(stats["ann_per_img"])
        median = statistics.median(stats["ann_per_img"])
        print(f"    boxes/annotated image : mean={mean:.2f}  median={median:.1f}")
    if class_ann_counts:
        print("    per-class annotation counts:")
        for cls, cnt in sorted(class_ann_counts.items(), key=lambda x: -x[1]):
            print(f"      {cls:<25s} {cnt}")

    present = set(class_ann_counts.keys())
    missing = {normalize_class(c["name"]) for c in categories} - present
    if missing:
        print(f"    zero-annotation classes (stay in the 12-class head, never predicted yet): "
              f"{sorted(missing)}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generator", metavar="NAME")
    parser.add_argument("--prompt-regime", choices=["full", "compressed", "maxlen"])
    parser.add_argument("--all", action="store_true", help="Export every cell found under train/*/*/index.jsonl")
    args = parser.parse_args()

    if not TEST_ANNOTATIONS.exists():
        raise SystemExit(f"ERROR: {TEST_ANNOTATIONS} not found — build the test subset first "
                          f"(scripts/synthetic_model_comparison/0-build_test_subset.py)")
    categories, cat_id_map = load_fixed_categories()
    print(f"Loaded {len(categories)} fixed categories from {TEST_ANNOTATIONS.relative_to(REPO_ROOT)}")

    if args.all:
        cells = discover_cells()
    elif args.generator and args.prompt_regime:
        cell_dir = TRAIN_ROOT / args.generator / args.prompt_regime
        if not (cell_dir / "index.jsonl").exists():
            raise SystemExit(f"ERROR: no index.jsonl found at {cell_dir}")
        cells = [(args.generator, args.prompt_regime, cell_dir)]
    else:
        parser.error("pass --all, or both --generator and --prompt-regime")
        return

    print("── Stage 5: COCO export ─────────────────────────────────────")
    for generator, regime, cell_dir in cells:
        export_cell(generator, regime, cell_dir, categories, cat_id_map)
    print("\n── Done ─────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
