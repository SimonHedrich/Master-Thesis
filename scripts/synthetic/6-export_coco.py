#!/usr/bin/env python3
"""
Stage 6 — COCO export

Merges md_detections, single_detect_flags, and manual_labels into
COCO-format JSON files for training, validation, and test splits.

Usage:
    cd /home/debian/Master-Thesis
    python3 scripts/synthetic/6-export_coco.py              # all splits
    python3 scripts/synthetic/6-export_coco.py --split train
    python3 scripts/synthetic/6-export_coco.py --split val
    python3 scripts/synthetic/6-export_coco.py --split test

Outputs:
    data/synthetic/annotations_train.json
    data/synthetic/annotations_val.json
    data/synthetic/annotations_test.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────

REPO_ROOT          = Path(__file__).resolve().parent.parent.parent
DATA_DIR           = REPO_ROOT / "data" / "synthetic"
INDEX              = DATA_DIR / "index.jsonl"
TEST_INDEX         = DATA_DIR / "test_index.jsonl"
MD_DETECTIONS      = DATA_DIR / "md_detections.jsonl"
MD_DETECTIONS_TEST = DATA_DIR / "md_detections_test.jsonl"
FLAGS_FILE         = DATA_DIR / "single_detect_flags.jsonl"
LABELS_FILE        = DATA_DIR / "manual_labels.jsonl"

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
        fp  = entry.get("filepath", "")
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


def yolo_to_coco(bbox: list, img_w: int, img_h: int) -> list[int]:
    xc, yc, w, h = bbox
    x1   = round((xc - w / 2) * img_w)
    y1   = round((yc - h / 2) * img_h)
    w_px = round(w * img_w)
    h_px = round(h * img_h)
    # Clamp to image bounds
    x1   = max(0, x1)
    y1   = max(0, y1)
    w_px = min(w_px, img_w - x1)
    h_px = min(h_px, img_h - y1)
    return [x1, y1, w_px, h_px]


def get_bboxes(
    fp: str,
    md_entry: dict,
    flags: dict[str, str],
    manual: dict[str, dict],
    is_test: bool,
) -> tuple[list[dict] | None, str]:
    """
    Returns (bboxes_or_None, warning_str) per merge logic.
    bboxes_or_None == None means skip the image entirely (no annotation).
    Each bbox dict: {"bbox": [xc,yc,w,h], "source": str, "conf": float|None}
    """
    n_sig        = md_entry.get("n_significant", 0)
    manual_entry = manual.get(fp)
    flag         = flags.get(fp)

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
        if flag == "single" or is_test:
            dets = md_entry.get("detections", [])
            sig  = [d for d in dets if d["conf"] >= CONF_PASS] or dets[:1]
            if sig:
                d = sig[0]
                return [{"bbox": d["bbox"], "source": "megadetector", "conf": d["conf"]}], ""
            return None, f"SKIP  n_sig=1  no detections stored: {fp}"
        # train/val, no flag yet
        dets = md_entry.get("detections", [])
        sig  = [d for d in dets if d["conf"] >= CONF_PASS] or dets[:1]
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


# ── Export ─────────────────────────────────────────────────────────────────────

def export_split(
    split: str,
    md_by_fp: dict[str, dict],
    idx_by_name: dict[str, dict],
    flags: dict[str, str],
    manual: dict[str, dict],
    cat_id_map: dict[str, int],
    categories: list[dict],
    out_path: Path,
    is_test: bool,
) -> dict:
    images:      list[dict] = []
    annotations: list[dict] = []
    img_id  = 1
    ann_id  = 1
    warns:   list[str] = []

    stats = {"total": 0, "annotated": 0, "skipped": 0, "ann_per_img": []}
    class_ann_counts: dict[str, int] = defaultdict(int)

    for fp in sorted(md_by_fp):
        md    = md_by_fp[fp]
        fname = Path(fp).name
        idx   = idx_by_name.get(fname)
        if not idx:
            continue
        if split in ("train", "val") and idx.get("split") != split:
            continue

        stats["total"] += 1

        bboxes, warn = get_bboxes(fp, md, flags, manual, is_test)
        if warn:
            warns.append(warn)

        img_entry: dict = {
            "id":        img_id,
            "file_name": fp,
            "width":     md["width"],
            "height":    md["height"],
            "band":      idx.get("band", ""),
            "split":     idx.get("split", split),
            "shot_type": idx.get("shot_type", ""),
            "distance":  idx.get("distance", ""),
            "lighting":  idx.get("lighting", ""),
            "occlusion": idx.get("occlusion", ""),
        }
        if "behavior" in idx:
            img_entry["behavior"] = idx["behavior"]
        images.append(img_entry)

        if bboxes is None:
            stats["skipped"] += 1
        else:
            stats["annotated"] += 1
            cls_norm = normalize_class(md.get("class", ""))
            cat_id   = cat_id_map.get(cls_norm, 1)
            count    = 0
            for b in bboxes:
                coco_bbox = yolo_to_coco(b["bbox"], md["width"], md["height"])
                w_px, h_px = coco_bbox[2], coco_bbox[3]
                annotations.append({
                    "id":          ann_id,
                    "image_id":    img_id,
                    "category_id": cat_id,
                    "bbox":        coco_bbox,
                    "area":        w_px * h_px,
                    "iscrowd":     0,
                    "source":      b.get("source", ""),
                    "conf":        b.get("conf"),
                })
                ann_id  += 1
                count   += 1
                class_ann_counts[cls_norm] += 1
            stats["ann_per_img"].append(count)

        img_id += 1

    # Warnings
    if warns:
        print(f"\n  Warnings ({len(warns)}):")
        for w in warns[:50]:
            print(f"    {w}")
        if len(warns) > 50:
            print(f"    ... and {len(warns) - 50} more")

    coco = {
        "info": {
            "description":  f"Synthetic wildlife {split} images",
            "date_created": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "version":      "1.0",
        },
        "licenses":    [],
        "categories":  categories,
        "images":      images,
        "annotations": annotations,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)

    # Summary
    n_tot = stats["total"]
    n_ann = stats["annotated"]
    n_skp = stats["skipped"]
    n_bxs = len(annotations)
    print(f"\n  [{split}] {out_path.name}")
    print(f"    images : {n_tot} total  |  {n_ann} annotated  |  {n_skp} skipped")
    print(f"    boxes  : {n_bxs} total")
    if stats["ann_per_img"]:
        mean   = statistics.mean(stats["ann_per_img"])
        median = statistics.median(stats["ann_per_img"])
        print(f"    boxes/annotated image : mean={mean:.2f}  median={median:.1f}")

    if class_ann_counts:
        top = sorted(class_ann_counts.items(), key=lambda x: -x[1])[:20]
        print(f"    top classes by annotation count:")
        for cls, cnt in top:
            print(f"      {cls:<30s} {cnt}")

    stats["class_ann_counts"] = dict(class_ann_counts)
    return stats


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 6 — COCO export")
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="all",
        help="Which split(s) to export (default: all)",
    )
    args = parser.parse_args()
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    print("── Stage 6: COCO export ─────────────────────────────────────", flush=True)

    # Load data
    print("  Loading index files ...", end=" ", flush=True)
    idx_by_name      = {e["filename"]: e for e in read_jsonl(INDEX)}
    test_idx_by_name = {e["filename"]: e for e in read_jsonl(TEST_INDEX)}
    print(f"{len(idx_by_name)} train/val  |  {len(test_idx_by_name)} test", flush=True)

    print("  Loading MD detections ...", end=" ", flush=True)
    md_train = {e["filepath"]: e for e in read_jsonl(MD_DETECTIONS)}
    md_test  = {e["filepath"]: e for e in read_jsonl(MD_DETECTIONS_TEST)}
    print(f"{len(md_train)} train/val  |  {len(md_test)} test", flush=True)

    print("  Loading single-detect flags ...", end=" ", flush=True)
    flags = load_flags(FLAGS_FILE)
    print(f"{len(flags)} flags", flush=True)

    print("  Loading manual labels ...", end=" ", flush=True)
    manual = load_last_per_key(LABELS_FILE, "filepath")
    print(f"{len(manual)} labeled", flush=True)

    # Build unified category list from both index files
    all_classes: set[str] = set()
    for entries in (idx_by_name.values(), test_idx_by_name.values()):
        for e in entries:
            all_classes.add(normalize_class(e.get("class", "")))
    all_classes.discard("")
    categories  = [
        {"id": i + 1, "name": cls, "supercategory": "animal"}
        for i, cls in enumerate(sorted(all_classes))
    ]
    cat_id_map  = {c["name"]: c["id"] for c in categories}
    print(f"  {len(categories)} categories", flush=True)

    # Export each requested split
    for split in splits:
        is_test    = split == "test"
        md_by_fp   = md_test  if is_test else md_train
        idx_lookup = test_idx_by_name if is_test else idx_by_name
        out_path   = DATA_DIR / f"annotations_{split}.json"
        export_split(
            split      = split,
            md_by_fp   = md_by_fp,
            idx_by_name= idx_lookup,
            flags      = flags,
            manual     = manual,
            cat_id_map = cat_id_map,
            categories = categories,
            out_path   = out_path,
            is_test    = is_test,
        )

    print("\n── Done ─────────────────────────────────────────────────────", flush=True)


if __name__ == "__main__":
    main()
