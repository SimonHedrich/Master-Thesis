#!/usr/bin/env python3
"""
Stage 12 — Dataset split assignment for real images

Scores all quality-passed real images with a composite quality metric Q,
then assigns train/val/test splits using per-band allocation rules.

Usage:
    cd /home/debian/Master-Thesis
    python3 scripts/dataset_quality/12-assign_dataset_splits.py
    python3 scripts/dataset_quality/12-assign_dataset_splits.py --dry-run
    python3 scripts/dataset_quality/12-assign_dataset_splits.py --include-negatives
    python3 scripts/dataset_quality/12-assign_dataset_splits.py --max-workers 16

Outputs:
    reports/dataset_split_manifest.json
    reports/dataset_split_summary.json
    data/real/annotations_train.json
    data/real/annotations_val.json
    data/real/annotations_test.json
    reports/dataset_split_report.md
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────

REPO_ROOT      = Path(__file__).resolve().parents[2]
SOURCES        = ["gbif", "inaturalist", "wikimedia", "openimages", "images_cv"]
CLASSES_CSV    = REPO_ROOT / "reports" / "class_distribution_reviewed.csv"
CLASSES_225    = REPO_ROOT / "reports" / "classes_225.csv"
STUDENT_LABELS = REPO_ROOT / "resources" / "2026-03-19_student_model_labels.txt"
DATA_REAL_DIR  = REPO_ROOT / "data" / "real"
MANIFEST_PATH  = REPO_ROOT / "reports" / "dataset_split_manifest.json"
SUMMARY_PATH   = REPO_ROOT / "reports" / "dataset_split_summary.json"
REPORT_PATH    = REPO_ROOT / "reports" / "dataset_split_report.md"
COCO_HUMANS_DIR = REPO_ROOT / "data" / "coco_humans"
BLANKS_DIR      = REPO_ROOT / "data" / "blanks"

SEED     = 42
CONF_SIG = 0.5  # MegaDetector significance threshold

# ── JSONL helper ───────────────────────────────────────────────────────────────

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

# ── Class name utilities ───────────────────────────────────────────────────────

def normalize_class(name: str) -> str:
    return name.strip().lower().replace("_", " ")


def _strip_apostrophe(name: str) -> str:
    return name.replace("'", "").replace("’", "")


def _build_canonical_lookup(classes_csv: Path) -> dict[str, str]:
    """Return {stripped_name: canonical_name} for apostrophe-containing class names."""
    if not classes_csv.exists():
        return {}
    lookup: dict[str, str] = {}
    with open(classes_csv, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            name    = row["common_name"].strip().lower()
            stripped = _strip_apostrophe(name)
            if stripped != name:
                lookup[stripped] = name
    return lookup

# ── Input loading ──────────────────────────────────────────────────────────────

def load_student_labels(path: Path) -> tuple[set[str], list[dict]]:
    """Parse student_model_labels.txt → (valid_classes set, COCO categories list)."""
    names: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts  = line.split(";")
            common = parts[-1].strip().lower()
            if common:
                names.append(common)
    names_sorted = sorted(set(names))
    categories = [
        {"id": i + 1, "name": n, "supercategory": "animal"}
        for i, n in enumerate(names_sorted)
    ]
    return set(names_sorted), categories


def load_class_bands(csv_path: Path) -> dict[str, dict]:
    """Read class_distribution_reviewed.csv → {class: {effective_pool, band}}."""
    result: dict[str, dict] = {}
    with open(csv_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            cls = normalize_class(row["class"])
            if cls == "unmatched":
                continue
            raw_pool = row["effective_pool"].strip()
            pool = int(raw_pool) if raw_pool.lstrip("-").isdigit() else 0
            pool = max(0, pool)
            if pool < 150:
                band = "A"
            elif pool < 250:
                band = "B"
            elif pool < 400:
                band = "C"
            else:
                band = "D"
            result[cls] = {"effective_pool": pool, "band": band}
    return result


def load_all_passed_images(
    valid_classes: set[str],
    canonical_lookup: dict[str, str],
) -> dict[str, list[dict]]:
    """Stream all filter_results.jsonl files → {class: [image_dict, ...]}."""
    per_class: dict[str, list[dict]] = defaultdict(list)
    grand_total = 0

    for source in SOURCES:
        results_path = REPO_ROOT / "data" / source / "filter_results.jsonl"
        entries = read_jsonl(results_path)
        count = 0
        for entry in entries:
            if not entry.get("passed"):
                continue
            fp       = entry["filepath"]
            raw_cls  = normalize_class(Path(fp).parent.name)
            cls      = canonical_lookup.get(_strip_apostrophe(raw_cls), raw_cls)
            if cls not in valid_classes:
                continue
            bbox       = entry.get("bbox")
            bbox_conf  = entry.get("bbox_conf")
            detections = entry.get("detections") or []
            Q, comps   = compute_quality_score(bbox, bbox_conf, detections)
            per_class[cls].append({
                "filepath":         fp,
                "source":           source,
                "Q":                Q,
                "score_components": comps,
                "bbox":             bbox,
                "bbox_conf":        bbox_conf,
            })
            count += 1
        grand_total += count
        print(f"  {source:<14s}: {count:>6,} passed images")

    print(f"  {'Total':<14s}: {grand_total:>6,} images across {len(per_class)} classes")
    return dict(per_class)

# ── Quality scoring ────────────────────────────────────────────────────────────

def compute_quality_score(
    bbox: list | None,
    bbox_conf: float | None,
    detections: list[dict],
) -> tuple[float, dict]:
    """Compute composite quality score Q ∈ [0, 1] with component breakdown."""
    n_sig        = len([d for d in detections if d.get("conf", 0) >= CONF_SIG])
    single_score = {0: 0.50, 1: 1.00, 2: 0.60}.get(n_sig, 0.30)
    conf_score   = float(bbox_conf) if bbox_conf is not None else 0.30

    if bbox is None:
        Q = 0.20 * single_score + 0.25 * conf_score
        return round(Q, 6), {
            "area_score":    0.0, "area_frac":     0.0,
            "edge_score":    0.0, "min_margin":    0.0,
            "single_score":  round(single_score, 6), "n_significant": n_sig,
            "conf_score":    round(conf_score, 6),
            "hard_excluded": False,
        }

    xc, yc, w, h = bbox
    area_frac = w * h
    xmin = xc - w / 2
    xmax = xc + w / 2
    ymin = yc - h / 2
    ymax = yc + h / 2
    margin = min(xmin, 1.0 - xmax, ymin, 1.0 - ymax)

    if area_frac < 0.01 or margin < 0.0:
        return 0.0, {
            "area_score":    0.0, "area_frac":     round(area_frac, 6),
            "edge_score":    0.0, "min_margin":    round(margin, 6),
            "single_score":  round(single_score, 6), "n_significant": n_sig,
            "conf_score":    round(conf_score, 6),
            "hard_excluded": True,
        }

    if area_frac < 0.02:
        area_score = 0.0
    elif area_frac < 0.04:
        area_score = (area_frac - 0.02) / 0.02
    elif area_frac <= 0.40:
        area_score = 1.0
    elif area_frac <= 0.70:
        area_score = (0.70 - area_frac) / 0.30
    else:
        area_score = 0.0

    edge_score = 1.0 if margin >= 0.02 else margin / 0.02

    Q = 0.30 * area_score + 0.25 * edge_score + 0.20 * single_score + 0.25 * conf_score
    return round(Q, 6), {
        "area_score":    round(area_score, 6), "area_frac":     round(area_frac, 6),
        "edge_score":    round(edge_score, 6), "min_margin":    round(margin, 6),
        "single_score":  round(single_score, 6), "n_significant": n_sig,
        "conf_score":    round(conf_score, 6),
        "hard_excluded": False,
    }

# ── Allocation helpers ─────────────────────────────────────────────────────────

def _percentile_bounds(images: list[dict], p_lo: float, p_hi: float) -> tuple[float, float]:
    qs = [img["Q"] for img in images]
    if len(qs) < 2:
        return (0.0, 1.0)
    return float(np.percentile(qs, p_lo)), float(np.percentile(qs, p_hi))


def _select_val(
    pool: list[dict],
    val_n: int,
    exclude_fps: set[str],
    rng: random.Random,
) -> list[dict]:
    """Sample val_n images from 30th–70th Q percentile, excluding exclude_fps."""
    candidates = [img for img in pool if img["filepath"] not in exclude_fps]
    if not candidates:
        return []
    p30, p70 = _percentile_bounds(candidates, 30, 70)
    mid_pool  = [img for img in candidates if p30 <= img["Q"] <= p70]
    if not mid_pool:
        mid_pool = candidates
    if len(mid_pool) < val_n:
        print(f"    WARNING: only {len(mid_pool)} val candidates, requested {val_n}")
    take = min(val_n, len(mid_pool))
    return rng.sample(sorted(mid_pool, key=lambda x: x["filepath"]), take)


def stratified_sample_by_source(
    images: list[dict],
    n: int,
    rng: random.Random,
) -> list[dict]:
    """Proportional stratified sample across sources using largest-remainder allocation."""
    if n <= 0:
        return []
    n = min(n, len(images))

    by_source: dict[str, list[dict]] = defaultdict(list)
    for img in images:
        by_source[img["source"]].append(img)

    sources = sorted(by_source.keys())
    total   = len(images)
    exact   = {src: n * len(by_source[src]) / total for src in sources}
    alloc   = {src: math.floor(v) for src, v in exact.items()}
    remainder = n - sum(alloc.values())
    order     = sorted(sources, key=lambda s: -(exact[s] - alloc[s]))
    for src in order[:remainder]:
        alloc[src] += 1

    result: list[dict] = []
    for src in sources:
        pool = sorted(by_source[src], key=lambda x: x["filepath"])
        take = min(alloc[src], len(pool))
        result.extend(rng.sample(pool, take) if take < len(pool) else pool[:])

    # Fill shortfall if any source had fewer images than its allocation
    if len(result) < n:
        assigned = {img["filepath"] for img in result}
        leftover = sorted(
            [img for img in images if img["filepath"] not in assigned],
            key=lambda x: x["filepath"],
        )
        extra = min(n - len(result), len(leftover))
        result.extend(rng.sample(leftover, extra) if extra < len(leftover) else leftover[:])
    return result


def greedy_train_with_cap(images: list[dict], train_n: int) -> list[dict]:
    """Top-Q greedy selection with source diversity cap."""
    if train_n <= 0 or not images:
        return []
    n_sources    = len({img["source"] for img in images})
    cap          = 0.60 if n_sources >= 3 else 0.75 if n_sources == 2 else 1.0
    max_per_src  = math.floor(train_n * cap)
    sorted_imgs  = sorted(images, key=lambda x: (-x["Q"], x["filepath"]))
    src_counts: dict[str, int] = defaultdict(int)
    train: list[dict] = []

    for img in sorted_imgs:
        if len(train) >= train_n:
            break
        src = img["source"]
        if n_sources > 1 and src_counts[src] >= max_per_src:
            continue
        train.append(img)
        src_counts[src] += 1

    # Relax cap if quota wasn't filled
    if len(train) < train_n:
        assigned = {img["filepath"] for img in train}
        for img in sorted_imgs:
            if len(train) >= train_n:
                break
            if img["filepath"] not in assigned:
                train.append(img)
                assigned.add(img["filepath"])
    return train


def compute_band_d_sizes(pool: int) -> tuple[int, int, int]:
    """Return (test_n, val_n, train_n) for Band D sub-ranges."""
    if pool < 1000:
        test_n = min(max(math.floor(pool * 0.20), 50), 200)
        val_n  = min(max(math.floor(pool * 0.07), 20), 70)
    elif pool < 5000:
        test_n = min(math.floor(pool * 0.15), 500)
        val_n  = 100
    else:
        test_n = 500
        val_n  = 150
    train_n = max(0, min(pool - test_n - val_n, 1500))
    return test_n, val_n, train_n

# ── Band allocators ────────────────────────────────────────────────────────────

def assign_splits(
    images: list[dict],
    csv_pool: int,
    band: str,
    rng: random.Random,
) -> dict:
    """Assign train/val/test splits for one class using band-specific rules."""
    if band == "A":
        return _assign_band_a(images)
    elif band == "B":
        return _assign_band_b(images, csv_pool, rng)
    elif band == "C":
        return _assign_band_c(images, rng)
    else:
        return _assign_band_d(images, csv_pool, rng)


def _assign_band_a(images: list[dict]) -> dict:
    # All passed images → test; no quality filtering for Band A (data is too scarce)
    return {"train": [], "val": [], "test": images[:], "surplus": []}


def _assign_band_b(images: list[dict], csv_pool: int, rng: random.Random) -> dict:
    active  = [img for img in images if not img["score_components"]["hard_excluded"]]
    val_n   = max(10, csv_pool - 85 - 50) if csv_pool < 155 else 20
    train_n = 85

    sorted_active = sorted(active, key=lambda x: (-x["Q"], x["filepath"]))
    train         = sorted_active[:train_n]
    train_fps     = {img["filepath"] for img in train}

    val           = _select_val(active, val_n, train_fps, rng)
    val_fps       = {img["filepath"] for img in val}

    test          = [
        img for img in active
        if img["filepath"] not in train_fps and img["filepath"] not in val_fps
    ]
    return {"train": train, "val": val, "test": test, "surplus": []}


def _assign_band_c(images: list[dict], rng: random.Random) -> dict:
    active    = [img for img in images if not img["score_components"]["hard_excluded"]]
    train_n   = 170
    val_n     = 30

    sorted_active = sorted(active, key=lambda x: (-x["Q"], x["filepath"]))
    train         = sorted_active[:train_n]
    train_fps     = {img["filepath"] for img in train}

    val           = _select_val(active, val_n, train_fps, rng)
    val_fps       = {img["filepath"] for img in val}

    test          = [
        img for img in active
        if img["filepath"] not in train_fps and img["filepath"] not in val_fps
    ]
    return {"train": train, "val": val, "test": test, "surplus": []}


def _assign_band_d(images: list[dict], csv_pool: int, rng: random.Random) -> dict:
    active   = [img for img in images if not img["score_components"]["hard_excluded"]]
    test_n, val_n, train_n = compute_band_d_sizes(csv_pool)

    test     = stratified_sample_by_source(active, min(test_n, len(active)), rng)
    test_fps = {img["filepath"] for img in test}

    remaining = [img for img in active if img["filepath"] not in test_fps]
    val       = _select_val(remaining, val_n, set(), rng)
    val_fps   = {img["filepath"] for img in val}

    train_pool = [img for img in remaining if img["filepath"] not in val_fps]
    train      = greedy_train_with_cap(train_pool, train_n)
    train_fps  = {img["filepath"] for img in train}

    assigned = test_fps | val_fps | train_fps
    surplus  = [img for img in active if img["filepath"] not in assigned]
    return {"train": train, "val": val, "test": test, "surplus": surplus}

# ── Image dimension reading ────────────────────────────────────────────────────

def read_image_dimensions(
    filepaths: list[str],
    max_workers: int = 32,
) -> dict[str, tuple[int, int] | None]:
    """Read (width, height) for each filepath using PIL lazy header reads."""
    dims: dict[str, tuple[int, int] | None] = {}
    failures = 0

    def _read_one(fp: str) -> tuple[str, tuple[int, int] | None]:
        try:
            with Image.open(REPO_ROOT / fp) as img:
                return fp, img.size
        except Exception:
            return fp, None

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_read_one, fp): fp for fp in filepaths}
        for future in tqdm(as_completed(futures), total=len(futures), desc="  image dims", unit="img"):
            fp, size = future.result()
            dims[fp] = size
            if size is None:
                failures += 1

    if failures:
        print(f"  WARNING: {failures}/{len(filepaths)} dimension reads failed (those images excluded from COCO)")
    return dims

# ── COCO export ────────────────────────────────────────────────────────────────

def yolo_to_coco(bbox: list, img_w: int, img_h: int) -> list[int]:
    xc, yc, w, h = bbox
    x1   = max(0, round((xc - w / 2) * img_w))
    y1   = max(0, round((yc - h / 2) * img_h))
    w_px = min(round(w * img_w), img_w - x1)
    h_px = min(round(h * img_h), img_h - y1)
    return [x1, y1, w_px, h_px]


def build_coco_split(
    split: str,
    all_assignments: dict[str, dict],
    band_info: dict[str, dict],
    categories: list[dict],
    cat_id_map: dict[str, int],
    dims: dict[str, tuple[int, int] | None],
    human_assignments: dict[str, list[dict]] | None = None,
    blank_assignments: dict[str, list[dict]] | None = None,
) -> dict:
    images:       list[dict] = []
    annotations:  list[dict] = []
    img_id        = 1
    ann_id        = 1
    skipped_dims  = 0
    no_annotation = 0

    for cls in sorted(all_assignments):
        band = band_info.get(cls, {}).get("band", "?")
        for img in all_assignments[cls][split]:
            size = dims.get(img["filepath"])
            if size is None:
                skipped_dims += 1
                continue
            img_w, img_h = size
            images.append({
                "id":            img_id,
                "file_name":     img["filepath"],
                "width":         img_w,
                "height":        img_h,
                "band":          band,
                "source":        img["source"],
                "split":         split,
                "quality_score": round(img["Q"], 6),
            })
            bbox = img.get("bbox")
            if bbox is not None:
                cat_id = cat_id_map.get(cls)
                if cat_id is not None:
                    coco_bbox = yolo_to_coco(bbox, img_w, img_h)
                    w_px, h_px = coco_bbox[2], coco_bbox[3]
                    annotations.append({
                        "id":          ann_id,
                        "image_id":    img_id,
                        "category_id": cat_id,
                        "bbox":        coco_bbox,
                        "area":        w_px * h_px,
                        "iscrowd":     0,
                        "source":      "megadetector",
                        "conf":        img.get("bbox_conf"),
                    })
                    ann_id += 1
            else:
                no_annotation += 1
            img_id += 1

    # Human images (annotated, class "human")
    if human_assignments:
        human_cat_id = cat_id_map.get("human")
        for img in human_assignments.get(split, []):
            size = dims.get(img["filepath"])
            if size is None:
                skipped_dims += 1
                continue
            img_w, img_h = size
            images.append({
                "id":            img_id,
                "file_name":     img["filepath"],
                "width":         img_w,
                "height":        img_h,
                "band":          "negative",
                "source":        "coco_humans",
                "split":         split,
                "quality_score": 1.0,
            })
            bbox = img.get("bbox")
            if bbox is not None and human_cat_id is not None:
                coco_bbox = yolo_to_coco(bbox, img_w, img_h)
                w_px, h_px = coco_bbox[2], coco_bbox[3]
                annotations.append({
                    "id":          ann_id,
                    "image_id":    img_id,
                    "category_id": human_cat_id,
                    "bbox":        coco_bbox,
                    "area":        w_px * h_px,
                    "iscrowd":     0,
                    "source":      "coco_humans",
                    "conf":        1.0,
                })
                ann_id += 1
            img_id += 1

    # Blank images (no annotations — true-negative background samples)
    if blank_assignments:
        for img in blank_assignments.get(split, []):
            size = dims.get(img["filepath"])
            if size is None:
                skipped_dims += 1
                continue
            img_w, img_h = size
            images.append({
                "id":            img_id,
                "file_name":     img["filepath"],
                "width":         img_w,
                "height":        img_h,
                "band":          "negative",
                "source":        "blanks",
                "split":         split,
                "quality_score": 0.5,
            })
            no_annotation += 1
            img_id += 1

    n_imgs = len(images)
    n_anns = len(annotations)
    print(f"  [{split}] {n_imgs:>6,} images  {n_anns:>6,} annotations"
          f"  (skipped dims: {skipped_dims}  no-bbox: {no_annotation})")
    return {
        "info": {
            "description":  f"Wildlife 226-class real {split} images",
            "date_created": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "version":      "1.1",
        },
        "licenses":    [],
        "categories":  categories,
        "images":      images,
        "annotations": annotations,
    }

# ── Manifest ───────────────────────────────────────────────────────────────────

def write_manifest(
    all_assignments: dict[str, dict],
    band_info: dict[str, dict],
    human_assignments: dict[str, list[dict]] | None = None,
    blank_assignments: dict[str, list[dict]] | None = None,
) -> None:
    flat: list[dict] = []
    for cls in sorted(all_assignments):
        band = band_info.get(cls, {}).get("band", "?")
        for split_name in ("train", "val", "test"):
            for img in all_assignments[cls][split_name]:
                flat.append({
                    "filepath":         img["filepath"],
                    "class":            cls,
                    "band":             band,
                    "source":           img["source"],
                    "split":            split_name,
                    "quality_score":    round(img["Q"], 6),
                    "score_components": img["score_components"],
                })

    if human_assignments:
        for split_name in ("train", "val", "test"):
            for img in human_assignments.get(split_name, []):
                flat.append({
                    "filepath":         img["filepath"],
                    "class":            "human",
                    "band":             "negative",
                    "source":           "coco_humans",
                    "split":            split_name,
                    "quality_score":    1.0,
                    "score_components": img["score_components"],
                })

    if blank_assignments:
        for split_name in ("train", "val", "test"):
            for img in blank_assignments.get(split_name, []):
                flat.append({
                    "filepath":         img["filepath"],
                    "class":            "blank",
                    "band":             "negative",
                    "source":           "blanks",
                    "split":            split_name,
                    "quality_score":    0.5,
                    "score_components": img["score_components"],
                })

    by_split = {s: sum(1 for x in flat if x["split"] == s) for s in ("train", "val", "test")}
    manifest = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "seed":       SEED,
            "version":    "1.0",
            "scoring_weights": {"area": 0.30, "edge": 0.25, "single": 0.20, "conf": 0.25},
            "counts": {
                "total_assigned": len(flat),
                "train": by_split["train"],
                "val":   by_split["val"],
                "test":  by_split["test"],
            },
        },
        "splits": flat,
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest written: {len(flat):,} records → {MANIFEST_PATH.name}")

# ── Summary ────────────────────────────────────────────────────────────────────

def _source_counts(images: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for img in images:
        counts[img["source"]] += 1
    return dict(sorted(counts.items()))


def write_summary(
    all_assignments: dict[str, dict],
    band_info: dict[str, dict],
    per_class_images: dict[str, list[dict]],
    human_assignments: dict[str, list[dict]] | None = None,
    blank_assignments: dict[str, list[dict]] | None = None,
) -> None:
    summary: dict[str, dict] = {}
    for cls in sorted(all_assignments):
        info    = band_info.get(cls, {"band": "?", "effective_pool": 0})
        splits  = all_assignments[cls]
        train   = splits["train"]
        val     = splits["val"]
        test    = splits["test"]
        surplus = splits["surplus"]

        # All passed images for this class (including hard-excluded)
        all_imgs   = per_class_images.get(cls, [])
        hard_excl  = sum(1 for img in all_imgs if img["score_components"]["hard_excluded"])
        active     = [img for img in all_imgs if not img["score_components"]["hard_excluded"]]

        pool_actual = len(train) + len(val) + len(test) + len(surplus) + hard_excl

        qs = [img["Q"] for img in active] if active else [0.0]
        q_stats = {
            "mean": round(float(np.mean(qs)), 4),
            "p25":  round(float(np.percentile(qs, 25)), 4),
            "p50":  round(float(np.percentile(qs, 50)), 4),
            "p75":  round(float(np.percentile(qs, 75)), 4),
        }

        summary[cls] = {
            "band":           info["band"],
            "csv_pool":       info["effective_pool"],
            "pool":           pool_actual,
            "hard_excluded":  hard_excl,
            "train":          len(train),
            "val":            len(val),
            "test":           len(test),
            "surplus":        len(surplus),
            "q_stats":        q_stats,
            "train_sources":  _source_counts(train),
            "val_sources":    _source_counts(val),
            "test_sources":   _source_counts(test),
        }

    if human_assignments:
        summary["human"] = {
            "band":          "negative",
            "csv_pool":      0,
            "pool":          sum(len(human_assignments.get(s, [])) for s in ("train", "val", "test")),
            "hard_excluded": 0,
            "train":         len(human_assignments.get("train", [])),
            "val":           len(human_assignments.get("val", [])),
            "test":          len(human_assignments.get("test", [])),
            "surplus":       0,
            "q_stats":       {"mean": 1.0, "p25": 1.0, "p50": 1.0, "p75": 1.0},
            "train_sources": {"coco_humans": len(human_assignments.get("train", []))},
            "val_sources":   {"coco_humans": len(human_assignments.get("val", []))},
            "test_sources":  {"coco_humans": len(human_assignments.get("test", []))},
        }

    if blank_assignments:
        summary["__blanks__"] = {
            "band":          "negative",
            "csv_pool":      0,
            "pool":          sum(len(blank_assignments.get(s, [])) for s in ("train", "val", "test")),
            "hard_excluded": 0,
            "train":         len(blank_assignments.get("train", [])),
            "val":           len(blank_assignments.get("val", [])),
            "test":          len(blank_assignments.get("test", [])),
            "surplus":       0,
            "q_stats":       {"mean": 0.5, "p25": 0.5, "p50": 0.5, "p75": 0.5},
            "train_sources": {"blanks": len(blank_assignments.get("train", []))},
            "val_sources":   {"blanks": len(blank_assignments.get("val", []))},
            "test_sources":  {"blanks": len(blank_assignments.get("test", []))},
        }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary written → {SUMMARY_PATH.name}")

# ── Markdown report ────────────────────────────────────────────────────────────

def write_report_md(
    all_assignments: dict[str, dict],
    band_info: dict[str, dict],
    per_class_images: dict[str, list[dict]],
    seed: int,
) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Per-band aggregates
    band_stats: dict[str, dict] = {b: {"classes": 0, "train": 0, "val": 0, "test": 0, "surplus": 0, "hard_excluded": 0} for b in "ABCD"}
    for cls, splits in all_assignments.items():
        band = band_info.get(cls, {}).get("band", "?")
        if band not in band_stats:
            continue
        bs = band_stats[band]
        bs["classes"]  += 1
        bs["train"]    += len(splits["train"])
        bs["val"]      += len(splits["val"])
        bs["test"]     += len(splits["test"])
        bs["surplus"]  += len(splits["surplus"])
        all_imgs        = per_class_images.get(cls, [])
        bs["hard_excluded"] += sum(1 for img in all_imgs if img["score_components"]["hard_excluded"])

    # Val representativeness per band
    val_q_by_band: dict[str, list[float]] = defaultdict(list)
    all_q_by_band: dict[str, list[float]] = defaultdict(list)
    for cls, splits in all_assignments.items():
        band     = band_info.get(cls, {}).get("band", "?")
        active   = [img for img in per_class_images.get(cls, []) if not img["score_components"]["hard_excluded"]]
        all_q_by_band[band].extend(img["Q"] for img in active)
        val_q_by_band[band].extend(img["Q"] for img in splits["val"])

    # Test-limited classes (<30 real test images)
    test_limited = [
        cls for cls, splits in all_assignments.items()
        if len(splits["test"]) < 30
    ]

    # Source distribution per split
    source_split: dict[str, dict[str, int]] = {s: defaultdict(int) for s in ("train", "val", "test")}
    for cls, splits in all_assignments.items():
        for split_name in ("train", "val", "test"):
            for img in splits[split_name]:
                source_split[split_name][img["source"]] += 1

    lines = [
        f"# Dataset Split Report",
        f"",
        f"**Generated:** {now}  ",
        f"**Seed:** {seed}  ",
        f"**Scoring weights:** area=0.30 · edge=0.25 · single=0.20 · conf=0.25",
        f"",
        f"---",
        f"",
        f"## Band Summary",
        f"",
        f"| Band | Pool threshold | Classes | Train | Val | Test | Surplus | Hard excluded |",
        f"|------|---------------|---------|-------|-----|------|---------|---------------|",
    ]
    thresholds = {"A": "< 150", "B": "150–249", "C": "250–399", "D": "≥ 400"}
    for band in "ABCD":
        bs = band_stats[band]
        lines.append(
            f"| {band} | {thresholds[band]} | {bs['classes']} | {bs['train']:,} | "
            f"{bs['val']:,} | {bs['test']:,} | {bs['surplus']:,} | {bs['hard_excluded']:,} |"
        )

    totals = {k: sum(band_stats[b][k] for b in "ABCD") for k in ("classes", "train", "val", "test", "surplus", "hard_excluded")}
    lines.append(
        f"| **Total** | | **{totals['classes']}** | **{totals['train']:,}** | "
        f"**{totals['val']:,}** | **{totals['test']:,}** | "
        f"**{totals['surplus']:,}** | **{totals['hard_excluded']:,}** |"
    )

    lines += [
        f"",
        f"---",
        f"",
        f"## Source Distribution by Split",
        f"",
        f"| Source | Train | Val | Test |",
        f"|--------|-------|-----|------|",
    ]
    all_sources = sorted(set(
        src for d in source_split.values() for src in d
    ))
    for src in all_sources:
        lines.append(
            f"| {src} | {source_split['train'].get(src, 0):,} | "
            f"{source_split['val'].get(src, 0):,} | {source_split['test'].get(src, 0):,} |"
        )

    lines += [
        f"",
        f"---",
        f"",
        f"## Val Set Representativeness",
        f"",
        f"Val images are sampled from the 30th–70th percentile Q range (representative quality, not cherry-picked).",
        f"",
        f"| Band | All-active mean Q | Val mean Q | Δ |",
        f"|------|-------------------|------------|---|",
    ]
    for band in "ABCD":
        aq = all_q_by_band.get(band, [])
        vq = val_q_by_band.get(band, [])
        if aq:
            all_mean = float(np.mean(aq))
            val_mean = float(np.mean(vq)) if vq else float("nan")
            delta    = val_mean - all_mean if vq else float("nan")
            lines.append(
                f"| {band} | {all_mean:.3f} | "
                f"{'—' if not vq else f'{val_mean:.3f}'} | "
                f"{'—' if not vq else f'{delta:+.3f}'} |"
            )

    if test_limited:
        lines += [
            f"",
            f"---",
            f"",
            f"## Test-Limited Classes (< 30 real test images)",
            f"",
            f"These classes have fewer than 30 real test images; interpret per-class test metrics with caution.",
            f"",
        ]
        for cls in sorted(test_limited):
            n = len(all_assignments[cls]["test"])
            band = band_info.get(cls, {}).get("band", "?")
            lines.append(f"- **{cls}** (Band {band}): {n} test images")

    lines += ["", "---", ""]

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Report written → {REPORT_PATH.name}")

# ── Verification ───────────────────────────────────────────────────────────────

def verify_no_overlap(all_assignments: dict[str, dict]) -> None:
    """Assert no filepath appears in more than one split."""
    train_fps = {img["filepath"] for cls_data in all_assignments.values() for img in cls_data["train"]}
    val_fps   = {img["filepath"] for cls_data in all_assignments.values() for img in cls_data["val"]}
    test_fps  = {img["filepath"] for cls_data in all_assignments.values() for img in cls_data["test"]}

    tv = train_fps & val_fps
    tt = train_fps & test_fps
    vt = val_fps   & test_fps
    if tv or tt or vt:
        if tv: print(f"  ERROR: {len(tv)} filepath(s) in both train and val")
        if tt: print(f"  ERROR: {len(tt)} filepath(s) in both train and test")
        if vt: print(f"  ERROR: {len(vt)} filepath(s) in both val and test")
    else:
        total = len(train_fps) + len(val_fps) + len(test_fps)
        print(f"  OK: no split overlap across {total:,} assigned images")

# ── Dry-run summary ────────────────────────────────────────────────────────────

def print_dry_run(all_assignments: dict[str, dict], band_info: dict[str, dict]) -> None:
    band_counts: dict[str, dict] = {b: {"classes": 0, "train": 0, "val": 0, "test": 0} for b in "ABCD"}
    for cls, splits in all_assignments.items():
        band = band_info.get(cls, {}).get("band", "?")
        if band not in band_counts:
            continue
        bc = band_counts[band]
        bc["classes"] += 1
        bc["train"]   += len(splits["train"])
        bc["val"]     += len(splits["val"])
        bc["test"]    += len(splits["test"])

    print("\n── Dry-run split counts ─────────────────────────────────────")
    print(f"  {'Band':<6} {'Classes':>8} {'Train':>8} {'Val':>6} {'Test':>8}")
    print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*6} {'-'*8}")
    for band in "ABCD":
        bc = band_counts[band]
        print(f"  {band:<6} {bc['classes']:>8} {bc['train']:>8,} {bc['val']:>6,} {bc['test']:>8,}")
    totals = {k: sum(band_counts[b][k] for b in "ABCD") for k in ("classes", "train", "val", "test")}
    print(f"  {'Total':<6} {totals['classes']:>8} {totals['train']:>8,} {totals['val']:>6,} {totals['test']:>8,}")

# ── Negative-class loaders ─────────────────────────────────────────────────────

def load_human_images_by_split() -> dict[str, list[dict]]:
    """Load coco_humans images from metadata_catalog.csv and assign 80/10/10 splits."""
    catalog = COCO_HUMANS_DIR / "metadata_catalog.csv"
    if not catalog.exists():
        print("  WARNING: coco_humans/metadata_catalog.csv not found, skipping humans")
        return {"train": [], "val": [], "test": [], "surplus": []}

    seen: dict[str, dict] = {}
    with open(catalog, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            fname = row["filename"]
            if fname in seen:
                continue
            xmin, ymin, xmax, ymax = [float(x) for x in row["bbox"].split(",")]
            w_n = xmax - xmin
            h_n = ymax - ymin
            xc  = xmin + w_n / 2
            yc  = ymin + h_n / 2
            seen[fname] = {
                "filepath":         f"data/coco_humans/images/human/{fname}",
                "source":           "coco_humans",
                "Q":                1.0,
                "score_components": {
                    "area_score": 1.0, "area_frac": round(w_n * h_n, 6),
                    "edge_score": 1.0, "min_margin": 0.02,
                    "single_score": 1.0, "n_significant": 1,
                    "conf_score": 1.0,
                    "hard_excluded": False,
                },
                "bbox":      [xc, yc, w_n, h_n],
                "bbox_conf": 1.0,
            }

    images = sorted(seen.values(), key=lambda x: x["filepath"])
    rng = random.Random(SEED)
    rng.shuffle(images)
    n = len(images)
    n_test, n_val, n_train = compute_band_d_sizes(n)
    print(f"  {n:,} human images → train={n_train} val={n_val} test={n_test} (Band D sizing)")
    return {
        "train":   images[:n_train],
        "val":     images[n_train : n_train + n_val],
        "test":    images[n_train + n_val :],
        "surplus": [],
    }


def load_blank_images_by_split() -> dict[str, list[dict]]:
    """Load blank (empty-scene) images from data/blanks and assign ~70/15/15 splits."""
    images_dir = BLANKS_DIR / "images"
    if not images_dir.exists():
        print("  WARNING: data/blanks/images not found, skipping blanks")
        return {"train": [], "val": [], "test": [], "surplus": []}

    images = [
        {
            "filepath":         f"data/blanks/images/{f.name}",
            "source":           "blanks",
            "Q":                0.5,
            "score_components": {
                "area_score": 0.0, "area_frac": 0.0,
                "edge_score": 0.0, "min_margin": 0.0,
                "single_score": 0.5, "n_significant": 0,
                "conf_score": 0.5,
                "hard_excluded": False,
            },
            "bbox":      None,
            "bbox_conf": None,
        }
        for f in sorted(images_dir.glob("*.jpg"))
    ]

    rng = random.Random(SEED + 1)
    rng.shuffle(images)
    n = len(images)
    n_val   = max(1, round(n * 0.15))
    n_test  = max(1, round(n * 0.15))
    n_train = n - n_val - n_test
    print(f"  {n:,} blank images → train={n_train} val={n_val} test={n_test}")
    return {
        "train":   images[:n_train],
        "val":     images[n_train : n_train + n_val],
        "test":    images[n_train + n_val :],
        "surplus": [],
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 12 — real image dataset split assignment")
    parser.add_argument("--dry-run",          action="store_true", help="Print counts only, write nothing")
    parser.add_argument("--include-negatives", action="store_true", help="(reserved) include no-bbox images explicitly")
    parser.add_argument("--max-workers",       type=int, default=32, help="Thread count for image dimension reads")
    args = parser.parse_args()

    rng = random.Random(SEED)

    print("── Stage 12: dataset split assignment ───────────────────────", flush=True)

    # ── Load inputs ────────────────────────────────────────────────────────────
    print("\n[1] Loading class metadata ...", flush=True)
    canonical_lookup = _build_canonical_lookup(CLASSES_225)
    if canonical_lookup:
        print(f"  {len(canonical_lookup)} apostrophe-name canonical mappings")
    valid_classes, categories = load_student_labels(STUDENT_LABELS)
    cat_id_map = {c["name"]: c["id"] for c in categories}
    if "human" not in cat_id_map:
        human_cat_id = max(c["id"] for c in categories) + 1
        categories.append({"id": human_cat_id, "name": "human", "supercategory": "person"})
        cat_id_map["human"] = human_cat_id
        print(f"  Added 'human' as category id {human_cat_id}")
    else:
        print(f"  'human' already in labels as category id {cat_id_map['human']}")
    print(f"  {len(valid_classes)} valid classes, {len(categories)} COCO categories")

    band_info = load_class_bands(CLASSES_CSV)
    band_dist  = {b: sum(1 for v in band_info.values() if v["band"] == b) for b in "ABCD"}
    print(f"  Band distribution: A={band_dist['A']} B={band_dist['B']} C={band_dist['C']} D={band_dist['D']}")

    # ── Load and score images ──────────────────────────────────────────────────
    print("\n[2] Loading quality-passed images ...", flush=True)
    per_class_images = load_all_passed_images(valid_classes, canonical_lookup)

    print("\n[2b] Loading human images ...", flush=True)
    human_assignments = load_human_images_by_split()

    print("\n[2c] Loading blank images ...", flush=True)
    blank_assignments = load_blank_images_by_split()

    # ── Assign splits ──────────────────────────────────────────────────────────
    print("\n[3] Assigning splits ...", flush=True)
    all_assignments: dict[str, dict] = {}
    for cls, info in sorted(band_info.items()):
        images   = per_class_images.get(cls, [])
        all_assignments[cls] = assign_splits(images, info["effective_pool"], info["band"], rng)

    verify_no_overlap(all_assignments)

    if args.dry_run:
        print_dry_run(all_assignments, band_info)
        print("\n── Dry run complete (no files written) ──────────────────────", flush=True)
        return

    # ── Write manifest and summary ─────────────────────────────────────────────
    print("\n[4] Writing manifest and summary ...", flush=True)
    write_manifest(all_assignments, band_info, human_assignments, blank_assignments)
    write_summary(all_assignments, band_info, per_class_images, human_assignments, blank_assignments)

    # ── Read image dimensions ──────────────────────────────────────────────────
    print("\n[5] Reading image dimensions ...", flush=True)
    assigned_fps: list[str] = []
    for cls_data in all_assignments.values():
        for split_name in ("train", "val", "test"):
            for img in cls_data[split_name]:
                assigned_fps.append(img["filepath"])
    for split_name in ("train", "val", "test"):
        for img in human_assignments.get(split_name, []):
            assigned_fps.append(img["filepath"])
        for img in blank_assignments.get(split_name, []):
            assigned_fps.append(img["filepath"])
    print(f"  {len(assigned_fps):,} assigned images (incl. human + blank)", flush=True)
    dims = read_image_dimensions(assigned_fps, args.max_workers)

    # ── Export COCO ────────────────────────────────────────────────────────────
    print("\n[6] Exporting COCO annotations ...", flush=True)
    DATA_REAL_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        coco     = build_coco_split(
            split, all_assignments, band_info, categories, cat_id_map, dims,
            human_assignments=human_assignments,
            blank_assignments=blank_assignments,
        )
        out_path = DATA_REAL_DIR / f"annotations_{split}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(coco, f, indent=2)

    # ── Write report ───────────────────────────────────────────────────────────
    print("\n[7] Writing methodology report ...", flush=True)
    write_report_md(all_assignments, band_info, per_class_images, SEED)

    print("\n── Done ─────────────────────────────────────────────────────", flush=True)


if __name__ == "__main__":
    main()
