#!/usr/bin/env python3
"""Run MegaDetector v5 on all generated synthetic images.

Reads data/synthetic/index.jsonl, runs MD v5 on every image with
status='generated', and writes per-image detection results to
data/synthetic/md_detections.jsonl.

All animal detections above a secondary confidence threshold are recorded
(YOLO-normalised [xc, yc, w, h]).  n_significant counts detections at or
above the primary (pass) threshold and is used by downstream stages to
route images:

  n_significant == 0  →  no animal detected; Stage 5 bbox labeling (draw from scratch)
  n_significant == 1  →  Stage 4 single-detection triage review
  n_significant >= 2  →  Stage 5 multi-animal bbox labeling (directly)

Output format (one JSON object per line):
  {
    "filepath":      "data/synthetic/images/band_a/walrus/a_walrus_001.png",
    "class":         "walrus",
    "band":          "a",
    "width":         592,
    "height":        448,
    "detections":    [{"bbox": [xc, yc, w, h], "conf": 0.87}, ...],
    "n_significant": 1
  }

Usage:
    python scripts/synthetic/3-run_megadetector.py
    python scripts/synthetic/3-run_megadetector.py --batch-size 16 --num-workers 2
    python scripts/synthetic/3-run_megadetector.py --force   # rerun all

Requirements:
    pip install pytorchwildlife
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

REPO_ROOT         = Path(__file__).resolve().parents[2]
SYNTHETIC_DIR     = REPO_ROOT / "data" / "synthetic"
INDEX_JSONL       = SYNTHETIC_DIR / "index.jsonl"
OUTPUT_JSONL      = SYNTHETIC_DIR / "md_detections.jsonl"
TEST_INDEX_JSONL  = SYNTHETIC_DIR / "test_index.jsonl"
TEST_OUTPUT_JSONL = SYNTHETIC_DIR / "md_detections_test.jsonl"

MD_CONF_PASS      = 0.5   # minimum conf for n_significant counter
MD_CONF_SECONDARY = 0.2   # lower bound for recording any detection
MD_BBOX_MIN_AREA  = 0.01  # minimum fractional bbox area (1 %)


# ── Helpers ───────────────────────────────────────────────────────────────────

def megadetector_to_yolo(bbox: list) -> list:
    """MegaDetector/COCO [xmin, ymin, w, h] (normalised) → YOLO [xc, yc, w, h]."""
    xmin, ymin, w, h = bbox
    return [xmin + w / 2.0, ymin + h / 2.0, w, h]


def bbox_area(bbox: list) -> float:
    return bbox[2] * bbox[3]


def image_path(record: dict) -> Path:
    """Derive absolute image path from an index.jsonl record."""
    band = record["band"].lower()
    fname = record["filename"]
    parts = fname.split("_")
    class_slug = "_".join(parts[1:-1])
    return SYNTHETIC_DIR / "images" / f"band_{band}" / class_slug / fname


def test_image_path(record: dict) -> Path:
    """Derive absolute image path from a test_index.jsonl record."""
    slug = Path(record["prompt_file"]).parent.name
    return SYNTHETIC_DIR / "images" / "test" / slug / record["filename"]


def load_done(jsonl_path: Path) -> set:
    """Return the set of filepaths already written to the output file."""
    done: set[str] = set()
    if not jsonl_path.exists():
        return done
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["filepath"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--batch-size",     type=int,   default=32,              metavar="N")
    parser.add_argument("--num-workers",    type=int,   default=4,               metavar="N")
    parser.add_argument("--conf",           type=float, default=MD_CONF_PASS,    metavar="T",
                        help=f"Min conf counted toward n_significant (default: {MD_CONF_PASS})")
    parser.add_argument("--conf-secondary", type=float, default=MD_CONF_SECONDARY, metavar="T",
                        help=f"Min conf for recording any detection (default: {MD_CONF_SECONDARY})")
    parser.add_argument("--force", action="store_true",
                        help="Truncate output file(s) and rerun all images")
    parser.add_argument("--split", choices=["train", "test", "all"], default="all",
                        help="Which dataset split to process (default: all)")
    args = parser.parse_args()

    # ── Build pending list across active splits ───────────────────────────────
    split_specs: list[tuple] = []
    if args.split in ("train", "all"):
        split_specs.append(("train", INDEX_JSONL, image_path, OUTPUT_JSONL))
    if args.split in ("test", "all"):
        split_specs.append(("test", TEST_INDEX_JSONL, test_image_path, TEST_OUTPUT_JSONL))

    pending: list[dict] = []
    missing = 0
    active_outputs: list[Path] = []

    for split_name, index_path, path_fn, out_path in split_specs:
        if not index_path.exists():
            print(f"Warning: {index_path.name} not found — skipping {split_name} split")
            continue
        active_outputs.append(out_path)
        if args.force and out_path.exists():
            out_path.unlink()
            print(f"--force: cleared existing {out_path.name}")
        split_done = set() if args.force else load_done(out_path)

        records: list[dict] = []
        with open(index_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    if rec.get("status") == "generated":
                        records.append(rec)
        print(f"[{split_name}] {len(records):,} generated images in {index_path.name}")

        for rec in records:
            abs_p = path_fn(rec)
            rel_p = abs_p.relative_to(REPO_ROOT).as_posix()
            if rel_p in split_done:
                continue
            if not abs_p.exists():
                missing += 1
                continue
            pending.append({
                "abs":    str(abs_p),
                "rel":    rel_p,
                "class":  rec["class"],
                "band":   rec["band"].lower(),
                "output": out_path,
            })

    if missing:
        print(f"Warning: {missing:,} images listed in index files not found on disk — skipped")
    if not pending:
        print("Nothing to process — all images already done or no index files found")
        for out_path in active_outputs:
            _print_stats(out_path)
        return
    print(f"Processing {len(pending):,} images ({missing:,} missing) …")

    # ── Load model ────────────────────────────────────────────────────────────
    try:
        import torch
        from torch.utils.data import Dataset, DataLoader
        from PytorchWildlife.models import detection as pw_detection
        from yolov5.utils.general import non_max_suppression, scale_boxes
    except ImportError as exc:
        print(f"ERROR: missing dependency: {exc}\n  Run: pip install pytorchwildlife")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model = pw_detection.MegaDetectorV5(device=device, pretrained=True)
    model.model.eval()

    conf_pass      = args.conf
    conf_secondary = args.conf_secondary
    img_size       = model.IMAGE_SIZE  # 1280

    # ── Dataset ───────────────────────────────────────────────────────────────
    class _FileListDataset(Dataset):
        def __init__(self, items, transform):
            self.items     = items
            self.transform = transform

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, idx):
            it  = self.items[idx]
            img = Image.open(it["abs"]).convert("RGB")
            size = torch.tensor(img.size[::-1])  # PIL (W,H) → tensor (H,W)
            if self.transform:
                img = self.transform(img)
            return img, idx, size

    dataset = _FileListDataset(pending, model.transform)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=(args.num_workers > 0),
        shuffle=False,
        drop_last=False,
    )

    # ── Inference loop ────────────────────────────────────────────────────────
    save_interval = max(1, 500 // args.batch_size)

    for p in active_outputs:
        p.parent.mkdir(parents=True, exist_ok=True)

    out_handles = {p: open(p, "a", encoding="utf-8") for p in active_outputs}
    out_buffers = {p: [] for p in active_outputs}
    counters    = {p: {"zero": 0, "single": 0, "multi": 0} for p in active_outputs}

    try:
        with torch.no_grad():
            for batch_idx, (imgs, indices, sizes) in enumerate(
                    tqdm(loader, desc="MegaDetector")):

                imgs = imgs.to(device, non_blocking=True)
                with torch.autocast("cuda", enabled=(device == "cuda")):
                    raw = model.model(imgs)[0]
                raw   = raw.float().detach().cpu()
                preds = non_max_suppression(raw, conf_thres=conf_secondary)

                for i, pred in enumerate(preds):
                    item       = pending[indices[i].item()]
                    H, W       = sizes[i].tolist()
                    out_path   = item["output"]
                    detections: list[dict] = []

                    if pred is not None and len(pred) > 0:
                        pred_np = pred.numpy().copy()
                        pred_np[:, :4] = scale_boxes(
                            [img_size] * 2, pred_np[:, :4], (H, W)
                        ).round()

                        animal_dets = sorted(
                            [{"bbox": [float(x1 / W), float(y1 / H),
                                       float((x2 - x1) / W), float((y2 - y1) / H)],
                              "conf": float(c)}
                             for x1, y1, x2, y2, c, cls in pred_np if int(cls) == 0],
                            key=lambda d: d["conf"],
                            reverse=True,
                        )
                        for d in animal_dets:
                            b = megadetector_to_yolo(d["bbox"])
                            if bbox_area(b) >= MD_BBOX_MIN_AREA:
                                detections.append({"bbox": b, "conf": d["conf"]})

                    n_sig = sum(1 for d in detections if d["conf"] >= conf_pass)
                    c = counters[out_path]
                    if n_sig == 0:
                        c["zero"] += 1
                    elif n_sig == 1:
                        c["single"] += 1
                    else:
                        c["multi"] += 1

                    out_buffers[out_path].append({
                        "filepath":      item["rel"],
                        "class":         item["class"],
                        "band":          item["band"],
                        "width":         int(W),
                        "height":        int(H),
                        "detections":    detections,
                        "n_significant": n_sig,
                    })

                if (batch_idx + 1) % save_interval == 0:
                    for p, buf in out_buffers.items():
                        for e in buf:
                            out_handles[p].write(json.dumps(e) + "\n")
                        out_handles[p].flush()
                        buf.clear()

        for p, buf in out_buffers.items():
            for e in buf:
                out_handles[p].write(json.dumps(e) + "\n")
    finally:
        for fh in out_handles.values():
            fh.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    for split_name, _, _, out_path in split_specs:
        if out_path not in counters:
            continue
        c     = counters[out_path]
        total = c["zero"] + c["single"] + c["multi"]
        pct   = lambda n: f"{100 * n / total:.1f}%" if total else "—"
        print(f"\n[{split_name}] Wrote {total:,} entries to {out_path.name}")
        print(f"  n_significant == 0:  {c['zero']:>6,}  ({pct(c['zero'])})  → Stage 5 bbox labeling (draw from scratch)")
        print(f"  n_significant == 1:  {c['single']:>6,}  ({pct(c['single'])})  → Stage 4 triage review")
        print(f"  n_significant >= 2:  {c['multi']:>6,}  ({pct(c['multi'])})  → Stage 5 bbox labeling")


def _print_stats(jsonl_path: Path) -> None:
    """Print n_significant distribution from an existing output file."""
    from collections import Counter
    c: Counter = Counter()
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                n = json.loads(line).get("n_significant", 0)
                c[min(n, 2)] += 1
    total = sum(c.values())
    if not total:
        return
    pct = lambda n: f"{100 * n / total:.1f}%"
    print(f"\nExisting {jsonl_path.name} — {total:,} entries:")
    print(f"  n_significant == 0:  {c[0]:>6,}  ({pct(c[0])})")
    print(f"  n_significant == 1:  {c[1]:>6,}  ({pct(c[1])})")
    print(f"  n_significant >= 2:  {c[2]:>6,}  ({pct(c[2])})")


if __name__ == "__main__":
    main()
