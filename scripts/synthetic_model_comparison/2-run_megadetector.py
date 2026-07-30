#!/usr/bin/env python3
"""Run MegaDetector v5 on one or all synthetic-generator-comparison train cells.

Adapted from scripts/synthetic/3-run_megadetector.py for this experiment's
per-cell layout (data/synthetic_model_comparison/train/<generator>/<regime>/)
instead of production's single flat data/synthetic/ tree. Same detection
thresholds as production (fixed control per
docs/synthetic-model-comparison/01_experiment-design.md §5 point 2 — "same
MegaDetector pass + same review rules").

Unlike production's index.jsonl (which needs a status=="generated" filter and
a filename-parsing helper to locate images), this experiment's index.jsonl
(written by 1-select_train_subset_incumbent.py) already lists every entry as
generated and already stores the full repo-relative image path in
"file_name" — so every entry is processed directly, no path reconstruction
needed.

Reads:
    data/synthetic_model_comparison/train/<generator>/<regime>/index.jsonl
Writes:
    data/synthetic_model_comparison/train/<generator>/<regime>/md_detections.jsonl

Output format (one JSON object per line, same schema as production):
    {
      "filepath":      "data/synthetic_model_comparison/train/<gen>/<regime>/images/<slug>/<file>",
      "class":         "grevy's zebra",
      "band":          "b",
      "width":         832,
      "height":        624,
      "detections":    [{"bbox": [xc, yc, w, h], "conf": 0.87}, ...],
      "n_significant": 1
    }

Usage:
    uv run python scripts/synthetic_model_comparison/2-run_megadetector.py \\
        --generator gemini-3.1-flash-image-preview --prompt-regime full
    uv run python scripts/synthetic_model_comparison/2-run_megadetector.py --all
    uv run python scripts/synthetic_model_comparison/2-run_megadetector.py \\
        --generator gemini-3.1-flash-image-preview --prompt-regime full --force

Requirements:
    pip install pytorchwildlife
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

REPO_ROOT  = Path(__file__).resolve().parents[2]
TRAIN_ROOT = REPO_ROOT / "data" / "synthetic_model_comparison" / "train"

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


def discover_cells() -> list[tuple[str, str, Path]]:
    """Return (generator, prompt_regime, cell_dir) for every existing cell."""
    cells = []
    for index_path in sorted(TRAIN_ROOT.glob("*/*/index.jsonl")):
        cell_dir = index_path.parent
        generator = cell_dir.parent.name
        regime = cell_dir.name
        cells.append((generator, regime, cell_dir))
    return cells


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


def process_cell(
    generator: str,
    regime: str,
    cell_dir: Path,
    model,
    device: str,
    args: argparse.Namespace,
) -> None:
    index_path = cell_dir / "index.jsonl"
    out_path = cell_dir / "md_detections.jsonl"

    if args.force and out_path.exists():
        out_path.unlink()
        print(f"[{generator}/{regime}] --force: cleared existing {out_path.name}")
    done = set() if args.force else load_done(out_path)

    records: list[dict] = []
    with open(index_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"[{generator}/{regime}] {len(records):,} images in {index_path.name}")

    pending: list[dict] = []
    missing = 0
    for rec in records:
        abs_p = REPO_ROOT / rec["file_name"]
        rel_p = rec["file_name"]
        if rel_p in done:
            continue
        if not abs_p.exists():
            missing += 1
            continue
        pending.append({"abs": str(abs_p), "rel": rel_p, "class": rec["class"], "band": rec["band"].lower()})

    if missing:
        print(f"[{generator}/{regime}] Warning: {missing:,} images listed in index not found on disk — skipped")
    if not pending:
        print(f"[{generator}/{regime}] Nothing to process — all images already done")
        _print_stats(out_path)
        return
    print(f"[{generator}/{regime}] Processing {len(pending):,} images ({missing:,} missing) …")

    import torch
    from torch.utils.data import Dataset, DataLoader
    from yolov5.utils.general import non_max_suppression, scale_boxes

    conf_pass = args.conf
    conf_secondary = args.conf_secondary
    img_size = model.IMAGE_SIZE  # 1280

    class _FileListDataset(Dataset):
        def __init__(self, items, transform):
            self.items = items
            self.transform = transform

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, idx):
            it = self.items[idx]
            img = Image.open(it["abs"]).convert("RGB")
            size = torch.tensor(img.size[::-1])  # PIL (W,H) → tensor (H,W)
            if self.transform:
                img = self.transform(img)
            return img, idx, size

    dataset = _FileListDataset(pending, model.transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=(args.num_workers > 0),
        shuffle=False,
        drop_last=False,
    )

    save_interval = max(1, 500 // args.batch_size)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counters = {"zero": 0, "single": 0, "multi": 0}
    buffer: list[dict] = []

    with open(out_path, "a", encoding="utf-8") as fh, torch.no_grad():
        for batch_idx, (imgs, indices, sizes) in enumerate(
                tqdm(loader, desc=f"MegaDetector[{generator}/{regime}]")):

            imgs = imgs.to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=(device == "cuda")):
                raw = model.model(imgs)[0]
            raw = raw.float().detach().cpu()
            preds = non_max_suppression(raw, conf_thres=conf_secondary)

            for i, pred in enumerate(preds):
                item = pending[indices[i].item()]
                H, W = sizes[i].tolist()
                detections: list[dict] = []

                if pred is not None and len(pred) > 0:
                    pred_np = pred.numpy().copy()
                    pred_np[:, :4] = scale_boxes([img_size] * 2, pred_np[:, :4], (H, W)).round()

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
                if n_sig == 0:
                    counters["zero"] += 1
                elif n_sig == 1:
                    counters["single"] += 1
                else:
                    counters["multi"] += 1

                buffer.append({
                    "filepath": item["rel"],
                    "class": item["class"],
                    "band": item["band"],
                    "width": int(W),
                    "height": int(H),
                    "detections": detections,
                    "n_significant": n_sig,
                })

            if (batch_idx + 1) % save_interval == 0:
                for e in buffer:
                    fh.write(json.dumps(e) + "\n")
                fh.flush()
                buffer.clear()

        for e in buffer:
            fh.write(json.dumps(e) + "\n")

    total = sum(counters.values())
    pct = lambda n: f"{100 * n / total:.1f}%" if total else "—"
    print(f"\n[{generator}/{regime}] Wrote {total:,} entries to {out_path.name}")
    print(f"  n_significant == 0:  {counters['zero']:>6,}  ({pct(counters['zero'])})  → stage 4 bbox labeling (draw from scratch)")
    print(f"  n_significant == 1:  {counters['single']:>6,}  ({pct(counters['single'])})  → stage 3 triage review")
    print(f"  n_significant >= 2:  {counters['multi']:>6,}  ({pct(counters['multi'])})  → stage 4 bbox labeling")


def _print_stats(jsonl_path: Path) -> None:
    """Print n_significant distribution from an existing output file."""
    from collections import Counter
    c: Counter = Counter()
    if not jsonl_path.exists():
        return
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
    print(f"  Existing {jsonl_path.name} — {total:,} entries:")
    print(f"    n_significant == 0:  {c[0]:>6,}  ({pct(c[0])})")
    print(f"    n_significant == 1:  {c[1]:>6,}  ({pct(c[1])})")
    print(f"    n_significant >= 2:  {c[2]:>6,}  ({pct(c[2])})")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--generator", metavar="NAME",
                        help="Generator slug, e.g. gemini-3.1-flash-image-preview")
    parser.add_argument("--prompt-regime", choices=["full", "compressed"],
                        help="Prompt regime for the selected generator")
    parser.add_argument("--all", action="store_true",
                        help="Process every cell found under train/*/*/index.jsonl")
    parser.add_argument("--batch-size", type=int, default=32, metavar="N")
    parser.add_argument("--num-workers", type=int, default=4, metavar="N")
    parser.add_argument("--conf", type=float, default=MD_CONF_PASS, metavar="T",
                        help=f"Min conf counted toward n_significant (default: {MD_CONF_PASS})")
    parser.add_argument("--conf-secondary", type=float, default=MD_CONF_SECONDARY, metavar="T",
                        help=f"Min conf for recording any detection (default: {MD_CONF_SECONDARY})")
    parser.add_argument("--force", action="store_true",
                        help="Truncate output file(s) and rerun all images")
    args = parser.parse_args()

    if args.all:
        cells = discover_cells()
    elif args.generator and args.prompt_regime:
        cell_dir = TRAIN_ROOT / args.generator / args.prompt_regime
        if not (cell_dir / "index.jsonl").exists():
            print(f"ERROR: no index.jsonl found at {cell_dir}")
            sys.exit(1)
        cells = [(args.generator, args.prompt_regime, cell_dir)]
    else:
        parser.error("pass --all, or both --generator and --prompt-regime")
        return

    if not cells:
        print("No cells found under", TRAIN_ROOT)
        return

    try:
        import torch
        from PytorchWildlife.models import detection as pw_detection
    except ImportError as exc:
        print(f"ERROR: missing dependency: {exc}\n  Run: pip install pytorchwildlife")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model = pw_detection.MegaDetectorV5(device=device, pretrained=True)
    model.model.eval()

    for generator, regime, cell_dir in cells:
        process_cell(generator, regime, cell_dir, model, device, args)


if __name__ == "__main__":
    main()
