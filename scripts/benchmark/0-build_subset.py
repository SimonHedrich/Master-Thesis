#!/usr/bin/env python3
"""
Stage 2 — Build the fixed, seeded benchmark subset shipped to the Pi.

Materializes a small self-contained image set plus a COCO annotation file in
the same schema as `data/real/annotations_test.json`, so that the Pi-side
parity run can emit predictions against it and the host can score them with
the existing `eval_suite` scorer.

Composition (plan §4.4): 120 real images from each of bands A/B/C/D + 20
negatives + 100 synthetic images sampled uniformly across the 225 classes.
Real and synthetic image IDs are disjoint (synthetic offset by SYNTH_ID_OFFSET)
so the two domains can live in one annotation file and be scored as one set.

Also copies the two fixed single images used by the timing loops: the
median-sized real test image, and a native AX Visio still (4192x3120) that
gives the realistic on-device JPEG-decode cost.

Usage:
    uv run python scripts/benchmark/0-build_subset.py
    uv run python scripts/benchmark/0-build_subset.py --dry-run

Outputs:
    data/benchmark_subset/images/*.jpg|png
    data/benchmark_subset/fixed/{median_real.jpg,ax_visio.jpeg}
    data/benchmark_subset/annotations_subset.json
    data/benchmark_subset/subset_manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import constants as C  # noqa: E402  (script-dir import; see package README)

SYNTH_ID_OFFSET = 1_000_000


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _sample_real(data: dict, rng: random.Random) -> list[dict]:
    """120 per band A/B/C/D + 20 negatives, sampled uniformly within band."""
    by_band: dict[str, list[dict]] = defaultdict(list)
    for img in data["images"]:
        by_band[img.get("band")].append(img)

    picked: list[dict] = []
    for band in C.SUBSET_BANDS:
        pool = sorted(by_band.get(band, []), key=lambda r: r["id"])
        if len(pool) < C.SUBSET_PER_BAND:
            print(f"  ! band {band}: only {len(pool)} images available, taking all")
            picked.extend(pool)
        else:
            picked.extend(rng.sample(pool, C.SUBSET_PER_BAND))

    neg_pool = sorted(by_band.get("negative", []), key=lambda r: r["id"])
    n_neg = min(C.SUBSET_NEGATIVES, len(neg_pool))
    picked.extend(rng.sample(neg_pool, n_neg) if neg_pool else [])
    return picked


def _sample_synth(data: dict, rng: random.Random) -> list[dict]:
    """100 synthetic images spread as evenly as possible over the 225 classes.

    The synthetic test set is a balanced 225x50 grid, so "uniform across
    classes" means picking 100 distinct classes and taking one image from each.
    """
    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in data["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    by_class: dict[int, list[dict]] = defaultdict(list)
    for img in data["images"]:
        anns = anns_by_image.get(img["id"], [])
        if not anns:
            continue
        by_class[anns[0]["category_id"]].append(img)

    classes = sorted(by_class)
    chosen = rng.sample(classes, min(C.SUBSET_SYNTHETIC, len(classes)))
    return [rng.choice(sorted(by_class[c], key=lambda r: r["id"])) for c in sorted(chosen)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="select and report, copy nothing")
    ap.add_argument("--out", type=Path, default=C.SUBSET_DIR)
    args = ap.parse_args()

    rng = random.Random(C.SEED)

    print(f"seed={C.SEED}")
    real = _load(C.ANNOTATIONS_TEST_REAL)
    synth = _load(C.ANNOTATIONS_TEST_SYNTH)
    print(f"real test: {len(real['images'])} images   synth test: {len(synth['images'])} images")

    if [c["id"] for c in real["categories"]] != [c["id"] for c in synth["categories"]]:
        print("FATAL: real and synthetic category id lists differ — cannot merge", file=sys.stderr)
        return 1

    real_pick = _sample_real(real, rng)
    synth_pick = _sample_synth(synth, rng)
    print(f"selected: {len(real_pick)} real + {len(synth_pick)} synthetic")

    real_ids = {r["id"] for r in real_pick}
    synth_ids = {r["id"] for r in synth_pick}

    out_images: list[dict] = []
    out_anns: list[dict] = []
    copies: list[tuple[Path, Path]] = []
    ann_id = 1
    missing = 0

    for domain, picks, ids, src_data, offset in (
        ("real", real_pick, real_ids, real, 0),
        ("synthetic", synth_pick, synth_ids, synth, SYNTH_ID_OFFSET),
    ):
        anns_by_image: dict[int, list[dict]] = defaultdict(list)
        for ann in src_data["annotations"]:
            if ann["image_id"] in ids:
                anns_by_image[ann["image_id"]].append(ann)

        for rec in picks:
            src = C.REPO_ROOT / rec["file_name"]
            if not src.exists():
                missing += 1
                continue
            new_id = rec["id"] + offset
            # Flat, collision-proof, domain-tagged name; keep the real suffix
            # so cv2 picks the right decoder.
            dst_name = f"{domain[:1]}{new_id:08d}{src.suffix.lower()}"
            out_images.append(
                {
                    "id": new_id,
                    "file_name": f"images/{dst_name}",
                    "width": rec["width"],
                    "height": rec["height"],
                    "band": rec.get("band"),
                    "domain": domain,
                    "source": rec.get("source", "synthetic"),
                    "orig_file_name": rec["file_name"],
                }
            )
            for ann in anns_by_image.get(rec["id"], []):
                out_anns.append(
                    {
                        "id": ann_id,
                        "image_id": new_id,
                        "category_id": ann["category_id"],
                        "bbox": ann["bbox"],
                        "area": float(ann.get("area", ann["bbox"][2] * ann["bbox"][3])),
                        "iscrowd": int(ann.get("iscrowd", 0)),
                    }
                )
                ann_id += 1
            copies.append((src, args.out / "images" / dst_name))

    if missing:
        print(f"  ! {missing} selected images not found on disk — skipped")

    # ── the two fixed timing images ──────────────────────────────────────────
    # Median real test image: the one whose area is closest to the median area
    # of the whole real test set, so the decode cost is representative.
    areas = sorted(r["width"] * r["height"] for r in real["images"])
    median_area = areas[len(areas) // 2]
    median_rec = min(
        (r for r in real_pick if (C.REPO_ROOT / r["file_name"]).exists()),
        key=lambda r: abs(r["width"] * r["height"] - median_area),
    )
    fixed: list[tuple[Path, Path]] = [
        (C.REPO_ROOT / median_rec["file_name"], args.out / "fixed" / "median_real.jpg")
    ]
    ax_dir = C.REPO_ROOT / "resources" / "AX_Visio_images"
    ax_candidates = sorted(ax_dir.glob("*.jpeg")) + sorted(ax_dir.glob("*.jpg"))
    if ax_candidates:
        fixed.append((ax_candidates[0], args.out / "fixed" / "ax_visio.jpeg"))
    else:
        print("  ! no AX Visio still found — the native-resolution decode cell will be skipped")

    print(
        f"subset: {len(out_images)} images, {len(out_anns)} annotations, "
        f"median real image {median_rec['width']}x{median_rec['height']}"
    )
    if args.dry_run:
        print("--dry-run: nothing written")
        return 0

    # ── write ────────────────────────────────────────────────────────────────
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    (args.out / "fixed").mkdir(parents=True, exist_ok=True)
    for src, dst in copies + fixed:
        shutil.copy2(src, dst)

    ann_out = {
        "info": {
            "description": "Fixed benchmark subset for on-device latency + parity checks",
            "seed": C.SEED,
            "synth_id_offset": SYNTH_ID_OFFSET,
            "source_real": str(C.ANNOTATIONS_TEST_REAL.relative_to(C.REPO_ROOT)),
            "source_synth": str(C.ANNOTATIONS_TEST_SYNTH.relative_to(C.REPO_ROOT)),
        },
        "images": out_images,
        "annotations": out_anns,
        "categories": real["categories"],
    }
    with (args.out / "annotations_subset.json").open("w") as f:
        json.dump(ann_out, f)

    bands = defaultdict(int)
    for r in out_images:
        bands[f"{r['domain']}/{r['band']}"] += 1
    manifest = {
        "seed": C.SEED,
        "n_images": len(out_images),
        "n_annotations": len(out_anns),
        "composition": dict(sorted(bands.items())),
        "total_bytes": sum(dst.stat().st_size for _, dst in copies),
        "fixed_images": {
            dst.name: {"orig": str(src.relative_to(C.REPO_ROOT)), "sha256": _sha256(dst)}
            for src, dst in fixed
        },
        "images": [
            {"id": r["id"], "file_name": r["file_name"], "orig": r["orig_file_name"]}
            for r in out_images
        ],
    }
    with (args.out / "subset_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    mb = manifest["total_bytes"] / 1e6
    print(f"wrote {args.out}  ({mb:.1f} MB images, ~{mb / 2.4:.0f}s over the 2.4 MB/s link)")
    print("composition:", dict(sorted(bands.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
