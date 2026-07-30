#!/usr/bin/env python3
"""
Stage 1 — Select the incumbent-generator synthetic train subset

Materializes the first (incumbent) generator cell's synthetic train set for
the model-comparison experiment
(docs/synthetic-model-comparison/01_experiment-design.md,
docs/synthetic-model-comparison/02_class-selection.md), reusing images
already generated in production with the same model/prompt template
(gemini-3.1-flash-image-preview, full prompt regime) rather than
regenerating them:

- Grevy's zebra: exactly 100 images already exist (the fixed control
  count) -- reused as-is, no selection needed.
- The 6 Bucket-3 classes (kinkajou, water deer, ringtail, saiga, aye-aye,
  pangolin family): 200 images each exist (double the needed count). 100 of
  200 are selected per class via a stratify-by-environment,
  greedily-maximize-diversity algorithm (see
  docs/synthetic-model-comparison/10_train-subset-incumbent-selection.md)
  so pose/camera-position/environment variety survives the halving.

The other 5 classes in the 12-class comparison set (plains zebra, mountain
zebra, red fox, American black bear, lion) have no existing incumbent
synthetic images and are out of scope here -- they need fresh generation.

Labeling (MegaDetector) is deliberately deferred to a later stage that will
run once over every generator cell together, so this script only selects
and copies image files plus a provenance manifest -- it does not touch
data/synthetic/annotations_*.json or produce a COCO json.

Usage:
    uv run python scripts/synthetic_model_comparison/1-select_train_subset_incumbent.py
    uv run python scripts/synthetic_model_comparison/1-select_train_subset_incumbent.py --dry-run

Outputs:
    data/synthetic_model_comparison/train/gemini-3.1-flash-image-preview/full/images/<class_slug>/*
    data/synthetic_model_comparison/train/gemini-3.1-flash-image-preview/full/index.jsonl
    data/synthetic_model_comparison/train/prompts_full/<class_slug>/*
    reports/model_comparison_train_incumbent_selection.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_DIR = REPO_ROOT / "data" / "synthetic"
PROMPTS_DIR = SYNTHETIC_DIR / "prompts"
TRAIN_ROOT = REPO_ROOT / "data" / "synthetic_model_comparison" / "train"
OUT_DIR = TRAIN_ROOT / "gemini-3.1-flash-image-preview" / "full"
OUT_IMAGES_DIR = OUT_DIR / "images"
OUT_INDEX_PATH = OUT_DIR / "index.jsonl"
# Prompt text is shared across every generator that runs the "full" regime
# (only the model consuming it varies), so it lives one level up, outside
# any single generator's subdirectory.
OUT_PROMPTS_DIR = TRAIN_ROOT / "prompts_full"
SUMMARY_CSV_PATH = REPO_ROOT / "reports" / "model_comparison_train_incumbent_selection.csv"

TARGET_COUNT_PER_CLASS = 100

# class name (as used in data/synthetic/annotations_*.json) -> (band, image-dir slug, prompt-dir slug)
CLASSES = {
    "grevy's zebra": ("B", "grevy_s_zebra", "grevy_s_zebra"),
    "kinkajou": ("A", "kinkajou", "kinkajou"),
    "water deer": ("A", "water_deer", "water_deer"),
    "ringtail": ("A", "ringtail", "ringtail"),
    "saiga": ("A", "saiga", "saiga"),
    "aye-aye": ("A", "aye_aye", "aye_aye"),
    "pangolin family": ("A", "pangolin_family", "pangolin_family"),
}

# Classes whose full existing pool is reused as-is (no selection needed).
TAKE_ALL_CLASSES = {"grevy's zebra"}

DIVERSITY_FIELDS = ["pose", "shot_type", "distance", "lighting", "occlusion"]

POSE_RE = re.compile(r"Animal pose / behavior:\s*(.+)")
ENV_RE = re.compile(r"Environment / background:\s*(.+)")


def slugify(class_name: str) -> str:
    return class_name.replace(" ", "_")


def load_image_metadata_by_basename(band: str) -> dict[str, dict]:
    """Load image records (with band/split/shot_type/... fields) from the
    production synthetic annotations, keyed by file basename."""
    splits = ["train", "val"]
    by_basename: dict[str, dict] = {}
    for split in splits:
        with open(SYNTHETIC_DIR / f"annotations_{split}.json", encoding="utf-8") as f:
            coco = json.load(f)
        for image in coco["images"]:
            if image.get("band") != band:
                continue
            basename = Path(image["file_name"]).name
            by_basename[basename] = {**image, "source_split": split}
    return by_basename


def parse_prompt_scene_fields(prompt_path: Path) -> tuple[str, str]:
    text = prompt_path.read_text(encoding="utf-8")
    pose_match = POSE_RE.search(text)
    env_match = ENV_RE.search(text)
    if not pose_match or not env_match:
        raise RuntimeError(f"could not parse scene-spec fields from {prompt_path}")
    return pose_match.group(1).strip(), env_match.group(1).strip()


def collect_class_records(class_name: str, prompt_slug: str, band: str) -> list[dict]:
    """One record per generated image for this class: image metadata
    (from the production annotations) plus the free-text pose/environment
    parsed from that image's prompt file."""
    prompt_dir = PROMPTS_DIR / prompt_slug
    metadata_by_basename = load_image_metadata_by_basename(band)

    records = []
    for prompt_path in sorted(prompt_dir.glob("*.txt")):
        index = int(prompt_path.stem)
        pose, environment = parse_prompt_scene_fields(prompt_path)

        basename = f"{band.lower()}_{prompt_slug}_{index:03d}.png"
        meta = metadata_by_basename.get(basename)
        if meta is None:
            raise RuntimeError(
                f"no image metadata found for {class_name} index {index} "
                f"(expected basename '{basename}')"
            )

        records.append(
            {
                "index": index,
                "filename": basename,
                "src_path": REPO_ROOT / meta["file_name"],
                "class": class_name,
                "band": band,
                "source_split": meta["source_split"],
                "shot_type": meta["shot_type"],
                "distance": meta["distance"],
                "lighting": meta["lighting"],
                "occlusion": meta["occlusion"],
                "pose": pose,
                "environment": environment,
                "prompt_file": str(prompt_path.relative_to(REPO_ROOT)),
            }
        )
    return records


def environment_quotas(environments: list[str], total: int) -> dict[str, int]:
    """Largest-remainder allocation of `total` across the given environments,
    split as evenly as possible; deterministic tie-break by sorted string."""
    n = len(environments)
    base, remainder = divmod(total, n)
    ordered = sorted(environments)
    return {env: base + (1 if i < remainder else 0) for i, env in enumerate(ordered)}


def greedy_diversify(records: list[dict], quota: int) -> list[dict]:
    """Repeatedly pick the unpicked record whose field values add the most
    new values to the covered sets, tie-broken by lowest index."""
    covered = {field: set() for field in DIVERSITY_FIELDS}
    pool = sorted(records, key=lambda r: r["index"])
    selected: list[dict] = []

    while len(selected) < quota and pool:
        def score(record: dict) -> tuple[int, int]:
            new_values = sum(1 for f in DIVERSITY_FIELDS if record[f] not in covered[f])
            return (new_values, -record["index"])

        best = max(pool, key=score)
        pool.remove(best)
        selected.append(best)
        for field in DIVERSITY_FIELDS:
            covered[field].add(best[field])

    return selected


def select_diverse_subset(records: list[dict], target_count: int) -> list[dict]:
    by_environment: dict[str, list[dict]] = {}
    for record in records:
        by_environment.setdefault(record["environment"], []).append(record)

    quotas = environment_quotas(list(by_environment.keys()), target_count)

    selected: list[dict] = []
    for environment, group in by_environment.items():
        selected.extend(greedy_diversify(group, quotas[environment]))
    return selected


def copy_selected(class_name: str, image_slug: str, selected: list[dict], dry_run: bool) -> int:
    dest_images_dir = OUT_IMAGES_DIR / image_slug
    dest_prompts_dir = OUT_PROMPTS_DIR / image_slug
    if not dry_run:
        dest_images_dir.mkdir(parents=True, exist_ok=True)
        dest_prompts_dir.mkdir(parents=True, exist_ok=True)

    seen_basenames: dict[str, Path] = {}
    byte_count = 0
    for record in tqdm(selected, desc=f"{class_name} ({image_slug})", leave=False):
        basename = record["filename"]
        if basename in seen_basenames and seen_basenames[basename] != record["src_path"]:
            raise RuntimeError(
                f"filename collision for class '{class_name}': "
                f"'{basename}' comes from both '{seen_basenames[basename]}' and "
                f"'{record['src_path']}'"
            )
        seen_basenames[basename] = record["src_path"]

        dest_path = dest_images_dir / basename
        if not dry_run:
            shutil.copy2(record["src_path"], dest_path)
        byte_count += record["src_path"].stat().st_size
        record["dest_path"] = dest_path

        src_prompt_path = REPO_ROOT / record["prompt_file"]
        dest_prompt_path = dest_prompts_dir / f"{record['index']:03d}.txt"
        if not dry_run:
            shutil.copy2(src_prompt_path, dest_prompt_path)
        record["dest_prompt_path"] = dest_prompt_path

    return byte_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print the selection plan without copying files or writing outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    all_selected: list[dict] = []
    summary_rows: list[dict] = []

    for class_name, (band, image_slug, prompt_slug) in CLASSES.items():
        records = collect_class_records(class_name, prompt_slug, band)
        pool_size = len(records)

        if class_name in TAKE_ALL_CLASSES:
            selected = records
        else:
            selected = select_diverse_subset(records, TARGET_COUNT_PER_CLASS)

        byte_count = copy_selected(class_name, image_slug, selected, dry_run=args.dry_run)
        all_selected.extend(selected)

        n_environments_covered = len({r["environment"] for r in selected})
        n_environments_available = len({r["environment"] for r in records})
        n_poses_selected = len({r["pose"] for r in selected})
        n_poses_available = len({r["pose"] for r in records})

        summary_rows.append(
            {
                "class": class_name,
                "band": band,
                "pool_size": pool_size,
                "selected_count": len(selected),
                "n_environments_covered": n_environments_covered,
                "n_environments_available": n_environments_available,
                "n_distinct_poses_selected": n_poses_selected,
                "n_distinct_poses_available": n_poses_available,
                "byte_count": byte_count,
            }
        )

    print(
        f"\n{'class':20s} {'band':4s} {'pool':>6s} {'sel':>6s} "
        f"{'envs':>10s} {'poses':>12s} {'size (MB)':>10s}"
    )
    total_selected = 0
    total_bytes = 0
    for row in summary_rows:
        print(
            f"{row['class']:20s} {row['band']:4s} {row['pool_size']:6d} {row['selected_count']:6d} "
            f"{row['n_environments_covered']:>4d}/{row['n_environments_available']:<5d} "
            f"{row['n_distinct_poses_selected']:>5d}/{row['n_distinct_poses_available']:<6d} "
            f"{row['byte_count'] / 1e6:10.1f}"
        )
        total_selected += row["selected_count"]
        total_bytes += row["byte_count"]
    print(f"\nTotal selected: {total_selected} images, {total_bytes / 1e6:.1f} MB")

    if args.dry_run:
        print("\n--dry-run: no files copied, no index/CSV written.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_INDEX_PATH, "w", encoding="utf-8") as f:
        for record in all_selected:
            f.write(
                json.dumps(
                    {
                        "filename": record["filename"],
                        "class": record["class"],
                        "band": record["band"],
                        "source_split": record["source_split"],
                        "shot_type": record["shot_type"],
                        "distance": record["distance"],
                        "lighting": record["lighting"],
                        "occlusion": record["occlusion"],
                        "pose": record["pose"],
                        "environment": record["environment"],
                        "prompt_file": record["prompt_file"],
                        "file_name": str(record["dest_path"].relative_to(REPO_ROOT)),
                        "dest_prompt_file": str(record["dest_prompt_path"].relative_to(REPO_ROOT)),
                    }
                )
                + "\n"
            )

    with open(SUMMARY_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "class",
                "band",
                "pool_size",
                "selected_count",
                "n_environments_covered",
                "n_environments_available",
                "n_distinct_poses_selected",
                "n_distinct_poses_available",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({k: row[k] for k in writer.fieldnames})

    print(f"\nWrote {OUT_INDEX_PATH.relative_to(REPO_ROOT)}")
    print(f"Wrote {SUMMARY_CSV_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
