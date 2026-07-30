#!/usr/bin/env python3
"""
Stage 1f — Generate the `compressed` prompt regime (all 12 classes)

Materializes the ≤75-CLIP-token `compressed` prompt set for every class in
the model-comparison experiment
(docs/synthetic-model-comparison/05_prompt-strategy-and-length-limits.md),
the mandatory shared prerequisite for the local-model tier
(docs/synthetic-model-comparison/04_local-models-and-output-parameters.md):
SDXL-family models silently truncate at 77 CLIP tokens, so every model in
this experiment's local + compressed-ablation cells must be fed the
*identical* short prompt to keep the comparison apples-to-apples.

Modeled on 1b-generate_prompts_fresh.py's structure (per-class record
building, --dry-run, --force, resumable metadata) but self-contained: no
Wikipedia fetch and no LLM call, since a ≤75-token prompt is built entirely
from data already cached in reports/synthetic_scene_profiles.json plus a
small hand-written per-class habitat phrase (in the style of doc 05 §3's own
worked examples) and a short, generic, rotating pose list — there is no room
in the token budget for the six-axis scene grid the `full` regime uses, so
per doc 05 §3 rule 2 this only rotates pose, not habitat, across a class's
100 images.

Prompt shape (doc 05 §3):
    {class name}, {feature 1-3}, {pose}, {habitat}, wildlife photograph,
    telephoto, photorealistic, full body

Token count is verified against the SDXL CLIP-L tokenizer
(openai/clip-vit-large-patch14) and logged per image; prompts over budget
are trimmed (habitat, then pose, then features down to 2/1) until they fit,
per doc 05 §3 rule 4.

Downstream consumption mirrors 1b -> 1c exactly: this script only authors
prompt text + metadata (no images), like 1b does for the `full` regime's 5
fresh classes; 1g-generate_images_local.py is the first consumer/generator
(analogous to 1c), and later, per doc 01, the incumbent-compressed and
gpt-image-2-low-compressed ablation cells can reuse the exact same prompt
files via 1d/1e's --source-generator flag pointing at whichever local
generator's index.jsonl (all three carry identical prompt_file references).

Usage:
    uv run python scripts/synthetic_model_comparison/1f-generate_prompts_compressed.py
    uv run python scripts/synthetic_model_comparison/1f-generate_prompts_compressed.py --dry-run
    uv run python scripts/synthetic_model_comparison/1f-generate_prompts_compressed.py --classes "lion,red fox"
    uv run python scripts/synthetic_model_comparison/1f-generate_prompts_compressed.py --force

Outputs:
    data/synthetic_model_comparison/train/prompts_compressed/<class_slug>/<NNN>.txt
    reports/model_comparison_compressed_prompt_metadata.jsonl

Requirements:
    transformers (already in pyproject.toml, for the CLIP tokenizer)
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CLASSES_CSV = REPO_ROOT / "reports" / "model_comparison_classes.csv"
PROFILES_JSON = REPO_ROOT / "reports" / "synthetic_scene_profiles.json"
TRAIN_ROOT = REPO_ROOT / "data" / "synthetic_model_comparison" / "train"
PROMPTS_OUT_DIR = TRAIN_ROOT / "prompts_compressed"
METADATA_OUT_PATH = REPO_ROOT / "reports" / "model_comparison_compressed_prompt_metadata.jsonl"

TARGET_COUNT_PER_CLASS = 100
MAX_CLIP_TOKENS = 75  # 77 minus BOS/EOS, matching the SDXL/RealVisXL hard limit.
CLIP_TOKENIZER_ID = "openai/clip-vit-large-patch14"  # same tokenizer family as SDXL's CLIP-L.

STYLE_SUFFIX = "wildlife photograph, telephoto, photorealistic, full body"

# One short, hand-written habitat phrase per class — in the style of doc 05
# §3's own worked examples (Grevy's zebra / saiga / kinkajou). Kept fixed per
# class rather than rotated: at this token budget there's no room for the
# `full` regime's six-axis scene grid, and doc 05 §3 rule 2 only requires
# *pose* to vary across a class's 100 images, not habitat.
HABITAT_SHORT = {
    "plains zebra": "standing on open savanna grassland",
    "grevy's zebra": "standing on arid grassland",
    "mountain zebra": "on a rocky mountain slope",
    "red fox": "in a temperate forest clearing",
    "american black bear": "in a dense mixed forest",
    "lion": "on open savanna grassland",
    "kinkajou": "on a rainforest branch at night",
    "water deer": "in reedy wetland vegetation near a river",
    "ringtail": "on a rocky desert canyon ledge at night",
    "saiga": "standing on dry steppe",
    "aye-aye": "in a rainforest canopy at night",
    "pangolin family": "on a forest floor among leaf litter",
}

# Small fixed pose list, rotated across each class's 100 images (doc 05 §3
# rule 2) — condensed from the existing VAL_BEHAVIOR_DESCRIPTIONS codes used
# for the `full` regime's val-split shots, shortened to a few words each.
POSE_POOL = [
    "standing alert",
    "walking",
    "foraging",
    "resting",
    "looking at the camera",
]

# Metadata-only rotation (not reflected in the compressed prompt text itself
# — no token budget for it) kept purely for index-schema parity with the
# `full` regime's per-image records.
LIGHTING_POOL = ["golden_hour", "overcast", "midday", "dappled", "backlit"]

SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(name: str) -> str:
    return SLUG_RE.sub("_", name.lower()).strip("_")


def load_classes() -> dict[str, str]:
    """class common name -> band, from the frozen 12-class list."""
    if not CLASSES_CSV.exists():
        sys.exit(f"Error: {CLASSES_CSV} not found. Run 0-build_test_subset.py first.")
    classes: dict[str, str] = {}
    with open(CLASSES_CSV, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            classes[row["class"]] = row["band"]
    return classes


def load_profiles() -> dict[str, dict]:
    with open(PROFILES_JSON, encoding="utf-8") as f:
        return json.load(f)


def diagnostic_features(profile: dict, n: int) -> list[str]:
    raw = profile.get("key_diagnostic_features", "")
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return parts[:n]


def build_prompt_text(class_name: str, features: list[str], pose: str | None, habitat: str | None) -> str:
    parts = [class_name, *features]
    if pose:
        parts.append(pose)
    if habitat:
        parts.append(habitat)
    parts.append(STYLE_SUFFIX)
    return ", ".join(parts)


def fit_to_token_budget(
    tokenizer, class_name: str, features: list[str], pose: str, habitat: str
) -> tuple[str, int]:
    """Trim habitat, then pose, then features (3 -> 2 -> 1) until the prompt
    fits MAX_CLIP_TOKENS. Returns (final prompt text, token count)."""
    candidates = [
        (features, pose, habitat),
        (features, pose, None),
        (features, None, None),
        (features[:2], None, None),
        (features[:1], None, None),
    ]
    last_text, last_count = "", 0
    for feats, p, h in candidates:
        text = build_prompt_text(class_name, feats, p, h)
        count = len(tokenizer.encode(text, add_special_tokens=True))
        last_text, last_count = text, count
        if count <= MAX_CLIP_TOKENS:
            return text, count
    return last_text, last_count


def build_class_records(class_name: str, band: str, slug: str, profile: dict, tokenizer) -> list[dict]:
    features = diagnostic_features(profile, 3)
    habitat = HABITAT_SHORT.get(class_name)
    if habitat is None:
        sys.exit(f"Error: no HABITAT_SHORT entry for class '{class_name}' — add one.")

    records: list[dict] = []
    for i in range(TARGET_COUNT_PER_CLASS):
        image_num = i + 1
        pose = POSE_POOL[i % len(POSE_POOL)]
        lighting = LIGHTING_POOL[i % len(LIGHTING_POOL)]

        prompt_text, token_count = fit_to_token_budget(tokenizer, class_name, features, pose, habitat)

        prompt_rel_path = PROMPTS_OUT_DIR / slug / f"{image_num:03d}.txt"
        records.append({
            "class": class_name,
            "band": band,
            "index": image_num,
            "filename": f"{band.lower()}_{slug}_{image_num:03d}.png",
            "shot_type": "eye_level",
            "distance": "medium",
            "lighting": lighting,
            "occlusion": "none",
            "pose": pose,
            "environment": habitat,
            "source_split": "train",
            "prompt_file": str(prompt_rel_path.relative_to(REPO_ROOT)),
            "token_count": token_count,
            "prompt_text": prompt_text,
        })
    return records


def write_class_prompts(slug: str, records: list[dict], dry_run: bool) -> None:
    if dry_run:
        return
    class_dir = PROMPTS_OUT_DIR / slug
    class_dir.mkdir(parents=True, exist_ok=True)
    for rec in records:
        (REPO_ROOT / rec["prompt_file"]).write_text(rec["prompt_text"], encoding="utf-8")


def load_existing_metadata() -> list[dict]:
    if not METADATA_OUT_PATH.exists():
        return []
    records = []
    with open(METADATA_OUT_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_metadata(records: list[dict]) -> None:
    METADATA_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METADATA_OUT_PATH, "w", encoding="utf-8") as f:
        for rec in records:
            out = {k: v for k, v in rec.items() if k != "prompt_text"}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--classes",
        default=None,
        help="Comma-separated subset of class common names to process (default: all 12).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate prompt files even if 100 already exist for a class.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print the plan without loading the tokenizer or writing files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    all_classes = load_classes()
    classes = all_classes
    if args.classes:
        requested = {c.strip().lower() for c in args.classes.split(",")}
        classes = {name: band for name, band in all_classes.items() if name.lower() in requested}
        not_found = requested - {name.lower() for name in classes}
        if not_found:
            print(f"Warning: class(es) not found: {', '.join(sorted(not_found))}")
        if not classes:
            sys.exit("No matching classes found.")

    profiles = load_profiles()

    existing_metadata = load_existing_metadata()
    metadata_by_class: dict[str, list[dict]] = {}
    for rec in existing_metadata:
        metadata_by_class.setdefault(rec["class"], []).append(rec)

    if args.dry_run:
        print(f"{'class':22s} {'slug':22s} {'shots':>6s}")
        for class_name in classes:
            print(f"{class_name:22s} {slugify(class_name):22s} {TARGET_COUNT_PER_CLASS:6d}  (dry-run, not built)")
        print("\n--dry-run: no tokenizer loaded, no files written.")
        return

    print("Loading CLIP tokenizer for token-budget verification ...")
    from transformers import CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(CLIP_TOKENIZER_ID)

    print(f"\n{'class':22s} {'slug':22s} {'shots':>6s} {'max_tok':>8s}")
    all_new_records: dict[str, list[dict]] = {}

    for class_name, band in classes.items():
        slug = slugify(class_name)
        class_dir = PROMPTS_OUT_DIR / slug
        existing_count = len(list(class_dir.glob("*.txt"))) if class_dir.exists() else 0
        if not args.force and existing_count >= TARGET_COUNT_PER_CLASS:
            print(f"{class_name:22s} {slug:22s} {existing_count:6d}  (already complete, skipping)")
            continue

        profile = profiles.get(class_name)
        if profile is None:
            sys.exit(f"Error: '{class_name}' has no cached scene profile in {PROFILES_JSON}")

        records = build_class_records(class_name, band, slug, profile, tokenizer)
        write_class_prompts(slug, records, dry_run=False)
        all_new_records[class_name] = records
        max_tok = max(r["token_count"] for r in records)
        over_budget = sum(1 for r in records if r["token_count"] > MAX_CLIP_TOKENS)
        flag = f"  ({over_budget} over budget!)" if over_budget else ""
        print(f"{class_name:22s} {slug:22s} {len(records):6d} {max_tok:8d}{flag}")

    if not all_new_records:
        print("\nNothing to do.")
        return

    for class_name, records in all_new_records.items():
        metadata_by_class[class_name] = [
            {k: v for k, v in rec.items() if k != "prompt_text"} for rec in records
        ]

    combined = [rec for recs in metadata_by_class.values() for rec in recs]
    write_metadata(combined)

    total_new = sum(len(r) for r in all_new_records.values())
    print(f"\nWrote {total_new} prompt files across {len(all_new_records)} class(es).")
    print(f"Wrote {len(combined)} total records to {METADATA_OUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
