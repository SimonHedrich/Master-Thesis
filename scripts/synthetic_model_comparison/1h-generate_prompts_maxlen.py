#!/usr/bin/env python3
"""
Stage 1h — Generate the `maxlen` prompt regime (256-token and 512-token tiers)

Builds prompts sized to *actually use* each local model's real text-encoder
capacity, for the production 1,200-image/model dataset — a different goal
from 1f-generate_prompts_compressed.py's `compressed` regime, which
deliberately gives every model the *same* ≤75-token prompt for an
apples-to-apples fairness ablation
(docs/synthetic-model-comparison/05_prompt-strategy-and-length-limits.md).
Here, models with a larger text-encoder budget (T5-XXL's 256 tokens for the
SD 3.5 family; the ~512-token Qwen-family encoders in FLUX.2-klein-9B and
Qwen-Image) get a genuinely longer, richer prompt instead of leaving that
capacity unused. RealVisXL+SDXL-Lightning is untouched — its existing
`compressed` prompt is already engineered to fill its 77-token CLIP ceiling,
so this script doesn't touch or duplicate it.

Two tiers, not the naive "truncate the existing `full` prompt" approach
tried first: inspecting an actual full-regime prompt showed its own
structural sections (SCENE SPECIFICATION + PHOTOGRAPHY STYLE + CRITICAL
REQUIREMENTS — the parts that aren't free-text Wikipedia excerpts) already
total ~464 words (~600+ T5 tokens) on their own, more than the entire
256-token budget before any species description is added. So instead this
reuses the simpler, already-proven template shape from
scripts/synthetic/2-generate_synthetic_images_local.py's build_prompt()
("Realistic wildlife photograph of a {class}. Species characteristics:
{excerpt}. {STYLE_SUFFIX}") and extends it to actually fill each budget
(not just clear it, which is all that shape needed to do there), plus adds
genuine per-image scene variation:

  - The free-text species-description excerpt comes from the existing
    train/prompts_full/<slug>/001.txt (same content for every image of a
    class — parsed once per class from its SPECIES DESCRIPTION: section),
    accumulated sentence-by-sentence until the tier's token budget is hit
    (mirrors 1f's fit_to_token_budget cascading-trim pattern, just building
    up instead of trimming down).
  - Real per-image pose/environment sentences come directly from the
    canonical train/gemini-3.1-flash-image-preview/full/index.jsonl (1,200
    records, one per class x image, with clean single-sentence pose/
    environment fields already parsed out by 1-select_train_subset_incumbent.py
    and 1c-generate_images_fresh.py) — this gives real per-image angle/
    lighting/behavior variation the `compressed` regime's simple 5-item
    pose rotation doesn't have, with no need to re-derive the production
    SHOT_SCHEDULE.

Token verification: 256-tier against T5TokenizerFast (loaded from SD 3.5
Medium's tokenizer_3 subfolder — the same T5-XXL used across the whole SD
3.5 family); 512-tier against Qwen/Qwen-Image's own tokenizer (both
FLUX.2-klein-9B's "Qwen3" and Qwen-Image's Qwen2.5-VL encoders are
Qwen-family BPE, close enough for budgeting given the safety margin below
each tier's nominal ceiling).

Usage:
    uv run python scripts/synthetic_model_comparison/1h-generate_prompts_maxlen.py
    uv run python scripts/synthetic_model_comparison/1h-generate_prompts_maxlen.py --dry-run
    uv run python scripts/synthetic_model_comparison/1h-generate_prompts_maxlen.py --classes "lion,red fox"
    uv run python scripts/synthetic_model_comparison/1h-generate_prompts_maxlen.py --force

Outputs:
    data/synthetic_model_comparison/train/prompts_maxlen_256/<class_slug>/<NNN>.txt
    data/synthetic_model_comparison/train/prompts_maxlen_512/<class_slug>/<NNN>.txt
    reports/model_comparison_maxlen_prompt_metadata.jsonl

Requirements:
    transformers (already in pyproject.toml, for the T5/Qwen tokenizers)
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
# Shared credentials with the production pipeline — the T5 tokenizer lives
# in a gated stabilityai repo; loading HF_TOKEN here is required to fetch it.
load_dotenv(REPO_ROOT / "scripts" / "synthetic" / ".env")

CLASSES_CSV = REPO_ROOT / "reports" / "model_comparison_classes.csv"
WIKI_URLS_JSON = REPO_ROOT / "reports" / "wikipedia_urls.json"
PROFILES_JSON = REPO_ROOT / "reports" / "synthetic_scene_profiles.json"
FULL_PROMPTS_DIR = REPO_ROOT / "data" / "synthetic_model_comparison" / "train" / "prompts_full"
CANONICAL_INDEX_PATH = (
    REPO_ROOT / "data" / "synthetic_model_comparison" / "train"
    / "gemini-3.1-flash-image-preview" / "full" / "index.jsonl"
)
TRAIN_ROOT = REPO_ROOT / "data" / "synthetic_model_comparison" / "train"
METADATA_OUT_PATH = REPO_ROOT / "reports" / "model_comparison_maxlen_prompt_metadata.jsonl"

TARGET_COUNT_PER_CLASS = 100

# Safety margin under each tier's nominal ceiling, absorbing tokenizer
# mismatch between the two Qwen-family encoders sharing the 512 tier.
TIER_BUDGETS = {256: 245, 512: 495}
T5_TOKENIZER_SOURCE = ("stabilityai/stable-diffusion-3.5-medium", "tokenizer_3")
QWEN_TOKENIZER_SOURCE = ("Qwen/Qwen-Image", "tokenizer")

# Same fixed style sentence used by the production local-generation script
# (scripts/synthetic/2-generate_synthetic_images_local.py's STYLE_SUFFIX).
STYLE_SUFFIX = (
    "Professional wildlife photograph. Telephoto lens, sharp focus on the animal, "
    "natural lighting, photorealistic, high resolution. Full body of the animal visible, "
    "entire animal from head to tail fits within the frame. Natural habitat background."
)
CLOSING_REMINDER = (
    "Exactly one {class_name} in frame, diagnostic features clearly visible, "
    "no other individuals of the same species."
)

SLUG_RE = re.compile(r"[^a-z0-9]+")
SPECIES_DESC_RE = re.compile(
    r"^SPECIES DESCRIPTION:\s*\n(.*?)(?=\n[A-Z][A-Z /]+:\n|\Z)",
    re.MULTILINE | re.DOTALL,
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
FILENAME_INDEX_RE = re.compile(r"_(\d{3})\.png$")


def slugify(name: str) -> str:
    return SLUG_RE.sub("_", name.lower()).strip("_")


def load_classes() -> dict[str, str]:
    with open(CLASSES_CSV, encoding="utf-8", newline="") as f:
        return {row["class"]: row["band"] for row in csv.DictReader(f)}


def load_scientific_names() -> dict[str, str]:
    with open(WIKI_URLS_JSON, encoding="utf-8") as f:
        wiki_urls: dict = json.load(f)
    return {
        v.get("common_name", "").strip().lower(): v.get("scientific_name", "")
        for v in wiki_urls.values()
    }


def load_diagnostic_features() -> dict[str, list[str]]:
    with open(PROFILES_JSON, encoding="utf-8") as f:
        profiles: dict = json.load(f)
    out = {}
    for class_name, profile in profiles.items():
        raw = profile.get("key_diagnostic_features", "")
        out[class_name] = [p.strip() for p in raw.split(",") if p.strip()][:3]
    return out


def load_canonical_scene_records() -> dict[tuple[str, int], dict]:
    """(class, image_index) -> {pose, environment, shot_type, distance,
    lighting, occlusion, source_split} from the canonical 1,200-record
    incumbent index.jsonl — the single source of clean, already-parsed
    per-image scene fields for all 12 classes."""
    if not CANONICAL_INDEX_PATH.exists():
        sys.exit(f"Error: {CANONICAL_INDEX_PATH} not found.")
    out = {}
    with open(CANONICAL_INDEX_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            m = FILENAME_INDEX_RE.search(rec["filename"])
            if not m:
                continue
            index = int(m.group(1))
            out[(rec["class"], index)] = rec
    return out


def extract_species_description(slug: str) -> str:
    """Species description is identical across a class's 100 images —
    parse it once from image 001's full-regime prompt."""
    path = FULL_PROMPTS_DIR / slug / "001.txt"
    if not path.exists():
        sys.exit(f"Error: {path} not found. Full-regime prompts must exist for all 12 classes.")
    text = path.read_text(encoding="utf-8")
    m = SPECIES_DESC_RE.search(text)
    if not m:
        sys.exit(f"Error: could not find 'SPECIES DESCRIPTION:' section in {path}")
    # Collapse the (possibly multi-paragraph, blank-line-separated) Wikipedia
    # excerpt into a flat sentence stream for greedy sentence accumulation.
    desc = re.sub(r"\s+", " ", m.group(1)).strip()
    return desc


def build_prompt_text(
    class_name: str, scientific_name: str, features: list[str], desc_excerpt: str,
    pose: str, environment: str,
) -> str:
    feature_str = ", ".join(features)
    subject = f"{class_name} ({scientific_name})" if scientific_name else class_name
    article = "an" if class_name[:1].lower() in "aeiou" else "a"
    parts = [
        f"Realistic wildlife photograph of {article} {subject}.",
        f"Species characteristics: {feature_str}. {desc_excerpt}".strip(),
        pose,
        environment,
        STYLE_SUFFIX,
        CLOSING_REMINDER.format(class_name=class_name),
    ]
    return " ".join(p for p in parts if p)


def fit_desc_to_budget(
    tokenizer, budget: int, class_name: str, scientific_name: str,
    features: list[str], desc_sentences: list[str], pose: str, environment: str,
) -> tuple[str, int]:
    """Greedily accumulate description sentences until the next one would
    exceed the token budget — the inverse of 1f's cascading-trim pattern
    (there: start full, trim down; here: start empty, build up)."""
    excerpt = ""
    best_text = build_prompt_text(class_name, scientific_name, features, excerpt, pose, environment)
    best_count = len(tokenizer.encode(best_text))

    for sentence in desc_sentences:
        candidate_excerpt = (excerpt + " " + sentence).strip() if excerpt else sentence
        candidate_text = build_prompt_text(
            class_name, scientific_name, features, candidate_excerpt, pose, environment
        )
        count = len(tokenizer.encode(candidate_text))
        if count > budget:
            break
        excerpt, best_text, best_count = candidate_excerpt, candidate_text, count

    return best_text, best_count


def build_class_tier_records(
    class_name: str, band: str, slug: str, tier: int, tokenizer,
    scientific_names: dict[str, str], features_by_class: dict[str, list[str]],
    scene_records: dict[tuple[str, int], dict],
) -> list[dict]:
    desc_text = extract_species_description(slug)
    desc_sentences = SENTENCE_SPLIT_RE.split(desc_text)
    scientific_name = scientific_names.get(class_name.lower(), "")
    features = features_by_class.get(class_name, [class_name])
    budget = TIER_BUDGETS[tier]

    # Index numbers are NOT necessarily contiguous 1-100: the 6 Bucket-3
    # classes (kinkajou, water deer, ringtail, saiga, aye-aye, pangolin
    # family) keep their original index out of a 200-image pool (per
    # 1-select_train_subset_incumbent.py's stratify-diversify selection),
    # so this iterates whatever 100 indices the canonical index.jsonl
    # actually has for this class, rather than assuming range(1, 101).
    class_indices = sorted(idx for (name, idx) in scene_records if name == class_name)
    if len(class_indices) != TARGET_COUNT_PER_CLASS:
        sys.exit(
            f"Error: expected {TARGET_COUNT_PER_CLASS} canonical scene records for "
            f"'{class_name}', found {len(class_indices)}."
        )

    records = []
    for image_num in class_indices:
        scene = scene_records[(class_name, image_num)]

        prompt_text, token_count = fit_desc_to_budget(
            tokenizer, budget, class_name, scientific_name, features,
            desc_sentences, scene["pose"], scene["environment"],
        )

        prompt_rel_path = TRAIN_ROOT / f"prompts_maxlen_{tier}" / slug / f"{image_num:03d}.txt"
        records.append({
            "class": class_name,
            "band": band,
            "index": image_num,
            "filename": f"{band.lower()}_{slug}_{image_num:03d}.png",
            "tier": tier,
            "shot_type": scene["shot_type"],
            "distance": scene["distance"],
            "lighting": scene["lighting"],
            "occlusion": scene["occlusion"],
            "pose": scene["pose"],
            "environment": scene["environment"],
            "source_split": scene["source_split"],
            "prompt_file": str(prompt_rel_path.relative_to(REPO_ROOT)),
            "token_count": token_count,
            "prompt_text": prompt_text,
        })
    return records


def write_class_prompts(tier: int, slug: str, records: list[dict], dry_run: bool) -> None:
    if dry_run:
        return
    class_dir = TRAIN_ROOT / f"prompts_maxlen_{tier}" / slug
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
        help="Regenerate prompt files even if 100 already exist for a class/tier.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print the plan without loading tokenizers or writing files.",
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

    if args.dry_run:
        print(f"{'class':22s} {'slug':22s} {'tiers':>10s}")
        for class_name in classes:
            print(f"{class_name:22s} {slugify(class_name):22s} {'256, 512':>10s}  (dry-run, not built)")
        print("\n--dry-run: no tokenizers loaded, no files written.")
        return

    scientific_names = load_scientific_names()
    features_by_class = load_diagnostic_features()
    scene_records = load_canonical_scene_records()

    print("Loading T5 tokenizer (256-tier) ...")
    from transformers import AutoTokenizer

    t5_repo, t5_subfolder = T5_TOKENIZER_SOURCE
    t5_tokenizer = AutoTokenizer.from_pretrained(t5_repo, subfolder=t5_subfolder)

    print("Loading Qwen tokenizer (512-tier) ...")
    qwen_repo, qwen_subfolder = QWEN_TOKENIZER_SOURCE
    qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_repo, subfolder=qwen_subfolder)

    tokenizers = {256: t5_tokenizer, 512: qwen_tokenizer}

    existing_metadata = load_existing_metadata()
    metadata_by_key: dict[tuple[str, int], list[dict]] = {}
    for rec in existing_metadata:
        metadata_by_key.setdefault((rec["class"], rec["tier"]), []).append(rec)

    print(f"\n{'class':22s} {'slug':22s} {'tier':>6s} {'shots':>6s} {'max_tok':>8s}")
    all_new_records: dict[tuple[str, int], list[dict]] = {}

    for class_name, band in classes.items():
        slug = slugify(class_name)
        for tier in (256, 512):
            tier_dir = TRAIN_ROOT / f"prompts_maxlen_{tier}" / slug
            existing_count = len(list(tier_dir.glob("*.txt"))) if tier_dir.exists() else 0
            if not args.force and existing_count >= TARGET_COUNT_PER_CLASS:
                print(f"{class_name:22s} {slug:22s} {tier:6d} {existing_count:6d}  (already complete, skipping)")
                continue

            records = build_class_tier_records(
                class_name, band, slug, tier, tokenizers[tier],
                scientific_names, features_by_class, scene_records,
            )
            write_class_prompts(tier, slug, records, dry_run=False)
            all_new_records[(class_name, tier)] = records
            max_tok = max(r["token_count"] for r in records)
            over_budget = sum(1 for r in records if r["token_count"] > TIER_BUDGETS[tier])
            flag = f"  ({over_budget} over budget!)" if over_budget else ""
            print(f"{class_name:22s} {slug:22s} {tier:6d} {len(records):6d} {max_tok:8d}{flag}")

    if not all_new_records:
        print("\nNothing to do.")
        return

    for (class_name, tier), records in all_new_records.items():
        metadata_by_key[(class_name, tier)] = [
            {k: v for k, v in rec.items() if k != "prompt_text"} for rec in records
        ]

    combined = [rec for recs in metadata_by_key.values() for rec in recs]
    write_metadata(combined)

    total_new = sum(len(r) for r in all_new_records.values())
    print(f"\nWrote {total_new} prompt files across {len(all_new_records)} class/tier combination(s).")
    print(f"Wrote {len(combined)} total records to {METADATA_OUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
