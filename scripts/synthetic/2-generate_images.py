"""
Generate synthetic wildlife images by calling the Gemini API for every record in
data/synthetic/index.jsonl.

By default resumes where it left off — images that already exist on disk are skipped.
Use --force to regenerate existing images.

Usage:
    # Full run (all 12,600 images):
    uv run python scripts/synthetic/2-generate_images.py

    # Preview: 5 images per class, spread across shot groups:
    uv run python scripts/synthetic/2-generate_images.py --preview 5

    # Preview for specific classes only:
    uv run python scripts/synthetic/2-generate_images.py --preview 5 --classes "walrus,kinkajou"

    # Regenerate a class from scratch:
    uv run python scripts/synthetic/2-generate_images.py --classes walrus --force

Requirements:
    pip install google-genai pillow python-dotenv
"""

import argparse
import asyncio
import io
import json
import os
import sys
import time
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

try:
    from google import genai
    from google.genai import types
except ImportError:
    sys.exit(
        "Error: google-genai is not installed.\n"
        "Run: pip install google-genai"
    )

from dotenv import load_dotenv
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"
INDEX_JSONL = SYNTHETIC_DIR / "index.jsonl"

MODEL = "gemini-3.1-flash-image-preview"
MAX_TRIES = 3
RETRY_DELAY = 3  # seconds between retries on no-image response

DEFAULT_DELAY = 1.0
DEFAULT_FLUSH_EVERY = 25
DEFAULT_PREVIEW_N = 5

IMAGE_ASPECT_RATIO = "4:3"
IMAGE_SIZE = "512"    # minimum supported; "512" is not available via generate_content

# ---------------------------------------------------------------------------
# Index I/O
# ---------------------------------------------------------------------------

def load_index() -> list[dict]:
    if not INDEX_JSONL.exists():
        sys.exit(f"Error: {INDEX_JSONL} not found. Run 1-generate_image_list.py first.")
    records = []
    with open(INDEX_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_index(records: list[dict]) -> None:
    with open(INDEX_JSONL, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def image_output_path(record: dict) -> Path:
    band = record["band"].lower()
    # class slug: derive from filename prefix (e.g. "a_walrus_001.jpg" → "walrus")
    fname = record["filename"]
    # filename pattern: {band}_{class_slug}_{nnn}.jpg
    # strip band prefix and number suffix to get class_slug
    parts = fname.split("_")
    # parts[0] is band letter, parts[-1] is "NNN.jpg", everything in between is slug
    class_slug = "_".join(parts[1:-1])
    return SYNTHETIC_DIR / "images" / f"band_{band}" / class_slug / fname.replace(".jpg", ".png")


def prompt_path(record: dict) -> Path:
    return SYNTHETIC_DIR / record["prompt_file"]

# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------

async def generate_image(prompt: str, client: genai.Client) -> bytes | None:
    """Call Gemini and return raw image bytes, or None if all retries fail."""
    config = types.GenerateContentConfig(
        response_modalities=["IMAGE", "TEXT"],
        image_config=types.ImageConfig(
            aspect_ratio=IMAGE_ASPECT_RATIO,
            image_size=IMAGE_SIZE,
        ),
    )

    for attempt in range(1, MAX_TRIES + 1):
        try:
            response = await client.aio.models.generate_content(
                model=MODEL,
                contents=[prompt],
                config=config,
            )
        except Exception as exc:
            print(f" API error: {exc}", flush=True)
            if attempt < MAX_TRIES:
                await asyncio.sleep(RETRY_DELAY)
            continue

        try:
            parts = response.candidates[0].content.parts or []
        except (IndexError, AttributeError):
            parts = []

        for part in parts:
            if part.inline_data is not None:
                raw = part.inline_data.data
                pil_img = Image.open(io.BytesIO(raw))
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                return buf.getvalue()

        if attempt < MAX_TRIES:
            print(f" no image in response, retrying ({attempt}/{MAX_TRIES})", flush=True)
            await asyncio.sleep(RETRY_DELAY)

    return None

# ---------------------------------------------------------------------------
# Preview index selection
# ---------------------------------------------------------------------------

def select_preview_indices(class_records: list[dict], n: int) -> list[int]:
    """Pick n evenly-spaced indices into class_records to sample different shot groups."""
    total = len(class_records)
    if n >= total:
        return list(range(total))
    return [int(i * total / n) for i in range(n)]

# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

async def run(
    records_to_process: list[dict],
    client: genai.Client,
    force: bool,
    delay: float,
    flush_every: int,
    all_records: list[dict],
    workers: int,
) -> None:
    rec_index: dict[tuple, int] = {}
    for i, rec in enumerate(all_records):
        rec_index[(rec["class"], rec["filename"])] = i

    by_class: dict[str, list[dict]] = defaultdict(list)
    for rec in records_to_process:
        by_class[rec["class"]].append(rec)

    classes = list(by_class.keys())
    n_classes = len(classes)

    total_generated = 0
    total_failed = 0
    total_skipped = 0
    since_flush = 0
    start_time = time.time()

    sem = asyncio.Semaphore(workers)
    lock = asyncio.Lock()

    async def process_one(rec: dict, class_name: str, class_idx: int, n_in_class: int, img_seq: int) -> None:
        nonlocal total_generated, total_failed, total_skipped, since_flush

        out_path = image_output_path(rec)
        prefix = f"[{class_name} {class_idx}/{n_classes}]  img {img_seq}/{n_in_class}"

        async with sem:
            if not force and out_path.exists():
                async with lock:
                    total_skipped += 1
                    key = (rec["class"], rec["filename"])
                    if key in rec_index and all_records[rec_index[key]]["status"] != "generated":
                        all_records[rec_index[key]]["status"] = "generated"
                        since_flush += 1
                        if since_flush >= flush_every:
                            save_index(all_records)
                            since_flush = 0
                return

            p_path = prompt_path(rec)
            if not p_path.exists():
                print(f"{prefix}  SKIPPED (prompt file missing: {p_path.name})", flush=True)
                async with lock:
                    total_skipped += 1
                return

            prompt_text = p_path.read_text(encoding="utf-8")
            t0 = time.time()
            img_bytes = await generate_image(prompt_text, client)
            elapsed_img = time.time() - t0

            if delay > 0:
                await asyncio.sleep(delay)

            async with lock:
                key = (rec["class"], rec["filename"])
                if img_bytes is None:
                    print(f"{prefix}  FAILED  ({elapsed_img:.1f}s)", flush=True)
                    total_failed += 1
                    if key in rec_index:
                        all_records[rec_index[key]]["status"] = "failed"
                else:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.open(io.BytesIO(img_bytes)).save(out_path, "PNG")
                    print(f"{prefix}  saved → {out_path.name}  ({elapsed_img:.1f}s)", flush=True)
                    total_generated += 1
                    if key in rec_index:
                        all_records[rec_index[key]]["status"] = "generated"
                since_flush += 1
                if since_flush >= flush_every:
                    save_index(all_records)
                    since_flush = 0

    tasks = []
    for class_idx, class_name in enumerate(classes, 1):
        class_records = by_class[class_name]
        for img_seq, rec in enumerate(class_records, 1):
            tasks.append(process_one(rec, class_name, class_idx, len(class_records), img_seq))

    await asyncio.gather(*tasks)

    if since_flush > 0:
        save_index(all_records)

    elapsed = time.time() - start_time
    elapsed_str = str(timedelta(seconds=int(elapsed)))
    print(f"\nTotal: {total_generated} generated, {total_failed} failed, {total_skipped} skipped  |  elapsed: {elapsed_str}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic wildlife images using the Gemini API."
    )
    parser.add_argument(
        "--preview",
        type=int,
        metavar="N",
        default=None,
        help=(
            f"Generate N images per class (evenly spread across shot groups) "
            f"instead of all. Default when flag given: {DEFAULT_PREVIEW_N}."
        ),
    )
    parser.add_argument(
        "--classes",
        default=None,
        help="Comma-separated class names to process (default: all classes in index.jsonl).",
    )
    parser.add_argument(
        "--split",
        choices=["all", "train", "val"],
        default="all",
        help="Only process records with this split (default: all).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate images even if they already exist on disk.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help=f"Seconds to sleep between API calls (default: {DEFAULT_DELAY}).",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=DEFAULT_FLUSH_EVERY,
        dest="flush_every",
        help=f"Write index.jsonl every N images (default: {DEFAULT_FLUSH_EVERY}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help=(
            "Max concurrent Gemini API calls (default: 5). "
            "Reduce if you see 429 errors; --delay is a per-worker post-call cooldown."
        ),
    )
    args = parser.parse_args()

    # Handle bare --preview with no value
    preview_n = args.preview if args.preview is not None else None

    load_dotenv(PROJECT_ROOT / "scripts" / "synthetic" / ".env")
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        sys.exit(
            "Error: GEMINI_API_KEY is not set.\n"
            "Add it to scripts/synthetic/.env or set it as an environment variable."
        )

    client = genai.Client(api_key=api_key)

    all_records = load_index()
    print(f"Loaded {len(all_records)} records from {INDEX_JSONL}")

    # Filter by class if requested
    if args.classes:
        requested = {c.strip().lower() for c in args.classes.split(",")}
        filtered = [r for r in all_records if r["class"].lower() in requested]
        not_found = requested - {r["class"].lower() for r in filtered}
        if not_found:
            print(f"Warning: class(es) not found in index: {', '.join(sorted(not_found))}")
        records = filtered
    else:
        records = all_records

    # Filter by split
    if args.split != "all":
        records = [r for r in records if r.get("split") == args.split]

    # Apply preview: select N evenly-spaced records per class
    if preview_n is not None:
        by_class: dict[str, list[dict]] = defaultdict(list)
        for rec in records:
            by_class[rec["class"]].append(rec)
        selected = []
        for cls_recs in by_class.values():
            if args.force:
                # --force: pick N evenly-spaced from all records and regenerate them
                idxs = select_preview_indices(cls_recs, preview_n)
                selected.extend(cls_recs[i] for i in idxs)
            else:
                # top-up: count existing, pick only what's still needed
                not_yet = [r for r in cls_recs if not image_output_path(r).exists()]
                already_have = len(cls_recs) - len(not_yet)
                need = max(0, preview_n - already_have)
                if need > 0:
                    idxs = select_preview_indices(not_yet, need)
                    selected.extend(not_yet[i] for i in idxs)
        records = selected
        print(f"Preview mode: {preview_n} images/class → {len(records)} to generate")
    else:
        print(f"Processing {len(records)} records")

    if not records:
        print("Nothing to process.")
        return

    print(f"Model  : {MODEL}")
    print(f"Workers: {args.workers}")
    print(f"Force  : {args.force}")
    print(f"Delay  : {args.delay}s/worker\n")

    asyncio.run(run(
        records_to_process=records,
        client=client,
        force=args.force,
        delay=args.delay,
        flush_every=args.flush_every,
        all_records=all_records,
        workers=args.workers,
    ))


if __name__ == "__main__":
    main()
