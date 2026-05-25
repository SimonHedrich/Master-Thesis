"""
Retry failed test images using direct Gemini API calls (no batch).

Reads data/synthetic/test_index.jsonl, finds records with status "failed",
and regenerates them one by one with retries.

Usage:
    python scripts/synthetic/fix_failed_test_images.py
    python scripts/synthetic/fix_failed_test_images.py --workers 5
    python scripts/synthetic/fix_failed_test_images.py --force  # also retry if file exists
"""

import argparse
import asyncio
import io
import json
import os
import sys
import time
from pathlib import Path

try:
    from google import genai
    from google.genai import types
except ImportError:
    sys.exit("Error: google-genai is not installed.\nRun: pip install google-genai")

from dotenv import load_dotenv
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"
INDEX_JSONL   = SYNTHETIC_DIR / "test_index.jsonl"
IMAGES_DIR    = SYNTHETIC_DIR / "images" / "test"

MODEL              = "gemini-3.1-flash-image-preview"
IMAGE_ASPECT_RATIO = "4:3"
IMAGE_SIZE         = "512"
MAX_TRIES          = 3
RETRY_DELAY        = 3  # seconds between retries


def load_index() -> list[dict]:
    if not INDEX_JSONL.exists():
        sys.exit(f"Error: {INDEX_JSONL} not found.\nRun 1-generate_test_image_list.py first.")
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


def image_output_path(record: dict) -> Path:
    slug = Path(record["prompt_file"]).parent.name
    return IMAGES_DIR / slug / record["filename"]


def prompt_path(record: dict) -> Path:
    return SYNTHETIC_DIR / record["prompt_file"]


async def generate_image(prompt: str, client: genai.Client) -> bytes | None:
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
            print(f"  API error: {exc}", flush=True)
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
            print(f"  no image in response, retrying ({attempt}/{MAX_TRIES})", flush=True)
            await asyncio.sleep(RETRY_DELAY)

    return None


async def run(records_to_retry: list[dict], all_records: list[dict], client: genai.Client,
              workers: int, delay: float) -> None:
    rec_index = {rec["filename"]: i for i, rec in enumerate(all_records)}
    sem = asyncio.Semaphore(workers)
    lock = asyncio.Lock()
    generated = 0
    failed = 0

    async def process_one(rec: dict) -> None:
        nonlocal generated, failed
        filename = rec["filename"]
        p_path = prompt_path(rec)

        async with sem:
            if not p_path.exists():
                print(f"  SKIPPED {filename}: prompt file missing", flush=True)
                return

            prompt_text = p_path.read_text(encoding="utf-8")
            t0 = time.time()
            img_bytes = await generate_image(prompt_text, client)
            elapsed = time.time() - t0

            if delay > 0:
                await asyncio.sleep(delay)

            async with lock:
                idx = rec_index.get(filename)
                if img_bytes is None:
                    print(f"  FAILED  {filename}  ({elapsed:.1f}s)", flush=True)
                    failed += 1
                else:
                    out_path = image_output_path(rec)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.open(io.BytesIO(img_bytes)).save(out_path, "PNG")
                    print(f"  SAVED   {filename}  ({elapsed:.1f}s)", flush=True)
                    generated += 1
                    if idx is not None:
                        all_records[idx]["status"] = "generated"

    await asyncio.gather(*[process_one(r) for r in records_to_retry])

    save_index(all_records)
    print(f"\nDone: {generated} generated, {failed} failed")
    print(f"Index updated: {INDEX_JSONL}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Retry failed test image generation.")
    parser.add_argument("--workers", type=int, default=3,
                        help="Max concurrent API calls (default: 3).")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Seconds to sleep per worker after each call (default: 1.0).")
    parser.add_argument("--force", action="store_true",
                        help="Retry even if the image file already exists on disk.")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / "scripts" / "synthetic" / ".env")
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        sys.exit("Error: GEMINI_API_KEY is not set.\nAdd it to scripts/synthetic/.env")

    client = genai.Client(api_key=api_key)

    all_records = load_index()
    print(f"Loaded {len(all_records)} records from {INDEX_JSONL}")

    records_to_retry = [
        r for r in all_records
        if r.get("status") == "failed"
        and (args.force or not image_output_path(r).exists())
    ]

    if not records_to_retry:
        print("Nothing to retry — no failed records found.")
        return

    print(f"Retrying {len(records_to_retry)} failed image(s)  (workers={args.workers})\n")

    asyncio.run(run(records_to_retry, all_records, client, args.workers, args.delay))


if __name__ == "__main__":
    main()
