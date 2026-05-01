"""Generate <DETAILED_CAPTION> descriptions for all passed images using Florence-2-base.

Reads filter_results.jsonl for one or all dataset sources, runs batched
Florence-2-base inference on every entry where passed=true and no caption
field yet exists, then writes the caption back to filter_results.jsonl as a
new "caption" field.

The script is resumable: entries that already have a "caption" field are
skipped. Results are flushed to disk every 500 images to guard against
interruption — at most one flush interval of work is lost if killed.

Model:      microsoft/Florence-2-base
Task:       <DETAILED_CAPTION>
Batch size: 6
Image size: 768 × 768 px (DaViT encoder requires square feature maps)

Output fields written to filter_results.jsonl:
  caption        — Florence-2 detailed caption for passed images
  caption_error  — error message when an image could not be opened (soft
                   failure; does not change the passed flag)

Note: "caption" is distinct from "vlm_caption", which is written by Stage 3
of 1-filter_dataset_quality.py onto *failed* Wikimedia borderline entries as
part of the rescue decision. Do not conflate the two fields.

Usage:
    python scripts/dataset_quality/4-generate_captions.py --source wikimedia
    python scripts/dataset_quality/4-generate_captions.py --source all

Requirements:
    pip install transformers==4.48.3 timm einops pillow tqdm torch
    (transformers 5.x breaks Florence-2 remote code)
"""

import argparse
import concurrent.futures
import gc
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

# ── Constants ─────────────────────────────────────────────────────────────────

REPO_ROOT     = Path(__file__).resolve().parents[2]

MODEL_ID      = "microsoft/Florence-2-base"
BATCH_SIZE    = 6
IMAGE_SIZE    = 768
TASK          = "<DETAILED_CAPTION>"
CAPTION_FIELD = "caption"
FLUSH_EVERY   = 500   # flush to disk after this many images

RESULTS_PATHS = {
    "gbif":        REPO_ROOT / "data" / "gbif"        / "filter_results.jsonl",
    "inaturalist": REPO_ROOT / "data" / "inaturalist" / "filter_results.jsonl",
    "wikimedia":   REPO_ROOT / "data" / "wikimedia"   / "filter_results.jsonl",
    "openimages":  REPO_ROOT / "data" / "openimages"  / "filter_results.jsonl",
    "images_cv":   REPO_ROOT / "data" / "images_cv"   / "filter_results.jsonl",
}

# ── JSONL utilities ───────────────────────────────────────────────────────────

def load_results(jsonl_path: Path) -> list:
    if not jsonl_path.exists():
        return []
    entries = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def save_results(jsonl_path: Path, entries: list) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


# ── Image loading helpers ──────────────────────────────────────────────────────

def _load_one_image(entry: dict) -> tuple:
    """Load and resize one image. Returns (entry, image, None) or (entry, None, err_str).
    Pure function — safe to call from background threads."""
    fp = REPO_ROOT / entry["filepath"]
    try:
        img = Image.open(fp).convert("RGB")
        resized = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        img.close()
        return (entry, resized, None)
    except Exception as exc:
        return (entry, None, str(exc))


def _load_batch_images(batch_entries: list) -> tuple:
    """Load and resize all images in a batch in parallel.
    Returns (valid_entries, valid_images, failed_pairs).
    Must NOT write to shared state — called from background thread."""
    valid_entries, valid_images, failed_pairs = [], [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch_entries)) as pool:
        for entry, img, err in pool.map(_load_one_image, batch_entries):
            if err is None:
                valid_entries.append(entry)
                valid_images.append(img)
            else:
                failed_pairs.append((entry, err))
    return valid_entries, valid_images, failed_pairs


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model_and_processor(model_id: str) -> tuple:
    """Load Florence-2 onto CUDA and return (model, processor)."""
    model = (
        AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype="auto")
        .eval()
        .cuda()
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor


def caption_batch(
    model: AutoModelForCausalLM,
    processor: AutoProcessor,
    images: list,
) -> list:
    """Caption a batch of PIL images, returning one clean string per image.

    Strips <pad> tokens that appear when batched sequences are decoded with
    skip_special_tokens=False (required for post_process_generation).
    """
    inputs = processor(
        text=[TASK] * len(images),
        images=images,
        return_tensors="pt",
        padding=True,
    ).to("cuda", torch.float16)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=256,
        do_sample=False,
        num_beams=1,  # greedy — ~3× faster than beam=3, negligible quality loss
    )

    raw_texts = processor.batch_decode(generated_ids, skip_special_tokens=False)
    return [
        processor.post_process_generation(
            text, task=TASK, image_size=(IMAGE_SIZE, IMAGE_SIZE)
        ).get(TASK, text).replace("<pad>", "").strip()
        for text in raw_texts
    ]


# ── Per-source processing ─────────────────────────────────────────────────────

def process_source(source: str, model: AutoModelForCausalLM, processor: AutoProcessor) -> None:
    """Generate captions for all passed, uncaptioned images in one source.

    Reads filter_results.jsonl, captions entries with passed=True and no
    existing caption field, and writes results back. Flushes every FLUSH_EVERY
    images so progress survives interruption.
    """
    path = RESULTS_PATHS[source]
    if not path.exists():
        print(f"[{source}] filter_results.jsonl not found — run the filter pipeline first, skipping.")
        return

    entries = load_results(path)
    pending = [e for e in entries if e.get("passed") and CAPTION_FIELD not in e]

    if not pending:
        print(f"[{source}] nothing to do ({len(entries):,} entries, all captioned or failed).")
        return

    print(f"[{source}] {len(pending):,} images to caption ({len(entries):,} total entries) …")

    # O(1) mutation via filepath index
    by_filepath = {e["filepath"]: e for e in entries}

    batches = [pending[i : i + BATCH_SIZE] for i in range(0, len(pending), BATCH_SIZE)]
    flush_interval = max(1, FLUSH_EVERY // BATCH_SIZE)

    captioned = 0
    errors = 0

    if not batches:
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as prefetch_executor:
        # Submit batch 0 before the loop — gives CPU a head-start
        prefetch_future = prefetch_executor.submit(_load_batch_images, batches[0])

        for batch_idx, batch_entries in enumerate(tqdm(batches, desc=f"captioning {source}", unit="batch")):

            # Collect this batch's images (may block briefly if load not done yet)
            valid_entries, valid_images, failed_pairs = prefetch_future.result()

            # Submit NEXT batch loading NOW — runs while model.generate() occupies GPU
            if batch_idx + 1 < len(batches):
                prefetch_future = prefetch_executor.submit(
                    _load_batch_images, batches[batch_idx + 1]
                )

            # Write load errors in main thread (by_filepath is not thread-safe)
            for entry, err_str in failed_pairs:
                by_filepath[entry["filepath"]]["caption_error"] = err_str
                errors += 1

            # GPU inference — background thread loads next batch in parallel
            if valid_images:
                captions = caption_batch(model, processor, valid_images)
                for entry, caption in zip(valid_entries, captions):
                    by_filepath[entry["filepath"]][CAPTION_FIELD] = caption
                    captioned += 1

            if (batch_idx + 1) % flush_interval == 0:
                save_results(path, entries)

    save_results(path, entries)
    print(f"[{source}] done — {captioned:,} captioned, {errors:,} load errors.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        required=True,
        choices=list(RESULTS_PATHS.keys()) + ["all"],
        help="Dataset source to caption, or 'all' for every source in sequence.",
    )
    args = parser.parse_args()

    sources = list(RESULTS_PATHS.keys()) if args.source == "all" else [args.source]

    print(f"Loading {MODEL_ID} …")
    model, processor = load_model_and_processor(MODEL_ID)
    print("Model ready.\n")

    try:
        for source in sources:
            process_source(source, model, processor)
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()

    print("All done.")


if __name__ == "__main__":
    main()
