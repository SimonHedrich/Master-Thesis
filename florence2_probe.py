"""Probe Florence-2-large and Florence-2-base on a directory of images.

Runs <DETAILED_CAPTION> inference on up to N_IMAGES images per model using
batched greedy decoding. Both models see the same images; results are saved
side-by-side in the output JSON for direct comparison.

Usage:
    python florence2_probe.py

Configuration (top-level constants):
    IMAGE_DIR    — directory of .jpg/.jpeg/.png images to caption
    N_IMAGES     — number of images to sample (default 200 for stable timing)
    BATCH_SIZE   — images per forward pass (2 is safe for 12 GB; raise to 4–6 on 16+ GB)
    OUTPUT       — path for the combined JSON result file

Output JSON schema:
    {
      "task": "<DETAILED_CAPTION>",
      "n_images": int,
      "models": {
        "<model_id>": {
          "timing": { "min_s", "mean_s", "median_s", "max_s", "imgs_per_sec" },
          "results": [ { "filename", "elapsed_s", "caption" }, ... ]
        }
      }
    }

Requirements:
    transformers==4.48.3   (5.x breaks Florence-2 remote code)
    torch, Pillow
"""

import gc
import json
import statistics
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT  = Path(__file__).resolve().parent
IMAGE_DIR  = REPO_ROOT / "data" / "openimages" / "images" / "african_elephant"
OUTPUT     = REPO_ROOT / "florence2_probe_output.json"
N_IMAGES   = 200
BATCH_SIZE = 6 # for Florence-2-base
TASK       = "<DETAILED_CAPTION>"
MODEL_IDS  = [
    "microsoft/Florence-2-base",
]

# Florence-2's DaViT encoder requires a square feature map.
IMAGE_SIZE = 768


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def caption_batch(
    model: AutoModelForCausalLM,
    processor: AutoProcessor,
    images: list[Image.Image],
) -> list[str]:
    """Caption a batch of images and return parsed caption strings."""
    square = [img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS) for img in images]
    inputs = processor(
        text=[TASK] * len(square),
        images=square,
        return_tensors="pt",
        padding=True,
    ).to("cuda", torch.float16)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=256,
        do_sample=False,
        num_beams=1,  # greedy — ~3× faster than beam=3, negligible quality loss for captions
    )

    raw_texts = processor.batch_decode(generated_ids, skip_special_tokens=False)
    return [
        processor.post_process_generation(
            text, task=TASK, image_size=(IMAGE_SIZE, IMAGE_SIZE)
        ).get(TASK, text)
        for text in raw_texts
    ]


def run_model(model_id: str, image_paths: list[Path]) -> dict:
    """Load model, caption all images in batches, unload, return result dict."""
    print(f"\n{'='*60}")
    print(f"Model: {model_id}")
    print(f"{'='*60}")

    model = (
        AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype="auto")
        .eval()
        .cuda()
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    print("Model ready\n")

    results: list[dict] = []
    for batch_start in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[batch_start : batch_start + BATCH_SIZE]
        batch_images = [Image.open(p).convert("RGB") for p in batch_paths]

        t0 = time.perf_counter()
        captions = caption_batch(model, processor, batch_images)
        batch_elapsed = time.perf_counter() - t0
        per_image = batch_elapsed / len(batch_paths)

        for path, caption in zip(batch_paths, captions):
            results.append({
                "filename": path.name,
                "elapsed_s": round(per_image, 3),
                "caption": caption,
            })
            print(f"  [{per_image:5.2f}s/img]  {path.name}")
            print(f"           {caption[:120]}\n")

    # Release GPU memory before loading the next model.
    del model
    gc.collect()
    torch.cuda.empty_cache()

    times = [r["elapsed_s"] for r in results]
    timing = {
        "min_s":       round(min(times), 3),
        "mean_s":      round(statistics.mean(times), 3),
        "median_s":    round(statistics.median(times), 3),
        "max_s":       round(max(times), 3),
        "imgs_per_sec": round(1.0 / statistics.mean(times), 2),
    }
    print(
        f"Timing ({len(results)} images, batch={BATCH_SIZE}):  "
        f"min={timing['min_s']}s  mean={timing['mean_s']}s  "
        f"median={timing['median_s']}s  max={timing['max_s']}s  "
        f"({timing['imgs_per_sec']} img/s)"
    )
    return {"timing": timing, "results": results}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    image_paths = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )[:N_IMAGES]

    if not image_paths:
        print(f"No images found in {IMAGE_DIR}")
        return

    print(f"Found {len(image_paths)} images — running {len(MODEL_IDS)} models\n")

    payload: dict = {
        "task": TASK,
        "n_images": len(image_paths),
        "models": {},
    }

    for model_id in MODEL_IDS:
        payload["models"][model_id] = run_model(model_id, image_paths)

    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults written to {OUTPUT}")


if __name__ == "__main__":
    main()
