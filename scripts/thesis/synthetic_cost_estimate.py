"""List-price cost of the synthetic image corpus and of the API generator cells.

No invoice or billing export exists for the Gemini runs, so every figure here is a
list-price calculation: measured token usage per request times the vendors' published
per-token rates. Two things make that more than "images times price":

* Both vendors bill the prompt text as input tokens on top of the image, and the
  prompts here run to several thousand tokens each.
* ``gemini-3.1-flash-image-preview`` was asked for image and text output, and in
  the measured cell it reviews its own image and returns a second (sometimes a
  third) one in the same response, so a request bills 1.4 images on average.

Token usage per request is read from the batch-output files of the generator
comparison cells where they exist (they carry ``usageMetadata`` / ``usage``), and
falls back to the values measured on 2026-09-28 otherwise. The production runs
kept no usage records, so their prompt tokens are estimated from prompt length in
characters at the tokens-per-character ratio measured on the comparison cell, and
their image and text-output tokens are taken from that cell, which used the same
model, response modalities, size and aspect ratio.

Run from the repository root:

    uv run python -m scripts.thesis.synthetic_cost_estimate

Prints the production-corpus table and the comparison-cell table that back the
cost figures in 33-Synthetic_Data_Supplementation.tex and
35-Synthetic_Generator_Comparison.tex.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path

# USD per 1M tokens, standard tier. Batch tier is half of every rate.
#   https://ai.google.dev/gemini-api/docs/pricing        (checked 2026-09-28)
#   https://developers.openai.com/api/docs/pricing       (checked 2026-09-28)
PRICES_PER_M = {
    "gemini-3.1-flash-image-preview": {"text_in": 0.50, "image_out": 60.0, "text_out": 3.0},
    "gemini-3.1-flash-lite-image": {"text_in": 0.25, "image_out": 30.0, "text_out": 1.50},
    "gpt-image-2": {"text_in": 5.0, "image_out": 30.0, "text_out": 0.0},
}
BATCH_FACTOR = 0.5
EUR_PER_USD = 0.88  # late September 2026

# Production corpus (data/synthetic/index.jsonl, test_index.jsonl).
PRODUCTION = {"train/val (Bands A+B)": 12_600, "test (225 x 50)": 11_250}
PROMPT_DIRS = {
    "train/val (Bands A+B)": Path("data/synthetic/prompts"),
    "test (225 x 50)": Path("data/synthetic/test_prompts"),
}
PROMPT_CHARS_FALLBACK = {"train/val (Bands A+B)": 9_421.1, "test (225 x 50)": 10_923.5}

COMPARISON_ROOT = Path("data/synthetic_model_comparison/train")
COMPARISON_PROMPT_CHARS_FALLBACK = 11_884.4  # prompts_full/, mean characters
CELL_IMAGES = 1_200


@dataclass
class Usage:
    """Mean tokens per request."""

    prompt: float
    image_out: float
    text_out: float
    n: int
    source: str


# Measured on 2026-09-28 from the batch-output files; used when those are absent.
USAGE_FALLBACK = {
    "gemini-3.1-flash-image-preview": Usage(3939.7, 1038.8, 590.2, 494, "fallback"),
    "gemini-3.1-flash-lite-image": Usage(2609.2, 1120.0, 417.5, 1197, "fallback"),
    "gpt-image-2-low": Usage(5931.2, 134.0, 0.0, 47, "fallback"),
    "gpt-image-2-medium": Usage(5931.6, 1204.0, 0.0, 46, "fallback"),
}

# Amounts actually charged, where recorded.
#   docs/synthetic-model-comparison/12_additional-generator-cells-build-log.md
BILLED_USD = {"gpt-image-2-low": 10.11, "gpt-image-2-medium": 29.36}

CELLS = [
    ("gpt-image-2-low", "gpt-image-2", "1024x768"),
    ("gpt-image-2-medium", "gpt-image-2", "1024x768"),
    ("gemini-3.1-flash-image-preview", "gemini-3.1-flash-image-preview", "592x448"),
    ("gemini-3.1-flash-lite-image", "gemini-3.1-flash-lite-image", "1200x896"),
]


def _mean_prompt_chars(directory: Path, fallback: float) -> float:
    files = list(directory.rglob("*.txt")) if directory.is_dir() else []
    if not files:
        return fallback
    return statistics.mean(len(f.read_text(encoding="utf-8")) for f in files)


def _gemini_usage(cell: str) -> Usage:
    path = COMPARISON_ROOT / cell / "full" / "fresh_batch_output.jsonl"
    if not path.is_file():
        return USAGE_FALLBACK[cell]
    prompt, image, text = [], [], []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            meta = json.loads(line)["response"]["usageMetadata"]
            img = sum(
                d["tokenCount"]
                for d in meta.get("candidatesTokensDetails", [])
                if d["modality"] == "IMAGE"
            )
            prompt.append(meta["promptTokenCount"])
            image.append(img)
            text.append(meta["candidatesTokenCount"] - img)
    return Usage(
        statistics.mean(prompt), statistics.mean(image), statistics.mean(text), len(prompt), str(path)
    )


def _openai_usage(cell: str) -> Usage:
    path = COMPARISON_ROOT / cell / "full" / "openai_batch_output.jsonl"
    if not path.is_file():
        return USAGE_FALLBACK[cell]
    prompt, image = [], []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            usage = json.loads(line)["response"]["body"]["usage"]
            prompt.append(usage["input_tokens"])
            image.append(usage["output_tokens"])
    return Usage(statistics.mean(prompt), statistics.mean(image), 0.0, len(prompt), str(path))


def request_usd(model: str, usage: Usage, prompt_tokens: float | None = None) -> float:
    """Standard-tier list price of one request in USD."""
    p = PRICES_PER_M[model]
    prompt = usage.prompt if prompt_tokens is None else prompt_tokens
    return (
        prompt * p["text_in"] + usage.image_out * p["image_out"] + usage.text_out * p["text_out"]
    ) / 1e6


def main() -> None:
    flash = _gemini_usage("gemini-3.1-flash-image-preview")
    comparison_chars = _mean_prompt_chars(
        COMPARISON_ROOT / "prompts_full", COMPARISON_PROMPT_CHARS_FALLBACK
    )
    tokens_per_char = flash.prompt / comparison_chars
    flat_image_usd = 747 * PRICES_PER_M["gemini-3.1-flash-image-preview"]["image_out"] / 1e6

    print("Production corpus, gemini-3.1-flash-image-preview at 0.5K / 4:3")
    print(f"  usage source: {flash.source} (n={flash.n})")
    print(
        f"  per request: {flash.image_out:.0f} image tokens "
        f"({flash.image_out / 747:.2f} images), {flash.text_out:.0f} text-output tokens; "
        f"{tokens_per_char:.3f} prompt tokens per character"
    )
    print(f"  flat per-image list price: ${flat_image_usd:.4f} direct, "
          f"${flat_image_usd * BATCH_FACTOR:.4f} batch")
    print(f"  {'split':<24}{'images':>8}{'prompt tok':>12}{'$/req batch':>13}"
          f"{'batch $':>10}{'direct $':>10}{'flat batch $':>14}")
    tot_batch = tot_direct = tot_flat = 0.0
    for split, n_images in PRODUCTION.items():
        chars = _mean_prompt_chars(PROMPT_DIRS[split], PROMPT_CHARS_FALLBACK[split])
        prompt_tokens = chars * tokens_per_char
        direct = request_usd("gemini-3.1-flash-image-preview", flash, prompt_tokens) * n_images
        batch = direct * BATCH_FACTOR
        flat = flat_image_usd * BATCH_FACTOR * n_images
        tot_batch += batch
        tot_direct += direct
        tot_flat += flat
        print(f"  {split:<24}{n_images:>8}{prompt_tokens:>12.0f}{batch / n_images:>13.4f}"
              f"{batch:>10.2f}{direct:>10.2f}{flat:>14.2f}")
    n_total = sum(PRODUCTION.values())
    print(f"  {'total':<24}{n_total:>8}{'':>12}{tot_batch / n_total:>13.4f}"
          f"{tot_batch:>10.2f}{tot_direct:>10.2f}{tot_flat:>14.2f}")
    print(f"  total batch: ${tot_batch:.2f} = EUR {tot_batch * EUR_PER_USD:.0f}; "
          f"direct: ${tot_direct:.2f} = EUR {tot_direct * EUR_PER_USD:.0f}; "
          f"flat per-image batch: ${tot_flat:.2f} = EUR {tot_flat * EUR_PER_USD:.0f}")
    print()

    print(f"Generator comparison cells, {CELL_IMAGES} images each")
    print(f"  {'cell':<32}{'size':>10}{'prompt tok':>12}{'img tok':>9}{'txt tok':>9}"
          f"{'direct $':>10}{'batch $':>10}{'billed $':>10}  usage source")
    for cell, model, size in CELLS:
        usage = _openai_usage(cell) if model == "gpt-image-2" else _gemini_usage(cell)
        direct = request_usd(model, usage) * CELL_IMAGES
        billed = BILLED_USD.get(cell)
        billed_s = f"{billed:>10.2f}" if billed is not None else f"{'---':>10}"
        print(f"  {cell:<32}{size:>10}{usage.prompt:>12.0f}{usage.image_out:>9.0f}"
              f"{usage.text_out:>9.0f}{direct:>10.2f}{direct * BATCH_FACTOR:>10.2f}{billed_s}"
              f"  {usage.source} (n={usage.n})")
    print()
    print(f"EUR at {EUR_PER_USD} per USD. gpt-image-2 cached-input discount not modeled, "
          "which is why its billed amounts fall below the batch column.")


if __name__ == "__main__":
    main()
