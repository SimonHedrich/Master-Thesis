#!/usr/bin/env python3
"""
Stage 1i — Generate a local-model generator cell at the `maxlen` prompt regime

Duplicate of 1g-generate_images_local.py (same six model loaders/generators,
copied per this directory's convention rather than imported) adapted to
consume the `maxlen` prompt regime built by 1h-generate_prompts_maxlen.py
instead of the `compressed` regime. Where 1g gives every model the same
≤75-token prompt for a fairness ablation, this script gives each model the
richest prompt its own text encoder can actually hold — the goal here is
the best real production dataset per model, not a controlled comparison.
The `compressed`-regime cells 1g already built are untouched by this script
(different output folder — see below), so they stay available for a later
ablation against the best proprietary API model if needed.

Per-generator prompt tier (`GENERATOR_TIER` below):

    realvisxl-lightning   77 CLIP tokens — reuses 1g/1f's EXISTING
                          `compressed` prompt unchanged. It's already
                          engineered to fill this tier's tiny budget
                          (doc 05); there is no separate maxlen variant
                          for it, and none is needed.
    sd35m                 256 T5 tokens — reads 1h's `prompts_maxlen_256`
    sd35-large            metadata (reports/model_comparison_maxlen_prompt_metadata.jsonl,
    sd35-large-turbo      filtered to tier == 256).
    flux2-klein-9b        512 Qwen-family tokens — reads 1h's
    qwen-image            `prompts_maxlen_512` metadata (tier == 512).

Output goes to data/synthetic_model_comparison/train/<generator>/maxlen/
(a new prompt-regime folder, sibling to <generator>/compressed/) with its
own index.jsonl and its own timing/benchmark CSVs
(reports/model_comparison_local_generation_timing_maxlen.csv,
reports/model_comparison_local_generation_benchmark_maxlen.csv) so maxlen
and compressed regime data never mix.

Usage:
    # Cheap smoke test first:
    uv run python scripts/synthetic_model_comparison/1i-generate_images_local_maxlen.py \\
        --generator sd35-large-turbo --classes lion --limit 2

    # Full cell (one model, 1,200 images):
    uv run python scripts/synthetic_model_comparison/1i-generate_images_local_maxlen.py \\
        --generator sd35-large-turbo

Outputs:
    data/synthetic_model_comparison/train/<generator>/maxlen/images/<class_slug>/*.png
    data/synthetic_model_comparison/train/<generator>/maxlen/index.jsonl
    reports/model_comparison_local_generation_timing_maxlen.csv
    reports/model_comparison_local_generation_benchmark_maxlen.csv (only with --benchmark)

Requirements: same as 1g-generate_images_local.py.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path

# Note: unlike scripts/synthetic/2-generate_synthetic_images_local.py, this
# script does NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True — on
# this machine's GPU (an NVIDIA vGPU profile, "A40-24C", not a bare-metal
# card) that allocator mode raises "CUDA driver error: operation not
# supported" the first time a pipeline is moved to the device.
import torch
from dotenv import load_dotenv
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
# Shared credentials with the production pipeline — several of these repos
# are gated; loading HF_TOKEN here is required to download them.
load_dotenv(REPO_ROOT / "scripts" / "synthetic" / ".env")
CLASSES_CSV = REPO_ROOT / "reports" / "model_comparison_classes.csv"
COMPRESSED_METADATA_PATH = REPO_ROOT / "reports" / "model_comparison_compressed_prompt_metadata.jsonl"
MAXLEN_METADATA_PATH = REPO_ROOT / "reports" / "model_comparison_maxlen_prompt_metadata.jsonl"
TRAIN_ROOT = REPO_ROOT / "data" / "synthetic_model_comparison" / "train"
TIMING_CSV_PATH = REPO_ROOT / "reports" / "model_comparison_local_generation_timing_maxlen.csv"
BENCHMARK_CSV_PATH = REPO_ROOT / "reports" / "model_comparison_local_generation_benchmark_maxlen.csv"
FULL_CELL_SIZE = 1200  # 12 classes x 100 images/class
PROMPT_REGIME = "maxlen"

AVAILABLE_GENERATORS = (
    "flux2-klein-9b",
    "realvisxl-lightning",
    "sd35m",
    "sd35-large",
    "sd35-large-turbo",
    "qwen-image",
)
HEADLINE_GENERATORS = ("flux2-klein-9b", "realvisxl-lightning", "sd35m")

# Which prompt tier each generator's text encoder actually supports.
GENERATOR_TIER: dict[str, int] = {
    "realvisxl-lightning": 75,
    "sd35m": 256,
    "sd35-large": 256,
    "sd35-large-turbo": 256,
    "flux2-klein-9b": 512,
    "qwen-image": 512,
}

# Nearest 4:3 ~1MP native bucket per model family (doc 04 §3):
# FLUX/SD3/Qwen need only /16 divisibility; SDXL needs a /64 multi-aspect bucket.
RESOLUTIONS: dict[str, tuple[int, int]] = {
    "flux2-klein-9b": (1152, 864),
    "realvisxl-lightning": (1152, 896),
    "sd35m": (1152, 864),
    "sd35-large": (1152, 864),
    "sd35-large-turbo": (1152, 864),
    "qwen-image": (1152, 864),
}

# Negative prompt for the SDXL/SD3/Qwen families (FLUX ignores CFG-based
# negatives) — copied verbatim from the production local-generation script.
NEGATIVE_PROMPT = (
    "text, watermark, cartoon, illustration, painting, drawing, art, sketch, animated, CGI, render, "
    "3D, unrealistic, low quality, blurry, watermark, text, logo, multiple animals, "
    "duplicate, deformed, ugly, bad anatomy, unnatural pose, "
    "close-up, closeup, portrait, head shot, face only, cropped body, partial animal, cut off limbs"
)

# ---------------------------------------------------------------------------
# TF32 — free speed-up on Ampere+ with no quality loss.
# ---------------------------------------------------------------------------


def _enable_tf32() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ---------------------------------------------------------------------------
# Model 1 — FLUX.2-klein-9B (step-distilled, bf16 + mandatory cpu_offload)
# ---------------------------------------------------------------------------


def _load_flux2_klein():
    from diffusers import Flux2KleinPipeline

    _enable_tf32()

    print("  Loading FLUX.2-klein-9B (bf16) ...")
    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-9b",
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()
    return pipe


def _generate_flux2_klein(pipe, prompt: str, seed: int, width: int, height: int) -> Image.Image:
    generator = torch.Generator("cpu").manual_seed(seed)
    return pipe(
        prompt=prompt,
        num_inference_steps=4,
        guidance_scale=1.0,
        max_sequence_length=512,
        height=height,
        width=width,
        generator=generator,
    ).images[0]


# ---------------------------------------------------------------------------
# Model 2 — RealVisXL V5.0 + SDXL-Lightning 4-step LoRA
# ---------------------------------------------------------------------------


def _load_realvisxl_lightning():
    from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline
    from huggingface_hub import hf_hub_download

    _enable_tf32()

    print("  Loading RealVisXL V5.0 (fp16) ...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "SG161222/RealVisXL_V5.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to("cuda")

    print("  Fusing SDXL-Lightning 4-step LoRA ...")
    lora_path = hf_hub_download(
        repo_id="ByteDance/SDXL-Lightning",
        filename="sdxl_lightning_4step_lora.safetensors",
    )
    pipe.load_lora_weights(lora_path)
    pipe.fuse_lora()

    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="trailing",
    )
    return pipe


def _generate_sdxl(pipe, prompt: str, seed: int, width: int, height: int) -> Image.Image:
    generator = torch.Generator("cuda").manual_seed(seed)
    return pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=4,
        guidance_scale=0.0,
        height=height,
        width=width,
        generator=generator,
    ).images[0]


# ---------------------------------------------------------------------------
# Model 3 — Stable Diffusion 3.5 Medium
# ---------------------------------------------------------------------------


def _load_sd35m():
    from diffusers import StableDiffusion3Pipeline

    _enable_tf32()

    print("  Loading SD 3.5 Medium (bf16) ...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium",
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")
    return pipe


def _generate_sd3(pipe, prompt: str, seed: int, width: int, height: int) -> Image.Image:
    generator = torch.Generator("cpu").manual_seed(seed)
    return pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=40,
        guidance_scale=4.5,
        height=height,
        width=width,
        generator=generator,
        max_sequence_length=256,
    ).images[0]


# ---------------------------------------------------------------------------
# Model 4 — Stable Diffusion 3.5 Large / Large-Turbo
# ---------------------------------------------------------------------------


def _load_sd35_large():
    from diffusers import StableDiffusion3Pipeline

    _enable_tf32()

    print("  Loading SD 3.5 Large (bf16) ...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large",
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()
    return pipe


def _generate_sd35_large(pipe, prompt: str, seed: int, width: int, height: int) -> Image.Image:
    generator = torch.Generator("cpu").manual_seed(seed)
    return pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=40,
        guidance_scale=4.5,
        height=height,
        width=width,
        generator=generator,
        max_sequence_length=256,
    ).images[0]


def _load_sd35_large_turbo():
    from diffusers import StableDiffusion3Pipeline

    _enable_tf32()

    print("  Loading SD 3.5 Large Turbo (bf16) ...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large-turbo",
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()
    return pipe


def _generate_sd35_large_turbo(pipe, prompt: str, seed: int, width: int, height: int) -> Image.Image:
    generator = torch.Generator("cpu").manual_seed(seed)
    return pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=4,
        guidance_scale=1.0,
        height=height,
        width=width,
        generator=generator,
        max_sequence_length=256,
    ).images[0]


# ---------------------------------------------------------------------------
# Model 5 — Qwen-Image (own NF4 quantization from the official repo)
# ---------------------------------------------------------------------------


def _load_qwen_image():
    from diffusers import BitsAndBytesConfig as DiffusersBnbConfig
    from diffusers import QwenImagePipeline, QwenImageTransformer2DModel
    from transformers import BitsAndBytesConfig as TransformersBnbConfig
    from transformers import Qwen2_5_VLForConditionalGeneration

    _enable_tf32()

    nf4_diffusers = DiffusersBnbConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    nf4_transformers = TransformersBnbConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model_id = "Qwen/Qwen-Image"

    print("  Loading Qwen2.5-VL text encoder (NF4) ...")
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        subfolder="text_encoder",
        quantization_config=nf4_transformers,
        torch_dtype=torch.bfloat16,
    )

    print("  Loading Qwen-Image transformer (NF4) ...")
    transformer = QwenImageTransformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        quantization_config=nf4_diffusers,
        torch_dtype=torch.bfloat16,
    )

    print("  Assembling QwenImagePipeline ...")
    pipe = QwenImagePipeline.from_pretrained(
        model_id,
        transformer=transformer,
        text_encoder=text_encoder,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()
    return pipe


def _generate_qwen_image(pipe, prompt: str, seed: int, width: int, height: int) -> Image.Image:
    generator = torch.Generator("cpu").manual_seed(seed)
    return pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=50,
        true_cfg_scale=4.0,
        height=height,
        width=width,
        generator=generator,
        max_sequence_length=512,
    ).images[0]


_LOADERS = {
    "flux2-klein-9b": _load_flux2_klein,
    "realvisxl-lightning": _load_realvisxl_lightning,
    "sd35m": _load_sd35m,
    "sd35-large": _load_sd35_large,
    "sd35-large-turbo": _load_sd35_large_turbo,
    "qwen-image": _load_qwen_image,
}
_GENERATORS = {
    "flux2-klein-9b": _generate_flux2_klein,
    "realvisxl-lightning": _generate_sdxl,
    "sd35m": _generate_sd3,
    "sd35-large": _generate_sd35_large,
    "sd35-large-turbo": _generate_sd35_large_turbo,
    "qwen-image": _generate_qwen_image,
}

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_class_order() -> dict[str, int]:
    """class common name -> row position, for stable per-image seeds."""
    with open(CLASSES_CSV, encoding="utf-8", newline="") as f:
        return {row["class"]: i for i, row in enumerate(csv.DictReader(f))}


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_records_for_generator(generator: str) -> list[dict]:
    """Reads the tier-appropriate prompt metadata for this generator:
    RealVisXL reuses the existing `compressed` regime (already fills its
    77-token budget); everything else reads 1h's `maxlen` metadata,
    filtered to the generator's own tier (256 or 512)."""
    tier = GENERATOR_TIER[generator]
    if tier == 75:
        if not COMPRESSED_METADATA_PATH.exists():
            sys.exit(f"Error: {COMPRESSED_METADATA_PATH} not found. Run 1f-generate_prompts_compressed.py first.")
        return _load_jsonl(COMPRESSED_METADATA_PATH)

    if not MAXLEN_METADATA_PATH.exists():
        sys.exit(f"Error: {MAXLEN_METADATA_PATH} not found. Run 1h-generate_prompts_maxlen.py first.")
    all_records = _load_jsonl(MAXLEN_METADATA_PATH)
    records = [r for r in all_records if r["tier"] == tier]
    if not records:
        sys.exit(f"Error: no tier={tier} records found in {MAXLEN_METADATA_PATH}.")
    return records


def load_index(index_path: Path) -> list[dict]:
    if not index_path.exists():
        return []
    return _load_jsonl(index_path)


def save_index(index_path: Path, records: list[dict]) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def append_timing_rows(rows: list[dict]) -> None:
    TIMING_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not TIMING_CSV_PATH.exists()
    with open(TIMING_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["generator", "class", "index", "filename", "width", "height", "seconds"])
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def append_benchmark_rows(rows: list[dict]) -> None:
    BENCHMARK_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not BENCHMARK_CSV_PATH.exists()
    fieldnames = ["generator", "n_images", "total_seconds", "avg_seconds_per_image", "estimated_hours_for_1200"]
    with open(BENCHMARK_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def slug_from_filename(filename: str) -> str:
    """'d_plains_zebra_001.png' -> 'plains_zebra' (band prefix and NNN suffix stripped)."""
    parts = filename.rsplit(".", 1)[0].split("_")
    return "_".join(parts[1:-1])


def image_output_path(images_dir: Path, rec: dict) -> Path:
    return images_dir / slug_from_filename(rec["filename"]) / rec["filename"]


def index_entry(rec: dict, images_dir: Path) -> dict:
    return {
        "filename": rec["filename"],
        "class": rec["class"],
        "band": rec["band"],
        "source_split": rec["source_split"],
        "shot_type": rec["shot_type"],
        "distance": rec["distance"],
        "lighting": rec["lighting"],
        "occlusion": rec["occlusion"],
        "pose": rec["pose"],
        "environment": rec["environment"],
        "prompt_file": rec["prompt_file"],
        "file_name": str(image_output_path(images_dir, rec).relative_to(REPO_ROOT)),
        "dest_prompt_file": rec["prompt_file"],
    }


# ---------------------------------------------------------------------------
# Per-generator run
# ---------------------------------------------------------------------------


def run_generator(
    generator: str,
    records: list[dict],
    class_order: dict[str, int],
    seed_base: int,
    force: bool,
) -> list[dict]:
    cell_dir = TRAIN_ROOT / generator / PROMPT_REGIME
    images_dir = cell_dir / "images"
    index_path = cell_dir / "index.jsonl"
    width, height = RESOLUTIONS[generator]

    index_records = load_index(index_path)
    indexed_filenames = {rec["filename"] for rec in index_records}

    pending = []
    for rec in records:
        out_path = image_output_path(images_dir, rec)
        if not force and out_path.exists() and rec["filename"] in indexed_filenames:
            continue
        pending.append(rec)

    print(f"\n{'=' * 60}")
    print(f"Generator : {generator}  (tier={GENERATOR_TIER[generator]})")
    print(f"Resolution: {width}x{height}")
    print(f"Pending   : {len(pending)}/{len(records)} images")
    print(f"{'=' * 60}")

    if not pending:
        print("Nothing to do — all requested images already generated.")
        return []

    print("Loading pipeline ...")
    t_load = time.perf_counter()
    pipe = _LOADERS[generator]()
    print(f"Pipeline ready in {time.perf_counter() - t_load:.1f}s\n")

    generate_fn = _GENERATORS[generator]
    timing_rows = []

    for n, rec in enumerate(pending, start=1):
        prompt_text = (REPO_ROOT / rec["prompt_file"]).read_text(encoding="utf-8")
        global_index = class_order[rec["class"]] * 200 + (rec["index"] - 1)
        seed = seed_base + global_index

        print(f"  [{n}/{len(pending)}] {rec['class']} #{rec['index']:03d} seed={seed} ...", end=" ", flush=True)

        torch.cuda.empty_cache()
        t0 = time.perf_counter()
        image = generate_fn(pipe, prompt_text, seed, width, height)
        elapsed = time.perf_counter() - t0

        out_path = image_output_path(images_dir, rec)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path, "PNG")
        print(f"saved -> {out_path.relative_to(REPO_ROOT)} ({elapsed:.1f}s)")

        if rec["filename"] in indexed_filenames:
            index_records[:] = [e for e in index_records if e["filename"] != rec["filename"]]
        index_records.append(index_entry(rec, images_dir))
        indexed_filenames.add(rec["filename"])

        timing_rows.append({
            "generator": generator,
            "class": rec["class"],
            "index": rec["index"],
            "filename": rec["filename"],
            "width": width,
            "height": height,
            "seconds": round(elapsed, 2),
        })

    save_index(index_path, index_records)
    append_timing_rows(timing_rows)

    print(f"\nDone: {len(pending)} generated.")
    print(f"Index updated: {index_path.relative_to(REPO_ROOT)} ({len(index_records)} total records)")
    print(f"Timing appended: {TIMING_CSV_PATH.relative_to(REPO_ROOT)}")

    print("Unloading pipeline ...")
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    return timing_rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--generator",
        required=True,
        choices=[*AVAILABLE_GENERATORS, "all"],
        help=(
            "Local model to generate with, or 'all' to run the three headline "
            f"models sequentially ({', '.join(HEADLINE_GENERATORS)})."
        ),
    )
    parser.add_argument(
        "--classes",
        default=None,
        help="Comma-separated class common names to include (default: all 12).",
    )
    limit_group = parser.add_mutually_exclusive_group()
    limit_group.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Cap generation to the first N pending images (for smoke tests).",
    )
    limit_group.add_argument(
        "--benchmark",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Benchmark mode: force-generate the first N images per generator "
            "(always fresh, ignoring already-generated files), then print/save "
            "inference-only timing (measured after pipeline load) and an "
            f"extrapolated estimate for the full {FULL_CELL_SIZE}-image cell."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate images that already exist.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed; each image uses seed + global_index (default: 42).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("Error: no CUDA GPU detected. This script requires a CUDA-capable GPU.")

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")

    class_order = load_class_order()

    generators = HEADLINE_GENERATORS if args.generator == "all" else (args.generator,)
    force = args.force or args.benchmark is not None

    benchmark_rows = []
    for generator in generators:
        all_records = load_records_for_generator(generator)
        print(f"Loaded {len(all_records)} tier={GENERATOR_TIER[generator]} records for {generator}")

        records = all_records
        if args.classes:
            requested = {c.strip().lower() for c in args.classes.split(",")}
            records = [r for r in all_records if r["class"].lower() in requested]
            not_found = requested - {r["class"].lower() for r in records}
            if not_found:
                print(f"Warning: class(es) not found: {', '.join(sorted(not_found))}")
            print(f"Filtered to {len(records)} records for {len(requested)} class(es)")

        records = sorted(records, key=lambda r: (class_order[r["class"]], r["index"]))

        if args.limit is not None:
            records = records[: args.limit]
            print(f"--limit {args.limit}: capped to {len(records)} records")
        elif args.benchmark is not None:
            records = records[: args.benchmark]
            print(f"--benchmark {args.benchmark}: force-generating {len(records)} records")

        timing_rows = run_generator(generator, records, class_order, args.seed, force)
        if args.benchmark is not None and timing_rows:
            total_seconds = sum(row["seconds"] for row in timing_rows)
            avg_seconds = total_seconds / len(timing_rows)
            estimated_hours = avg_seconds * FULL_CELL_SIZE / 3600
            benchmark_rows.append({
                "generator": generator,
                "n_images": len(timing_rows),
                "total_seconds": round(total_seconds, 2),
                "avg_seconds_per_image": round(avg_seconds, 2),
                "estimated_hours_for_1200": round(estimated_hours, 2),
            })

    if args.benchmark is not None and benchmark_rows:
        header = f"{'generator':22s} {'n':>4s} {'total_s':>10s} {'avg_s/img':>10s} {'est. hrs/1200':>14s}"
        print(f"\n{'=' * len(header)}")
        print("BENCHMARK SUMMARY (inference-only, measured after pipeline load)")
        print(f"{'=' * len(header)}")
        print(header)
        for row in benchmark_rows:
            print(
                f"{row['generator']:22s} {row['n_images']:4d} {row['total_seconds']:10.2f} "
                f"{row['avg_seconds_per_image']:10.2f} {row['estimated_hours_for_1200']:14.2f}"
            )
        append_benchmark_rows(benchmark_rows)
        print(f"\nWrote {BENCHMARK_CSV_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
