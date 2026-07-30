#!/usr/bin/env python3
"""
Stage 1g — Generate a local-model generator cell via HuggingFace `diffusers`

Materializes a generator cell for the synthetic-model-comparison experiment's
local-model tier (docs/synthetic-model-comparison/01_experiment-design.md §6,
04_local-models-and-output-parameters.md), run at the `compressed` prompt
regime only (77-512 token limits make the `full` regime infeasible for all
of these models). Reads the shared compressed-prompt metadata built by
1f-generate_prompts_compressed.py (analogous to how 1c-generate_images_fresh.py
consumes 1b's fresh-prompt metadata), generates one image per record with a
local `diffusers` pipeline, and writes this generator's own index.jsonl.

Seven models are available (`--generator`):

    flux2-klein-9b        FLUX.2-klein-9B (step-distilled, 4 steps). Current
                          BFL-family pick, replacing FLUX.1-schnell — see
                          "GPU memory notes" below re: why offload is
                          mandatory despite being far smaller than FLUX.2-dev
                          (32B, evaluated and dropped from this roster — too
                          large/slow/OOM-risky on this GPU even 4-bit).
    realvisxl-lightning   RealVisXL V5.0 + SDXL-Lightning 4-step LoRA.
    sd35m                 Stable Diffusion 3.5 Medium.
    sd35-large            Stable Diffusion 3.5 Large (8B, non-distilled).
    sd35-large-turbo      Stable Diffusion 3.5 Large Turbo (8B, 4-step).
    qwen-image            Qwen-Image (20.4B DiT + 8.3B Qwen2.5-VL encoder,
                          Apache-2.0), quantized to NF4 at load time from
                          the official Qwen/Qwen-Image repo (no official
                          pre-quantized checkpoint exists, unlike FLUX.2-dev).
    hidream-i1            HiDream-I1-Full (17B DiT, MIT-licensed). Previously
                          researched and rejected (see below) — now adopted.
                          Its transformer and 4th text encoder (Llama-3.1-8B)
                          are quantized to NF4 at load time, same pattern and
                          rationale as qwen-image: unquantized bf16 would
                          total ~63.5GB, more than this machine's 47GB RAM,
                          and enable_model_cpu_offload() loads the full
                          pipeline onto the CPU before moving submodules to
                          GPU, so a naive load risks heavy swapping/OOM.

`--generator all` only runs the three models this experiment's docs actually
specify as the headline local tier (flux2-klein-9b, realvisxl-lightning,
sd35m) — sd35-large/sd35-large-turbo/qwen-image/hidream-i1 are evaluation
extras, selected explicitly by name, not folded into "all".

HiDream-I1 was originally researched and NOT included: it needs Meta's
gated meta-llama/Llama-3.1-8B-Instruct as a 4th text encoder (not bundled
in HiDream's own repo), and this account's HF token had no download access
to it (confirmed via a live 403 — Llama gating requires manual per-repo
approval, unlike the "auto" gates on every other repo used here). That
access has since been granted — reconfirmed with a live hf_hub_download()
of the repo's config.json, not just a model_info() metadata check (which
can succeed even without real access, see doc 13 §3/§4) — so it's now
adopted as the roster's 7th model.

GPU memory notes (this machine: a single ~23.8GB GPU): `enable_model_cpu_offload()`
was previously used on every model here as a blanket RTX-3060-12GB-era
default, which needlessly hurt throughput on models that actually fit
without it. RealVisXL (never offloaded) and SD3.5 Medium (offload removed)
both fit comfortably in bf16/fp16 and run with no offload. The four larger
models added since (flux2-klein-9b, sd35-large, sd35-large-turbo,
qwen-image) all structurally exceed 23.8GB even before activations, so they
keep `enable_model_cpu_offload()` regardless — there's no headroom left for
them to reclaim. If a run OOMs or is impractically slow anyway, that's
reported as-is (no cloud/remote-encoder fallback — keeps this tier
purely local).

Pipeline loaders/generators are copied (not imported) from
scripts/synthetic/2-generate_synthetic_images_local.py where applicable —
this experiment's code stays independent of the production pipeline per
this directory's existing convention (see 1b/1c's docstrings). Differences
from that script: (1) output goes into this experiment's
data/synthetic_model_comparison/train/<generator>/compressed/ cell layout
with an index.jsonl instead of a flat data/synthetic/<class>/ directory, and
(2) resolution is fixed per generator to each model's nearest 4:3 ~1MP
native bucket (doc 04 §3) and saved as-is — no forced downscale to an
arbitrary size, matching how every other generator cell in this experiment
stores images at whatever resolution the model natively produces.

Per-image seeds are `--seed + global_index`, where global_index is derived
from a fixed class ordering (reports/model_comparison_classes.csv row order)
so seeds stay stable regardless of which --classes subset a given run
covers (docs/synthetic-model-comparison/06_evaluation-methodology.md notes
local models are fully seed-reproducible, unlike the API models).

Because loading a pipeline takes real time (and these are all large
downloads — tens of GB each), --generator all loads/generates/unloads each
model sequentially (one resident in VRAM at a time), mirroring the
production script's run_model().

Usage:
    # Cheap smoke test first:
    uv run python scripts/synthetic_model_comparison/1g-generate_images_local.py \\
        --generator realvisxl-lightning --classes lion --limit 2

    # Full cell (one model):
    uv run python scripts/synthetic_model_comparison/1g-generate_images_local.py \\
        --generator sd35m

    # The three headline local models sequentially:
    uv run python scripts/synthetic_model_comparison/1g-generate_images_local.py --generator all

    # Benchmark: force-generate N images/model, print inference-only timing
    # + a 1,200-image estimate (time measured after pipeline load, i.e. pure
    # per-image inference):
    uv run python scripts/synthetic_model_comparison/1g-generate_images_local.py \\
        --generator qwen-image --benchmark 5

Outputs:
    data/synthetic_model_comparison/train/<generator>/compressed/images/<class_slug>/*.png
    data/synthetic_model_comparison/train/<generator>/compressed/index.jsonl
    reports/model_comparison_local_generation_timing.csv
    reports/model_comparison_local_generation_benchmark.csv (only with --benchmark)

Requirements:
    diffusers, transformers, accelerate, bitsandbytes, sentencepiece, protobuf,
    peft, python-dotenv (all uv-managed, see pyproject.toml; diffusers>=0.39.0
    /transformers>=5.14.0 already cover Flux2Pipeline/Flux2KleinPipeline/
    QwenImagePipeline/Qwen2_5_VLForConditionalGeneration/HiDreamImagePipeline,
    no upgrade needed).
    Requires a CUDA GPU and an HF_TOKEN in scripts/synthetic/.env with access
    to the gated stabilityai/stable-diffusion-3.5-*,
    black-forest-labs/FLUX.2-klein-9b, and meta-llama/Llama-3.1-8B-Instruct
    repos.
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
# Shared credentials with the production pipeline — SD 3.5 Medium is a gated
# HF repo; loading HF_TOKEN here is required to download it.
load_dotenv(REPO_ROOT / "scripts" / "synthetic" / ".env")
CLASSES_CSV = REPO_ROOT / "reports" / "model_comparison_classes.csv"
METADATA_PATH = REPO_ROOT / "reports" / "model_comparison_compressed_prompt_metadata.jsonl"
TRAIN_ROOT = REPO_ROOT / "data" / "synthetic_model_comparison" / "train"
TIMING_CSV_PATH = REPO_ROOT / "reports" / "model_comparison_local_generation_timing.csv"
BENCHMARK_CSV_PATH = REPO_ROOT / "reports" / "model_comparison_local_generation_benchmark.csv"
FULL_CELL_SIZE = 1200  # 12 classes x 100 images/class

AVAILABLE_GENERATORS = (
    "flux2-klein-9b",
    "realvisxl-lightning",
    "sd35m",
    "sd35-large",
    "sd35-large-turbo",
    "qwen-image",
    "hidream-i1",
)
# The three models this experiment's docs specify as the headline local
# tier — "--generator all" is scoped to these; the rest are explicit-only.
HEADLINE_GENERATORS = ("flux2-klein-9b", "realvisxl-lightning", "sd35m")

# Nearest 4:3 ~1MP native bucket per model family (doc 04 §3):
# FLUX/SD3/Qwen need only /16 divisibility; SDXL needs a /64 multi-aspect bucket.
RESOLUTIONS: dict[str, tuple[int, int]] = {
    "flux2-klein-9b": (1152, 864),
    "realvisxl-lightning": (1152, 896),
    "sd35m": (1152, 864),
    "sd35-large": (1152, 864),
    "sd35-large-turbo": (1152, 864),
    "qwen-image": (1152, 864),
    "hidream-i1": (1152, 864),
}

# Negative prompt for the SDXL/SD3 families (FLUX.1-schnell ignores CFG-based
# negatives at guidance_scale=0.0) — copied verbatim from the production
# local-generation script.
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
#
# klein-9B's transformer (~18GB bf16) plus its 8B "Qwen3" text encoder
# (~16GB bf16) total ~34GB — more than this GPU's 23.8GB VRAM, so offload
# is mandatory here regardless of the Part-1 GPU-utilization finding (that
# finding only applies to models that actually fit without it — see
# _load_sd35m). klein-9B's real advantage over the much larger FLUX.2-dev
# (32B, also offload-mandatory, dropped from this roster) is speed: it's
# step-distilled to 4 steps, so far fewer offload swap-cycles per image.
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
    # No negative_prompt: Flux2KleinPipeline only accepts pre-encoded
    # negative_prompt_embeds, not a plain string — same as the rest of the
    # FLUX family in this script.
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
    # 2.5B transformer + T5-XXL/CLIP encoders fit comfortably in this GPU's
    # 23.8GB without offload (~16.6GB total) — cpu_offload here was a
    # leftover from the RTX-3060-12GB-targeting production script this was
    # copied from and only slowed things down via unnecessary PCIe transfers.
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
#
# 8B transformer (~16GB bf16) + shared T5-XXL/CLIP encoders (~11.5GB) total
# ~27.6GB — unlike Medium, this exceeds 23.8GB VRAM, so (unlike the Part-1
# fix applied to _load_sd35m) cpu_offload stays mandatory for both variants.
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
    # Same params as Medium — same architecture, just a bigger transformer.
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
    # Step-distilled: 4 steps, low guidance (HF's own stated recommendation).
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
#
# 20.4B transformer + 8.3B Qwen2.5-VL text encoder (28.7B total) — no
# official or diffusers-org pre-quantized checkpoint exists (unlike
# FLUX.2-dev's diffusers/FLUX.2-dev-bnb-4bit), so this quantizes both
# components at load time from the official Qwen/Qwen-Image repo, mirroring
# the same BitsAndBytesConfig NF4 pattern already used for FLUX.1-schnell
# above. cpu_offload stays on as a safety margin — even 4-bit, the combined
# footprint is close to this GPU's 23.8GB ceiling.
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


# ---------------------------------------------------------------------------
# Model 6 — HiDream-I1-Full (own NF4 quantization of the transformer + the
# gated Llama-3.1-8B 4th text encoder)
#
# 17B transformer (~34.7GB bf16) + Llama-3.1-8B text encoder (~16GB bf16) +
# bundled CLIP/OpenCLIP/T5-XXL encoders (~12.8GB bf16) total ~63.5GB — more
# than this machine's 47GB RAM, and enable_model_cpu_offload() loads the
# full pipeline onto the CPU before moving submodules to GPU, so a naive
# bf16 load risks heavy swapping or an OOM before generation even starts.
# Quantizing the transformer and the Llama encoder to NF4 (same pattern as
# _load_qwen_image) cuts resident RAM to ~25.5GB, safely inside both RAM and
# close to (still over) this GPU's 23.8GB VRAM ceiling, so offload stays on.
# Originally rejected for lacking download access to the gated Llama repo —
# reconfirmed live via hf_hub_download() (not just model_info()) that access
# now exists, so it's adopted as this roster's 7th model.
# ---------------------------------------------------------------------------


def _load_hidream_i1():
    from diffusers import BitsAndBytesConfig as DiffusersBnbConfig
    from diffusers import HiDreamImagePipeline, HiDreamImageTransformer2DModel
    from transformers import AutoTokenizer
    from transformers import BitsAndBytesConfig as TransformersBnbConfig
    from transformers import LlamaForCausalLM

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

    model_id = "HiDream-ai/HiDream-I1-Full"
    llama_id = "meta-llama/Llama-3.1-8B-Instruct"

    print("  Loading Llama-3.1-8B text encoder (NF4) ...")
    tokenizer_4 = AutoTokenizer.from_pretrained(llama_id)
    text_encoder_4 = LlamaForCausalLM.from_pretrained(
        llama_id,
        output_hidden_states=True,
        quantization_config=nf4_transformers,
        torch_dtype=torch.bfloat16,
    )

    print("  Loading HiDream-I1 transformer (NF4) ...")
    transformer = HiDreamImageTransformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        quantization_config=nf4_diffusers,
        torch_dtype=torch.bfloat16,
    )

    print("  Assembling HiDreamImagePipeline ...")
    pipe = HiDreamImagePipeline.from_pretrained(
        model_id,
        transformer=transformer,
        tokenizer_4=tokenizer_4,
        text_encoder_4=text_encoder_4,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()
    return pipe


def _generate_hidream_i1(pipe, prompt: str, seed: int, width: int, height: int) -> Image.Image:
    # Defaults per HiDreamImagePipeline's own docstring example.
    generator = torch.Generator("cpu").manual_seed(seed)
    return pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=50,
        guidance_scale=5.0,
        height=height,
        width=width,
        generator=generator,
        max_sequence_length=128,
    ).images[0]


_LOADERS = {
    "flux2-klein-9b": _load_flux2_klein,
    "realvisxl-lightning": _load_realvisxl_lightning,
    "sd35m": _load_sd35m,
    "sd35-large": _load_sd35_large,
    "sd35-large-turbo": _load_sd35_large_turbo,
    "qwen-image": _load_qwen_image,
    "hidream-i1": _load_hidream_i1,
}
_GENERATORS = {
    "flux2-klein-9b": _generate_flux2_klein,
    "realvisxl-lightning": _generate_sdxl,
    "sd35m": _generate_sd3,
    "sd35-large": _generate_sd35_large,
    "sd35-large-turbo": _generate_sd35_large_turbo,
    "qwen-image": _generate_qwen_image,
    "hidream-i1": _generate_hidream_i1,
}

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_class_order() -> dict[str, int]:
    """class common name -> row position, for stable per-image seeds."""
    with open(CLASSES_CSV, encoding="utf-8", newline="") as f:
        return {row["class"]: i for i, row in enumerate(csv.DictReader(f))}


def load_metadata() -> list[dict]:
    if not METADATA_PATH.exists():
        sys.exit(f"Error: {METADATA_PATH} not found. Run 1f-generate_prompts_compressed.py first.")
    records = []
    with open(METADATA_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_index(index_path: Path) -> list[dict]:
    if not index_path.exists():
        return []
    records = []
    with open(index_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


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
    cell_dir = TRAIN_ROOT / generator / "compressed"
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
    print(f"Generator : {generator}")
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
        global_index = class_order[rec["class"]] * 100 + (rec["index"] - 1)
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
    all_records = load_metadata()
    print(f"Loaded {len(all_records)} records from {METADATA_PATH.relative_to(REPO_ROOT)}")

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
        print(f"--benchmark {args.benchmark}: force-generating {len(records)} records per generator")

    generators = HEADLINE_GENERATORS if args.generator == "all" else (args.generator,)
    force = args.force or args.benchmark is not None

    benchmark_rows = []
    for generator in generators:
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
