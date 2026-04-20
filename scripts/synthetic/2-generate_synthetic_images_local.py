"""
Generate synthetic wildlife images using local text-to-image models.

Three models are available:

  flux-schnell          FLUX.1-schnell (NF4 quantized) — best quality/throughput balance.
                        ~80–95 images/hour on RTX 3060 12 GB.  Apache-2.0 licence.

  realvisxl-lightning   RealVisXL V5.0 base + SDXL-Lightning 4-step LoRA — highest throughput.
                        ~240–300 images/hour on RTX 3060 12 GB.  openrail++ licence.

  sd35m                 Stable Diffusion 3.5 Medium — no quantization required, clean bf16.
                        ~80–120 images/hour on RTX 3060 12 GB.  Stability AI community licence.

Images are saved to: data/synthetic/<class_name>/

Usage:
    # Single model
    python scripts/synthetic/2-generate_synthetic_images_local.py \\
        --class-name "binturong" \\
        --description "The binturong is long and heavy..." \\
        --n-images 5 \\
        --model flux-schnell

    # Run all three models sequentially (for easy comparison)
python scripts/synthetic/2-generate_synthetic_images_local.py \\
--class-name "red_fox" \\
--description "..." \\
--n-images 3 \\
--model all

Requirements:
    pip install torch torchvision diffusers transformers accelerate \\
                bitsandbytes huggingface_hub pillow

    bitsandbytes >= 0.43.0 with CUDA support is required for FLUX NF4 quantization.
"""

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import torch

# Reduce CUDA memory fragmentation (helps with successive VAE decode allocations).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_BASE = Path(__file__).parent.parent.parent / "data" / "synthetic"

AVAILABLE_MODELS = ("flux-schnell", "realvisxl-lightning", "sd35m")

# Shared wildlife photography style suffix (mirrors the OpenRouter script).
STYLE_SUFFIX = (
    "Professional wildlife photograph. Telephoto lens, sharp focus on the animal, "
    "natural lighting, photorealistic, high resolution. Full body of the animal visible, "
    "entire animal from head to tail fits within the frame. Natural habitat background."
)

# Negative prompt used by SDXL- and SD3-family models (FLUX does not use one).
NEGATIVE_PROMPT = (
    "text, watermark, cartoon, illustration, painting, drawing, art, sketch, animated, CGI, render, "
    "3D, unrealistic, low quality, blurry, watermark, text, logo, multiple animals, "
    "duplicate, deformed, ugly, bad anatomy, unnatural pose, "
    "close-up, closeup, portrait, head shot, face only, cropped body, partial animal, cut off limbs"
)

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompt(class_name: str, description: str, max_chars: int = 800) -> str:
    """Build the generation prompt, capping the description excerpt to avoid token overflow."""
    excerpt = description.strip()[:max_chars]
    return (
        f"Realistic wildlife photograph of a {class_name}. "
        f"Species characteristics: {excerpt}. "
        f"{STYLE_SUFFIX}"
    )


def build_clip_safe_prompt(class_name: str, description: str, tokenizer) -> str:
    """Build a prompt that fits within CLIP's 77-token limit.

    Trims the description to preserve the full style suffix, which has more
    impact on image quality than extra description tokens.
    """
    MAX_TOKENS = 77
    prefix = f"Realistic wildlife photograph of a {class_name}. Species characteristics: "
    suffix = f". {STYLE_SUFFIX}"

    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    available = MAX_TOKENS - 2 - len(prefix_ids) - len(suffix_ids)  # 2 for BOS/EOS

    if available <= 0:
        # Prefix + suffix already exceed limit — hard-truncate the full prompt
        full = f"{prefix}{description.strip()}{suffix}"
        ids = tokenizer.encode(full, add_special_tokens=True)[:MAX_TOKENS]
        return tokenizer.decode(ids, skip_special_tokens=True)

    desc_ids = tokenizer.encode(description.strip(), add_special_tokens=False)
    if len(desc_ids) > available:
        desc_ids = desc_ids[:available]
        description = tokenizer.decode(desc_ids)

    return f"{prefix}{description}{suffix}"


# ---------------------------------------------------------------------------
# TF32 — free speed-up on Ampere (RTX 30xx) with no quality loss
# ---------------------------------------------------------------------------

def _enable_tf32() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ---------------------------------------------------------------------------
# Model 1 — FLUX.1-schnell + NF4 quantization
# ---------------------------------------------------------------------------

def _load_flux_schnell(compile_transformer: bool = False):
    """Load FLUX.1-schnell with NF4 quantization (mandatory for 12 GB GPUs).

    First run downloads ~25 GB of model weights from HuggingFace.
    NF4 reduces active VRAM to ~9 GB; CPU offload provides a safety margin.
    """
    from diffusers import FluxPipeline, FluxTransformer2DModel
    from diffusers import BitsAndBytesConfig as DiffusersBnbConfig
    from transformers import T5EncoderModel
    from transformers import BitsAndBytesConfig as TransformersBnbConfig

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

    model_id = "black-forest-labs/FLUX.1-schnell"

    print("  Loading T5 text encoder (NF4) …")
    text_encoder_2 = T5EncoderModel.from_pretrained(
        model_id,
        subfolder="text_encoder_2",
        quantization_config=nf4_transformers,
        torch_dtype=torch.bfloat16,
    )

    print("  Loading FLUX transformer (NF4) …")
    transformer = FluxTransformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        quantization_config=nf4_diffusers,
        torch_dtype=torch.bfloat16,
    )

    print("  Assembling FluxPipeline …")
    pipe = FluxPipeline.from_pretrained(
        model_id,
        transformer=transformer,
        text_encoder_2=text_encoder_2,
        torch_dtype=torch.bfloat16,
    )
    # CPU offload moves components to GPU only when needed, respecting the 12 GB limit.
    pipe.enable_model_cpu_offload()

    if compile_transformer:
        print("  Compiling transformer (this takes 2–10 min on first run) …")
        pipe.transformer = torch.compile(
            pipe.transformer,
            mode="reduce-overhead",
            fullgraph=True,
        )
        # Warm-up pass to trigger compilation before the generation loop.
        print("  Warm-up pass …")
        _ = pipe(
            prompt="test",
            num_inference_steps=1,
            guidance_scale=0.0,
            max_sequence_length=64,
            height=64,
            width=64,
        ).images[0]

    return pipe


def _generate_flux(pipe, prompt: str, seed: int, height: int, width: int) -> Image.Image:
    generator = torch.Generator("cpu").manual_seed(seed)
    return pipe(
        prompt=prompt,
        num_inference_steps=4,
        guidance_scale=0.0,          # FLUX.1-schnell: guidance disabled
        max_sequence_length=256,     # schnell limit (dev supports 512)
        height=height,
        width=width,
        generator=generator,
    ).images[0]


# ---------------------------------------------------------------------------
# Model 2 — RealVisXL V5.0 + SDXL-Lightning 4-step LoRA
# ---------------------------------------------------------------------------

def _load_realvisxl_lightning(compile_transformer: bool = False):
    """Load RealVisXL V5.0 with the SDXL-Lightning 4-step LoRA fused in.

    First run downloads ~7 GB (RealVisXL checkpoint) + ~400 MB (Lightning LoRA).
    Runs at fp16 without quantization; requires ~10–12 GB GPU VRAM.
    """
    from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
    from huggingface_hub import hf_hub_download

    _enable_tf32()

    print("  Loading RealVisXL V5.0 (fp16) …")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "SG161222/RealVisXL_V5.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to("cuda")

    print("  Fusing SDXL-Lightning 4-step LoRA …")
    lora_path = hf_hub_download(
        repo_id="ByteDance/SDXL-Lightning",
        filename="sdxl_lightning_4step_lora.safetensors",
    )
    pipe.load_lora_weights(lora_path)
    pipe.fuse_lora()

    # SDXL-Lightning requires trailing timestep spacing.
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="trailing",
    )

    if compile_transformer:
        print("  Compiling UNet …")
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

    return pipe


def _generate_sdxl(pipe, prompt: str, seed: int, height: int, width: int) -> Image.Image:
    generator = torch.Generator("cuda").manual_seed(seed)
    return pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=4,
        guidance_scale=0.0,          # Lightning is distilled — no CFG
        height=height,
        width=width,
        generator=generator,
    ).images[0]


# ---------------------------------------------------------------------------
# Model 3 — Stable Diffusion 3.5 Medium
# ---------------------------------------------------------------------------

def _load_sd35m(compile_transformer: bool = False):
    """Load SD 3.5 Medium in full bf16 — no quantization required for 12 GB GPUs.

    First run downloads ~10 GB from HuggingFace (requires accepting the Stability AI
    community licence at https://huggingface.co/stabilityai/stable-diffusion-3.5-medium).
    Keeps ~9.9 GB of VRAM; CPU offload is enabled as a safety margin.
    """
    from diffusers import StableDiffusion3Pipeline

    _enable_tf32()

    print("  Loading SD 3.5 Medium (bf16) …")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium",
        torch_dtype=torch.bfloat16,
    )
    # Enable offload to avoid OOM from activation spikes on the 12 GB limit.
    pipe.enable_model_cpu_offload()

    if compile_transformer:
        print("  Compiling transformer …")
        pipe.transformer = torch.compile(
            pipe.transformer,
            mode="reduce-overhead",
            fullgraph=True,
        )

    return pipe


def _generate_sd3(pipe, prompt: str, seed: int, height: int, width: int) -> Image.Image:
    generator = torch.Generator("cpu").manual_seed(seed)
    return pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=40,
        guidance_scale=4.5,
        height=height,
        width=width,
        generator=generator,
        max_sequence_length=256,     # SD 3.5 Medium: avoid T5 artifacts beyond 256 tokens
    ).images[0]


# ---------------------------------------------------------------------------
# Dispatch tables
# ---------------------------------------------------------------------------

_LOADERS = {
    "flux-schnell":         _load_flux_schnell,
    "realvisxl-lightning":  _load_realvisxl_lightning,
    "sd35m":                _load_sd35m,
}

_GENERATORS = {
    "flux-schnell":         _generate_flux,
    "realvisxl-lightning":  _generate_sdxl,
    "sd35m":                _generate_sd3,
}


# ---------------------------------------------------------------------------
# Per-model generation loop
# ---------------------------------------------------------------------------

def run_model(
    model_key: str,
    class_name: str,
    description: str,
    n_images: int,
    output_base: Path,
    base_seed: int,
    height: int,
    width: int,
    compile_model: bool,
) -> None:
    """Load one model, generate n_images, then unload to free VRAM."""

    print(f"\n{'=' * 60}")
    print(f"Model    : {model_key}")
    print(f"{'=' * 60}")

    output_dir = output_base / class_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Full prompt shown upfront; CLIP-safe variant rebuilt after pipeline load
    full_prompt = build_prompt(class_name, description)
    print(f"Prompt   : {full_prompt[:120]}…")
    print(f"Output   : {output_dir}")
    print()

    print("Loading pipeline …")
    t_load = time.perf_counter()
    pipe = _LOADERS[model_key](compile_transformer=compile_model)
    print(f"Pipeline ready in {time.perf_counter() - t_load:.1f} s\n")

    # CLIP encoders have a 77-token hard limit; rebuild the prompt using the
    # pipeline's own tokenizer so the style suffix is never truncated.
    if model_key in ("realvisxl-lightning", "sd35m"):
        prompt = build_clip_safe_prompt(class_name, description, pipe.tokenizer)
    else:
        prompt = full_prompt

    generate_fn = _GENERATORS[model_key]
    saved = 0

    for i in range(n_images):
        seed = base_seed + i
        print(f"  [{i + 1}/{n_images}] seed={seed} …", end=" ", flush=True)

        torch.cuda.empty_cache()
        t0 = time.perf_counter()
        image = generate_fn(pipe, prompt, seed, height, width)
        elapsed = time.perf_counter() - t0

        # Index against existing files for this model so reruns don't overwrite.
        existing = sorted(output_dir.glob(f"{class_name}_{model_key}_*.png"))
        next_idx = len(existing) + 1
        img_path = output_dir / f"{class_name}_{model_key}_{next_idx:04d}.png"
        image.save(img_path, "PNG")
        saved += 1
        print(f"saved → {img_path.name}  ({elapsed:.1f} s)")

    print(f"\nDone. {saved}/{n_images} images saved.")

    # Unload pipeline and free VRAM before loading the next model.
    print("Unloading pipeline …")
    del pipe
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic wildlife images using local diffusion models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--class-name", required=True,
        help="Species/class name used in filenames and the prompt (e.g. red_fox).",
    )
    parser.add_argument(
        "--description", required=True,
        help="Morphological or ecological description of the species.",
    )
    parser.add_argument(
        "--n-images", type=int, default=5,
        help="Number of images to generate per model (default: 5).",
    )
    parser.add_argument(
        "--model",
        choices=[*AVAILABLE_MODELS, "all"],
        default="flux-schnell",
        help=(
            "Which model to use. "
            "'all' runs all three sequentially for easy comparison (default: flux-schnell)."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed; each image uses seed + image_index (default: 42).",
    )
    parser.add_argument(
        "--height", type=int, default=1024,
        help="Output image height in pixels (default: 1024).",
    )
    parser.add_argument(
        "--width", type=int, default=1024,
        help="Output image width in pixels (default: 1024).",
    )
    parser.add_argument(
        "--compile", action="store_true",
        help=(
            "JIT-compile the model's transformer/UNet with torch.compile "
            "(adds 2–10 min warm-up, then ~20 %% faster). "
            "Recommended only when generating large batches."
        ),
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit(
            "Error: no CUDA GPU detected. "
            "This script requires a CUDA-capable GPU (tested: RTX 3060 12 GB)."
        )

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"GPU      : {gpu_name}  ({vram_gb:.1f} GB VRAM)")

    models_to_run = AVAILABLE_MODELS if args.model == "all" else (args.model,)

    total_t0 = time.perf_counter()

    for model_key in models_to_run:
        run_model(
            model_key=model_key,
            class_name=args.class_name,
            description=args.description,
            n_images=args.n_images,
            output_base=OUTPUT_BASE,
            base_seed=args.seed,
            height=args.height,
            width=args.width,
            compile_model=args.compile,
        )

    total_elapsed = time.perf_counter() - total_t0
    print(f"\nAll done in {total_elapsed / 60:.1f} min.")
    print(f"Images saved to: {OUTPUT_BASE / args.class_name}")


if __name__ == "__main__":
    main()
