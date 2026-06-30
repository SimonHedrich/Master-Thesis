# Local Text-to-Image Models for Wildlife Training Data Generation

**Research date:** 2026-04-02  
**Hardware target:** NVIDIA RTX 3060 12 GB GDDR6  
**Use case:** Generating realistic wildlife photographs (non-bird mammals) as training data for an object detection model  
**Access method:** HuggingFace `diffusers` library (Python)

---

## Table of Contents

1. [HuggingFace Landscape Overview](#1-huggingface-landscape-overview)
2. [Candidate Model Profiles](#2-candidate-model-profiles)
   - 2.1 [FLUX.1-dev](#21-flux1-dev)
   - 2.2 [FLUX.1-schnell](#22-flux1-schnell)
   - 2.3 [Stable Diffusion 3.5 Large](#23-stable-diffusion-35-large)
   - 2.4 [Stable Diffusion 3.5 Medium](#24-stable-diffusion-35-medium)
   - 2.5 [SDXL 1.0](#25-sdxl-10)
   - 2.6 [SDXL-Turbo](#26-sdxl-turbo)
   - 2.7 [SDXL-Lightning (ByteDance)](#27-sdxl-lightning-bytedance)
   - 2.8 [RealVisXL V5.0 (SG161222)](#28-realvisxl-v50-sg161222)
   - 2.9 [Stable Diffusion 2.1](#29-stable-diffusion-21)
   - 2.10 [Wildlife/Nature-Specific Fine-Tunes and LoRAs](#210-wildlifenature-specific-fine-tunes-and-loras)
3. [RTX 3060 12 GB Performance Summary](#3-rtx-3060-12-gb-performance-summary)
4. [Acceleration Techniques](#4-acceleration-techniques)
   - 4.1 [Precision: fp16 / bf16 / tf32](#41-precision-fp16--bf16--tf32)
   - 4.2 [Attention Backends: SDPA, xFormers, Flash Attention 2](#42-attention-backends-sdpa-xformers-flash-attention-2)
   - 4.3 [torch.compile](#43-torchcompile)
   - 4.4 [Quantization: INT8 and NF4 (bitsandbytes)](#44-quantization-int8-and-nf4-bitsandbytes)
   - 4.5 [Nunchaku / SVDQuant INT4](#45-nunchaku--svdquant-int4)
   - 4.6 [DeepCache](#46-deepcache)
   - 4.7 [Model CPU Offloading](#47-model-cpu-offloading)
   - 4.8 [Few-Step Distillation Variants (LCM/Turbo/Lightning/Hyper)](#48-few-step-distillation-variants-lcmturbolightninghyper)
5. [Batched Inference Considerations](#5-batched-inference-considerations)
6. [Recommendation](#6-recommendation)
7. [Proposed Implementation Stack](#7-proposed-implementation-stack)

---

## 1. HuggingFace Landscape Overview

As of early 2026, HuggingFace hosts ~94,000 text-to-image models. The models that rank highly across all three sort dimensions (trending, most-liked, most-downloaded) are:

| Rank | Model | Likes | Downloads/mo | Notes |
|------|-------|-------|--------------|-------|
| 1 | `black-forest-labs/FLUX.1-dev` | 12.5 k | 696 k | Non-commercial licence |
| 2 | `stabilityai/stable-diffusion-xl-base-1.0` | 7.58 k | 2 M | Open weights, most downloaded |
| 3 | `CompVis/stable-diffusion-v1-4` | 6.99 k | 464 k | Older SD 1.x family |
| 4 | `stabilityai/stable-diffusion-3-medium` | 4.92 k | 4.05 k | SD3 architecture |
| 5 | `black-forest-labs/FLUX.1-schnell` | 4.74 k | 725 k | Apache-2.0, fastest FLUX |
| 6 | `stabilityai/sdxl-turbo` | 2.54 k | 799 k | Non-commercial |
| 7 | `ByteDance/SDXL-Lightning` | 2.14 k | 86.3 k | openrail++, very fast |
| 8 | `stabilityai/stable-diffusion-3.5-large` | 3.39 k | 76.1 k | Community licence |
| 9 | `SG161222/RealVisXL_V5.0` | — | 71.5 k | Photorealism fine-tune |
| 10 | `ByteDance/Hyper-SD` | 1.33 k | 56.9 k | Multi-step distillation |

Models optimised for photorealistic wildlife imagery cluster into three architectural families: **FLUX** (DiT-based, 12 B params), **SDXL** (UNet-based, ~3.5 B params), and **SD 3.x** (MMDiT-based, 2.6–8 B params).

---

## 2. Candidate Model Profiles

### 2.1 FLUX.1-dev

| Property | Value |
|----------|-------|
| Architecture | 12 B-parameter Rectified Flow Transformer (DiT) |
| Full-precision VRAM | ~24 GB (FP16/BF16) — exceeds RTX 3060 |
| FP8 VRAM | ~13–14 GB — just over limit |
| NF4/INT4 VRAM | ~8–10 GB — fits comfortably |
| Steps needed | 20–50 (guidance-distilled, cfg=3.5) |
| RTX 3060 speed (FP8) | ~400 s / image (measured benchmark) |
| RTX 3060 speed (NF4) | ~138 s / image (measured benchmark) |
| License | FLUX.1-dev Non-Commercial (research, personal, and commercial outputs allowed; model weights non-commercial only) |
| diffusers support | `FluxPipeline` — full first-class support |
| Prompt adherence | Excellent — consistently cited as the best open-weight model for complex, multi-element prompts |
| Photorealism | Very high; superior skin/fur/texture rendering vs. SDXL |
| Batching | `num_images_per_prompt` supported; each extra image multiplies VRAM and time |

**Notes for wildlife use:** FLUX.1-dev produces the most convincing photorealistic animal fur, environmental detail, and lighting of any open model. Its T5-XXL text encoder handles very long, detailed prompts well (up to 512 tokens). The primary barrier on an RTX 3060 is the 2–7 minute generation time even with NF4 quantization, making large-scale batch generation slow.

---

### 2.2 FLUX.1-schnell

| Property | Value |
|----------|-------|
| Architecture | 12 B-parameter Rectified Flow Transformer (same as dev) |
| Full-precision VRAM | ~24 GB |
| NF4/INT4 VRAM | ~8–10 GB |
| Steps needed | 1–4 (latent adversarial distillation; guidance_scale=0.0) |
| RTX 3060 speed (NF4, ~4 steps) | ~38–41 s / image (measured community reports) |
| RTX 3060 speed (FP16/BF16 with offload) | ~80–90 s / image |
| License | **Apache-2.0** — fully commercial |
| diffusers support | `FluxPipeline` — full support |
| Prompt adherence | Good (~75–80 % of dev quality); less fine detail in complex compositions |
| Photorealism | High; same architecture as dev, somewhat fewer denoising steps |
| Batching | Supported |

**Notes for wildlife use:** Schnell offers the best quality-per-minute ratio in the FLUX family on limited hardware. The 4-step constraint means less denoising refinement; for dataset-level throughput this is typically acceptable because many varied images are needed rather than perfection in each. Max token context is 256 vs. 512 for dev, which limits the longest prompts somewhat.

---

### 2.3 Stable Diffusion 3.5 Large

| Property | Value |
|----------|-------|
| Architecture | MMDiT (Multimodal Diffusion Transformer), ~8 B params |
| Full-precision VRAM | ~18 GB |
| FP8 VRAM (NVIDIA TensorRT) | ~11 GB (40 % reduction) — barely fits |
| NF4 (4-bit bitsandbytes) | ~7–8 GB |
| Steps needed | 40 (recommended) |
| Speed | ~60–120 s / image at 1024×1024 on RTX 3060 (estimated from community reports; precise benchmark unavailable) |
| License | Stability AI Community Licence — free for commercial use up to $1 M annual revenue |
| diffusers support | `StableDiffusion3Pipeline` — full support |
| Prompt adherence | Strong; T5-XXL encoder, 512-token context |
| Photorealism | High; improved over SD 3.0, comparable to FLUX.1-dev on many prompts |
| Batching | Supported |

**Notes for wildlife use:** SD 3.5 Large requires either NF4 quantization or FP8 TensorRT to fit on the RTX 3060. NF4 loading through `bitsandbytes` is straightforward in `diffusers`. Quality is competitive with FLUX.1-dev. Prompt context up to 512 tokens via T5. The 40-step default and large model size make it slower than schnell-class models, but generates more refined images than SDXL.

---

### 2.4 Stable Diffusion 3.5 Medium

| Property | Value |
|----------|-------|
| Architecture | MMDiT, ~2.6 B params |
| Full-precision VRAM | ~9.9 GB (fits on RTX 3060 at full precision) |
| Steps needed | 40 (recommended) |
| RTX 3060 speed (fp16) | ~5 s / iteration reported; ~30–40 s per full 1024×1024 image (community estimate) |
| License | Stability AI Community Licence (same as Large) |
| diffusers support | `StableDiffusion3Pipeline` — full support |
| Prompt adherence | Good; same text encoder family |
| Photorealism | Good; slightly below Large on fine detail |
| Batching | Supported; batch size > 1 may push VRAM past limit |

**Notes for wildlife use:** SD 3.5 Medium is the only large-architecture model that fits in full FP16 precision within 12 GB. It does not match FLUX.1 quality but substantially exceeds SDXL on complex prompt adherence, and its generation speed on the RTX 3060 is roughly 3–5× faster than FLUX.1-dev NF4. A practical choice when quality does not need to be maximised.

Caveat: the T5 encoder is sensitive to long prompts — tokens beyond 256 can cause edge artifacts. Keep prompts under 256 tokens for best results with SD 3.5 Medium.

---

### 2.5 SDXL 1.0

| Property | Value |
|----------|-------|
| Architecture | UNet-based, ~3.5 B params (base) + ~3.5 B (refiner optional) |
| Full-precision VRAM (base only) | ~10–12 GB — fits with headroom at FP16 |
| Steps needed | 20–30 (base only) |
| RTX 3060 speed (fp16, 20 steps, 1024×1024) | ~25–35 s / image |
| RTX 3060 speed (fp16, 30 steps, 1024×1024) | ~40–50 s / image |
| License | openrail++ (permissive, commercial OK) |
| diffusers support | `StableDiffusionXLPipeline` — full support, widely tested |
| Prompt adherence | Moderate; frequently misses or loose-interprets long multi-attribute prompts |
| Photorealism | Moderate; solid baseline but clearly below FLUX.1 or SD 3.5 |
| Batching | Supported; batch 2 fits at FP16, batch 4+ requires NF4 or offloading |

**Notes for wildlife use:** SDXL is the most mature and ecosystem-rich option — large LoRA library, ControlNet support, widest community tooling. For short, simple prompts it produces decent photorealistic animals. For long, highly detailed species descriptions it regularly ignores or blends attributes. It is fast enough on the RTX 3060 to generate ~80–120 images/hour at 1024×1024. The right choice if throughput dominates over per-image quality.

---

### 2.6 SDXL-Turbo

| Property | Value |
|----------|-------|
| Architecture | Distilled SDXL via Adversarial Diffusion Distillation (ADD) |
| VRAM | Same as SDXL base (~10–12 GB FP16) |
| Steps needed | 1–4; guidance_scale=0 required |
| Native resolution | 512×512 (higher resolutions are off-distribution and noticeably degrade) |
| Speed | Sub-second per image on high-end GPUs; estimated 5–10 s at 512×512 on RTX 3060 |
| License | **Non-commercial** (sai-nc-community) — commercial use requires Stability AI membership |
| diffusers support | `AutoPipelineForText2Image` |
| Prompt adherence | Poor to moderate at 1 step; improves at 4 steps |
| Photorealism | Limited at native 512 px; loses fine fur/texture detail at upscale |

**Notes for wildlife use:** SDXL-Turbo's non-commercial licence and 512×512 native resolution make it unsuitable for this use case. Training images at 512 px can work for detection models but the licence restriction and lower photorealism quality eliminate it as the primary tool.

---

### 2.7 SDXL-Lightning (ByteDance)

| Property | Value |
|----------|-------|
| Architecture | Progressive adversarial distillation of SDXL; 1/2/4/8-step variants |
| VRAM | Same as SDXL base (~10–12 GB FP16) |
| Steps needed | 2 or 4 recommended (1-step is unstable) |
| Native resolution | 1024×1024 |
| RTX 3060 speed (4-step) | ~10–15 s / image (community estimate) |
| License | **openrail++** — commercial OK |
| diffusers support | Full; load via `StableDiffusionXLPipeline` + `EulerDiscreteScheduler` |
| Prompt adherence | Similar to SDXL base; limited by UNet capacity |
| Photorealism | Comparable to SDXL base at 4 steps; slight quality loss vs. full 20-step SDXL |

**Notes for wildlife use:** SDXL-Lightning gives the best throughput among commercially-licensed models that still produce 1024×1024 images. At 4 steps and ~12 s/image, the RTX 3060 can generate ~300 images/hour — the highest throughput option. The quality is SDXL-level (not FLUX-level). Available both as full UNet checkpoints and as LoRA weights that apply to any SDXL base, making it possible to combine with a photorealism fine-tune like RealVisXL.

---

### 2.8 RealVisXL V5.0 (SG161222)

| Property | Value |
|----------|-------|
| Architecture | SDXL checkpoint merge (DreamShaper XL + Juggernaut XL + SDXL 1.0) |
| VRAM | Same as SDXL (~10–12 GB FP16) |
| Steps needed | 30+ (DPM++ SDE Karras) or 50+ (DPM++ 2M Karras) |
| Speed | Similar to SDXL base: ~35–50 s / image at 1024×1024 on RTX 3060 |
| License | openrail++ |
| diffusers support | `StableDiffusionXLPipeline` (native safetensors) |
| Prompt adherence | Similar to SDXL but tuned for photorealism prompts |
| Photorealism | Substantially better than vanilla SDXL on people and naturalistic scenes |
| Batching | Supported |

**Notes for wildlife use:** RealVisXL is the best SDXL-family checkpoint for photorealism without resorting to FLUX or SD 3.5. It has 71 k monthly downloads and an "Overwhelmingly Positive" Civitai rating (>8500 reviews). Recommended negative prompt helps avoid common artefacts. Being SDXL-based, it inherits all SDXL ecosystem tools: ControlNet, IP-Adapter, SDXL-Lightning LoRA.

A practical hybrid: use SDXL-Lightning LoRA (4 step, fast) on top of RealVisXL base for high-throughput photorealistic generation.

---

### 2.9 Stable Diffusion 2.1

| Property | Value |
|----------|-------|
| Architecture | UNet-based, ~900 M params |
| VRAM | ~4–6 GB |
| Speed | Very fast — ~5–10 s / image at 768×768 on RTX 3060 |
| License | openrail |
| Photorealism | Clearly below SDXL; textures are notably less detailed |
| Prompt adherence | Weak on long, complex prompts |

**Notes for wildlife use:** SD 2.1 is too limited in photorealism and prompt adherence for high-quality wildlife training data generation. Its main advantage (low VRAM / high speed) is no longer necessary given that SDXL and SDXL-Lightning run comfortably on the RTX 3060. Not recommended.

---

### 2.10 Wildlife/Nature-Specific Fine-Tunes and LoRAs

No high-quality, wildlife-specific text-to-image models exist as standalone checkpoints at the FLUX or SD3 scale. The available options are SDXL-based LoRAs:

| Model | Base | Notes |
|-------|------|-------|
| `RalFinger/smol-animals-sdxl-lora` | SDXL | Trigger: "zhibi"; trained on diverse real animal photos |
| Animals SDXL (Civitai) | SDXL | 88-image training set; "hyper-realistic animals in natural habitat" |
| `MaxNoichl/dierenleven-sdxl-lora-001` | SDXL | Focus on birds of paradise; less relevant |
| `XLabs-AI/flux-RealismLora` | FLUX.1-dev | 1.22 k likes; general photorealism booster for FLUX |

These LoRAs are additive — they merge with the base model at inference time with no VRAM overhead beyond the base model. For wildlife generation the most practical approach is to use a photorealism LoRA (e.g. `XLabs-AI/flux-RealismLora` with FLUX.1-schnell, or an animal LoRA with SDXL) rather than relying on a specialist model.

**Prompt engineering is more impactful than species-specific LoRAs** for this use case: detailed prompts specifying species name, body posture, habitat, lighting condition, camera angle, and lens type already push FLUX.1-schnell or RealVisXL to produce high-quality wildlife photographs without additional fine-tuning.

---

## 3. RTX 3060 12 GB Performance Summary

| Model | VRAM needed | Steps | Time / image (est.) | Images / hour | License |
|-------|------------|-------|---------------------|---------------|---------|
| FLUX.1-dev (NF4) | ~9 GB | 30–50 | 138–200 s | 18–26 | Non-commercial weights |
| FLUX.1-schnell (NF4) | ~9 GB | 4 | 38–45 s | 80–95 | Apache-2.0 |
| SD 3.5 Large (NF4) | ~8 GB | 40 | ~90–120 s | 30–40 | Community (free <$1M rev) |
| SD 3.5 Medium (fp16) | ~10 GB | 40 | ~30–45 s | 80–120 | Community (free <$1M rev) |
| SDXL 1.0 (fp16) | ~10–12 GB | 20 | 25–35 s | 103–144 | openrail++ |
| RealVisXL V5.0 (fp16) | ~10–12 GB | 30 | 35–50 s | 72–103 | openrail++ |
| SDXL-Lightning 4-step (fp16) | ~10–12 GB | 4 | 10–15 s | 240–360 | openrail++ |
| SDXL-Turbo (fp16) | ~10–12 GB | 1–4 | 5–12 s (512 px) | 300–720 | Non-commercial |
| SD 2.1 (fp16) | ~5 GB | 20 | 5–10 s | 360–720 | openrail |

Notes on VRAM estimates:
- "NF4" = 4-bit quantization via bitsandbytes `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")`.
- FLUX.1 full-precision FP16 requires ~24 GB — does not fit on RTX 3060. NF4 is mandatory.
- SDXL-based models fit at FP16. Batch size > 1 requires NF4 or sequential offloading.
- Times are community-measured estimates. Individual variation is ±30 % depending on resolution, scheduler, and system configuration.
- RTX 3060 has 112 Tensor Cores (Ampere); significantly slower than RTX 4090 (512 Tensor Cores). FP8 and Flash Attention 2 speedups are less pronounced than on Ada/Hopper.

---

## 4. Acceleration Techniques

### 4.1 Precision: fp16 / bf16 / tf32

- **fp16** is the standard for SDXL on the RTX 3060. Load with `torch_dtype=torch.float16`.
- **bf16** is preferred for FLUX and SD 3.5 (better numerical stability). Ampere supports bf16 natively.
- **TF32** (TensorFloat-32) is an Ampere GPU feature. Enable with:
  ```python
  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True
  ```
  This provides meaningful speedups (~10–20 %) on convolutions and matmuls with no code changes and no quality loss.

### 4.2 Attention Backends: SDPA, xFormers, Flash Attention 2

PyTorch 2.0+ automatically enables **Scaled Dot Product Attention (SDPA)** which selects the best backend (Flash Attention 2, xFormers, or native C++) at runtime. No manual configuration needed if using PyTorch >= 2.0.

- **xFormers** provides 20–40 % memory and speed improvement on older PyTorch. Install with `pip install xformers`. Enable with `pipe.enable_xformers_memory_efficient_attention()`.
- **Flash Attention 2** provides up to 44 % total generation time reduction vs. pytorch's native attention. On PyTorch >= 2.0, SDPA selects FA2 automatically if installed.
- On the RTX 3060 (Ampere), a community-maintained build of Flash Attention + patched xFormers is available specifically for FLUX inference.

In practice, with PyTorch >= 2.2, xFormers is redundant — SDPA handles attention optimally. xFormers is still useful for older PyTorch versions.

### 4.3 torch.compile

`torch.compile` JIT-compiles the UNet or transformer to optimised CUDA kernels. Typical speedup: **20–35 %** after warm-up. Key caveats:

- First compilation is slow (2–10 min). Subsequent runs reuse the compiled graph.
- Use `mode="max-autotune"` for maximum speed at the cost of a longer initial compilation.
- Dynamic shape inputs (varying resolution) re-trigger compilation unless `dynamic=True` is set.
- For FLUX, compile only the transformer: `pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)`.
- For SDXL, compile both UNet and VAE decoder.
- **Regional compilation** (`compile_repeated_blocks`) reduces compile time by 8–10× at similar runtime speedup — preferred for development.

RTX 3060 (Ampere) benchmark: on an RTX 4090, 4-bit FLUX generation went from 32.6 s to 25.8 s with compilation. Proportionally the RTX 3060 gain will be similar in percentage terms.

### 4.4 Quantization: INT8 and NF4 (bitsandbytes)

For **FLUX on RTX 3060**, NF4 quantization is mandatory to fit within 12 GB:

```python
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig

# Quantize T5 text encoder (largest component)
t5_config = TransformersBitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
# Quantize FLUX transformer
flux_config = DiffusersBitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
```

Impact summary:

| Quantization | VRAM saved | Speed vs. fp16 | Quality loss |
|-------------|-----------|----------------|-------------|
| INT8 (FLUX transformer only) | ~50 % | Slight slowdown on Ampere | Minimal |
| NF4 (transformer + T5) | ~75 % | ~10–20× faster vs. fp16+offload | Noticeable but acceptable |
| Double quantization (nested NF4) | +0.4 bits/param extra | ~same | Slightly more |

Important: do not load an FP8 checkpoint and then apply NF4 — this double-quantizes and degrades quality more than either alone.

For **SDXL**: fp16 fits natively in 12 GB, so quantization is only needed to free headroom for larger batch sizes or to use higher-resolution generation.

### 4.5 Nunchaku / SVDQuant INT4

[SVDQuant (ICLR 2025 Spotlight)](https://github.com/nunchaku-ai/nunchaku) is a research method that absorbs quantization outliers via low-rank decomposition, achieving cleaner 4-bit FLUX inference than bitsandbytes NF4:

- Reduces 12 B FLUX model size by **3.6×**.
- On RTX 4090: **3× faster** than NF4 W4A16 baseline; 8.7× faster than 16-bit with CPU offloading.
- RTX 3080 10 GB reported: ~10–12 s / image (vs. 40+ s without SVDQuant).
- RTX 3060 (INT4, not FP4): works — INT4 is the correct variant for pre-50xx GPUs.
- Integrates with ComfyUI; diffusers compatibility through model classes that inherit from diffusers base classes.

**Practical implication for RTX 3060:** Nunchaku could reduce FLUX.1-schnell generation to potentially 25–35 s/image (extrapolating from RTX 3080 data), though the RTX 3060's memory bandwidth limits the maximum speedup. This is likely the fastest path to high-quality FLUX inference on the RTX 3060.

Pre-quantized model: `mit-han-lab/nunchaku-flux.1-dev` and `mit-han-lab/nunchaku-flux.1-schnell` on HuggingFace.

### 4.6 DeepCache

[DeepCache (CVPR 2024)](https://github.com/horseee/DeepCache) caches high-level UNet/DiT features across adjacent denoising steps, reusing expensive computations. Natively supported in `diffusers`:

```python
from diffusers.utils import DeepCacheSDHelper
helper = DeepCacheSDHelper(pipe=pipe)
helper.enable()
```

Speedups:
- **SD 1.5:** 2.3× speedup, −0.05 CLIP Score
- **LDM-4-G (ImageNet):** 4.1× speedup, −0.22 FID
- Effective for SDXL as well

DeepCache works by skipping the deep UNet computation every N steps (configurable `cache_interval`). Quality trade-off is small at `cache_interval=3`. For dataset generation where slight quality variation is acceptable, this is a free speedup on SDXL.

DeepCache is **less effective for FLUX** (DiT architecture uses a different cache pattern) and has limited official support for FLUX in diffusers as of early 2026.

### 4.7 Model CPU Offloading

Two strategies in diffusers:

- `pipe.enable_model_cpu_offload()` — moves model components to CPU when not in use, restoring to GPU one at a time. Reduces peak VRAM by 30–50 % but adds CPU↔GPU transfer overhead (~10–40 % slower).
- `pipe.enable_sequential_cpu_offload()` — more aggressive, offloads each sub-module. Lowest VRAM usage but slowest.

For FLUX on RTX 3060, `enable_model_cpu_offload()` combined with NF4 quantization is the recommended approach — quantization handles the VRAM constraint and offloading provides a safety margin.

For SDXL, offloading is usually not needed at fp16 with a single image at a time.

### 4.8 Few-Step Distillation Variants (LCM/Turbo/Lightning/Hyper)

| Technique | Base | Steps | Quality | Commercial |
|-----------|------|-------|---------|-----------|
| FLUX.1-schnell | FLUX.1 | 1–4 | High (75–80 % of dev) | Apache-2.0 |
| SDXL-Lightning (ByteDance) | SDXL | 2–4 | Good (SDXL-level) | openrail++ |
| Hyper-SDXL | SDXL | 1–8 | Slightly above Lightning | openrail++ |
| SDXL-Turbo | SDXL | 1–4 | Good at 512 px | Non-commercial |
| LCM-SDXL | SDXL | 4–8 | Below Lightning | openrail++ |
| SD 3.5 Large Turbo | SD 3.5 | 4 | High | Community |

For this use case, **FLUX.1-schnell** is the distilled variant of choice if quality is the priority. **SDXL-Lightning** is the choice if throughput dominates. Both are integrated into `diffusers` with minimal configuration changes.

---

## 5. Batched Inference Considerations

All `diffusers` pipelines support `num_images_per_prompt` for within-batch generation and `prompt` as a list for cross-prompt batching.

VRAM implications:

- **SDXL fp16, batch=1:** ~10–12 GB — fits.
- **SDXL fp16, batch=2:** ~14–16 GB — exceeds RTX 3060. Requires NF4 (reduces to ~8 GB) to fit batch=2.
- **FLUX NF4, batch=1:** ~9–10 GB — fits.
- **FLUX NF4, batch=2:** ~14–15 GB — exceeds limit.

Practical recommendation: **batch size = 1** with a tight generation loop is more reliable on 12 GB than attempting batch=2. The generation loop itself is IO-bound between images (saving files, etc.) so single-image generation with a tight `for` loop approaches pipeline-batched throughput without the VRAM risk.

For throughput, the most effective strategy is to pre-compute all prompts, use `pipe()` in a loop with seeds varied by index, and write images asynchronously.

---

## 6. Recommendation

### Primary Recommendation: FLUX.1-schnell + NF4 + Nunchaku

For the best balance of photorealism, prompt adherence, and throughput on the RTX 3060 12 GB:

**Model:** `black-forest-labs/FLUX.1-schnell`  
**Quantization:** NF4 (bitsandbytes) or SVDQuant INT4 (Nunchaku) for the transformer and T5 encoder  
**Steps:** 4  
**Expected throughput:** 80–95 images/hour (NF4), potentially 120–140 images/hour (Nunchaku INT4)  
**License:** Apache-2.0 — fully commercial  
**Diffusers integration:** `FluxPipeline` with `BitsAndBytesConfig`  

Rationale:
1. FLUX architecturally outperforms SDXL and SD 3.5 Medium on complex, long prompts — critical for detailed species descriptions with habitat, lighting, and pose attributes.
2. schnell's 4-step inference is 3–5× faster than FLUX.1-dev on the same hardware.
3. Apache-2.0 licence has no revenue restrictions (unlike FLUX.1-dev's non-commercial weight restriction).
4. NF4 quantization via `bitsandbytes` is a single-line change in `diffusers` and is well-documented.
5. The `XLabs-AI/flux-RealismLora` LoRA can be layered on top with no VRAM overhead to further enhance photographic quality.

Optionally, add `torch.compile` with regional compilation on the transformer for an additional ~20 % speedup after the initial warm-up.

### Secondary Recommendation: SDXL-Lightning + RealVisXL (throughput-first)

When generating at scale (tens of thousands of images) and quality requirements can tolerate SDXL-level photorealism:

**Model:** `SG161222/RealVisXL_V5.0` as base + `ByteDance/SDXL-Lightning` 4-step LoRA  
**Steps:** 4  
**Expected throughput:** ~240–300 images/hour  
**License:** openrail++ (commercial)  
**Diffusers integration:** `StableDiffusionXLPipeline` with LoRA merging  

Rationale:
- 3× higher throughput than FLUX.1-schnell NF4.
- RealVisXL provides substantially better photorealism than vanilla SDXL.
- SDXL-Lightning LoRA can be applied to any SDXL checkpoint without changing the base model.
- DeepCache can provide an additional 2× speedup at minimal quality cost.
- Entire stack is open-source, commercial-licensed, and well-supported by diffusers.

### Tertiary Recommendation: SD 3.5 Medium (quality + fits without quantization)

If avoiding quantization entirely (for cleaner outputs) while still fitting in 12 GB:

**Model:** `stabilityai/stable-diffusion-3.5-medium`  
**Steps:** 40  
**Expected throughput:** ~80–120 images/hour  
**License:** Stability AI Community (free for research and <$1M revenue commercial)  

---

## 7. Proposed Implementation Stack

The existing `scripts/generate_synthetic_images.py` uses OpenRouter/Gemini (API-based). A local diffusers replacement would follow this structure:

```python
import torch
from diffusers import FluxPipeline, BitsAndBytesConfig as DiffusersBnbConfig
from transformers import BitsAndBytesConfig as TransformersBnbConfig, T5EncoderModel
from diffusers import AutoModel

# --- FLUX.1-schnell + NF4 ---
t5_config = TransformersBnbConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
flux_config = DiffusersBnbConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

text_encoder_2 = T5EncoderModel.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    subfolder="text_encoder_2",
    quantization_config=t5_config,
    torch_dtype=torch.bfloat16,
)
transformer = AutoModel.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    subfolder="transformer",
    quantization_config=flux_config,
    torch_dtype=torch.bfloat16,
)

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    transformer=transformer,
    text_encoder_2=text_encoder_2,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Optional: torch.compile for ~20% speedup after warm-up
# pipe.transformer.compile_repeated_blocks(fullgraph=True)

# Optional: TF32 on Ampere
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def generate(prompt: str, seed: int = 42) -> "PIL.Image.Image":
    generator = torch.Generator().manual_seed(seed)
    return pipe(
        prompt=prompt,
        num_inference_steps=4,
        guidance_scale=0.0,          # schnell: classifier-free guidance disabled
        max_sequence_length=256,
        height=1024,
        width=1024,
        generator=generator,
    ).images[0]
```

Key differences from the current API-based script:
- No API key or internet connection required after the one-time model download (~25 GB for FLUX schnell + NF4 quantized weights).
- Full control over seeds (reproducibility for ablations).
- Can generate hundreds of images per hour vs. API rate limits.
- Cost is electricity rather than per-image API charges.
- Works offline — important for experiments on isolated lab hardware.

Model download size reference:
- FLUX.1-schnell full (BF16): ~25 GB on disk; NF4 quantized variant: ~7 GB
- SD 3.5 Medium: ~10 GB
- SDXL + RealVisXL checkpoint: ~7 GB

---

*Research compiled from HuggingFace model cards, community benchmarks (Stable Diffusion WebUI Forge, ComfyUI discussions, PromptingPixels GPU benchmarks), official diffusers documentation, NVIDIA technical blogs, and academic publications (SVDQuant/Nunchaku ICLR 2025, DeepCache CVPR 2024).*
