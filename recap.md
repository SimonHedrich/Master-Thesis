# Session Recap — 2026-04-28

## Scripts created

### `scripts/dataset_quality/4-generate_captions.py`
Standalone Florence-2-base caption generator for all passed images in any dataset source.

- Model: `microsoft/Florence-2-base`, batch size 6, image size 768 px
- CLI: `--source {gbif|inaturalist|wikimedia|openimages|images_cv|all}`
- Reads `filter_results.jsonl`, captions every entry with `passed=true` and no existing `caption` field
- Writes `caption` field back to `filter_results.jsonl`; writes `caption_error` on unreadable images
- Resumable (skips already-captioned entries), flushes every 500 images
- Fixed `<pad>` token artifact from batched decoding via `.replace("<pad>", "").strip()` after `post_process_generation()`

### `scripts/dataset_quality/5-evaluate_captions.py`
LLM-based quality filter that reads Florence-2 captions and applies three rejection criteria:
1. Real, living animal (not a painting / figurine / taxidermy / fossil / footprint)
2. Quality photograph (not a camera trap / night vision / IR / thermal image)
3. Whole animal visible (not just a head close-up or isolated body part)

Follows the existing filter pipeline pattern: rejections set `passed=false`, `stage_failed="caption_eval"`, `reason=<llm reason>` — identical to how the megadetector stage works. All evaluated entries receive a `caption_eval: {"pass": bool, "reason": str}` field and `"caption_eval"` in `stages_done`.

**Backends:**
- `--backend vllm` (default): `Qwen/Qwen2.5-7B-Instruct-AWQ` via vLLM offline batch API; XGrammar-constrained JSON output (100% valid); ~12–18 h for 543k images on RTX 3060
- `--backend vllm --model Qwen/Qwen2.5-3B-Instruct-AWQ`: faster (~8–10 h), marginal quality loss
- `--backend openrouter`: async aiohttp requests to OpenRouter API; ~$5 for 543k images; 30–90 min at 30+ concurrent requests; reads `OPENROUTER_API_KEY` from `.env`

Additional features: `--force` to reset and re-evaluate, soft-pass on infrastructure errors (so network hiccups never silently reject images), flush every 5,000 entries.

## Documentation updated

### `docs/automatic_image_qualitiy_filtering.md`
Two new sections added:
- **Standalone Caption Generation** — documents script 4 (model, CLI, output fields, `<pad>` fix, `transformers==4.48.3` requirement)
- **Caption-Based LLM Evaluation** — documents script 5 (backends table with runtimes/costs, usage examples, field schema, soft-pass policy)

## Dataset scale (discovered during session)
- 543,829 images currently have `passed=true` across all sources
- iNaturalist: 459,850 — GBIF: 50,504 — Wikimedia: 15,166 — OpenImages: 10,901 — images_cv: 7,408
- No captions generated yet (script 4 still to be run)

---

# Session Recap — 2026-04-28 (session 2)

## Documentation updated

### `docs/dataset-supplementation-plan.md`
New section **"Minimum Per-Class Image Threshold"** added between the Current State table and Step 1. Documents the working per-class instance targets derived from transfer-learning and KD literature:

| Scenario | Minimum instances/class |
|---|---|
| KD from SpeciesNet teacher | ~200–400 |
| Direct fine-tuning of student | ~400–800 |
| Visually similar / long-tail species | 800+ |

Rationale: COCO-pretrained backbones already encode animal-like features; KD from SpeciesNet amplifies each training image via soft labels; nano-scale students overfit less readily. Class inclusion threshold set at **≥ 300 GBIF images** as proxy for iNaturalist/supplementary availability. Classes below ~200 instances after real-data steps are candidates for synthetic generation (Step 5).

## Human class — source evaluation and final approach

The `human / homo sapiens` class has 0 images after quality filtering (existing GBIF images all rejected as "no animal detected" — MegaDetector uses a separate human detection class and the quality filter pipeline is therefore incompatible with human images).

### Sources investigated

| Source | Outcome | Reason |
|---|---|---|
| GBIF (existing) | 0 after filter | MegaDetector rejects humans as "not animal" |
| COCO 2017 (first attempt) | Rejected by inspection | Vehicle-heavy, crowd shots, tiny background persons — `vehicle` supercategory was never excluded, area floor was 100 absolute px² |
| iNaturalist S3 open data | Not viable | Public bulk export deliberately excludes Homo sapiens observations (privacy) |
| Open Images V7 | Rejected by inspection | Close-up portraits, low quality; OI's tightly framed annotation style does not match binocular deployment scenario |
| COCO 2017 (revised, strict filters) | **Adopted** | See below |

### Adopted approach: COCO 2017 with strict selection + quality filters

**Script:** `scripts/download_coco_humans.py`  
**Output:** `data/coco_humans/images/human/coco_{image_id}.jpg` — isolated from quality-filter pipeline  
**Full documentation:** `docs/dataset-supplementation-plan.md` § "Special Case: Human Class"

Selection filters (pre-download, from COCO annotation JSON):
- Hard exclude: `vehicle` supercategory (cars, buses, trucks, motorcycles, bicycles, …)
- Hard exclude: `animal` supercategory
- Hard exclude: `iscrowd = 1` person annotations
- Max 3 total person annotations per image
- Person bbox normalized area: 5%–60% of image
- Bbox edge margin ≥ 2% from all four edges
- Person bbox aspect ratio h/w ≥ 0.5

Post-download quality checks (same thresholds as `1-filter_dataset_quality.py`):
- Resolution ≥ 256 px (shorter side)
- Laplacian variance ≥ 100 (blur)
- Mean HSV saturation ≥ 15 (grayscale/IR)

**Usage:**
```bash
python scripts/download_coco_humans.py --reset --target 50   # smoke test
python scripts/download_coco_humans.py --target 2000         # full run / resume
```

---

# Session Recap — 2026-04-28 (session 3)

## Script created

### `scripts/synthetic/1-generate_synthetic_images_gemini.py`
Duplicate of `1-generate_synthetic_images_openrounter.py` rewritten to call the Gemini API directly via the `google-genai` SDK, removing the OpenRouter proxy dependency.

**Key differences from the OpenRouter version:**

| Area | OpenRouter version | Gemini version |
|---|---|---|
| HTTP client | `requests` (raw REST to OpenRouter) | `google-genai` SDK |
| Auth env var | `OPENROUTER_API_KEY` | `GEMINI_API_KEY` |
| Response parsing | `choices[0].message.images[0]` → base64 decode | `response.candidates[0].content.parts` → `part.inline_data` → `part.as_image()` |
| Cost tracking | `_fetch_cost()` via OpenRouter stats endpoint | Removed — not available from Gemini API |
| `--image-size` arg | Present (OpenRouter-specific 0.5K/1K/2K/4K tiers) | Removed |

**Usage:**
```bash
pip install google-genai
# add GEMINI_API_KEY=... to .env

python scripts/synthetic/1-generate_synthetic_images_gemini.py \
  --class-name "binturong" \
  --description "A large viverrid with long dark hair and a prehensile tail." \
  --n-images 5 \
  [--scenarios-file reports/animal_scenario_prompts.json]
```

Output images saved to `data/synthetic/<class_name>/<class_name>_gemini_NNNN.png`.
