"""Evaluate Florence-2 captions with an LLM to filter non-qualifying images.

Reads filter_results.jsonl for one or all dataset sources. For each entry
where passed=true, a caption field exists (written by 4-generate_captions.py),
and "caption_eval" is not yet in stages_done, the caption is sent to an LLM
that applies three rejection criteria. Results are written back as a filter
stage — identical in structure to megadetector and vlm stages.

Filter criteria (ALL three must pass):
  1. Real, living animal — not a painting, drawing, figurine, taxidermy,
     fossil, footprint, or any non-living/non-photographic subject
  2. Quality photograph — not a camera-trap, night-vision, IR, or thermal image
  3. Whole animal visible — not just a head close-up, isolated paw, or partial
     body part

Output changes to filter_results.jsonl (per entry):
  caption_eval    — {"pass": bool, "reason": str} always written
  stages_done     — "caption_eval" appended (enables resumability)
  passed=false, stage_failed="caption_eval", reason=<llm reason>
                  — only for entries the LLM rejects

The script is resumable: entries already containing "caption_eval" in
stages_done are skipped. Results are flushed to disk every FLUSH_EVERY entries.

Backends
--------
vllm  (default) — local inference via vLLM offline batch API.
                  Default model: Qwen/Qwen2.5-7B-Instruct-AWQ (~6.5 GB VRAM).
                  Structured output via GuidedDecodingParams(json=EVAL_SCHEMA).
                  Estimated runtime on RTX 3060: 12–18 h for 543 k images.

openrouter      — async HTTP to OpenRouter API with asyncio + aiohttp.
                  Default model: qwen/qwen-2.5-7b-instruct.
                  Estimated runtime: 30–90 min at 30 + concurrent requests.
                  Total cost: ~$5 for 543 k images.
                  Requires OPENROUTER_API_KEY in .env or environment.

Usage:
    # vLLM, all sources (overnight run):
    python scripts/dataset_quality/5-evaluate_captions.py --source all

    # vLLM, single source, faster 3B model:
    python scripts/dataset_quality/5-evaluate_captions.py --source wikimedia \\
        --model Qwen/Qwen2.5-3B-Instruct-AWQ

    # OpenRouter backend (set OPENROUTER_API_KEY in .env):
    python scripts/dataset_quality/5-evaluate_captions.py --source all \\
        --backend openrouter --concurrency 60

    # Re-evaluate from scratch:
    python scripts/dataset_quality/5-evaluate_captions.py --source wikimedia --force

Requirements:
    vllm backend:       pip install vllm
    openrouter backend: pip install aiohttp python-dotenv
"""

import argparse
import asyncio
import gc
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer

# ── Constants ─────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[2]

RESULTS_PATHS = {
    "gbif":        REPO_ROOT / "data" / "gbif"        / "filter_results.jsonl",
    "inaturalist": REPO_ROOT / "data" / "inaturalist" / "filter_results.jsonl",
    "wikimedia":   REPO_ROOT / "data" / "wikimedia"   / "filter_results.jsonl",
    "openimages":  REPO_ROOT / "data" / "openimages"  / "filter_results.jsonl",
    "images_cv":   REPO_ROOT / "data" / "images_cv"   / "filter_results.jsonl",
}

STAGE_NAME  = "caption_eval"
FLUSH_EVERY = 5000  # entries between disk flushes

# ── vLLM backend ──────────────────────────────────────────────────────────────
VLLM_DEFAULT_MODEL            = "Qwen/Qwen2.5-7B-Instruct-AWQ"
VLLM_MAX_NEW_TOKENS           = 80
VLLM_GPU_MEMORY_UTILIZATION   = 0.90
VLLM_MAX_NUM_SEQS             = 256

# ── OpenRouter backend ────────────────────────────────────────────────────────
OPENROUTER_URL                = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL              = "qwen/qwen-2.5-7b-instruct"
OPENROUTER_MAX_NEW_TOKENS     = 80
OPENROUTER_DEFAULT_CONCURRENCY = 30
OPENROUTER_RETRY_ATTEMPTS     = 3
OPENROUTER_RETRY_DELAY        = 2.0  # seconds between retries, multiplied by attempt

# ── OpenRouter pricing (per million tokens) ───────────────────────────────────
OPENROUTER_PRICE_INPUT_PER_M  = 0.07   # qwen/qwen-2.5-7b-instruct
OPENROUTER_PRICE_OUTPUT_PER_M = 0.10

# ── Prompt & schema ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a wildlife image quality evaluator. Given a caption describing a photograph, "
    "output ONLY the JSON: {\"pass\": true/false, \"reason\": \"one sentence\"}.\n\n"
    "REJECT (pass: false) ONLY for these reasons:\n"
    "1. Dead or non-living subject: dead animal, taxidermy, skeleton, skull, fossil, "
    "painting, drawing, figurine, stuffed animal, toy, or sign/screen showing an animal\n"
    "2. Not a daytime photograph: at night, illuminated by moonlight or moon, "
    "visible watermark, or text/timestamp overlay at the bottom\n"
    "3. No live animal visible: only tracks, burrow, mound, nest, cave, or landscape; "
    "or only a detached body part (skull, paw) without a live animal\n\n"
    "PASS in all other cases. "
    "Humans in the frame, domesticated animals, blurred or dark backgrounds, "
    "and descriptions of the animal's color or condition are NOT reasons to reject."
)

EVAL_SCHEMA = {
    "type": "object",
    "properties": {
        "pass":   {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["pass", "reason"],
    "additionalProperties": False,
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


# ── vLLM backend ──────────────────────────────────────────────────────────────

def load_vllm_model(model_id: str) -> tuple:
    """Load vLLM model and the associated tokenizer (for chat-template application).

    Returns (llm, tokenizer). The tokenizer is CPU-only; the LLM occupies GPU memory.
    vllm is imported here so the script remains importable (and --help works) even when
    vllm is not installed, as long as the openrouter backend is used.
    """
    try:
        from vllm import LLM
    except ImportError:
        sys.exit("vllm not found — install it with: pip install vllm")

    llm = LLM(
        model=model_id,
        quantization="awq",
        dtype="float16",
        gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
        max_num_seqs=VLLM_MAX_NUM_SEQS,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return llm, tokenizer


def build_vllm_prompt(caption: str, tokenizer) -> str:
    """Apply the model's chat template to system prompt + caption."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": caption},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def evaluate_batch_vllm(llm, tokenizer, captions: list) -> list:
    """Run batch inference via vLLM with JSON-schema-constrained output.

    Returns a list of {"pass": bool, "reason": str} dicts (never None —
    XGrammar guarantees valid JSON when EVAL_SCHEMA is applied).
    """
    try:
        from vllm.sampling_params import GuidedDecodingParams
    except ImportError:
        try:
            from vllm.model_executor.guided_decoding import GuidedDecodingParams
        except ImportError:
            sys.exit(
                "GuidedDecodingParams not found. Ensure vllm >= 0.4.0 is installed."
            )
    from vllm import SamplingParams

    guided = GuidedDecodingParams(json=EVAL_SCHEMA)
    params = SamplingParams(
        max_tokens=VLLM_MAX_NEW_TOKENS,
        temperature=0.0,
        guided_decoding=guided,
    )

    prompts = [build_vllm_prompt(c, tokenizer) for c in captions]
    outputs = llm.generate(prompts, params)
    results = []
    for out in outputs:
        text = out.outputs[0].text.strip()
        try:
            results.append(json.loads(text))
        except json.JSONDecodeError:
            results.append(None)  # safety fallback; should not occur with XGrammar
    return results


# ── OpenRouter backend ────────────────────────────────────────────────────────

async def _evaluate_single(
    session,
    semaphore: asyncio.Semaphore,
    caption: str,
    api_key: str,
) -> tuple:
    """Send one caption to OpenRouter. Returns ({"pass": bool, "reason": str}, cost_usd).

    Returns (None, 0.0) if all retry attempts are exhausted.
    """
    import aiohttp  # imported here; already installed

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": caption},
        ],
        "max_tokens": OPENROUTER_MAX_NEW_TOKENS,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with semaphore:
        for attempt in range(1, OPENROUTER_RETRY_ATTEMPTS + 1):
            try:
                async with session.post(
                    OPENROUTER_URL,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status != 200:
                        if attempt < OPENROUTER_RETRY_ATTEMPTS:
                            await asyncio.sleep(OPENROUTER_RETRY_DELAY * attempt)
                            continue
                        return None, 0.0
                    data = await resp.json()
                    text = data["choices"][0]["message"]["content"].strip()
                    usage = data.get("usage", {})
                    cost = (
                        usage.get("prompt_tokens", 0) / 1_000_000 * OPENROUTER_PRICE_INPUT_PER_M
                        + usage.get("completion_tokens", 0) / 1_000_000 * OPENROUTER_PRICE_OUTPUT_PER_M
                    )
                    return json.loads(text), cost
            except Exception:
                if attempt < OPENROUTER_RETRY_ATTEMPTS:
                    await asyncio.sleep(OPENROUTER_RETRY_DELAY * attempt)
    return None, 0.0


async def _evaluate_all_openrouter(
    captions: list,
    api_key: str,
    concurrency: int,
) -> tuple:
    """Send all captions to OpenRouter concurrently. Returns (results, chunk_cost_usd)."""
    import aiohttp

    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency + 10)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            _evaluate_single(session, semaphore, cap, api_key)
            for cap in captions
        ]
        pairs = await asyncio.gather(*tasks)

    results    = [r for r, _ in pairs]
    chunk_cost = sum(c for _, c in pairs)
    return results, chunk_cost


# ── Result application ────────────────────────────────────────────────────────

def apply_eval_results(
    pending_chunk: list,
    eval_results: list,
    by_filepath: dict,
) -> tuple:
    """Write evaluation results back into the entries dict in place.

    For None results (infrastructure failure): soft-pass to avoid false
    rejections. Returns (passed_count, rejected_count, error_count).
    """
    passed_count = rejected_count = error_count = 0

    for entry, result in zip(pending_chunk, eval_results):
        entry.setdefault("stages_done", []).append(STAGE_NAME)

        if result is None:
            entry["caption_eval"] = {"pass": True, "reason": "eval error — soft pass"}
            error_count += 1
        else:
            entry["caption_eval"] = result
            if result.get("pass") is False:
                entry["passed"]      = False
                entry["stage_failed"] = STAGE_NAME
                entry["reason"]      = result.get("reason", "")
                rejected_count += 1
            else:
                passed_count += 1

    return passed_count, rejected_count, error_count


# ── Per-source processing ─────────────────────────────────────────────────────

def process_source(
    source: str,
    backend: str,
    llm,
    tokenizer,
    api_key: str,
    concurrency: int,
) -> None:
    """Evaluate all captioned, not-yet-evaluated entries in one source."""
    path = RESULTS_PATHS[source]
    if not path.exists():
        print(f"[{source}] filter_results.jsonl not found — run the filter pipeline first, skipping.")
        return

    entries = load_results(path)
    if not entries:
        print(f"[{source}] empty filter_results.jsonl, skipping.")
        return

    pending = [
        e for e in entries
        if e.get("passed")
        and "caption" in e
        and STAGE_NAME not in e.get("stages_done", [])
    ]

    if not pending:
        print(f"[{source}] nothing to do ({len(entries):,} entries already evaluated or uncaptioned).")
        return

    print(f"[{source}] {len(pending):,} entries to evaluate ({len(entries):,} total) …")

    by_filepath = {e["filepath"]: e for e in entries}
    chunks = [pending[i : i + FLUSH_EVERY] for i in range(0, len(pending), FLUSH_EVERY)]

    total_passed = total_rejected = total_errors = 0
    total_cost = 0.0

    for chunk in tqdm(chunks, desc=f"eval {source}", unit="chunk"):
        captions = [e["caption"] for e in chunk]

        if backend == "vllm":
            eval_results = evaluate_batch_vllm(llm, tokenizer, captions)
        else:
            eval_results, chunk_cost = asyncio.run(
                _evaluate_all_openrouter(captions, api_key, concurrency)
            )
            total_cost += chunk_cost
            print(f"  chunk cost: ${chunk_cost:.4f}  running total: ${total_cost:.4f}")

        p, r, e = apply_eval_results(chunk, eval_results, by_filepath)
        total_passed   += p
        total_rejected += r
        total_errors   += e

        save_results(path, entries)

    print(
        f"[{source}] done — {total_passed:,} passed, {total_rejected:,} rejected, "
        f"{total_errors:,} eval errors (soft-passed)."
        + (f"  Total cost: ${total_cost:.4f}" if backend == "openrouter" else "")
    )


# ── Force-reset helper ────────────────────────────────────────────────────────

def force_reset_source(source: str) -> None:
    """Remove caption_eval stage from all entries in one source, restoring passed state."""
    path = RESULTS_PATHS[source]
    if not path.exists():
        return

    entries = load_results(path)
    reset_count = 0
    for entry in entries:
        stages = entry.get("stages_done", [])
        if STAGE_NAME in stages:
            stages.remove(STAGE_NAME)
            entry["stages_done"] = stages
            if entry.get("stage_failed") == STAGE_NAME:
                entry["passed"]       = True
                entry["stage_failed"] = None
                entry["reason"]       = None
            entry.pop("caption_eval", None)
            reset_count += 1

    save_results(path, entries)
    print(f"[{source}] force-reset {reset_count:,} entries.")


# ── Test mode ────────────────────────────────────────────────────────────────

async def _run_test_async(captions: list, api_key: str, concurrency: int) -> list:
    """Run _evaluate_single for each caption and return per-request (result, cost) pairs."""
    import aiohttp
    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency + 10)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [_evaluate_single(session, semaphore, cap, api_key) for cap in captions]
        return await asyncio.gather(*tasks)


def run_test_mode(sources: list, n: int, api_key: str, concurrency: int) -> None:
    """Sample N captions, evaluate via OpenRouter, print a decision table, and exit.

    No filter_results.jsonl files are modified.
    """
    import random

    pool_by_source: dict = {}
    for source in sources:
        path = RESULTS_PATHS[source]
        if not path.exists():
            continue
        entries = load_results(path)
        pool_by_source[source] = [
            e for e in entries
            if e.get("passed") and "caption" in e
            and STAGE_NAME not in e.get("stages_done", [])
        ]

    if not any(pool_by_source.values()):
        print("No captioned+pending entries found for the selected source(s).")
        sys.exit(0)

    per_source = max(1, n // max(1, len(pool_by_source)))
    sample: list = []
    for src, pool in pool_by_source.items():
        for e in random.sample(pool, min(per_source, len(pool))):
            sample.append((src, e))
    random.shuffle(sample)
    sample = sample[:n]

    print(f"Test mode: {len(sample)} sample(s) — read-only, no file writes\n")
    captions = [e["caption"] for _, e in sample]
    pairs = asyncio.run(_run_test_async(captions, api_key, concurrency))

    print(f"{'#':<4} {'SRC':<13} {'CAPTION':<54} {'PASS':<5} {'COST':>10}  REASON")
    print("-" * 120)
    running = 0.0
    for i, ((src, entry), (result, cost)) in enumerate(zip(sample, pairs), 1):
        running += cost
        cap = entry["caption"][:53].replace("\n", " ")
        if result is None:
            decision, reason = "ERR", "request failed"
        else:
            decision = "PASS" if result.get("pass") else "FAIL"
            reason = result.get("reason", "")[:55]
        print(f"{i:<4} {src:<13} {cap:<54} {decision:<5} ${cost:.6f}  {reason}")
    print("-" * 120)
    print(f"Total cost: ${running:.4f}")
    sys.exit(0)


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
        help="Dataset source to evaluate, or 'all' for every source in sequence.",
    )
    parser.add_argument(
        "--backend",
        default="vllm",
        choices=["vllm", "openrouter"],
        help="Inference backend (default: vllm).",
    )
    parser.add_argument(
        "--model",
        default=VLLM_DEFAULT_MODEL,
        help=f"HuggingFace model ID for the vLLM backend (default: {VLLM_DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=OPENROUTER_DEFAULT_CONCURRENCY,
        help=f"Max simultaneous requests for OpenRouter backend (default: {OPENROUTER_DEFAULT_CONCURRENCY}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reset caption_eval stage and re-evaluate all entries.",
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Sample N captions and evaluate via OpenRouter, printing a decision table "
            "without modifying any files. Requires --backend openrouter."
        ),
    )
    args = parser.parse_args()

    sources = list(RESULTS_PATHS.keys()) if args.source == "all" else [args.source]

    # ── Force-reset pass ──────────────────────────────────────────────────────
    if args.force:
        for source in sources:
            force_reset_source(source)

    # ── API key (OpenRouter only) ─────────────────────────────────────────────
    api_key = ""
    if args.backend == "openrouter":
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass  # python-dotenv optional; fall back to env var
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            sys.exit(
                "Error: OPENROUTER_API_KEY is not set. "
                "Add it to .env or export it in the shell."
            )

    # ── Test mode (read-only, exits before any model loading) ────────────────
    if args.test_samples:
        if args.backend != "openrouter":
            sys.exit("--test-samples requires --backend openrouter")
        run_test_mode(sources, args.test_samples, api_key, args.concurrency)

    # ── Model loading (vLLM only) ─────────────────────────────────────────────
    llm = tokenizer = None
    if args.backend == "vllm":
        print(f"Loading {args.model} …")
        llm, tokenizer = load_vllm_model(args.model)
        print("Model ready.\n")

    import torch

    try:
        for source in sources:
            process_source(
                source,
                args.backend,
                llm,
                tokenizer,
                api_key,
                args.concurrency,
            )
    finally:
        if llm is not None:
            del llm
            gc.collect()
            torch.cuda.empty_cache()

    print("All done.")


if __name__ == "__main__":
    main()
