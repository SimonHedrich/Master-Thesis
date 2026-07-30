#!/usr/bin/env python3
"""
Stage 1e — Generate a gpt-image-2 generator cell via the OpenAI Batch API,
reusing an existing cell's metadata and shared prompt text

Materializes a full 12-class x 100-images/class generator cell for the
synthetic-model-comparison experiment
(docs/synthetic-model-comparison/01_experiment-design.md), analogous to
scripts/synthetic_model_comparison/1d-generate_images_new_generator.py but
for OpenAI's gpt-image-2 (docs/synthetic-model-comparison/03_api-models-landscape-and-pricing.md
§0/§2: gpt-image-2 low + medium are two of the five decided API
generators). Reads per-image class/band/shot_type/distance/lighting/
occlusion/pose/environment metadata from an existing cell's index.jsonl and
prompt text from each record's `dest_prompt_file` (the shared
data/synthetic_model_comparison/train/prompts_full/<slug>/<NNN>.txt
location).

Quality tier (low/medium/high) is a distinct axis from the model name per
doc 01, so each tier gets its own generator-cell directory:
train/gpt-image-2-<quality>/<prompt-regime>/. Only the `full` prompt regime
exists today (`compressed` prompts haven't been authored for any model
yet, per docs/synthetic-model-comparison/05_prompt-strategy-and-length-limits.md
§6) — passing --prompt-regime compressed will fail fast with a clear error
rather than silently reusing full-regime prompts.

Size defaults to 1024x768 (4:3, matching every other generator cell's fixed
aspect ratio per doc 01 §3) — satisfies gpt-image-2's constraints (divisible
by 16, aspect <=3:1, 655,360-8,294,400 total pixels; confirmed via a live
test call).

Because batch jobs can take up to 24 hours, the script is split into modes:

  submit   (default) — build JSONL, upload, create batch job, save state, exit
  status   — print status of the last submitted job (or --job-name)
  retrieve — download completed results, decode images, write the cell's
             index.jsonl
  run      — submit + poll + retrieve in one blocking call (good for
             smoke tests / small subsets)

Usage:
    # Cheap smoke test first:
    uv run python scripts/synthetic_model_comparison/1e-generate_images_openai.py \\
        --quality low --mode run --classes lion --limit 3

    # Full async workflow:
    uv run python scripts/synthetic_model_comparison/1e-generate_images_openai.py --quality low --mode submit
    # ... wait ...
    uv run python scripts/synthetic_model_comparison/1e-generate_images_openai.py --quality low --mode status
    uv run python scripts/synthetic_model_comparison/1e-generate_images_openai.py --quality low --mode retrieve

Requirements:
    pip install openai pillow python-dotenv
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import openai
except ImportError:
    sys.exit("Error: openai is not installed.\nRun: pip install openai")

import tiktoken
from dotenv import load_dotenv
from PIL import Image
import io
import os

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_ROOT = REPO_ROOT / "data" / "synthetic_model_comparison" / "train"

# Shared credentials with the production pipeline — not duplicated per experiment.
ENV_PATH = REPO_ROOT / "scripts" / "synthetic" / ".env"

MODEL = "gpt-image-2"
DEFAULT_SIZE = "1024x768"
POLL_INTERVAL = 60  # seconds

# The org's "enqueued tokens" cap for gpt-image-2 batches is 1,000,000 — a
# single ~1200-request batch of these ~1,300-word (avg ~2,600-token) prompts
# blows straight through that (observed empirically: a same-size submission
# failed instantly with token_limit_exceeded). Chunk submissions to stay
# safely under it; --mode submit only ever sends the next chunk that fits.
TOKEN_ENCODING = "o200k_base"
DEFAULT_MAX_TOKENS_PER_BATCH = 700_000

# Approximate direct (non-batch) per-image prices at 1024x1024, per
# docs/synthetic-model-comparison/03_api-models-landscape-and-pricing.md §2 —
# for cost-awareness printouts only; actual billing applies the Batch -50%
# discount automatically and may vary slightly with --size.
APPROX_DIRECT_PRICE = {"low": 0.006, "medium": 0.053, "high": 0.211}

TERMINAL_STATES = frozenset({"completed", "failed", "expired", "cancelled"})

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_source_index(source_generator: str, prompt_regime: str) -> list[dict]:
    path = TRAIN_ROOT / source_generator / prompt_regime / "index.jsonl"
    if not path.exists():
        sys.exit(f"Error: {path} not found (--source-generator cell must already exist).")
    records = []
    with open(path, encoding="utf-8") as f:
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

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def slug_from_filename(filename: str) -> str:
    """'d_plains_zebra_001.png' -> 'plains_zebra' (band prefix and NNN suffix stripped)."""
    parts = filename.rsplit(".", 1)[0].split("_")
    return "_".join(parts[1:-1])


def image_output_path(images_dir: Path, rec: dict) -> Path:
    slug = slug_from_filename(rec["filename"])
    return images_dir / slug / rec["filename"]


def new_index_entry(rec: dict, images_dir: Path) -> dict:
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
        "dest_prompt_file": rec["dest_prompt_file"],
    }

# ---------------------------------------------------------------------------
# State file
# ---------------------------------------------------------------------------

def save_state(state_path: Path, batch_id: str, input_file_id: str, n_requests: int) -> None:
    state = {
        "batch_id": batch_id,
        "input_file_id": input_file_id,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "n_requests": n_requests,
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def load_state(state_path: Path) -> dict:
    if not state_path.exists():
        sys.exit(f"Error: no state file found at {state_path}.\nRun --mode submit first.")
    with open(state_path, encoding="utf-8") as f:
        return json.load(f)

# ---------------------------------------------------------------------------
# Build batch requests
# ---------------------------------------------------------------------------

# OpenAI hard-caps the images/generations prompt at 32,000 characters (per
# docs/synthetic-model-comparison/05_prompt-strategy-and-length-limits.md
# §1's "OpenAI 32k chars" row). A handful of species have Wikipedia articles
# long enough to push the shared full-regime template over that (observed:
# every red_fox/*.txt prompt is ~33.3-33.5k chars). Per doc 05 §2, the `full`
# regime is defined as "each model at its *maximum usable* prompt" — so
# trimming just enough of the free-text SPECIES DESCRIPTION / NATURAL
# BEHAVIOR AND HABITAT block (the two Wikipedia-sourced sections; everything
# else, especially SCENE SPECIFICATION and CRITICAL REQUIREMENTS, is
# generation-critical and must stay intact) to fit OpenAI's actual capacity
# is exactly what that principle calls for. This only ever changes the
# in-memory request body sent to OpenAI — the shared prompts_full/*.txt
# files on disk (and every other model's request) are never modified.
OPENAI_MAX_PROMPT_CHARS = 32_000
_TRUNCATION_NOTICE = "\n[...truncated to fit OpenAI's 32,000-character prompt limit...]\n"


def truncate_for_openai(prompt_text: str, max_chars: int = OPENAI_MAX_PROMPT_CHARS) -> str:
    if len(prompt_text) <= max_chars:
        return prompt_text

    start = prompt_text.find("SPECIES DESCRIPTION:")
    end = prompt_text.find("SCENE SPECIFICATION:")
    if start == -1 or end == -1 or end <= start:
        # Template markers not found (shouldn't happen for our prompts) — fall
        # back to a plain tail truncation rather than crashing.
        return prompt_text[: max_chars - len(_TRUNCATION_NOTICE)] + _TRUNCATION_NOTICE

    overflow = len(prompt_text) - max_chars + len(_TRUNCATION_NOTICE)
    span_end = end - overflow
    if span_end <= start:
        span_end = start  # extreme case: drop the whole bulk section
    return prompt_text[:span_end] + _TRUNCATION_NOTICE + prompt_text[end:]


def build_batch_requests(
    records: list[dict], images_dir: Path, size: str, quality: str, force: bool
) -> list[dict]:
    requests_list = []
    missing_prompts = 0
    already_done = 0
    truncated = 0

    for rec in records:
        out_path = image_output_path(images_dir, rec)
        if not force and out_path.exists():
            already_done += 1
            continue

        prompt_path = REPO_ROOT / rec["dest_prompt_file"]
        if not prompt_path.exists():
            print(f"  WARN: prompt file missing: {prompt_path}", flush=True)
            missing_prompts += 1
            continue

        prompt_text = prompt_path.read_text(encoding="utf-8")
        if len(prompt_text) > OPENAI_MAX_PROMPT_CHARS:
            prompt_text = truncate_for_openai(prompt_text)
            truncated += 1

        requests_list.append({
            "custom_id": rec["filename"],
            "method": "POST",
            "url": "/v1/images/generations",
            "body": {
                "model": MODEL,
                "prompt": prompt_text,
                "size": size,
                "quality": quality,
                "n": 1,
            },
        })

    if already_done:
        print(f"  Skipped {already_done} already-generated images (use --force to regenerate)")
    if missing_prompts:
        print(f"  Skipped {missing_prompts} records with missing prompt files")
    if truncated:
        print(f"  Truncated {truncated} prompts over OpenAI's {OPENAI_MAX_PROMPT_CHARS:,}-char limit "
              f"(trimmed the Wikipedia-sourced description/behavior text only)")

    return requests_list


def chunk_by_token_budget(requests_list: list[dict], max_tokens: int) -> tuple[list[dict], int, int]:
    """Greedily take a prefix of requests_list whose summed prompt-token count
    stays under max_tokens. Returns (chunk, chunk_tokens, n_remaining_after)."""
    enc = tiktoken.get_encoding(TOKEN_ENCODING)
    chunk: list[dict] = []
    total = 0
    for i, req in enumerate(requests_list):
        n = len(enc.encode(req["body"]["prompt"]))
        if chunk and total + n > max_tokens:
            return chunk, total, len(requests_list) - i
        chunk.append(req)
        total += n
    return chunk, total, 0

# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------

def submit_batch(
    client: "openai.OpenAI",
    requests_list: list[dict],
    display_name: str,
    batch_jsonl: Path,
    state_path: Path,
) -> object:
    print(f"Writing {len(requests_list)} requests to {batch_jsonl.relative_to(REPO_ROOT)} ...", flush=True)
    batch_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(batch_jsonl, "w", encoding="utf-8") as f:
        for req in requests_list:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")

    print("Uploading JSONL to Files API ...", flush=True)
    with open(batch_jsonl, "rb") as fh:
        uploaded = client.files.create(file=fh, purpose="batch")
    print(f"  Uploaded: {uploaded.id}", flush=True)

    print(f"Creating batch job (model={MODEL}) ...", flush=True)
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/images/generations",
        completion_window="24h",
        metadata={"display_name": display_name},
    )
    print(f"  Batch created: {batch.id}", flush=True)

    save_state(state_path, batch.id, uploaded.id, len(requests_list))
    print(f"  State saved to {state_path.relative_to(REPO_ROOT)}", flush=True)

    return batch

# ---------------------------------------------------------------------------
# Poll
# ---------------------------------------------------------------------------

def poll_job(client: "openai.OpenAI", batch_id: str, interval: int) -> object:
    print(f"Polling batch {batch_id} every {interval}s ...", flush=True)
    while True:
        batch = client.batches.retrieve(batch_id)
        if batch.status in TERMINAL_STATES:
            print(f"  Final status: {batch.status}", flush=True)
            return batch
        counts = batch.request_counts
        progress = f" ({counts.completed}/{counts.total} done, {counts.failed} failed)" if counts else ""
        print(f"  {batch.status}{progress} — waiting {interval}s ...", flush=True)
        time.sleep(interval)

# ---------------------------------------------------------------------------
# Retrieve
# ---------------------------------------------------------------------------

def retrieve_results(
    client: "openai.OpenAI",
    batch: object,
    all_source_records: list[dict],
    images_dir: Path,
    index_path: Path,
    batch_output_jsonl: Path,
    batch_error_jsonl: Path,
) -> None:
    if batch.status != "completed":
        sys.exit(f"Error: batch is in status {batch.status}, not completed.\nCannot retrieve results.")

    rec_by_filename = {rec["filename"]: rec for rec in all_source_records}
    index_records = load_index(index_path)
    indexed_filenames = {rec["filename"] for rec in index_records}

    generated = 0
    failed = 0
    skipped = 0

    if batch.output_file_id:
        print(f"Downloading output file {batch.output_file_id} -> {batch_output_jsonl.relative_to(REPO_ROOT)} ...", flush=True)
        content = client.files.content(batch.output_file_id)
        batch_output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        content.write_to_file(batch_output_jsonl)

        with open(batch_output_jsonl, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parsed = json.loads(line)
                key = parsed.get("custom_id", "")

                if key not in rec_by_filename:
                    print(f"  WARN: custom_id not in source metadata: {key}", flush=True)
                    skipped += 1
                    continue
                rec = rec_by_filename[key]

                error = parsed.get("error")
                response = parsed.get("response") or {}
                if error or response.get("status_code") != 200:
                    print(f"  FAILED  {key}: {error or response}", flush=True)
                    failed += 1
                    continue

                try:
                    b64 = response["body"]["data"][0]["b64_json"]
                except (KeyError, IndexError, TypeError):
                    print(f"  NO IMAGE  {key}", flush=True)
                    failed += 1
                    continue

                img_bytes = base64.b64decode(b64)
                out_path = image_output_path(images_dir, rec)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                Image.open(io.BytesIO(img_bytes)).save(out_path, "PNG")
                generated += 1

                if key in indexed_filenames:
                    index_records[:] = [e for e in index_records if e["filename"] != key]
                index_records.append(new_index_entry(rec, images_dir))
                indexed_filenames.add(key)

    if batch.error_file_id:
        print(f"Downloading error file {batch.error_file_id} -> {batch_error_jsonl.relative_to(REPO_ROOT)} ...", flush=True)
        client.files.content(batch.error_file_id).write_to_file(batch_error_jsonl)
        print(f"  See {batch_error_jsonl.relative_to(REPO_ROOT)} for per-request error details.")

    save_index(index_path, index_records)
    print(
        f"\nDone: {generated} generated, {failed} failed, {skipped} skipped\n"
        f"Index updated: {index_path.relative_to(REPO_ROOT)} ({len(index_records)} total records)",
        flush=True,
    )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--quality",
        required=True,
        choices=["low", "medium", "high"],
        help="gpt-image-2 quality tier — also determines the generator-cell directory name.",
    )
    parser.add_argument(
        "--size",
        default=DEFAULT_SIZE,
        help=f"gpt-image-2 'WxH' size, divisible by 16 (default: {DEFAULT_SIZE}).",
    )
    parser.add_argument(
        "--source-generator",
        default="gemini-3.1-flash-image-preview",
        dest="source_generator",
        help="Existing cell to read per-image metadata + prompt paths from "
             "(default: the incumbent, gemini-3.1-flash-image-preview).",
    )
    parser.add_argument(
        "--prompt-regime",
        default="full",
        dest="prompt_regime",
        choices=["full", "compressed"],
        help="Prompt regime (default: full — 'compressed' prompts don't exist yet for any model).",
    )
    parser.add_argument(
        "--mode",
        choices=["submit", "status", "retrieve", "run"],
        default="submit",
        help="Operation mode (default: submit).",
    )
    parser.add_argument(
        "--classes",
        default=None,
        help="Comma-separated class common names to include (default: all 12).",
    )
    parser.add_argument(
        "--job-name",
        default=None,
        dest="job_name",
        help="Batch id to use for status/retrieve (overrides state file).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Include already-generated images in the batch (regenerate them).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Cap the batch to the first N pending requests (for cheap smoke tests).",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=POLL_INTERVAL,
        dest="poll_interval",
        help=f"Seconds between status checks in run mode (default: {POLL_INTERVAL}).",
    )
    parser.add_argument(
        "--max-tokens-per-batch",
        type=int,
        default=DEFAULT_MAX_TOKENS_PER_BATCH,
        dest="max_tokens_per_batch",
        help=f"Cap a single batch's summed prompt tokens to stay under the org's "
             f"1,000,000 enqueued-token limit for gpt-image-2 (default: {DEFAULT_MAX_TOKENS_PER_BATCH}).",
    )
    args = parser.parse_args()

    if args.prompt_regime == "compressed":
        sys.exit(
            "Error: no 'compressed' prompts exist yet for any generator "
            "(docs/synthetic-model-comparison/05_prompt-strategy-and-length-limits.md §6). "
            "Build them first; this script only consumes existing prompt files, it "
            "does not author new ones."
        )

    generator_slug = f"gpt-image-2-{args.quality}"
    cell_dir = TRAIN_ROOT / generator_slug / args.prompt_regime
    images_dir = cell_dir / "images"
    index_path = cell_dir / "index.jsonl"
    batch_jsonl = cell_dir / "openai_batch_input.jsonl"
    state_path = cell_dir / "openai_batch_state.json"
    batch_output_jsonl = cell_dir / "openai_batch_output.jsonl"
    batch_error_jsonl = cell_dir / "openai_batch_error.jsonl"

    load_dotenv(ENV_PATH)
    api_key = os.getenv("OPEN_AI_API_KEY", "").strip()
    if not api_key:
        sys.exit(f"Error: OPEN_AI_API_KEY is not set.\nAdd it to {ENV_PATH} or set it as an environment variable.")

    client = openai.OpenAI(api_key=api_key)

    # --- status mode (no metadata needed) ---
    if args.mode == "status":
        batch_id = args.job_name or load_state(state_path)["batch_id"]
        batch = client.batches.retrieve(batch_id)
        print(f"Batch  : {batch.id}")
        print(f"Status : {batch.status}")
        if batch.request_counts:
            c = batch.request_counts
            print(f"Counts : {c.completed}/{c.total} completed, {c.failed} failed")
        if batch.errors:
            print(f"Errors : {batch.errors}")
        return

    all_source_records = load_source_index(args.source_generator, args.prompt_regime)
    print(f"Loaded {len(all_source_records)} records from "
          f"{args.source_generator}/{args.prompt_regime}/index.jsonl")

    records = all_source_records
    if args.classes:
        requested = {c.strip().lower() for c in args.classes.split(",")}
        records = [r for r in all_source_records if r["class"].lower() in requested]
        not_found = requested - {r["class"].lower() for r in records}
        if not_found:
            print(f"Warning: class(es) not found: {', '.join(sorted(not_found))}")
        print(f"Filtered to {len(records)} records for {len(requested)} class(es)")

    if args.mode == "retrieve":
        batch_id = args.job_name or load_state(state_path)["batch_id"]
        batch = client.batches.retrieve(batch_id)
        retrieve_results(client, batch, all_source_records, images_dir, index_path, batch_output_jsonl, batch_error_jsonl)
        return

    # --- submit / run ---
    requests_list = build_batch_requests(records, images_dir, args.size, args.quality, force=args.force)
    if args.limit is not None:
        requests_list = requests_list[:args.limit]
        print(f"--limit {args.limit}: capped batch to {len(requests_list)} requests")
    if not requests_list:
        print("Nothing to submit — all images already generated.")
        return

    total_pending = len(requests_list)
    requests_list, chunk_tokens, n_remaining = chunk_by_token_budget(requests_list, args.max_tokens_per_batch)
    if n_remaining:
        print(
            f"Token budget: this batch uses {chunk_tokens:,} of the "
            f"{args.max_tokens_per_batch:,}-token cap ({len(requests_list)}/{total_pending} pending "
            f"requests) — {n_remaining} more will remain pending after this batch; "
            f"re-run the same command once it completes to submit the next chunk."
        )
    else:
        print(f"Token budget: this batch uses {chunk_tokens:,} of the {args.max_tokens_per_batch:,}-token cap "
              f"({len(requests_list)}/{total_pending} pending requests — all of them).")

    direct_price = APPROX_DIRECT_PRICE.get(args.quality)
    cost_note = ""
    if direct_price is not None:
        direct_total = direct_price * len(requests_list)
        cost_note = f"  (approx ${direct_total:.2f} direct / ${direct_total / 2:.2f} batch-discounted)"

    print(f"\nGenerator : {generator_slug}")
    print(f"Size      : {args.size}")
    print(f"Images    : {len(requests_list)}{cost_note}\n")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    display_name = f"model-comparison-{generator_slug}-{timestamp}"

    batch = submit_batch(client, requests_list, display_name, batch_jsonl, state_path)

    if args.mode == "submit":
        print(
            f"\nBatch job submitted: {batch.id}\n"
            f"Check status with:   uv run python scripts/synthetic_model_comparison/1e-generate_images_openai.py --quality {args.quality} --mode status\n"
            f"Retrieve results with: uv run python scripts/synthetic_model_comparison/1e-generate_images_openai.py --quality {args.quality} --mode retrieve"
        )
        return

    # run mode: poll then retrieve
    batch = poll_job(client, batch.id, args.poll_interval)
    retrieve_results(client, batch, all_source_records, images_dir, index_path, batch_output_jsonl, batch_error_jsonl)


if __name__ == "__main__":
    main()
