#!/usr/bin/env python3
"""
Stage 1c — Generate images for the 5 fresh incumbent-cell classes via the
Gemini Batch API, and merge them into the cell's index.jsonl

Reads reports/model_comparison_fresh_prompt_metadata.jsonl (written by
1b-generate_prompts_fresh.py) and submits a Gemini Batch API job for all
pending images — the Batch API is half the price of direct calls
(docs/synthetic-model-comparison/03_api-models-landscape-and-pricing.md §4:
"use Batch for all API generation to halve cost"). Adapted from
scripts/synthetic/2-generate_test_images.py's batch submit/status/retrieve
workflow, copied here (not imported) to keep this experiment's code
independent of the production pipeline.

Because batch jobs can take up to 24 hours, the script is split into modes:

  submit   (default) — build JSONL, upload, create batch job, save state, exit
  status   — print status of the last submitted job (or --job-name)
  retrieve — download completed results, decode images, merge into the
             cell's index.jsonl
  run      — submit + poll + retrieve in one blocking call (good for
             smoke tests / small subsets)

Usage:
    # Full async workflow (recommended for the ~497 remaining images):
    uv run python scripts/synthetic_model_comparison/1c-generate_images_fresh.py --mode submit
    # ... wait (minutes to hours) ...
    uv run python scripts/synthetic_model_comparison/1c-generate_images_fresh.py --mode status
    uv run python scripts/synthetic_model_comparison/1c-generate_images_fresh.py --mode retrieve

    # Smoke test (blocking, small subset):
    uv run python scripts/synthetic_model_comparison/1c-generate_images_fresh.py --mode run --classes lion

    # Resume from a specific job:
    uv run python scripts/synthetic_model_comparison/1c-generate_images_fresh.py --mode retrieve --job-name batches/123

Requirements:
    pip install google-genai httpx pillow python-dotenv
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import httpx
except ImportError:
    sys.exit("Error: httpx is not installed.\nRun: pip install httpx")

try:
    from google import genai
    from google.genai import types
except ImportError:
    sys.exit("Error: google-genai is not installed.\nRun: pip install google-genai")

from dotenv import load_dotenv
from PIL import Image

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
METADATA_PATH = REPO_ROOT / "reports" / "model_comparison_fresh_prompt_metadata.jsonl"
CELL_DIR = REPO_ROOT / "data" / "synthetic_model_comparison" / "train" / "gemini-3.1-flash-image-preview" / "full"
IMAGES_DIR = CELL_DIR / "images"
INDEX_JSONL = CELL_DIR / "index.jsonl"
BATCH_JSONL = CELL_DIR / "fresh_batch_input.jsonl"
STATE_FILE = CELL_DIR / "fresh_batch_state.json"
BATCH_OUTPUT_JSONL = CELL_DIR / "fresh_batch_output.jsonl"

# Shared credentials with the production pipeline — not duplicated per experiment.
ENV_PATH = REPO_ROOT / "scripts" / "synthetic" / ".env"

MODEL = "gemini-3.1-flash-image-preview"
IMAGE_ASPECT_RATIO = "4:3"
IMAGE_SIZE = "512"
POLL_INTERVAL = 60  # seconds

TERMINAL_STATES = frozenset({
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
})

_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_metadata() -> list[dict]:
    if not METADATA_PATH.exists():
        sys.exit(f"Error: {METADATA_PATH} not found. Run 1b-generate_prompts_fresh.py first.")
    records = []
    with open(METADATA_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_index() -> list[dict]:
    if not INDEX_JSONL.exists():
        return []
    records = []
    with open(INDEX_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_index(records: list[dict]) -> None:
    INDEX_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_JSONL, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def slug_from_filename(filename: str) -> str:
    """'d_plains_zebra_001.png' -> 'plains_zebra' (band prefix and NNN suffix stripped)."""
    parts = filename.rsplit(".", 1)[0].split("_")
    return "_".join(parts[1:-1])


def image_output_path(rec: dict) -> Path:
    slug = slug_from_filename(rec["filename"])
    return IMAGES_DIR / slug / rec["filename"]


def index_entry(rec: dict) -> dict:
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
        "file_name": str(image_output_path(rec).relative_to(REPO_ROOT)),
        "dest_prompt_file": rec["prompt_file"],
    }

# ---------------------------------------------------------------------------
# State file
# ---------------------------------------------------------------------------

def save_state(job_name: str, input_file: str, n_requests: int) -> None:
    state = {
        "job_name": job_name,
        "input_file": input_file,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "n_requests": n_requests,
    }
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def load_state() -> dict:
    if not STATE_FILE.exists():
        sys.exit(f"Error: no state file found at {STATE_FILE}.\nRun --mode submit first.")
    with open(STATE_FILE, encoding="utf-8") as f:
        return json.load(f)

# ---------------------------------------------------------------------------
# Build batch requests
# ---------------------------------------------------------------------------

def build_batch_requests(records: list[dict], force: bool) -> list[dict]:
    requests_list = []
    missing_prompts = 0
    already_done = 0

    for rec in records:
        out_path = image_output_path(rec)
        if not force and out_path.exists():
            already_done += 1
            continue

        prompt_path = REPO_ROOT / rec["prompt_file"]
        if not prompt_path.exists():
            print(f"  WARN: prompt file missing: {prompt_path}", flush=True)
            missing_prompts += 1
            continue

        prompt_text = prompt_path.read_text(encoding="utf-8")
        requests_list.append({
            "key": rec["filename"],
            "request": {
                "contents": [{
                    "parts": [{"text": prompt_text}],
                    "role": "user",
                }],
                "generation_config": {
                    "responseModalities": ["IMAGE", "TEXT"],
                    "imageConfig": {
                        "aspectRatio": IMAGE_ASPECT_RATIO,
                        "imageSize": IMAGE_SIZE,
                    },
                },
            },
        })

    if already_done:
        print(f"  Skipped {already_done} already-generated images (use --force to regenerate)")
    if missing_prompts:
        print(f"  Skipped {missing_prompts} records with missing prompt files")

    return requests_list

# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------

def submit_batch(client: "genai.Client", requests_list: list[dict], display_name: str) -> object:
    print(f"Writing {len(requests_list)} requests to {BATCH_JSONL.relative_to(REPO_ROOT)} ...", flush=True)
    BATCH_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(BATCH_JSONL, "w", encoding="utf-8") as f:
        for req in requests_list:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")

    print("Uploading JSONL to File API ...", flush=True)
    uploaded = client.files.upload(
        file=str(BATCH_JSONL),
        config=types.UploadFileConfig(display_name=display_name, mime_type="jsonl"),
    )
    print(f"  Uploaded: {uploaded.name}", flush=True)

    print(f"Creating batch job (model={MODEL}) ...", flush=True)
    job = client.batches.create(
        model=MODEL,
        src=uploaded.name,
        config={"display_name": display_name},
    )
    print(f"  Job created: {job.name}", flush=True)

    save_state(job.name, uploaded.name, len(requests_list))
    print(f"  State saved to {STATE_FILE.relative_to(REPO_ROOT)}", flush=True)

    return job

# ---------------------------------------------------------------------------
# Poll
# ---------------------------------------------------------------------------

def poll_job(client: "genai.Client", job_name: str, interval: int) -> object:
    print(f"Polling job {job_name} every {interval}s ...", flush=True)
    while True:
        job = client.batches.get(name=job_name)
        state = job.state.name
        if state in TERMINAL_STATES:
            print(f"  Final state: {state}", flush=True)
            return job
        print(f"  {state} — waiting {interval}s ...", flush=True)
        time.sleep(interval)

# ---------------------------------------------------------------------------
# Retrieve
# ---------------------------------------------------------------------------

def _stream_file_to_disk(result_file: str, dest: Path, api_key: str) -> None:
    """Download a Files API result to disk in chunks, avoiding loading it all into memory."""
    name = result_file.removeprefix("files/")
    url = f"{_GEMINI_BASE}/files/{name}:download?alt=media"
    headers = {"x-goog-api-key": api_key}
    dest.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=300) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_bytes(chunk_size=65536):
                f.write(chunk)
                written += len(chunk)
    print(f"  Downloaded {written / 1e6:.1f} MB", flush=True)


def retrieve_results(
    job: object,
    all_planned: list[dict],
    index_records: list[dict],
    indexed_filenames: set[str],
    api_key: str,
) -> None:
    if job.state.name != "JOB_STATE_SUCCEEDED":
        sys.exit(
            f"Error: job is in state {job.state.name}, not JOB_STATE_SUCCEEDED.\n"
            "Cannot retrieve results."
        )

    result_file = job.dest.file_name
    print(f"Streaming results from {result_file} -> {BATCH_OUTPUT_JSONL.relative_to(REPO_ROOT)} ...", flush=True)
    _stream_file_to_disk(result_file, BATCH_OUTPUT_JSONL, api_key)

    rec_by_filename = {rec["filename"]: rec for rec in all_planned}

    generated = 0
    failed = 0
    skipped = 0

    with open(BATCH_OUTPUT_JSONL, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parsed = json.loads(line)
            key = parsed.get("key", "")

            if key not in rec_by_filename:
                print(f"  WARN: key not in planned metadata: {key}", flush=True)
                skipped += 1
                continue
            rec = rec_by_filename[key]

            if "error" in parsed:
                print(f"  FAILED  {key}: {parsed['error']}", flush=True)
                failed += 1
                continue

            img_bytes = None
            try:
                parts = parsed["response"]["candidates"][0]["content"]["parts"]
                for part in parts:
                    if "inlineData" in part:
                        img_bytes = base64.b64decode(part["inlineData"]["data"])
                        break
            except (KeyError, IndexError):
                pass

            if img_bytes is None:
                print(f"  NO IMAGE  {key}", flush=True)
                failed += 1
                continue

            out_path = image_output_path(rec)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            Image.open(io.BytesIO(img_bytes)).save(out_path, "PNG")
            generated += 1

            if key in indexed_filenames:
                index_records[:] = [e for e in index_records if e["filename"] != key]
            index_records.append(index_entry(rec))
            indexed_filenames.add(key)

    save_index(index_records)
    print(
        f"\nDone: {generated} generated, {failed} failed, {skipped} skipped\n"
        f"Index updated: {INDEX_JSONL.relative_to(REPO_ROOT)} ({len(index_records)} total records)",
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
        "--mode",
        choices=["submit", "status", "retrieve", "run"],
        default="submit",
        help="Operation mode (default: submit).",
    )
    parser.add_argument(
        "--classes",
        default=None,
        help="Comma-separated class common names to include (default: all 5).",
    )
    parser.add_argument(
        "--job-name",
        default=None,
        dest="job_name",
        help="Batch job name to use for status/retrieve (overrides state file).",
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
    args = parser.parse_args()

    load_dotenv(ENV_PATH)
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        sys.exit(f"Error: GEMINI_API_KEY is not set.\nAdd it to {ENV_PATH} or set it as an environment variable.")

    client = genai.Client(api_key=api_key)

    # --- status mode (no metadata/index needed) ---
    if args.mode == "status":
        job_name = args.job_name or load_state()["job_name"]
        job = client.batches.get(name=job_name)
        print(f"Job  : {job.name}")
        print(f"State: {job.state.name}")
        if hasattr(job, "create_time") and job.create_time:
            print(f"Created : {job.create_time}")
        if job.state.name == "JOB_STATE_FAILED" and job.error:
            print(f"Error: {job.error}")
        return

    all_planned = load_metadata()
    print(f"Loaded {len(all_planned)} planned records from {METADATA_PATH.relative_to(REPO_ROOT)}")

    records = all_planned
    if args.classes:
        requested = {c.strip().lower() for c in args.classes.split(",")}
        records = [r for r in all_planned if r["class"].lower() in requested]
        not_found = requested - {r["class"].lower() for r in records}
        if not_found:
            print(f"Warning: class(es) not found: {', '.join(sorted(not_found))}")
        print(f"Filtered to {len(records)} records for {len(requested)} class(es)")

    index_records = load_index()
    indexed_filenames = {rec["filename"] for rec in index_records}
    print(f"Loaded {len(index_records)} existing records from {INDEX_JSONL.relative_to(REPO_ROOT)}")

    if args.mode == "retrieve":
        job_name = args.job_name or load_state()["job_name"]
        job = client.batches.get(name=job_name)
        retrieve_results(job, all_planned, index_records, indexed_filenames, api_key)
        return

    # --- submit / run ---
    requests_list = build_batch_requests(records, force=args.force)
    if args.limit is not None:
        requests_list = requests_list[:args.limit]
        print(f"--limit {args.limit}: capped batch to {len(requests_list)} requests")
    if not requests_list:
        print("Nothing to submit — all images already generated.")
        return

    print(f"\nModel : {MODEL}")
    print(f"Images: {len(requests_list)}\n")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    display_name = f"model-comparison-fresh-images-{timestamp}"

    job = submit_batch(client, requests_list, display_name)

    if args.mode == "submit":
        print(
            f"\nBatch job submitted: {job.name}\n"
            "Check status with:   uv run python scripts/synthetic_model_comparison/1c-generate_images_fresh.py --mode status\n"
            "Retrieve results with: uv run python scripts/synthetic_model_comparison/1c-generate_images_fresh.py --mode retrieve"
        )
        return

    # run mode: poll then retrieve
    job = poll_job(client, job.name, args.poll_interval)
    retrieve_results(job, all_planned, index_records, indexed_filenames, api_key)


if __name__ == "__main__":
    main()
