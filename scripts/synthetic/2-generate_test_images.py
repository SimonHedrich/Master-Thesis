"""
Generate synthetic wildlife test images using the Gemini Batch API.

Reads data/synthetic/test_index.jsonl (produced by 1-generate_test_image_list.py)
and submits a batch job for all pending images.  Results are saved to
data/synthetic/images/test/{class_slug}/.

Because batch jobs can take up to 24 hours, the script is split into modes:

  submit   (default) — build JSONL, upload, create batch job, save state, exit
  status   — print status of the last submitted job (or --job-name)
  retrieve — download completed results, decode images, update test_index.jsonl
  run      — submit + poll + retrieve in one blocking call (good for smoke tests)

Usage:
    # Full async workflow (recommended for 11,250 images):
    uv run python scripts/synthetic/2-generate_test_images.py --mode submit
    # ... wait up to 24 h ...
    uv run python scripts/synthetic/2-generate_test_images.py --mode status
    uv run python scripts/synthetic/2-generate_test_images.py --mode retrieve

    # Smoke test (blocking, small subset):
    uv run python scripts/synthetic/2-generate_test_images.py --mode run --classes walrus,kinkajou

    # Resume from a specific job:
    uv run python scripts/synthetic/2-generate_test_images.py --mode retrieve --job-name batches/123

Requirements:
    pip install google-genai pillow python-dotenv
"""

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
# Paths & constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"
INDEX_JSONL   = SYNTHETIC_DIR / "test_index.jsonl"
IMAGES_DIR    = SYNTHETIC_DIR / "images" / "test"
BATCH_JSONL   = SYNTHETIC_DIR / "test_batch_input.jsonl"
STATE_FILE    = SYNTHETIC_DIR / "test_batch_state.json"

MODEL              = "gemini-3.1-flash-image-preview"
IMAGE_ASPECT_RATIO = "4:3"
IMAGE_SIZE         = "512"
POLL_INTERVAL      = 60  # seconds

TERMINAL_STATES = frozenset({
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
})

# ---------------------------------------------------------------------------
# Index I/O
# ---------------------------------------------------------------------------

def load_index() -> list[dict]:
    if not INDEX_JSONL.exists():
        sys.exit(
            f"Error: {INDEX_JSONL} not found.\n"
            "Run 1-generate_test_image_list.py first."
        )
    records = []
    with open(INDEX_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_index(records: list[dict]) -> None:
    with open(INDEX_JSONL, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def image_output_path(record: dict) -> Path:
    # slug is the directory name in prompt_file, e.g. "walrus" from "test_prompts/walrus/001.txt"
    slug = Path(record["prompt_file"]).parent.name
    return IMAGES_DIR / slug / record["filename"]


def prompt_path(record: dict) -> Path:
    return SYNTHETIC_DIR / record["prompt_file"]

# ---------------------------------------------------------------------------
# State file
# ---------------------------------------------------------------------------

def save_state(job_name: str, input_file: str, n_requests: int) -> None:
    state = {
        "job_name":     job_name,
        "input_file":   input_file,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "n_requests":   n_requests,
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
    requests = []
    missing_prompts = 0
    already_done = 0

    for rec in records:
        out_path = image_output_path(rec)
        if not force and out_path.exists():
            already_done += 1
            continue

        p_path = prompt_path(rec)
        if not p_path.exists():
            print(f"  WARN: prompt file missing: {p_path}", flush=True)
            missing_prompts += 1
            continue

        prompt_text = p_path.read_text(encoding="utf-8")
        requests.append({
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

    return requests

# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------

def submit_batch(client: genai.Client, requests_list: list[dict], display_name: str) -> object:
    print(f"Writing {len(requests_list)} requests to {BATCH_JSONL} ...", flush=True)
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
    print(f"  State saved to {STATE_FILE}", flush=True)

    return job

# ---------------------------------------------------------------------------
# Poll
# ---------------------------------------------------------------------------

def poll_job(client: genai.Client, job_name: str, interval: int) -> object:
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

BATCH_OUTPUT_JSONL = SYNTHETIC_DIR / "test_batch_output.jsonl"

_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"


def _stream_file_to_disk(result_file: str, dest: Path, api_key: str) -> None:
    """Download a Files API result to disk in chunks, avoiding loading it all into memory."""
    # SDK strips "files/" prefix, then constructs "files/{name}:download?alt=media"
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


def retrieve_results(job: object, all_records: list[dict], api_key: str) -> None:
    if job.state.name != "JOB_STATE_SUCCEEDED":
        sys.exit(
            f"Error: job is in state {job.state.name}, not JOB_STATE_SUCCEEDED.\n"
            "Cannot retrieve results."
        )

    result_file = job.dest.file_name
    print(f"Streaming results from {result_file} → {BATCH_OUTPUT_JSONL} ...", flush=True)
    _stream_file_to_disk(result_file, BATCH_OUTPUT_JSONL, api_key)

    # Build a filename → record index map for fast status updates
    rec_index: dict[str, int] = {rec["filename"]: i for i, rec in enumerate(all_records)}

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

            if "error" in parsed:
                print(f"  FAILED  {key}: {parsed['error']}", flush=True)
                failed += 1
                if key in rec_index:
                    all_records[rec_index[key]]["status"] = "failed"
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
                if key in rec_index:
                    all_records[rec_index[key]]["status"] = "failed"
                continue

            if key not in rec_index:
                print(f"  WARN: key not in index: {key}", flush=True)
                skipped += 1
                continue

            rec = all_records[rec_index[key]]
            out_path = image_output_path(rec)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            Image.open(io.BytesIO(img_bytes)).save(out_path, "PNG")
            all_records[rec_index[key]]["status"] = "generated"
            generated += 1

    save_index(all_records)
    print(
        f"\nDone: {generated} generated, {failed} failed, {skipped} skipped\n"
        f"Index updated: {INDEX_JSONL}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic test images using the Gemini Batch API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
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
        help="Comma-separated class common names to include (default: all).",
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
        "--poll-interval",
        type=int,
        default=POLL_INTERVAL,
        dest="poll_interval",
        help=f"Seconds between status checks in run/poll modes (default: {POLL_INTERVAL}).",
    )
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / "scripts" / "synthetic" / ".env")
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        sys.exit(
            "Error: GEMINI_API_KEY is not set.\n"
            "Add it to scripts/synthetic/.env or set it as an environment variable."
        )

    client = genai.Client(api_key=api_key)

    # --- status mode (no index needed) ---
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

    # --- modes that need the index ---
    all_records = load_index()
    print(f"Loaded {len(all_records)} records from {INDEX_JSONL}")

    records = all_records
    if args.classes:
        requested = {c.strip().lower() for c in args.classes.split(",")}
        records = [r for r in all_records if r["class"].lower() in requested]
        not_found = requested - {r["class"].lower() for r in records}
        if not_found:
            print(f"Warning: class(es) not found in index: {', '.join(sorted(not_found))}")
        print(f"Filtered to {len(records)} records for {len(requested)} class(es)")

    if args.mode == "retrieve":
        job_name = args.job_name or load_state()["job_name"]
        job = client.batches.get(name=job_name)
        retrieve_results(job, all_records, api_key)
        return

    # --- submit / run ---
    requests_list = build_batch_requests(records, force=args.force)
    if not requests_list:
        print("Nothing to submit — all images already generated.")
        return

    print(f"\nModel : {MODEL}")
    print(f"Images: {len(requests_list)}\n")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    display_name = f"test-images-{timestamp}"

    job = submit_batch(client, requests_list, display_name)

    if args.mode == "submit":
        print(
            f"\nBatch job submitted: {job.name}\n"
            "Check status with:   python scripts/synthetic/2-generate_test_images.py --mode status\n"
            "Retrieve results with: python scripts/synthetic/2-generate_test_images.py --mode retrieve"
        )
        return

    # run mode: poll then retrieve
    job = poll_job(client, job.name, args.poll_interval)
    retrieve_results(job, all_records, api_key)


if __name__ == "__main__":
    main()
