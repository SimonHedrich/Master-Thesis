"""
One-time migration: back-fill `bokeh` field in data/synthetic/index.jsonl and update
the PHOTOGRAPHY STYLE sentence in all pending prompt .txt files to use the
binocular/sharp-background style instead of the telephoto/bokeh style.

Already-generated images (status="generated") are left untouched.

Usage:
    python scripts/synthetic/1b-migrate_bokeh.py --dry-run   # preview counts only
    python scripts/synthetic/1b-migrate_bokeh.py             # apply changes
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"
INDEX_JSONL = SYNTHETIC_DIR / "index.jsonl"

OLD_STYLE = (
    "Telephoto lens (400–600 mm equivalent), natural shallow depth of field "
    "with background softly blurred, authentic field conditions."
)

NEW_STYLE = (
    "Wildlife observation through optical binoculars (8–10× magnification), "
    "full-scene sharp focus, authentic field conditions. "
    "No background blur — the animal is integrated into its environment, "
    "not isolated against a softened background."
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Back-fill bokeh field in index.jsonl and update pending prompts.")
    parser.add_argument("--dry-run", action="store_true", help="Print counts without writing any files.")
    args = parser.parse_args()

    if not INDEX_JSONL.exists():
        sys.exit(f"Error: {INDEX_JSONL} not found")

    records = []
    with open(INDEX_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    n_generated = sum(1 for r in records if r.get("status") == "generated")
    n_pending = sum(1 for r in records if r.get("status") == "pending")
    print(f"Total records: {len(records)}  (generated={n_generated}, pending={n_pending})")

    prompt_updated = 0
    prompt_already_new = 0
    prompt_missing = 0

    for rec in records:
        is_generated = rec.get("status") == "generated"
        rec["bokeh"] = is_generated

        if is_generated:
            continue

        txt_path = SYNTHETIC_DIR / rec["prompt_file"]
        if not txt_path.exists():
            prompt_missing += 1
            continue

        text = txt_path.read_text(encoding="utf-8")
        if OLD_STYLE in text:
            if not args.dry_run:
                txt_path.write_text(text.replace(OLD_STYLE, NEW_STYLE, 1), encoding="utf-8")
            prompt_updated += 1
        elif NEW_STYLE in text:
            prompt_already_new += 1
        else:
            print(f"  WARNING: expected style string not found in {rec['prompt_file']}")

    print(f"Prompt files: {prompt_updated} updated, {prompt_already_new} already new style, {prompt_missing} missing")

    if not args.dry_run:
        with open(INDEX_JSONL, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Wrote {len(records)} records to {INDEX_JSONL}")
    else:
        print("Dry run — no files written.")


if __name__ == "__main__":
    main()
