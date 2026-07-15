"""
One-time migration: convert all pending closeup_head/close shots to eye_level/medium
and add a fully-visible requirement to every pending prompt.

Already-generated images are left untouched.

Usage:
    uv run python scripts/synthetic/1c-migrate_closeup.py --dry-run   # preview counts only
    uv run python scripts/synthetic/1c-migrate_closeup.py             # apply changes
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"
INDEX_JSONL = SYNTHETIC_DIR / "index.jsonl"

OLD_ANGLE = "Tight frame on head and face; head fills approximately 60–70% of the frame"
NEW_ANGLE = "Camera at the same height as the animal's mid-body, horizontal perspective"

OLD_DISTANCE = "Very close — animal fills approximately 50–70% of frame (telephoto at ~20–80 m)"
NEW_DISTANCE = "Standard field view — animal fills approximately 20–40% of frame (~80–250 m telephoto)"

OLD_REQ_ENDING = (
    "5. The photograph must be suitable for training an automated species identification system."
)
NEW_REQ_ENDING = (
    "5. The photograph must be suitable for training an automated species identification system.\n"
    "6. The animal must be fully visible within the frame — no part of its body, head, or limbs "
    "should be cut off at the image edges."
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert pending closeup_head shots to eye_level/medium and add fully-visible requirement."
    )
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

    closeup_updated = 0
    closeup_already_done = 0
    req_updated = 0
    req_already_done = 0
    missing = 0

    for rec in records:
        if rec.get("status") != "pending":
            continue

        txt_path = SYNTHETIC_DIR / rec["prompt_file"]
        if not txt_path.exists():
            missing += 1
            continue

        text = txt_path.read_text(encoding="utf-8")
        changed = False

        # Step 1: convert closeup_head → eye_level for affected entries
        if rec.get("shot_type") == "closeup_head":
            if OLD_ANGLE in text and OLD_DISTANCE in text:
                text = text.replace(OLD_ANGLE, NEW_ANGLE, 1)
                text = text.replace(OLD_DISTANCE, NEW_DISTANCE, 1)
                rec["shot_type"] = "eye_level"
                rec["distance"] = "medium"
                changed = True
                closeup_updated += 1
            else:
                closeup_already_done += 1

        # Step 2: add fully-visible requirement to all pending prompts
        if OLD_REQ_ENDING in text:
            text = text.replace(OLD_REQ_ENDING, NEW_REQ_ENDING, 1)
            changed = True
            req_updated += 1
        elif NEW_REQ_ENDING in text:
            req_already_done += 1
        else:
            print(f"  WARNING: requirement ending not found in {rec['prompt_file']}")

        if changed and not args.dry_run:
            txt_path.write_text(text, encoding="utf-8")

    print(f"Closeup→eye_level: {closeup_updated} updated, {closeup_already_done} already done")
    print(f"Fully-visible req:  {req_updated} updated, {req_already_done} already done")
    if missing:
        print(f"Missing prompt files: {missing}")

    if not args.dry_run:
        with open(INDEX_JSONL, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Wrote {len(records)} records to {INDEX_JSONL}")
    else:
        print("Dry run — no files written.")


if __name__ == "__main__":
    main()
