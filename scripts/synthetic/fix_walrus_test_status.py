"""
Set walrus test image statuses from 'pending' to 'generated' in test_index.jsonl
so that 3-run_megadetector.py picks them up on the next run.

Usage:
    python scripts/synthetic/fix_walrus_test_status.py
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "synthetic"
TEST_INDEX_FILE = DATA_DIR / "test_index.jsonl"
IMAGES_DIR = DATA_DIR / "images"


def main():
    lines = TEST_INDEX_FILE.read_text().splitlines()
    updated, already_done, missing = 0, 0, 0
    new_lines = []

    for line in lines:
        if not line.strip():
            new_lines.append(line)
            continue
        record = json.loads(line)
        if record.get("class") == "walrus" and record.get("status") == "pending":
            slug = Path(record["prompt_file"]).parent.name
            img_path = IMAGES_DIR / "test" / slug / record["filename"]
            if img_path.exists():
                record["status"] = "generated"
                updated += 1
                print(f"  UPDATED  {img_path.relative_to(REPO_ROOT)}")
            else:
                missing += 1
                print(f"  MISSING  {img_path.relative_to(REPO_ROOT)} — image not on disk, leaving pending")
        elif record.get("class") == "walrus":
            already_done += 1
        new_lines.append(json.dumps(record, ensure_ascii=False))

    TEST_INDEX_FILE.write_text("\n".join(new_lines) + "\n")
    print(f"\n→ {updated} updated, {already_done} already generated, {missing} missing on disk")
    if updated:
        print("\nNext steps:")
        print("  python scripts/synthetic/3-run_megadetector.py --split test")
        print("  python scripts/synthetic/6-export_coco.py --split test")


if __name__ == "__main__":
    main()
