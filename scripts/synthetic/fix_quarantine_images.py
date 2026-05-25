"""
Move defective generated images to quarantine, fix problematic prompts,
and reset their index.jsonl status to 'pending' so they get regenerated.

Usage:
    python scripts/synthetic/fix_quarantine_images.py
"""

import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "synthetic"
IMAGES_DIR = DATA_DIR / "images"
PROMPTS_DIR = DATA_DIR / "prompts"
INDEX_FILE = DATA_DIR / "index.jsonl"
QUARANTINE_DIR = IMAGES_DIR / "quarantine"

# ---------------------------------------------------------------------------
# Defective images: (band_subdir, class_slug, filename)
# ---------------------------------------------------------------------------
DEFECTIVE = [
    ("band_a", "african_civet",      "a_african_civet_165.png"),
    ("band_b", "african_wild_dog",   "b_african_wild_dog_032.png"),
    ("band_b", "african_wild_dog",   "b_african_wild_dog_098.png"),
    ("band_b", "american_mink",      "b_american_mink_096.png"),
    ("band_a", "black_backed_jackal","a_black_backed_jackal_024.png"),
    ("band_a", "brown_hyaena",       "a_brown_hyaena_034.png"),
    ("band_a", "clouded_leopard",    "a_clouded_leopard_024.png"),
    ("band_b", "gerenuk",            "b_gerenuk_033.png"),
    ("band_a", "maned_wolf",         "a_maned_wolf_157.png"),
    ("band_a", "striped_hyaena",     "a_striped_hyaena_023.png"),
]

DEFECTIVE_FILENAMES = {f for _, _, f in DEFECTIVE}

# ---------------------------------------------------------------------------
# Prompt patches: list of (prompt_path, old_line, new_line)
# ---------------------------------------------------------------------------
PROMPT_PATCHES = [
    (
        PROMPTS_DIR / "african_civet" / "165.txt",
        "The civet investigates a piece of carrion, using its front claws to hold the prey steady.",
        "The civet sniffs at bare, bleached bones and a dry skull on the ground — the fully desiccated skeletal remains of long-dead prey.",
    ),
    (
        PROMPTS_DIR / "black_backed_jackal" / "024.txt",
        "black-backed jackal actively foraging and feeding naturally on species-appropriate food in its primary habitat",
        "The jackal stands alert on the dry riverbed, ears pricked forward and nose raised, scanning the open terrain for danger.",
    ),
    (
        PROMPTS_DIR / "clouded_leopard" / "024.txt",
        "clouded leopard actively foraging and feeding naturally on species-appropriate food in its primary habitat",
        "The clouded leopard crouches low on a thick, moss-covered branch in a dense forest, tail draped downward, gaze fixed on the canopy below.",
    ),
    (
        PROMPTS_DIR / "striped_hyaena" / "023.txt",
        "striped hyaena actively foraging and feeding naturally on species-appropriate food in its primary habitat",
        "The striped hyaena stands motionless at the edge of a rocky canyon, dorsal mane erect, head raised and alert in the moonlit terrain.",
    ),
]


def move_images():
    print("=== Moving defective images to quarantine ===")
    moved, skipped = 0, 0
    for band, cls, fname in DEFECTIVE:
        src = IMAGES_DIR / band / cls / fname
        dst = QUARANTINE_DIR / band / cls / fname
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.move(str(src), str(dst))
            print(f"  MOVED   {src.relative_to(REPO_ROOT)}  →  {dst.relative_to(REPO_ROOT)}")
            moved += 1
        elif dst.exists():
            print(f"  ALREADY  {fname} already in quarantine, skipping source move")
            skipped += 1
        else:
            print(f"  MISSING  {src.relative_to(REPO_ROOT)} not found (skipped)")
            skipped += 1
    print(f"  → {moved} moved, {skipped} skipped\n")


def patch_prompts():
    print("=== Patching prompts ===")
    patched = 0
    for path, old, new in PROMPT_PATCHES:
        if not path.exists():
            print(f"  MISSING  {path.relative_to(REPO_ROOT)}")
            continue
        text = path.read_text()
        if old in text:
            path.write_text(text.replace(old, new, 1))
            print(f"  PATCHED  {path.relative_to(REPO_ROOT)}")
            patched += 1
        elif new in text:
            print(f"  ALREADY  {path.relative_to(REPO_ROOT)} already patched")
        else:
            print(f"  NOMATCH  {path.relative_to(REPO_ROOT)} — old text not found!")
    print(f"  → {patched} prompts patched\n")


def reset_index():
    print("=== Resetting index.jsonl status to 'pending' ===")
    lines = INDEX_FILE.read_text().splitlines()
    updated = 0
    new_lines = []
    for line in lines:
        if not line.strip():
            new_lines.append(line)
            continue
        record = json.loads(line)
        if record.get("filename") in DEFECTIVE_FILENAMES:
            if record.get("status") != "pending":
                record["status"] = "pending"
                updated += 1
                print(f"  RESET    {record['filename']}")
            else:
                print(f"  ALREADY  {record['filename']} already pending")
        new_lines.append(json.dumps(record, ensure_ascii=False))
    INDEX_FILE.write_text("\n".join(new_lines) + "\n")
    print(f"  → {updated} records reset\n")


def main():
    move_images()
    patch_prompts()
    reset_index()
    print("Done. Regenerate with:")
    classes = " ".join(sorted({cls for _, cls, _ in DEFECTIVE}))
    print(f"  python scripts/synthetic/2-generate_images.py --classes {classes}")


if __name__ == "__main__":
    main()
