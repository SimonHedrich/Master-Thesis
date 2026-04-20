"""Filter Wikimedia Commons category trees to remove entries unusable for training.

Reads .txt files from reports/wikimedia_categories/, applies keyword-based filtering,
and writes cleaned files to reports/wikimedia_categories_filtered/.

Two modes of exclusion:
  1. ZERO-FILE REMOVAL  — lines with "(0 files)" are dropped (no children removed).
  2. KEYWORD CASCADE    — lines whose category name matches a keyword are dropped
                          together with ALL their indented subcategories.

Usage:
    python scripts/wikimedia/2-filter_wikimedia_categories.py
    python scripts/wikimedia/2-filter_wikimedia_categories.py --input-dir reports/wikimedia_categories
    python scripts/wikimedia/2-filter_wikimedia_categories.py --dry-run
"""

import argparse
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Root path
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

INPUT_DIR  = REPO_ROOT / "reports" / "wikimedia_categories"
OUTPUT_DIR = REPO_ROOT / "reports" / "wikimedia_categories_filtered"

# ---------------------------------------------------------------------------
# Keyword list — any category whose name contains one of these strings
# (case-insensitive) is removed together with ALL its subcategories.
#
# Rationale for each group is commented inline.
# ---------------------------------------------------------------------------
FILTER_KEYWORDS = [
    # ── Artwork, illustrations and cultural depictions ────────────────────
    " in art",          # "Lions in art", "Cheetahs in art", "X in art by …"
    "(in art)",
    "in art by",
    "illustration",     # "(illustrations)", "illustrations of X", "anatomy illustrations"
    "engraving",        # historical engravings
    "sculpture",        # sculptures of X
    " icons",           # "Cheetah icons" (graphic icons, not photos)
    "(icons)",
    "mosaic",
    "anthropomorphic",  # cartoon/anthropomorphized animals
    " tattoo",          # tattoos of X
    "life restoration", # paleoart
    "silhouette",       # silhouette graphics
    "fictional",        # fictional animals (Simba etc.)
    "heraldry",         # coats-of-arms
    "coat of arm",      # "coat of arms"
    " emblem",          # emblems / logos
    " logo",            # logos
    " logos",
    "costume",          # animal costumes
    "cladogram",        # phylogenetic diagrams
    " drawing",         # drawings / line-art
    " drawings",
    " painting",        # paintings / artwork
    " paintings",
    "in mythology",
    "mythological",
    "in popular culture",
    "in culture",
    "in religion",
    "in fables",
    "in fairy tale",
    "(rite)",           # cultural rites (e.g. "Bear guiding (rite)")
    "things named",     # "Things named after lions" → city names, beer brands, etc.
    "named after",

    # ── Anatomy and isolated body parts ───────────────────────────────────
    "anatomy",          # "Acinonyx jubatus anatomy"
    " bones",           # "X bones"
    " bone",            # singular
    "skull",            # "skulls", "skull illustrations"
    "skeleton",
    " teeth",
    " tooth",
    " paws",
    " paw ",
    " claws",
    " claw ",
    " horns",
    " horn ",
    "heads",            # "(heads)", "X heads" — head-only close-ups
    "(head)",           # "rothschildi (head)"
    "(head ",           # "(heads)"
    "fur-skin",         # pelts used as clothing/fashion items
    "fur and skin",

    # ── Distribution and range maps ───────────────────────────────────────
    "distribution map",
    "range map",

    # ── Philately, numismatics, currency ──────────────────────────────────
    " stamps",          # "on stamps", "X stamps"
    " stamp",           # singular "on stamp"
    " coins",           # "on coins"
    " coin ",           # singular
    "banknote",
    "on currency",
    " medals",
    " medal ",
    "postal card",

    # ── Feces, waste, excrement ───────────────────────────────────────────
    "feces",
    " dung",
    " scat",            # scat = droppings
    "droppings",
    "defecating",

    # ── Tracks, footprints, spoor ─────────────────────────────────────────
    "footprint",
    " tracks",          # "Loxodonta africana tracks"
    " track ",
    "spoor",

    # ── Museum specimens and taxidermy ────────────────────────────────────
    "taxidermied",
    "taxidermy",
    "museum specimen",

    # ── Dead, killed, wounded animals ─────────────────────────────────────
    "(dead)",
    " dead ",           # "X dead Y" (middle of name)
    "dead ",            # "Dead X" (beginning of name)
    " carcass",
    "hunting trophy",
    " trophies",
    " trophy",
    "wounded",
    "dying ",           # "Dying lions"
    "roadkill",
    "poaching",

    # ── Audio-only content ────────────────────────────────────────────────
    "audio file",
    "pronunciation",

    # ── Human activities / interactions ───────────────────────────────────
    "riding on",        # "Riding on Loxodonta africana"
    "(clothing)",       # "Cheetah (clothing)"
    " pelts",
    " pelt ",
    "people with",      # "People with cheetahs" — humans dominate these shots

    # ── Food and animal products ──────────────────────────────────────────
    "as food",
    " meat",
    "castoreum",        # beaver secretion product
    "ivory",            # ivory products
    " products",        # "Lion products", "Bos taurus proteins" products

    # ── Molecular/biochemical content (domestic species) ─────────────────
    " proteins",        # "Bos taurus proteins"
    "ribbon diagram",   # protein structure diagrams

    # ── Other clearly non-photographic content ────────────────────────────
    "fossil",           # fossils / paleontology
    " mummies",
    "mummified",
    " signs",           # "Ursus arctos signs" = spoor/marks, not animal photos
    " traps",           # snares/traps
    "(rite)",           # cultural ceremonies
    "size comparison",
    "information graphic",
    "diseases and disorder",  # lesion/disease documentation
    "in advertisement",
    "scrimshaw",        # carved ivory/bone
]

# Compile lowercase set for fast lookup
_KEYWORDS_LOWER = [kw.lower() for kw in FILTER_KEYWORDS]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CATEGORY_RE = re.compile(r'^( *)Category:(.+?)\s+\((\d+) files?\)\s*$')


def parse_line(line: str):
    """Return (indent_spaces, name, file_count) or None if not a Category line."""
    m = _CATEGORY_RE.match(line.rstrip('\n'))
    if not m:
        return None
    indent = len(m.group(1))
    name   = m.group(2)
    count  = int(m.group(3))
    return indent, name, count


def matches_keyword(name: str) -> bool:
    name_lower = name.lower()
    return any(kw in name_lower for kw in _KEYWORDS_LOWER)


def filter_file(lines: list[str]) -> tuple[list[str], dict]:
    """Filter a list of lines from a category .txt file.

    Returns:
        filtered_lines: lines to keep
        stats: dict with counts of removed categories and reasons
    """
    result   = []
    stats    = {"zero_files": 0, "keyword_match": 0, "cascaded": 0, "kept": 0}

    # Track the minimum depth of the active cascade exclusion.
    # When we encounter a category at depth <= cascade_depth, the cascade ends.
    cascade_min_depth: int | None = None

    for line in lines:
        # Always keep header / blank lines
        parsed = parse_line(line)
        if parsed is None:
            result.append(line)
            continue

        indent, name, count = parsed
        # depth in logical units (2 spaces per level in the generated files)
        depth = indent // 2

        # ── Cascade boundary ──────────────────────────────────────────────
        if cascade_min_depth is not None:
            if depth <= cascade_min_depth:
                # We stepped back out of the excluded subtree
                cascade_min_depth = None
            else:
                # Still inside the excluded subtree → skip
                stats["cascaded"] += 1
                continue

        # ── Keyword match → remove + cascade ─────────────────────────────
        if matches_keyword(name):
            cascade_min_depth = depth
            stats["keyword_match"] += 1
            continue

        # ── Zero files → remove line only, keep children ─────────────────
        if count == 0:
            stats["zero_files"] += 1
            continue

        # ── Keep ──────────────────────────────────────────────────────────
        stats["kept"] += 1
        result.append(line)

    return result, stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", default=str(INPUT_DIR),
        help="Directory containing raw .txt category files",
    )
    parser.add_argument(
        "--output-dir", default=str(OUTPUT_DIR),
        help="Directory to write filtered .txt files (created if needed)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print stats only, do not write output files",
    )
    args = parser.parse_args()

    in_dir  = Path(args.input_dir)
    out_dir = Path(args.output_dir)

    if not in_dir.is_dir():
        print(f"ERROR: input directory not found: {in_dir}")
        raise SystemExit(1)

    txt_files = sorted(in_dir.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {in_dir}")
        raise SystemExit(1)

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    total_kept    = 0
    total_removed = 0
    summary_rows  = []

    for path in txt_files:
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        filtered, stats = filter_file(lines)

        removed = stats["zero_files"] + stats["keyword_match"] + stats["cascaded"]
        kept    = stats["kept"]
        total_kept    += kept
        total_removed += removed

        summary_rows.append(
            f"  {path.stem:<45}  kept={kept:>4}  removed={removed:>4}"
            f"  (zero={stats['zero_files']}, kw={stats['keyword_match']}, cascade={stats['cascaded']})"
        )

        if not args.dry_run:
            out_path = out_dir / path.name
            out_path.write_text("".join(filtered), encoding="utf-8")

    print("=" * 70)
    print(f"Filter report  ({len(txt_files)} files)")
    print("=" * 70)
    for row in summary_rows:
        print(row)
    print("-" * 70)
    print(f"  TOTAL  kept={total_kept}  removed={total_removed}")
    if not args.dry_run:
        print(f"\nFiltered files written to: {out_dir}")
    else:
        print("\n[dry-run] No files written.")


if __name__ == "__main__":
    main()
