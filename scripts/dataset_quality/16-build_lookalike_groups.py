"""Build the coarse look-alike grouping table for evaluation.

Produces ``reports/lookalike_groups.csv`` with columns:
    coco_id, class_name, group_id, group_label, group_source

This table is the fixed, frozen label→group map used by the evaluation
suite (see ``scripts/training/yolov5s/eval_suite/grouping.py``) to remap
fine-grained 225-way category IDs to a coarser set of look-alike groups.
The remapping enables the granularity-gap decomposition described in
``docs/plans/2026-06-10_model-evaluation-strategy.md`` §4–§5.

Strategy (§5 of the evaluation plan)
-------------------------------------
1. **Genus backbone** — every species-level class is assigned the genus from
   its scientific name (first word).  Genus-level and family-level classes
   already carry their genus/family scientific name directly.  Classes that
   share a genus_scientific value are automatically merged into one group.
2. **Curated overrides (FROZEN)** — a small, hand-authored dict of
   cross-genus look-alike clusters that taxonomy splits or merges wrongly.
   Overrides are applied *before* genus assignment; any class listed in an
   override is forced into that override group regardless of genus.
3. **Singleton fallback** — any class not resolved by override or shared
   genus becomes its own singleton group (group_label = class_name,
   group_source = 'singleton').  This is the safe, documented default: we
   never merge classes when uncertain.

The output CSV is sorted by coco_id.  group_ids are assigned by lexical sort
of unique group_labels so the mapping is deterministic across reruns.

Usage
-----
    python scripts/dataset_quality/16-build_lookalike_groups.py
    python scripts/dataset_quality/16-build_lookalike_groups.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# ── I/O paths ─────────────────────────────────────────────────────────────────

ANNOTATIONS_TEST = REPO_ROOT / "data" / "real" / "annotations_test.json"
CLASSES_225_CSV  = REPO_ROOT / "reports" / "classes_225.csv"
# v2 = reviewed/refined table (see docs/plans/2026-06-11_lookalike-groups-review.md).
# The v1 genus-only table (reports/lookalike_groups.csv) is preserved untouched
# as an audit artifact; v2 is the table the evaluation suite actually uses.
OUTPUT_CSV       = REPO_ROOT / "reports" / "lookalike_groups_v2.csv"

OUTPUT_COLUMNS = ["coco_id", "class_name", "group_id", "group_label", "group_source"]

# ── Curated override groups (FROZEN — do not change after first evaluation) ───
#
# Each key is a human-readable override-group label; the value is the frozenset
# of class names (exactly as they appear in annotations_test.json) that are
# forced into that group, overriding whatever genus they would fall into.
#
# Design rules for overrides:
#   - Only include classes that ACTUALLY EXIST in the 225.
#   - Group iff confusing the classes in a single still frame is a FORGIVABLE
#     mistake (it should be charged to Δ_fine, not Δ_coarse).  Keep apart iff
#     confusing them is a GENUINE failure (it should surface in Δ_coarse).
#   - Overrides may WIDEN genus groups (add cross-genus look-alikes) AND SPLIT
#     genus groups whose members are visually distinct (the strategy doc §5
#     "keep visually-distinct same-genus pairs split if warranted" clause).
#   - A SPLIT is expressed by overriding each resulting sub-group explicitly,
#     INCLUDING singleton sub-groups (e.g. "lion"/"tiger" below) — otherwise the
#     genus backbone would re-merge the leftover members.
#   - Keep the list minimal and conservative; every override must be
#     individually justifiable from domain knowledge.
#   - This dict is FROZEN once evaluation begins.  Any change invalidates
#     previously computed coarse mAP numbers.
#
# Full review & justification: docs/plans/2026-06-11_lookalike-groups-review.md
#
# Justifications:
#   "elephant"  (cross-genus MERGE)
#       African elephant (Loxodonta africana) and Asian elephant (Elephas
#       maximus) are different genera but visually near-identical in field
#       photos (size/ear shape distinguishable only at close range with
#       good lighting). Named in §5 of the evaluation plan.
#
#   "lynx_caracal_cluster"  (cross-genus MERGE)
#       Bobcat (Lynx rufus), Canada lynx (Lynx canadensis), and Eurasian lynx
#       (Lynx lynx) are already in the same genus; caracal (Caracal caracal)
#       is a different genus but is visually very similar (tufted ears, long
#       legs, sandy coat). Named in §5 of the evaluation plan.  The genus
#       backbone already groups the three lynx species; this override adds
#       caracal to them.
#
#   "hyaena"  (cross-genus MERGE)
#       Spotted hyaena (Crocuta crocuta), striped hyaena (Hyaena hyaena), and
#       brown hyaena (Parahyaena brunnea) span three genera yet are easily
#       confused in degraded or distant photos.  All three have the
#       characteristic sloped back, coarse coat, and similar body plan.
#
#   "lion" / "tiger" / "panthera_rosette"  (SPLIT of genus `panthera`)
#       The genus panthera mixes three visually unrelated coat types: lion
#       (uniform tawny + mane), tiger (orange-and-black stripes), and the three
#       ROSETTE cats leopard (P. pardus) / jaguar (P. onca) / snow leopard
#       (P. uncia).  Lion and tiger are unmistakable — confusing them is a real
#       failure that must reach Δ_coarse, so each is forced to its own singleton
#       group.  The genuine fine-grained confusion is leopard↔jaguar (snow
#       leopard shares the rosette pattern, differing mainly in coat colour),
#       so those three form one look-alike group.
#
#   "zebra" / "equine_unstriped"  (SPLIT of genus `equus`)
#       Stripes are an unmissable feature: the three zebras (grevy's / mountain
#       / plains) are never confused with the unstriped equids (asiatic wild
#       ass / domestic donkey / domestic horse) in a clear frame, so zebra↔horse
#       must reach Δ_coarse.  The strategy doc §1 itself lists "the three zebra
#       species" and "the Equus asses" as TWO distinct look-alike clusters.
#       Within-group confusion (zebra↔zebra; ass↔donkey↔horse) stays in Δ_fine.
#
#   "gazelle"  (cross-genus MERGE; reverses the v1 "keep split" decision)
#       Grant's gazelle (Nanger granti) and Thomson's gazelle (Eudorcas
#       thomsonii) are a textbook look-alike pair (tan coat, white belly, dark
#       flank band, lyre horns; confused even by experienced observers).
#       Springbok (Antidorcas marsupialis) shares the same template.  Doc §1
#       lists gazelles as a look-alike concern.  Deliberately EXCLUDED: gerenuk
#       (long neck), blackbuck (males black/white), impala (reddish, distinct
#       build) — each separable, so kept in their own groups.
#
# NOTE (considered but NOT merged — keep split, charged to Δ_coarse if confused):
#   river otters (lutra/lontra), small spotted cats (leopardus/prionailurus/
#   leptailurus), cheetah, clouded leopard, and the foxes.  See the review doc
#   §4 for the per-case reasoning.  When uncertain, do NOT merge.

CURATED_OVERRIDES: dict[str, frozenset[str]] = {
    # fmt: off
    "elephant": frozenset({
        "african elephant",   # Loxodonta africana  — genus loxodonta
        "asian elephant",     # Elephas maximus     — genus elephas  (different genus!)
    }),
    "lynx_caracal_cluster": frozenset({
        "bobcat",             # Lynx rufus
        "canada lynx",        # Lynx canadensis
        "eurasian lynx",      # Lynx lynx
        "caracal",            # Caracal caracal     — different genus, tufted ears
    }),
    "hyaena": frozenset({
        "spotted hyaena",     # Crocuta crocuta     — genus crocuta
        "striped hyaena",     # Hyaena hyaena       — genus hyaena
        "brown hyaena",       # Parahyaena brunnea  — genus parahyaena
    }),
    # ── SPLIT of genus `panthera` (lion/tiger are visually unique outliers) ──
    "lion": frozenset({
        "lion",               # Panthera leo        — uniform tawny + mane; kept apart
    }),
    "tiger": frozenset({
        "tiger",              # Panthera tigris     — orange/black stripes; kept apart
    }),
    "panthera_rosette": frozenset({
        "leopard",            # Panthera pardus     — rosettes
        "jaguar",             # Panthera onca       — rosettes (textbook ↔leopard pair)
        "snow leopard",       # Panthera uncia      — rosettes, pale coat (weakest member)
    }),
    # ── SPLIT of genus `equus` (zebras are striped; equids are not) ──
    "zebra": frozenset({
        "grevy's zebra",      # Equus grevyi
        "mountain zebra",     # Equus zebra
        "plains zebra",       # Equus quagga
    }),
    "equine_unstriped": frozenset({
        "asiatic wild ass",   # Equus hemionus
        "domestic donkey",    # Equus asinus
        "domestic horse",     # Equus caballus
    }),
    # ── cross-genus MERGE of the true gazelles ──
    "gazelle": frozenset({
        "grant's gazelle",    # Nanger granti       — genus nanger
        "thomson's gazelle",  # Eudorcas thomsonii  — genus eudorcas
        "springbok",          # Antidorcas marsupialis — genus antidorcas
    }),
    # fmt: on
}

# ── Logging setup ─────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(levelname)s  %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_categories(path: Path) -> list[tuple[int, str]]:
    """Return [(coco_id, class_name), ...] from an annotations JSON.

    Sorts by coco_id so the output CSV row order is deterministic.
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    cats = [(int(c["id"]), str(c["name"])) for c in data["categories"]]
    cats.sort(key=lambda x: x[0])
    return cats


def _load_classes_225(path: Path) -> dict[str, str]:
    """Return {common_name: scientific_name} from classes_225.csv.

    scientific_name is already lowercased in that file.
    """
    mapping: dict[str, str] = {}
    with open(path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            name = row["common_name"].strip()
            sci  = row["scientific_name"].strip().lower()
            mapping[name] = sci
    return mapping


def _genus_from_scientific(sci: str) -> str:
    """Extract the genus from a scientific name (first space-separated token).

    For family-level entries (e.g. 'cricetidae'), the entire string is
    returned as-is (it is already a single token and serves as the group key).
    """
    return sci.split()[0] if sci else sci


# ── Core builder ──────────────────────────────────────────────────────────────

def build_groups(
    categories: list[tuple[int, str]],
    name_to_sci: dict[str, str],
) -> list[dict]:
    """Assign each category to a group; return rows for the output CSV.

    Assignment logic (in priority order):
    1. Curated override:  class_name is in CURATED_OVERRIDES → use override.
    2. Genus/family backbone: class_name has a scientific name in
       name_to_sci → use the genus (first word of scientific name) as
       group_label; family-level entries use the family name directly.
    3. Singleton fallback: group_label = class_name.
    """
    # Build reverse map: class_name -> override_group_label
    name_to_override: dict[str, str] = {}
    for group_label, members in CURATED_OVERRIDES.items():
        for member in members:
            name_to_override[member] = group_label

    # First pass: assign group_label + group_source to every class
    raw_rows: list[dict] = []
    for coco_id, class_name in categories:
        if class_name in name_to_override:
            group_label  = name_to_override[class_name]
            group_source = "override"
        elif class_name in name_to_sci:
            sci          = name_to_sci[class_name]
            group_label  = _genus_from_scientific(sci)
            group_source = "genus"
        else:
            group_label  = class_name   # safe singleton
            group_source = "singleton"
            log.warning(
                "class '%s' (id=%d) not found in classes_225.csv — "
                "falling back to singleton group.",
                class_name, coco_id,
            )
        raw_rows.append({
            "coco_id":      coco_id,
            "class_name":   class_name,
            "group_label":  group_label,
            "group_source": group_source,
        })

    # Promote genus singletons: a genus-group that ends up with only ONE member
    # (after overrides have pulled some classes away) is a singleton by effect.
    # We keep group_source = 'genus' for these because the *reason* is
    # taxonomic, even though the group is size-1.  They are NOT relabelled as
    # 'singleton' because: (a) it keeps the source traceable, and (b) future
    # classes added to the same genus would naturally join the group.
    # Exception: class_name already used as group_label for truly orphaned
    # entries (no scientific name lookup) stay as 'singleton'.

    # Second pass: assign stable integer group_ids by sorted unique group_label.
    unique_labels = sorted({r["group_label"] for r in raw_rows})
    label_to_id   = {lbl: idx for idx, lbl in enumerate(unique_labels)}

    rows: list[dict] = []
    for r in raw_rows:
        rows.append({
            "coco_id":      r["coco_id"],
            "class_name":   r["class_name"],
            "group_id":     label_to_id[r["group_label"]],
            "group_label":  r["group_label"],
            "group_source": r["group_source"],
        })

    return rows


# ── Reporting ─────────────────────────────────────────────────────────────────

def _print_summary(rows: list[dict]) -> None:
    """Print a human-readable review summary to stdout."""
    from collections import Counter

    n_classes = len(rows)
    group_to_members: dict[str, list[str]] = defaultdict(list)
    source_counts: Counter[str] = Counter()

    for r in rows:
        group_to_members[r["group_label"]].append(r["class_name"])
        source_counts[r["group_source"]] += 1

    multi_groups = {g: m for g, m in group_to_members.items() if len(m) > 1}
    singleton_groups = {g: m for g, m in group_to_members.items() if len(m) == 1}
    n_merged = sum(len(m) for m in multi_groups.values())

    sep = "─" * 72
    print()
    print(sep)
    print("LOOKALIKE GROUPS SUMMARY")
    print(sep)
    print(f"  Total classes   : {n_classes}")
    print(f"  Total groups    : {len(group_to_members)}")
    print(f"    multi-member  : {len(multi_groups)}")
    print(f"    singleton     : {len(singleton_groups)}")
    print(f"  Classes merged  : {n_merged}  (in multi-member groups)")
    print()
    print("  Source breakdown:")
    for src in ("override", "genus", "singleton"):
        print(f"    {src:<12}  {source_counts[src]:>3} classes")
    print()
    print("  Multi-member groups (review these):")
    print()
    for grp_label, members in sorted(multi_groups.items(), key=lambda x: -len(x[1])):
        # Find group_source for display (all members in a group share the same source)
        src = next(r["group_source"] for r in rows if r["group_label"] == grp_label)
        gid = next(r["group_id"] for r in rows if r["group_label"] == grp_label)
        members_sorted = sorted(members)
        print(f"  [{gid:3d}] {grp_label}  ({src}, {len(members)} members)")
        for m in members_sorted:
            print(f"        • {m}")
    print(sep)
    print()


# ── Writer ────────────────────────────────────────────────────────────────────

def _write_csv(rows: list[dict], path: Path, dry_run: bool) -> None:
    """Write rows to CSV; atomic via tmp-rename; no-op if dry_run."""
    if dry_run:
        log.info("DRY RUN — would write %d rows to %s", len(rows), path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)
    log.info("Written %d rows → %s", len(rows), path.relative_to(REPO_ROOT))


# ── Validation ────────────────────────────────────────────────────────────────

def _validate(rows: list[dict], categories: list[tuple[int, str]]) -> None:
    """Sanity-check the output before writing."""
    errors: list[str] = []

    # Exact 225 rows
    if len(rows) != 225:
        errors.append(f"Expected 225 rows, got {len(rows)}")

    # Every coco_id 1..225 present exactly once
    ids_in_rows = sorted(r["coco_id"] for r in rows)
    expected_ids = list(range(1, 226))
    if ids_in_rows != expected_ids:
        errors.append(f"coco_id set mismatch: {ids_in_rows[:5]}... vs 1..225")

    # group_ids contiguous from 0
    gids = sorted({r["group_id"] for r in rows})
    if gids != list(range(len(gids))):
        errors.append(f"group_ids not contiguous from 0: {gids[:5]}...")
    if gids[0] != 0:
        errors.append(f"group_ids do not start at 0")

    # group_source values valid
    valid_sources = {"override", "genus", "singleton"}
    bad_sources = {r["group_source"] for r in rows} - valid_sources
    if bad_sources:
        errors.append(f"Unknown group_source values: {bad_sources}")

    # All override members present in rows
    all_class_names = {r["class_name"] for r in rows}
    for group_label, members in CURATED_OVERRIDES.items():
        for member in members:
            if member not in all_class_names:
                errors.append(
                    f"Override member '{member}' (group '{group_label}') "
                    "not found in categories"
                )

    if errors:
        for e in errors:
            log.error("VALIDATION FAILED: %s", e)
        raise SystemExit("Validation failed — see errors above.")

    log.info(
        "Validation passed: %d rows, %d groups, group_ids 0..%d",
        len(rows),
        len(gids),
        gids[-1],
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary and validate but do not write the CSV.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=ANNOTATIONS_TEST,
        help=f"Path to COCO annotations JSON (default: {ANNOTATIONS_TEST.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--classes-csv",
        type=Path,
        default=CLASSES_225_CSV,
        help=f"Path to classes_225.csv (default: {CLASSES_225_CSV.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_CSV,
        help=f"Path to write lookalike_groups.csv (default: {OUTPUT_CSV.relative_to(REPO_ROOT)})",
    )
    args = parser.parse_args()

    log.info("Loading categories from %s", args.annotations.relative_to(REPO_ROOT))
    categories = _load_categories(args.annotations)
    log.info("  %d categories loaded.", len(categories))

    log.info("Loading scientific names from %s", args.classes_csv.relative_to(REPO_ROOT))
    name_to_sci = _load_classes_225(args.classes_csv)
    log.info("  %d entries.", len(name_to_sci))

    log.info("Building groups …")
    rows = build_groups(categories, name_to_sci)

    _validate(rows, categories)
    _print_summary(rows)
    _write_csv(rows, args.output, dry_run=args.dry_run)

    if args.dry_run:
        log.info("Dry run complete — no file written.")
    else:
        log.info("Done.")


if __name__ == "__main__":
    main()
