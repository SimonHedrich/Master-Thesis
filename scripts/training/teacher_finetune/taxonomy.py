"""Group-table construction for grouped/marginal cross-entropy (implementation
plan §2.2).

Of the 225 project classes, 178 are species-level (a 1:1 match to exactly one
of SpeciesNet's 2,498 leaf classes) but 35 are genus-level and 12 are
family-level rollups with no single correct leaf index — e.g. a genus-level
label like "weasel species" maps to every ``mustela *`` leaf class at once.
``build_group_table()`` inverts the existing species→genus→family lookup
dicts already implemented in ``scripts/dataset_quality/7-filter_speciesnet.py``
(``load_classes_225``) against the classifier's own label list
(``load_speciesnet_labels``) to produce ``idx_225 -> list[leaf_idx]`` — the
same priority order ``compute_probs_225`` already uses when projecting scores
onto the 225-class vector, just inverted to build membership instead of
summing probabilities. No new taxonomy parsing code — only a new small
inversion function, per the implementation plan's explicit instruction.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import scripts.training.teacher_finetune.constants as constants

_SCRIPT7_PATH = (
    Path(__file__).resolve().parents[2] / "dataset_quality" / "7-filter_speciesnet.py"
)


def _load_script7():
    """Import 7-filter_speciesnet.py via importlib (numeric name blocks regular
    import) — same pattern already established in
    ``scripts/dataset_quality/8-class_distribution_report.py``.
    """
    spec = importlib.util.spec_from_file_location("filter_speciesnet", _SCRIPT7_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["filter_speciesnet"] = mod
    spec.loader.exec_module(mod)
    return mod


_cache: dict = {}


def _load_projection_tables() -> dict:
    """Load + cache the (idx_to_label, genus_species_to_225, genus_to_225,
    family_to_225) lookup tables shared by `build_group_table()` (loss) and
    `projection_tables()` (eval-time `compute_probs_225` projection).

    Cached at module scope because `load_speciesnet_labels()` instantiates the
    full SpeciesNet classifier just to read its label list — expensive enough
    that it must only happen once per process, not once per call site.
    """
    if not _cache:
        s7 = _load_script7()
        s7._check_environment()
        idx_to_label = s7.load_speciesnet_labels()
        by_common, genus_species_to_225, genus_to_225, family_to_225 = s7.load_classes_225(
            constants.CLASSES_225_PATH
        )
        _cache.update(
            idx_to_label=idx_to_label,
            genus_species_to_225=genus_species_to_225,
            genus_to_225=genus_to_225,
            family_to_225=family_to_225,
            by_common=by_common,
        )
    return _cache


def projection_tables() -> tuple[dict[int, str], dict[str, int], dict[str, int], dict[str, int]]:
    """Public accessor for `evaluate.py`: the exact 4-tuple `compute_probs_225`
    expects (`idx_to_label, genus_species_to_225, genus_to_225, family_to_225`).
    """
    t = _load_projection_tables()
    return t["idx_to_label"], t["genus_species_to_225"], t["genus_to_225"], t["family_to_225"]


def build_group_table() -> tuple[dict[int, list[int]], dict[int, str]]:
    """Build the ``idx_225 -> list[leaf_idx]`` group table plus each
    ``idx_225``'s taxonomic level ("species" | "genus" | "family").

    Requires the ``speciesnet`` package (loads the full classifier once, to
    read its 2,498-class label list via ``load_speciesnet_labels()`` — the
    on-disk ``data/speciesnet_classes.json`` manifest is not usable here since
    it stores integer indices, not the "uuid;class;order;family;genus;species;
    common" label strings this function needs to resolve each leaf class's
    taxonomy).
    """
    t = _load_projection_tables()
    idx_to_label = t["idx_to_label"]
    genus_species_to_225 = t["genus_species_to_225"]
    genus_to_225 = t["genus_to_225"]
    family_to_225 = t["family_to_225"]

    groups: dict[int, list[int]] = {idx: [] for idx in range(constants.NUM_CLASSES_225)}
    for leaf_idx, label in idx_to_label.items():
        parts = label.split(";")
        if len(parts) < 6:
            continue
        family = parts[3].lower().strip()
        genus = parts[4].lower().strip()
        species = parts[5].lower().strip()

        idx_225 = genus_species_to_225.get(f"{genus} {species}")
        if idx_225 is None:
            idx_225 = genus_to_225.get(genus)
        if idx_225 is None:
            idx_225 = family_to_225.get(family)
        if idx_225 is not None:
            groups[idx_225].append(leaf_idx)

    levels = {row["idx_225"]: row["level"] for row in t["by_common"].values()}
    return groups, levels
