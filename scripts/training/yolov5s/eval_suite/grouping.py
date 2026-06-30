"""Label-remap helpers for the three granularity levels of the evaluation suite.

This module provides the label→group maps described in
``docs/plans/2026-06-10_model-evaluation-strategy.md`` §4–§5.  It is a pure
utility module: it loads CSV/JSON files from ``reports/`` and returns plain
Python dicts.  It does NOT import any training or heavy-ML modules.

Granularity levels (§4 of the evaluation plan)
------------------------------------------------
``fine``    — identity remap (category_id → itself); 225-way classification.
``coarse``  — category_id → group_id; look-alike clusters merged.
``detect``  — every category_id → 1; class-agnostic animal detection.

Usage example
-------------
::

    from scripts.training.yolov5s.eval_suite.grouping import (
        load_coarse_remap,
        load_detect_remap,
        identity_remap,
        load_group_labels,
        load_class_to_band,
        lookalike_group_ids,
    )

    cat_ids = list(range(1, 226))
    coarse = load_coarse_remap()        # {1: 115, 2: 139, …}  (id → group_id)
    detect = load_detect_remap(cat_ids) # {1: 1, 2: 1, …}
    fine   = identity_remap(cat_ids)    # {1: 1, 2: 2, …}

    labels    = load_group_labels()         # {115: 'orycteropus', …}
    band_info = load_class_to_band()        # {'by_name': {…}, 'by_id': {…}}
    la_gids   = lookalike_group_ids()       # [50, 69, 84, …]  (multi-member groups)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

# Repo root is 4 levels up from this file:
#   eval_suite/grouping.py → eval_suite/ → yolov5s/ → training/ → scripts/ → repo_root
# This mirrors constants.py: REPO_ROOT = Path(__file__).resolve().parents[3]
_REPO_ROOT = Path(__file__).resolve().parents[4]

# v2 = the reviewed/refined coarse-grouping table (genus backbone + curated
# splits/merges); see docs/plans/2026-06-11_lookalike-groups-review.md. The v1
# genus-only table (reports/lookalike_groups.csv) is preserved as an audit
# artifact but is no longer the one evaluation scores against.
_DEFAULT_GROUPS_CSV    = _REPO_ROOT / "reports" / "lookalike_groups_v2.csv"
_DEFAULT_SUMMARY_JSON  = _REPO_ROOT / "reports" / "dataset_split_summary.json"

# ── Public API ────────────────────────────────────────────────────────────────


def load_coarse_remap(
    path: str | Path = _DEFAULT_GROUPS_CSV,
) -> dict[int, int]:
    """Return COCO category_id (1..225) → coarse group_id.

    Reads ``reports/lookalike_groups.csv`` produced by
    ``scripts/dataset_quality/16-build_lookalike_groups.py``.

    Raises
    ------
    FileNotFoundError
        If the CSV is absent.  The error message instructs the user to run the
        builder script first.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Look-alike groups CSV not found: {path}\n"
            "  Run the builder first:\n"
            "    python scripts/dataset_quality/16-build_lookalike_groups.py"
        )
    remap: dict[int, int] = {}
    with open(path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            remap[int(row["coco_id"])] = int(row["group_id"])
    return remap


def load_detect_remap(cat_ids: list[int]) -> dict[int, int]:
    """Return every category_id → 1 (single 'animal' class).

    This implements the ``detect`` granularity level: predictions and ground
    truth are remapped to a single class before scoring, so the evaluator
    measures localisation + object-presence only, ignoring all species labels.

    Parameters
    ----------
    cat_ids:
        The list of COCO category IDs present in the evaluation set.
        Typically ``list(range(1, 226))``.
    """
    return {cid: 1 for cid in cat_ids}


def identity_remap(cat_ids: list[int]) -> dict[int, int]:
    """Return category_id → itself (fine granularity, no remapping).

    This is a no-op pass-through included for API symmetry so all three
    granularity levels can be driven by the same scoring code path:
    ``remap = fine | coarse | detect; then apply remap[cat_id]``.

    Parameters
    ----------
    cat_ids:
        The list of COCO category IDs present in the evaluation set.
    """
    return {cid: cid for cid in cat_ids}


def load_group_labels(
    path: str | Path = _DEFAULT_GROUPS_CSV,
) -> dict[int, str]:
    """Return group_id → human-readable group_label, for report tables.

    Reads the same CSV as :func:`load_coarse_remap`.  The group_label is the
    scientific genus name (for genus-backed groups), the override-group name
    (for curated overrides), or the class name itself (for singletons).

    Raises
    ------
    FileNotFoundError
        If the CSV is absent (same message as :func:`load_coarse_remap`).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Look-alike groups CSV not found: {path}\n"
            "  Run the builder first:\n"
            "    python scripts/dataset_quality/16-build_lookalike_groups.py"
        )
    labels: dict[int, str] = {}
    with open(path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            gid   = int(row["group_id"])
            label = row["group_label"]
            # Multiple rows share the same group_id; the label is identical
            # for all of them, so any assignment is idempotent.
            labels[gid] = label
    return labels


def load_class_to_band(
    summary_path: str | Path = _DEFAULT_SUMMARY_JSON,
    cat_id_to_name: dict[int, str] | None = None,
) -> dict:
    """Return band assignments for all 225 classes.

    Reads ``reports/dataset_split_summary.json``, whose top-level keys are
    class names and each value is a dict containing a ``'band'`` field
    (one of ``'A'`` / ``'B'`` / ``'C'`` / ``'D'``).

    Parameters
    ----------
    summary_path:
        Path to ``dataset_split_summary.json``.
    cat_id_to_name:
        Optional mapping of COCO category_id → class_name.  When provided,
        the returned dict also contains a ``'by_id'`` sub-dict keyed by
        integer category IDs.  When absent, ``'by_id'`` is an empty dict.

    Returns
    -------
    dict with two keys:

    ``'by_name'``: ``{class_name: band}``  — e.g. ``{'aardvark': 'A', …}``
    ``'by_id'``:   ``{coco_id: band}``     — populated only if *cat_id_to_name*
                   is supplied; empty dict otherwise.

    Raises
    ------
    FileNotFoundError
        If the JSON is absent.
    """
    summary_path = Path(summary_path)
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Dataset split summary not found: {summary_path}\n"
            "  This file is produced by the dataset splitting pipeline."
        )
    with open(summary_path, encoding="utf-8") as fh:
        data = json.load(fh)

    by_name: dict[str, str] = {}
    for class_name, info in data.items():
        band = info.get("band")
        if band is not None:
            by_name[class_name] = str(band)

    by_id: dict[int, str] = {}
    if cat_id_to_name is not None:
        for cid, name in cat_id_to_name.items():
            if name in by_name:
                by_id[int(cid)] = by_name[name]

    return {"by_name": by_name, "by_id": by_id}


def lookalike_group_ids(
    path: str | Path = _DEFAULT_GROUPS_CSV,
) -> list[int]:
    """Return group_ids that contain more than one fine class.

    These are the actual look-alike clusters — the groups where the
    within-group confusion rate and block-diagonal confusion matrix are
    meaningful diagnostics.

    Returns group_ids sorted in ascending order.

    Raises
    ------
    FileNotFoundError
        If the CSV is absent (same message as :func:`load_coarse_remap`).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Look-alike groups CSV not found: {path}\n"
            "  Run the builder first:\n"
            "    python scripts/dataset_quality/16-build_lookalike_groups.py"
        )
    from collections import Counter

    counts: Counter[int] = Counter()
    with open(path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            counts[int(row["group_id"])] += 1
    return sorted(gid for gid, cnt in counts.items() if cnt > 1)
