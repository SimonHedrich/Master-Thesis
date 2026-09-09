"""Diff two ``evaluation_report.json`` files, flagging any numeric leaf that
differs by more than a float-noise tolerance. Used to validate the scoring-
engine swap (TODO.md 2.1): run a smoke evaluation before and after a change
to ``scoring.py``, then diff the two reports.

Run command
-----------
    uv run python -m scripts.training.yolov5s.eval_suite.tests.diff_smoke_reports \
        /path/to/before/evaluation_report.json /path/to/after/evaluation_report.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

DEFAULT_TOL = 1e-4


def _walk_diff(a, b, path: str, tol: float, out: list[str]) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        keys = set(a) | set(b)
        for k in sorted(keys, key=str):
            if k not in a:
                out.append(f"{path}.{k}: missing in BEFORE (after={b[k]!r})")
            elif k not in b:
                out.append(f"{path}.{k}: missing in AFTER (before={a[k]!r})")
            else:
                _walk_diff(a[k], b[k], f"{path}.{k}", tol, out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length differs (before={len(a)}, after={len(b)})")
            return
        for i, (av, bv) in enumerate(zip(a, b)):
            _walk_diff(av, bv, f"{path}[{i}]", tol, out)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        af, bf = float(a), float(b)
        if math.isnan(af) and math.isnan(bf):
            return
        if not math.isclose(af, bf, abs_tol=tol, rel_tol=tol):
            out.append(f"{path}: {af} -> {bf} (delta={bf - af:.6g})")
    else:
        if a != b:
            out.append(f"{path}: {a!r} -> {b!r}")


def diff_reports(before_path: Path, after_path: Path, tol: float = DEFAULT_TOL) -> list[str]:
    with Path(before_path).open() as f:
        before = json.load(f)
    with Path(after_path).open() as f:
        after = json.load(f)
    diffs: list[str] = []
    _walk_diff(before, after, "report", tol, diffs)
    return diffs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("before", type=Path)
    p.add_argument("after", type=Path)
    p.add_argument("--tol", type=float, default=DEFAULT_TOL)
    args = p.parse_args()

    diffs = diff_reports(args.before, args.after, args.tol)
    if not diffs:
        print(f"[OK] no material differences beyond tol={args.tol}")
        return
    print(f"{len(diffs)} material difference(s) beyond tol={args.tol}:")
    for d in diffs:
        print(f"  {d}")
    sys.exit(1)


if __name__ == "__main__":
    main()
