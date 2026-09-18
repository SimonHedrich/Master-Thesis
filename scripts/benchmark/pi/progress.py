#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
Sweep progress + live validity check. RUNS ON THE PI.

Summarises `latency_raw.jsonl` while the sweep is still running, so a run that
is quietly producing throttled (i.e. worthless) cells can be caught early
rather than at the end of a multi-hour sweep.

Usage (on the Pi):
    uv run --script ~/benchmark/progress.py
    python3 ~/benchmark/progress.py --expected 76
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=Path.home() / "benchmark/results/latency_raw.jsonl")
    ap.add_argument("--expected", type=int, default=0)
    ap.add_argument("--tail", type=int, default=6)
    args = ap.parse_args()

    rows = []
    if args.raw.exists():
        rows = [json.loads(line) for line in args.raw.read_text().splitlines() if line.strip()]

    total = f" / {args.expected}" if args.expected else ""
    print(f"records: {len(rows)}{total}")

    def sh(c: str) -> str:
        try:
            return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:  # noqa: BLE001
            return "?"

    print(f"temp: {sh('vcgencmd measure_temp')}   {sh('vcgencmd get_throttled')}")

    for r in rows[-args.tail:]:
        w = r["windows"]["infer"]
        mon = r.get("monitor", {})
        print(
            f"  {r['cell']:<46} infer {w['median_ms']:9.1f} ms "
            f"n={r['protocol']['iters']:<4} "
            f"peak {mon.get('temp_max_c', 0):.0f}C "
            f"valid={r['gate']['valid']}"
        )

    invalid = [r["cell"] for r in rows if r["gate"]["valid"] is False]
    if invalid:
        print(f"\n  !! {len(invalid)} INVALID cell(s):")
        for c in invalid:
            print(f"     {c}")
    else:
        print("\n  all cells passed the thermal validity gate")

    if rows:
        peak = max(r["monitor"].get("temp_max_c", 0) for r in rows)
        bits = 0
        for r in rows:
            bits |= max(0, r["monitor"].get("throttled_bits", 0))
        print(f"  sweep peak temperature: {peak:.1f} C   throttle bits seen: 0x{bits:x}")


if __name__ == "__main__":
    main()
