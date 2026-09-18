#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
Device monitor — thermal / frequency / throttle / memory sampler (runs ON the Pi).

Started as a sibling process by `2-bench_latency.py` for the duration of every
measured cell and stopped when it finishes. Writes one CSV row every
`--interval` seconds; the benchmark script then folds a summary (max temp,
median frequency, throttle bits, peak RSS) into that cell's record and applies
the validity gate.

The gate matters more than it sounds: the Pi 400 is passively cooled with only
an internal heatspreader, so a long sweep can silently drop from 1.8 GHz and
turn a latency table into a record of the room temperature. Sampling the
throttle bitmask is the only way to tell the difference after the fact.

Stdlib only, no dependencies — it must add no measurable load of its own.

Usage (normally invoked by 2-bench_latency.py, not by hand):
    uv run --script scripts/benchmark/pi/monitor.py --out monitor_cell.csv --interval 0.2
    uv run --script scripts/benchmark/pi/monitor.py --out m.csv --pid 12345
"""
from __future__ import annotations

import argparse
import signal
import subprocess
import time
from pathlib import Path

THERMAL = Path("/sys/class/thermal/thermal_zone0/temp")
CPU_DIRS = sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_cur_freq"))

_running = True


def _stop(signum, frame) -> None:  # noqa: ANN001, ARG001
    global _running
    _running = False


def _read_int(path: Path) -> int:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return -1


def _throttled() -> int:
    """`vcgencmd get_throttled` as an int. Bit 0 under-voltage, bit 1 arm-freq
    capped, bit 2 currently throttled, bit 3 soft temp limit; bits 16-19 are
    the same conditions latched since boot."""
    try:
        out = subprocess.run(
            ["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=2
        ).stdout.strip()
        return int(out.split("=")[1], 16)
    except Exception:  # noqa: BLE001 — a missing vcgencmd must not kill the run
        return -1


def _mem_available_kb() -> int:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    except OSError:
        pass
    return -1


def _vm_hwm_kb(pid: int | None) -> int:
    if not pid:
        return -1
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    except OSError:
        pass
    return -1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--interval", type=float, default=0.2)
    ap.add_argument("--pid", type=int, default=None, help="benchmark PID to track VmHWM for")
    ap.add_argument("--max-seconds", type=float, default=86400)
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = (
        ["t", "temp_c"]
        + [f"freq{i}_khz" for i in range(len(CPU_DIRS))]
        + ["throttled", "mem_avail_kb", "vmhwm_kb"]
    )
    t0 = time.time()
    with args.out.open("w", buffering=1) as f:
        f.write(",".join(cols) + "\n")
        # vcgencmd is a subprocess call; sample it every 5th tick so the monitor
        # itself stays cheap enough not to perturb what it is measuring.
        tick = 0
        throttled = _throttled()
        while _running and (time.time() - t0) < args.max_seconds:
            if tick % 5 == 0:
                throttled = _throttled()
            row = [
                f"{time.time() - t0:.3f}",
                f"{_read_int(THERMAL) / 1000.0:.1f}",
                *[str(_read_int(p)) for p in CPU_DIRS],
                str(throttled),
                str(_mem_available_kb()),
                str(_vm_hwm_kb(args.pid)),
            ]
            f.write(",".join(row) + "\n")
            tick += 1
            time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
