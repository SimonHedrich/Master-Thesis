#!/usr/bin/env bash
# Phase B — MegaDetector + parity. RUNS ON THE PI, after the Phase A sweep.
#
# MegaDetector gets a deliberately REDUCED grid, and the reason is recorded
# here rather than left implicit: at 140M parameters and 831 GFLOPs per frame
# at 1280x1280 it is roughly two orders of magnitude more work than the
# students, so the full 3-threads x 3-invocations grid would run for hours
# without changing any conclusion. It is measured to establish that the
# teacher's detector cannot ship on this class of device, which the headline
# 4-thread figure settles on its own.
#
#   4 threads  -> 3 invocations (the headline, full protocol)
#   1, 2 threads -> 1 invocation each (thread-scaling data points, n=1)
#
# Then the parity runs, which must come last: anything executing concurrently
# with the latency sweep would contaminate it.
set -uo pipefail

ROOT="${ROOT:-$HOME/benchmark}"
export PATH="$HOME/.local/bin:$PATH"
cd "$ROOT"

echo "[phaseB] === MegaDetector, headline grid (4 threads, 3 invocations) ==="
MODELS="megadetector" THREADS="4" INVOCATIONS="1 2 3" BUDGET_S=45 \
  OUT="$ROOT/results/latency_raw.jsonl" SUSTAINED_ONLY=0 bash run_all.sh

echo "[phaseB] === MegaDetector, thread-scaling points (1 and 2 threads, n=1) ==="
MODELS="megadetector" THREADS="1 2" INVOCATIONS="1" BUDGET_S=45 \
  OUT="$ROOT/results/latency_raw.jsonl" SUSTAINED_ONLY=0 bash run_all.sh

echo "[phaseB] === Phase C: recover the TorchScript multi-thread cells ==="
bash run_phase_c.sh

echo "[phaseB] === parity predictions ==="
bash run_parity.sh

echo "[phaseB] complete"
python3 progress.py
