#!/usr/bin/env bash
# Phase C — recover the TorchScript multi-thread cells. RUNS ON THE PI.
#
# Every TorchScript cell at 2 and 4 threads died with SIGILL (exit 132) in
# Phase A, on both detectors, while the 1-thread cells ran fine. The Cortex-A72
# is ARMv8.0 -- no `asimddp`, no `asimdhp` -- and PyTorch dispatches
# multi-threaded convolutions to oneDNN/ACL, which appears to select a kernel
# assuming those extensions are present.
#
# `--no-mkldnn` forces the ISA-safe reference path. If a cell still dies the
# finding stands as a runtime-portability result: on this device PyTorch is
# usable single-threaded only, and ONNX Runtime is the only runtime that scales
# across cores.
#
# Diagnostic first, then the re-runs.
set -uo pipefail

ROOT="${ROOT:-$HOME/benchmark}"
export PATH="$HOME/.local/bin:$PATH"
cd "$ROOT"

echo "[phaseC] === diagnostic: is it mkldnn? ==="
for flag in "" "--no-mkldnn"; do
  echo "[phaseC] torchscript 2 threads ${flag:-(default)}"
  taskset -c 0-1 env OMP_NUM_THREADS=2 \
    uv run --script "$ROOT/2-bench_latency.py" \
      --model yolo26n-direct --runtime torchscript --threads 2 --invocation 90 \
      --exports "$ROOT/exports" --subset "$ROOT/benchmark_subset" \
      --out "$ROOT/results/diag.jsonl" --monitor-dir "$ROOT/results/monitor" \
      --budget-s 15 $flag
  echo "[phaseC]   -> exit $?"
done

echo "[phaseC] === re-running TorchScript multi-thread cells with --no-mkldnn ==="
MODELS="${MODELS:-yolo26n-direct yolo26n-kd yolov5s speciesnet megadetector}"
for model in $MODELS; do
  [ -d "$ROOT/exports/$model" ] || continue
  for th in 2 4; do
    for inv in 1 2 3; do
      RUNTIMES=torchscript THREADS="$th" INVOCATIONS="$inv" MODELS="$model" \
        BUDGET_S=45 SUSTAINED_ONLY=0 EXTRA="--no-mkldnn" \
        OUT="$ROOT/results/latency_raw.jsonl" bash run_cell_one.sh "$model" torchscript "$th" "$inv"
    done
  done
done

echo "[phaseC] complete"
python3 progress.py
