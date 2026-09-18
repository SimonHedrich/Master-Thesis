#!/usr/bin/env bash
# Run a single benchmark cell with the thermal pre-gate. RUNS ON THE PI.
# Factored out of run_all.sh so Phase C can re-run individual cells without
# re-entering the full sweep loop.
# Usage: run_cell_one.sh <model> <runtime> <threads> <invocation> [extra args...]
set -uo pipefail
ROOT="${ROOT:-$HOME/benchmark}"
export PATH="$HOME/.local/bin:$PATH"
model="$1"; rt="$2"; th="$3"; inv="$4"; shift 4
EXTRA="${EXTRA:-}"
OUT="${OUT:-$ROOT/results/latency_raw.jsonl}"
COOL_TO_C="${COOL_TO_C:-55}"

case "$th" in 1) cpus="0";; 2) cpus="0-1";; *) cpus="0-3";; esac

waited=0
while [ "$waited" -lt 300 ]; do
  t=$(vcgencmd measure_temp 2>/dev/null | sed 's/[^0-9.]//g' | cut -d. -f1)
  [ -z "$t" ] && break
  [ "$t" -le "$COOL_TO_C" ] && break
  sleep 10; waited=$((waited + 10))
done

echo "[cell] $(date +%H:%M:%S)  $model / $rt / ${th}t / inv$inv  $EXTRA $*"
taskset -c "$cpus" env OMP_NUM_THREADS="$th" OPENBLAS_NUM_THREADS="$th" MKL_NUM_THREADS="$th" \
  uv run --script "$ROOT/2-bench_latency.py" \
    --model "$model" --runtime "$rt" --threads "$th" --invocation "$inv" \
    --exports "$ROOT/exports" --subset "$ROOT/benchmark_subset" \
    --out "$OUT" --monitor-dir "$ROOT/results/monitor" \
    --budget-s "${BUDGET_S:-45}" $EXTRA "$@"
rc=$?
[ $rc -ne 0 ] && echo "[cell] FAILED rc=$rc"
sleep "${COOLDOWN:-20}"
exit $rc
