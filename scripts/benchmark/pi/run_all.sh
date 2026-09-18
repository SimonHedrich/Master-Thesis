#!/usr/bin/env bash
# Stage 3/4 driver — the full latency sweep. RUNS ON THE RASPBERRY PI.
#
# One `2-bench_latency.py` process per cell, which is what makes the
# "3 independent process invocations" in the protocol mean something: thread
# pools, allocator state and page cache all start fresh each time.
#
# Holds fixed the two things the thesis's Literature Review 2.3.2 says a
# benchmark protocol must hold fixed: core affinity (taskset) and thread count
# (passed explicitly to each runtime, never autodetected). Forces the
# `performance` governor for the duration and restores `ondemand` on exit,
# including on Ctrl-C, so an aborted sweep doesn't silently leave the machine
# in a different power state for the next one.
#
# Usage (on the Pi, inside tmux):
#   bash ~/benchmark/run_all.sh
#   MODELS="yolo26n-direct" INVOCATIONS="1" bash ~/benchmark/run_all.sh   # smoke
#   SUSTAINED_ONLY=1 bash ~/benchmark/run_all.sh
set -uo pipefail

ROOT="${ROOT:-$HOME/benchmark}"
export PATH="$HOME/.local/bin:$PATH"

MODELS="${MODELS:-yolo26n-direct yolo26n-kd yolov5s speciesnet megadetector}"
RUNTIMES="${RUNTIMES:-onnxruntime torchscript}"
THREADS="${THREADS:-1 2 4}"
INVOCATIONS="${INVOCATIONS:-1 2 3}"
BUDGET_S="${BUDGET_S:-45}"
COOLDOWN="${COOLDOWN:-20}"
COOL_TO_C="${COOL_TO_C:-55}"
COOL_MAX_WAIT="${COOL_MAX_WAIT:-300}"
COOLDOWN_INVALID="${COOLDOWN_INVALID:-120}"
SUSTAINED_ONLY="${SUSTAINED_ONLY:-0}"
OUT="${OUT:-$ROOT/results/latency_raw.jsonl}"

mkdir -p "$ROOT/results/monitor"

restore_governor() {
  echo "[driver] restoring ondemand governor"
  for c in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor; do
    echo ondemand | sudo tee "$c" >/dev/null 2>&1
  done
}
trap restore_governor EXIT INT TERM

echo "[driver] forcing performance governor"
for c in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor; do
  echo performance | sudo tee "$c" >/dev/null 2>&1
done
grep -h . /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor | sort -u | sed 's/^/[driver] governor: /'

affinity_for() {  # thread count -> taskset cpu list
  case "$1" in
    1) echo "0" ;;
    2) echo "0-1" ;;
    *) echo "0-3" ;;
  esac
}

# Fairness pre-gate: never START a cell on a hot SoC. The post-hoc thermal
# gate catches a cell that throttled while running; this stops the previous
# cell's heat from biasing the next one before it begins.
wait_until_cool() {
  local waited=0 t
  while [ "$waited" -lt "$COOL_MAX_WAIT" ]; do
    t=$(vcgencmd measure_temp 2>/dev/null | sed "s/[^0-9.]//g" | cut -d. -f1)
    [ -z "$t" ] && return 0
    [ "$t" -le "$COOL_TO_C" ] && { [ "$waited" -gt 0 ] && echo "[driver] cooled to ${t}C after ${waited}s"; return 0; }
    sleep 10; waited=$((waited + 10))
  done
  echo "[driver] WARNING: still ${t}C after ${waited}s wait — proceeding, gate will flag if it throttles"
}

run_cell() {  # model runtime threads invocation extra...
  local model="$1" rt="$2" th="$3" inv="$4"; shift 4
  wait_until_cool
  local cpus; cpus="$(affinity_for "$th")"
  echo "[driver] $(date +%H:%M:%S)  $model / $rt / ${th}t / inv$inv  (cpus $cpus)"
  taskset -c "$cpus" env OMP_NUM_THREADS="$th" OPENBLAS_NUM_THREADS="$th" \
      MKL_NUM_THREADS="$th" \
    uv run --script "$ROOT/2-bench_latency.py" \
      --model "$model" --runtime "$rt" --threads "$th" --invocation "$inv" \
      --exports "$ROOT/exports" --subset "$ROOT/benchmark_subset" \
      --out "$OUT" --monitor-dir "$ROOT/results/monitor" \
      --budget-s "$BUDGET_S" "$@"
  local rc=$?
  if [ $rc -eq 3 ]; then
    echo "[driver] cell INVALID (thermal gate) — cooling ${COOLDOWN_INVALID}s and retrying once"
    sleep "$COOLDOWN_INVALID"
    taskset -c "$cpus" env OMP_NUM_THREADS="$th" OPENBLAS_NUM_THREADS="$th" \
        MKL_NUM_THREADS="$th" \
      uv run --script "$ROOT/2-bench_latency.py" \
        --model "$model" --runtime "$rt" --threads "$th" --invocation "$inv" \
        --exports "$ROOT/exports" --subset "$ROOT/benchmark_subset" \
        --out "$OUT" --monitor-dir "$ROOT/results/monitor" \
        --budget-s "$BUDGET_S" "$@"
    rc=$?
    [ $rc -eq 3 ] && echo "[driver] still invalid — kept and flagged in the record"
  elif [ $rc -ne 0 ]; then
    echo "[driver] cell FAILED (rc=$rc) — continuing"
  fi
  sleep "$COOLDOWN"
}

started=$(date +%s)

if [ "$SUSTAINED_ONLY" != "1" ]; then
  for model in $MODELS; do
    [ -d "$ROOT/exports/$model" ] || { echo "[driver] skip $model (not exported)"; continue; }
    for rt in $RUNTIMES; do
      for th in $THREADS; do
        for inv in $INVOCATIONS; do
          run_cell "$model" "$rt" "$th" "$inv"
        done
      done
    done
  done
fi

# Thermal-drift run: headline thread count, one runtime, once per model.
echo "[driver] === sustained / thermal-drift runs ==="
for model in $MODELS; do
  [ -d "$ROOT/exports/$model" ] || continue
  run_cell "$model" onnxruntime 4 1 --sustained
done

echo "[driver] done in $(( ($(date +%s) - started) / 60 )) min — results in $OUT"
wc -l "$OUT" 2>/dev/null | sed 's/^/[driver] records: /'
