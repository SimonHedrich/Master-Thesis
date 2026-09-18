#!/usr/bin/env bash
# Stage 5 driver — parity predictions over the fixed subset. RUNS ON THE PI.
#
# Run this only AFTER the latency sweep has finished: anything else executing
# concurrently would contaminate the latency measurements.
#
# Uses the `eval` regime (conf 0.001 / IoU 0.6 / max_det 100), matching
# `constants.EVAL_*`, so the predictions are directly comparable with the host
# reference the project's own eval_suite produces.
#
# Usage (on the Pi, inside tmux):
#   bash ~/benchmark/run_parity.sh
set -uo pipefail

ROOT="${ROOT:-$HOME/benchmark}"
export PATH="$HOME/.local/bin:$PATH"

MODELS="${MODELS:-yolo26n-direct yolo26n-kd}"
RUNTIMES="${RUNTIMES:-onnxruntime torchscript}"
THREADS="${THREADS:-4}"

for model in $MODELS; do
  [ -d "$ROOT/exports/$model" ] || { echo "[parity] skip $model (not exported)"; continue; }
  for rt in $RUNTIMES; do
    # PyTorch's multi-threaded CPU path dies with SIGILL on this ARMv8.0 core
    # (see run_phase_c.sh), so TorchScript runs with mkldnn disabled here.
    extra=""
    [ "$rt" = "torchscript" ] && extra="--no-mkldnn"
    echo "[parity] $(date +%H:%M:%S)  $model / $rt $extra"
    taskset -c 0-3 env OMP_NUM_THREADS="$THREADS" \
      uv run --script "$ROOT/3-bench_parity.py" \
        --model "$model" --runtime "$rt" --threads "$THREADS" $extra \
        --exports "$ROOT/exports" --subset "$ROOT/benchmark_subset" \
        --out "$ROOT/results/predictions_${model}_${rt}_pi.json"
    sleep 20
  done
done
echo "[parity] done"
ls -la "$ROOT/results/"predictions_*_pi.json 2>/dev/null
