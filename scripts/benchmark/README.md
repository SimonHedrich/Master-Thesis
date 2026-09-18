# `scripts/benchmark` — on-device latency, memory and parity

Implements `docs/plans/2026-09-17_on-device-benchmarking-plan.md`: exports the
trained checkpoints to deployable artifacts, times them on a **Raspberry Pi 400**
as a QCS605 proxy, checks that the exported models still compute the same thing
on ARM, and emits the tables and plots `4-Results.tex`
§`sec:results_ax_visio_benchmark` needs.

This closes `TODO.md` §5.2 — before this package, nothing in the repo exported
or timed a model on target hardware.

## Two environments, on purpose

| Scripts | Where | How |
|---|---|---|
| `0-`, `1-`, `4-`, `5-` | this host | `uv run python scripts/benchmark/<script>.py` (project venv) |
| `1b-` | this host | `uv run --script scripts/benchmark/1b-convert_and_verify.py` |
| `2-`, `3-`, `pi/monitor.py` | the Pi | `uv run --script <script>.py` |

The Pi-side scripts are **PEP 723 standalone scripts** — the same exception
`CLAUDE.md` already carves out for `scripts/literature/`, and for the same class
of reason: `pyproject.toml` pins torch to the `pytorch-cu130` index for
`sys_platform == 'linux'`, which on aarch64 resolves to CUDA SBSA wheels that
cannot run on a Pi. They import **no repo code at all** — only the artifacts
`1-export_models.py` produced — so the device needs no checkout.

`1b-` is isolated for a different reason: it needs `onnxruntime` / `onnxslim` /
`pnnx`, and this host is often training, so mutating the shared venv and
lockfile to get them would be both unnecessary and risky.

## Run order

```bash
# ── host ──────────────────────────────────────────────────────────────────
uv run python scripts/benchmark/0-build_subset.py              # 600-image fixed subset
uv run python scripts/benchmark/1-export_models.py             # -> TorchScript + ONNX
uv run --script scripts/benchmark/1b-convert_and_verify.py     # export gate (must PASS)

rsync -az --exclude='*.slim.onnx' scripts/benchmark/exports data/benchmark_subset \
      debian@sh-rpi-400.taile550ef.ts.net:~/benchmark/
rsync -az scripts/benchmark/2-bench_latency.py scripts/benchmark/3-bench_parity.py \
      scripts/benchmark/pi/monitor.py scripts/benchmark/pi/run_all.sh \
      debian@sh-rpi-400.taile550ef.ts.net:~/benchmark/

# ── device (inside tmux; the full sweep takes hours) ──────────────────────
sudo apt-get install -y mesa-vulkan-drivers libvulkan1 vulkan-tools tmux
curl -LsSf https://astral.sh/uv/install.sh | sh
tmux new -s bench
bash ~/benchmark/run_all.sh                                     # latency matrix
uv run --script ~/benchmark/3-bench_parity.py --model yolo26n-direct --runtime onnxruntime
uv run --script ~/benchmark/3-bench_parity.py --model yolo26n-direct --runtime torchscript
uv run --script ~/benchmark/3-bench_parity.py --model yolo26n-kd     --runtime onnxruntime

# ── host ──────────────────────────────────────────────────────────────────
rsync -az debian@sh-rpi-400.taile550ef.ts.net:'~/benchmark/results/*' \
      reports/embedded_benchmark/pi_results/
uv run python scripts/benchmark/4-score_parity.py
uv run python scripts/benchmark/5-report.py
```

## What is measured, and what it is not

Four windows are timed **separately**, because collapsing them is what makes
most published embedded latency numbers incomparable:

| Window | Scope |
|---|---|
| `W_infer` | the runtime forward call only, on an already-preprocessed tensor in memory — **the MLPerf-Tiny-scoped headline** the thesis's §2.3.2 commits to |
| `W_pre` | JPEG decode → letterbox → BGR2RGB → `/255` → CHW, on a median test image *and* on a native 4192×3120 AX Visio still |
| `W_post` | NMS + rescale (YOLOv5s, MegaDetector), rescale only (YOLO26n, whose top-k is in-graph), softmax+top-k (SpeciesNet) |
| `W_e2e` | all three as one wall-clock loop — the number that actually answers "does this fit a 30 ms per-frame budget" |

Held fixed across every cell, per §2.3.2's explicit requirements: batch size 1,
core affinity (`taskset`), thread count (passed to each runtime, never
autodetected), the `performance` governor, and the export format.

**Two confidence regimes, never mixed.** `deployment` (conf 0.25 / IoU 0.45)
produces the headline and the 30 ms verdict; `eval` (conf 0.001 / IoU 0.6,
mirroring `constants.EVAL_*`) is used only for the parity run. The eval
regime's NMS cost over 225 classes at conf 0.001 is pathological on an A72 and
would badly misrepresent deployment latency.

**Adaptive iteration counts.** The protocol's nominal 20 warm-up / 100 measured
is affordable for the students but not for MegaDetector (831 GFLOPs at
1280×1280), so iteration counts shrink to fit a per-window wall-clock budget,
down to a floor that still supports a median. The realised `n` is in every
record; cells with small `n` are flagged in the report rather than quietly
averaged in.

**The thermal gate.** The Pi 400 is passively cooled. A cell is marked invalid
and re-run once after a cooldown if `vcgencmd get_throttled & 0x7` is set, if
median core frequency drops below 1.75 GHz, or if the peak temperature reaches
80 °C. A cell measured while throttled is a record of the room, not of the
model.

## Known limitation: no GPU number

NCNN-Vulkan was the only GPU compute path on this device — VideoCore VI has no
OpenCL driver and ONNX Runtime has no Vulkan execution provider. The device
side works (`vulkaninfo` reports **V3D 4.2.14.0**, an integrated GPU on Vulkan
1.3.354 / Mesa 26.2.2, and the `ncnn` aarch64 wheel ships with Vulkan compiled
in), but `pnnx` crashes converting both detectors, so no NCNN model can be
produced. Full evidence, including the controls that rule out the torch
version, `jit.freeze` and YOLO26's end-to-end head, is in
`reports/embedded_benchmark/ncnn_conversion_finding.md`.

## Parity, not re-evaluation

Predictions should be identical across hardware, so Stage 5 is a **numerical
equivalence** check — did the model survive export and a different instruction
set — not a re-scoring of accuracy. Three tiers: raw-tensor agreement (T1),
per-detection matching at IoU ≥ 0.99 (T2), and subset-mAP delta (T3). Only the
*delta* in T3 is interpreted; a 600-image subset cannot produce a meaningful
absolute 225-class mAP and is never used as one. The host reference is
generated by the project's own `eval_suite/predict.py`, on both GPU and host
CPU, so a delta can be attributed to device/ISA rather than lumped together.

Agreement is checked **scale-aware** (`max_abs <= atol + rtol * max|ref|`), not
against a bare absolute tolerance: the detectors' outputs mix pixel coordinates
(0–1280) with scores (0–1) in one array, so a fixed `atol` is meaningless.

Models exported without trained weights (`weights_source: architecture-only` —
currently YOLOv5s, whose post-Band-A-fix checkpoint is not on this host) are
**valid for latency and excluded from parity**; `3-bench_parity.py` refuses to
run against them and `4-score_parity.py` skips them.

## Artifacts

Everything lands in `reports/embedded_benchmark/` — see `5-report.py` and the
plan's §4.8 for the schema. `scripts/benchmark/exports/` is gitignored.
