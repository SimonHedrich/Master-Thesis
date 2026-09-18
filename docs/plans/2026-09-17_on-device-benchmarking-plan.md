# On-Device Benchmarking Plan — Raspberry Pi 400 as QCS605 Proxy

**Date:** 2026-09-17
**Status:** Plan only — no implementation yet.
**Closes:** `TODO.md` §5.2 (export + on-device benchmarking)
**Feeds:** `thesis/manuscript/chapters/4-Results.tex` §`sec:results_ax_visio_benchmark`

---

## 1. Context

The thesis is titled *"Optimizing Deep Learning Object Detection Models for
Real-Time Inference on Embedded Hardware"*, and
`2-Literature_Review.tex` §`sec:lit_embedded_inference` asserts a "fixed, tight
per-frame latency budget" as the premise that rules out two-stage detectors and
the MegaDetector+SpeciesNet ensemble. That premise is currently **unbacked by
any measurement**. `TODO.md` §5.2 says so plainly:

> "Nothing in the repo yet exports or times a model on real target hardware —
> this is the thesis's core 'real-time inference on embedded hardware' claim and
> currently has no artifact behind it."

Verified during the analysis behind this plan: there is no export code, no
`.onnx` / `.tflite` / `.param` artifact, and no inference-timing code anywhere
in `scripts/`. `model_exports/` is a misnomer — it holds training-run
checkpoints. The only file in the repo with "benchmark" in its name
(`scripts/training/yolov5s/eval_suite/benchmarks/bench_scoring_engines.py`)
times the *mAP scoring engine*, not a model. The only per-model latency figures
recorded anywhere are published third-party numbers and RTX 3060 forward passes
from a since-deleted smoke test.

A Raspberry Pi 400 is now reachable at `debian@sh-rpi-400.taile550ef.ts.net`.
This plan turns §5.2 into a measured artifact: an export pipeline, a benchmark
harness with a defensible protocol, an on-device correctness check, and the
numbers `sec:results_ax_visio_benchmark` needs.

**Outcome: the thesis's framing premise moves from *asserted* to *measured*.**

---

## 2. Device facts (measured over SSH, 2026-09-17)

| | Value |
|---|---|
| Model | Raspberry Pi 400 Rev 1.0 (BCM2711) |
| CPU | 4× Cortex-A72 @ 1.8 GHz (`arm_boost=1`), 1 MiB shared L2 |
| ISA features | `fp asimd evtstrm crc32 cpuid` — **no `asimddp` (dot product), no `fphp`/`asimdhp` (fp16 arithmetic)** |
| RAM | 3.7 GiB + 2.0 GiB swap |
| GPU | VideoCore VI (V3D 4.2), `/dev/dri/{card0,card1,renderD128}`, `vc4-kms-v3d` |
| OS | Debian 13 trixie, kernel 6.12.47 aarch64 |
| Python | 3.13.5 system; **no uv, no pip packages, no cmake, no docker** |
| Governor | `ondemand`, 600–1800 MHz; `performance` available |
| Thermal | 39.4 °C idle, `get_throttled=0x0`, passive heatspreader only |
| Disk | 108 GB free |
| sudo | passwordless |
| Link | Tailscale, **≈2.4 MB/s** (~19 Mbit/s) from the A40 host |

**Toolchain viability confirmed (not assumed):**

- `onnxruntime 1.30.0`, `torch 2.14.0`, `ncnn`, `ai-edge-litert` and `openvino`
  all publish **cp313 manylinux aarch64** wheels.
- The official `ncnn` aarch64 cp313 wheel is **built with Vulkan on** — 130
  `vk*` symbols found in its `.so`. No source build is needed for the GPU path.
- `mesa-vulkan-drivers 25.0.7-2+rpt4`, `libvulkan1 1.4.309.0` and `vulkan-tools`
  are installable from the Pi's apt repos → the V3D Vulkan path is available.
- `pnnx` ships both x86_64 and aarch64 wheels → ONNX→NCNN conversion runs on the
  GPU host, not on the Pi.

**Two ISA facts that shape the whole analysis**, and that must be stated in the
thesis because they are exactly what separates this device from both the
originally-planned Pi 5 and the real target:

1. The A72 lacks ARMv8.2 `SDOT`, so INT8 GEMM falls back to widening `SMLAL`
   paths. INT8 speedups here would be ~1.3–2×, not the 3–4× an A76 or the
   Hexagon 685's HVX would give.
2. The A72 has no fp16 arithmetic, so fp16 is storage-only — no speed benefit.

---

## 3. Decisions taken

1. **The Pi 400 is the proxy, and the proxy argument gets reframed around it.**
2. **Scope = students + teacher components:** YOLO26n (direct-FT and KD),
   YOLOv5s, the SpeciesNet classifier, MegaDetector v5, plus a composed MD+SN
   per-frame estimate.
3. **FP32 runtime matrix + a Vulkan GPU probe.** No PTQ/QAT — consistent with
   §2.3.2's existing "does not implement PTQ, QAT, or any other quantization"
   sentence, which therefore needs no change.

### 3.1 The reframed proxy argument

`docs/2026-03-09_hardware-proxy-selection.md` already contains everything
needed; it simply recommended a different device, and did so on
software-maturity grounds rather than on fidelity. Its own comparison table
rates the Pi 4/400 at **−10 % CPU / −20 % GPU vs. the QCS605** — and notes the
400 is the faster of that pair at 1.8 GHz vs 1.5 GHz — against the Pi 5's
**+60 % / +40 %**.

So the translation rule changes direction:

| | Pi 5 (planned) | Pi 400 (actual) |
|---|---|---|
| CPU vs QCS605 | +60 % faster | 10–15 % slower |
| Role | performance **ceiling** | near-parity, mild **floor** |
| Translation | `QCS605 ≈ Pi5 × 1.6` → discount to 60 % | `QCS605 ≈ Pi400 × 0.85–0.90` |
| Pass band for the ≤30 ms budget | Pi 5 ≤ ~18–19 ms | **Pi 400 ≤ ~33–35 ms** |

This is the *stronger* of the two claims: a model fast enough on the Pi 400 is
fast enough on the QCS605 with near-certainty, whereas the Pi 5 argument always
required trusting a 1.6× discount in the unfavourable direction. It must be
reported as a **project design estimate, not a vendor-certified figure**, per
the `thesis-writing` skill's `claims.md` taxonomy.

**Non-negotiable caveat to state alongside every number:** the Pi 400 has *no
analogue* to the QCS605's Hexagon 685 DSP (~2.1 TOPS), and its VideoCore VI is
far weaker than the Adreno 615. **The translation rule applies to the CPU path
only.** The QCS605's accelerated path is not proxied by this device at all and
remains unmeasured, as does any INT8/DSP speedup.

---

## 4. Design

### 4.1 Package layout and run conventions

New package `scripts/benchmark/`, structured as a numbered pipeline (per
`CLAUDE.md`, numbered scripts use the path form, not `-m`):

```
scripts/benchmark/
├── README.md                  # run order, protocol, artifact schema
├── constants.py               # protocol constants, model registry
├── 0-build_subset.py          # host: materialize the fixed benchmark subset
├── 1-export_models.py         # host: checkpoint → TorchScript + ONNX + NCNN
├── 2-bench_latency.py         # PI:   the latency matrix               (PEP 723)
├── 3-bench_parity.py          # PI:   predictions JSON over the subset (PEP 723)
├── 4-score_parity.py          # host: score Pi predictions vs host reference
├── 5-report.py                # host: tables + plots into reports/
└── pi/
    ├── monitor.py             # PI:   thermal/freq/throttle sampler    (PEP 723)
    └── run_all.sh             # PI:   tmux driver for the full sweep
```

**Host scripts** run the normal way:
`uv run python scripts/benchmark/1-export_models.py`.

**Pi-side scripts use PEP 723 inline dependencies with `uv run --script`.** This
is the same exception already carved out for `scripts/literature/` in
`CLAUDE.md`, and for the same class of reason: `pyproject.toml` pins `torch` to
the `pytorch-cu130` index for `sys_platform == 'linux'`, which on aarch64 would
try to pull CUDA SBSA wheels that cannot work on a Pi. A self-contained script
environment sidesteps this without touching the shared project lockfile.

The Pi scripts deliberately take **no dependency on the repo's model code** —
they consume TorchScript / ONNX / NCNN artifacts only. That keeps the Pi
environment to `torch` (as a TorchScript runtime), `onnxruntime`, `ncnn`,
`numpy`, `opencv-python-headless` and `psutil`, and nothing else. In particular
neither `ultralytics` nor `yolov5` needs to be installed on the device.

### 4.2 Stage 0 — Pi environment bring-up

One idempotent setup step, recorded in `scripts/benchmark/README.md`:

```bash
sudo apt-get install -y mesa-vulkan-drivers libvulkan1 vulkan-tools tmux
curl -LsSf https://astral.sh/uv/install.sh | sh     # uv, for --script
vulkaninfo --summary        # must report V3D 4.2 / Mesa 25.0.7
```

Everything else is resolved per-script by `uv run --script`.

### 4.3 Stage 1 — export (host)

`1-export_models.py` reconstructs each architecture and exports it. It **reuses
the existing checkpoint loaders rather than reimplementing them**:

- `scripts/training/yolov5s/eval_suite/predict.py::load_checkpoint`
- `scripts/training/yolo26n/eval_suite/predict.py::load_checkpoint`
- `scripts/training/yolo26n/yolo26n_model.py::yolo26n_model`
- `scripts/training/yolov5s/yolov5s_model.py::yolov5s_model`

This matters: both YOLO checkpoints are bare `state_dict`s for hand-built
`DetectionModel` / yolov5 `Model` instances (the pipelines bypass Ultralytics'
high-level `Model`/`Trainer` API entirely), so `YOLO("best.pt").export()` will
**not** work off them.

| Model | Source | Input | Export targets |
|---|---|---|---|
| YOLO26n direct-FT | `yolo26n/model_exports/yolo26n-bs32-20260910-212812/best.pt` | 1×3×640×640 | TorchScript, ONNX, NCNN |
| YOLO26n KD | `yolo26n-kd-bs16-20260916-101612/best.pt` *(on `gpu-server`)* | 1×3×640×640 | TorchScript, ONNX, NCNN |
| YOLOv5s | `yolov5s/model_exports/yolov5s-20260909-230900/best.pt` | 1×3×640×640 | TorchScript, ONNX, NCNN |
| SpeciesNet classifier | `teacher_finetune/model_exports/teacher-finetune-ff0.75-bs64-20260909-231034/best.pt` | 1×3×480×480 | TorchScript, ONNX |
| MegaDetector v5a | `~/.cache/torch/hub/checkpoints/md_v5a.0.0.pt` (280 MB) | 1×3×1280×1280 | TorchScript, ONNX |

Export details:

- `torch.onnx.export`, opset 17, **static** batch-1 shapes (embedded inference
  is sequential; dynamic axes only add runtime dispatch overhead),
  `do_constant_folding=True`, then `onnxslim`.
- **YOLO26n's head is `end2end=True` / NMS-free** — its score-ranked top-k runs
  *inside* the graph and exports with it. **YOLOv5s' NMS is external** and is
  timed separately as post-processing. This asymmetry is the single most
  important thing to hold straight when comparing the two, and must be stated
  explicitly in the results text.
- ONNX → NCNN (`.param` + `.bin`) via `pnnx` **on the host**, not on the Pi.
- Artifacts land in `scripts/benchmark/exports/<model>/<run_id>/` (gitignored,
  same convention as `*.pt`), each with a `manifest.json` recording the source
  checkpoint path + sha256, exporter versions, opset, and input shape.

**Export verification gate (host, before anything ships to the Pi):** for each
model, run the original PyTorch module and the exported ONNX (CPU
ExecutionProvider) on the same fixed input and assert `max|Δ| ≤ 1e-4` on raw
outputs. A failure here is an export bug, not a device finding, and must be
caught before it can contaminate the device numbers.

### 4.4 Stage 2 — the benchmark subset (host)

`0-build_subset.py` materializes one fixed, seeded subset (`SEED=42`, matching
the evaluation strategy's statistical-hygiene rules) and writes a COCO
annotation file for it in the same schema as `data/real/annotations_test.json`:

- **500 real** images: 120 each from bands A/B/C/D + 20 negatives, sampled
  uniformly within band.
- **100 synthetic** images sampled uniformly across the 225 classes.
- Output: `data/benchmark_subset/` (images copied flat, `file_name` rewritten)
  + `annotations_subset.json` + `subset_manifest.json` (ids, bands, sha256).

Size: ~250 MB at the measured ~420 KB/image mean → **≈2 min** over the 2.4 MB/s
link. Transferred once via `rsync -avz` alongside the exports.

Two fixed single images are also shipped for the fixed-input timing loops: one
median-sized real test image (500×375, the measured median of the real test set)
and one native AX Visio still (4192×3120, from `resources/AX_Visio_images/`) to
measure realistic on-device decode cost.

### 4.5 Stage 3 — the latency benchmark (Pi)

#### What is timed

Four separately-recorded windows, because collapsing them is precisely what
makes most published embedded latency numbers incomparable:

| Window | Definition |
|---|---|
| `W_infer` | Runtime forward call only (`session.run` / `module.forward`), on an already-preprocessed tensor resident in memory. **This is the MLPerf-Tiny-scoped headline**, matching the commitment already written into §2.3.2. |
| `W_pre` | JPEG decode → letterbox to target → BGR→RGB → `/255` → HWC→CHW → contiguous. Measured over real subset images *and* over the 4192×3120 AX Visio still. |
| `W_post` | YOLOv5s: `non_max_suppression` + coordinate rescale. YOLO26n: rescale only (top-k is in-graph). |
| `W_e2e` | The three in sequence, measured as one wall-clock loop (not the sum of medians) so allocator and cache effects are captured. |

Reporting `W_infer` satisfies the MLPerf discipline §2.3.2 commits to; reporting
`W_e2e` alongside it is what actually answers "does this meet a 30 ms per-frame
budget on a product". Both go in the table.

#### Two threshold regimes

| Regime | conf | iou | max_det | Used for |
|---|---|---|---|---|
| `deployment` | 0.25 | 0.45 | 100 | The headline latency and the ≤30 ms verdict |
| `eval` | 0.001 | 0.6 | 100 | The parity run only (matches `constants.EVAL_*`) |

These are reported separately and never mixed. The `eval` regime's NMS cost over
225 classes at conf 0.001 is pathological on an A72 and would misrepresent
deployment latency by a wide margin; measuring it anyway and showing the gap is
itself a useful result.

#### Protocol

Follows `docs/2026-03-12_research-and-experimentation-plan.md` §4.2, extended
where that protocol is silent:

- **Batch size 1** throughout (embedded inference is sequential).
- **20 warm-up iterations, 100 measured**, per cell.
- **3 independent process invocations** per cell. Headline = median of the
  pooled 300; between-invocation spread reported so process-level variance is
  visible rather than hidden.
- Report **median, IQR, p95, min, mean** — median ± IQR is the headline.
- **Thread sweep: 1 / 2 / 4.** Headline = 4 (full device). The **2-thread number
  is the more transferable one** and is called out as such: the QCS605 has 2 big
  A75 cores + 6 little A55s, so a 4×A72 figure and a 2-thread figure bracket it
  from different directions.
- **Core affinity pinned** with `taskset -c` (`0-3`, `0-1`, `0`); runtime thread
  count set explicitly per backend (`OMP_NUM_THREADS`,
  `sess_options.intra_op_num_threads`, `ncnn.Option.num_threads`) — never left
  to autodetect.
- **Governor forced to `performance`** before the sweep and restored to
  `ondemand` after; achieved frequency verified and recorded, not assumed.
- **Sustained/thermal run:** 600 additional iterations per model at 4 threads on
  ONNX Runtime. Reports the median of the first 100 vs the last 100 (the
  throttling drift) and the peak temperature reached.
- **Peak RSS** per cell via `resource.getrusage(RUSAGE_SELF).ru_maxrss`, plus
  the sampler's `VmHWM` trace. This is what gets checked against the **≤500 MB**
  memory budget from `docs/2026-03-10_object-detection-models-for-embedded-systems.md`.

#### The runtime × model matrix

| Model | TorchScript | ONNX Runtime | NCNN (CPU) | NCNN (Vulkan) |
|---|---|---|---|---|
| YOLO26n direct-FT | ✓ | ✓ | ✓ | probe |
| YOLO26n KD | ✓ | ✓ | ✓ | probe |
| YOLOv5s | ✓ | ✓ | ✓ | probe |
| SpeciesNet classifier (480) | ✓ | ✓ | — | — |
| MegaDetector v5 (1280) | ✓ | ✓ | — | — |

× {1, 2, 4} threads × {`deployment`} regime × 3 invocations.

NCNN options are forced to true fp32 (`use_fp16_packed` / `use_fp16_storage` /
`use_fp16_arithmetic` = `False`). The A72 has no fp16 arithmetic anyway, and
leaving NCNN's defaults on would silently change the numerics and invalidate the
parity comparison.

**The MD+SN ensemble is composed, not run as a pipeline.** Per-frame cost is
`t_MD(1280) + k × t_SN(480)`, where `k` = the mean animal crops per image, read
from the existing
`scripts/training/megadet_speciesnet_ensemble/model_exports/.../predictions_real.json`.
This is labelled a *derived estimate* in the report, not a measurement, and the
two measured components are reported beside it.

#### Monitoring

`pi/monitor.py` runs as a sibling process for the duration of every cell,
sampling at 200 ms into `monitor_<cell>.csv`:

- `/sys/class/thermal/thermal_zone0/temp`
- `/sys/devices/system/cpu/cpu{0..3}/cpufreq/scaling_cur_freq`
- `vcgencmd get_throttled` (raw bitmask)
- `MemAvailable` from `/proc/meminfo`, and the bench process's `VmHWM`

A per-cell summary (max temp, min/median freq, throttle bits, peak RSS) is
folded into the result record.

**Validity gate — a cell is marked invalid and re-run once after a 120 s
cooldown if:** `get_throttled & 0x7 != 0`, **or** median core frequency
< 1.75 GHz, **or** max temperature ≥ 80 °C. A cell that fails twice is reported
as throttled with its numbers marked accordingly, rather than silently kept.
Thermal throttling on a passively-cooled Pi 400 is the single biggest threat to
the validity of this benchmark, which is why it gets a hard gate rather than a
footnote.

The whole sweep runs under `tmux` on the Pi (matching `TERMINALS.md`), with a
`status.json` heartbeat the host-side driver polls over SSH.

### 4.6 Stage 4 — the Vulkan GPU probe

The Pi 400's only usable GPU compute path is **Vulkan via mesa's v3dv driver**,
consumed by NCNN. There is no OpenCL for VideoCore VI (VC4CL covers the Pi 3's
VC4 only), so no ARM Compute Library or OpenCL route exists.

- Enable with `ncnn.Net().opt.use_vulkan_compute = True` once
  `mesa-vulkan-drivers` is installed; the stock wheel already has Vulkan
  compiled in.
- Same protocol as the CPU cells, plus GPU-specific records: the `vulkaninfo`
  device/driver string, and whether any layer fell back to CPU.
- **Time-boxed.** If a model fails to run on Vulkan (an unsupported op in
  YOLO26n's end-to-end head is the likely failure mode), that is recorded as a
  result — "op X unsupported by the v3dv/NCNN Vulkan path" — and the probe moves
  on. No source builds, no op re-implementation.

**Expected outcome, stated up front so it is not later mistaken for a bug:**
V3D's fp32 throughput (~13.5 GFLOPS) is *below* four A72 NEON cores (~28 GFLOPS
peak), so the GPU path will most likely be **slower** than the CPU. That is a
legitimate measured negative result and worth reporting. It must come with the
explicit warning that **it does not transfer to the QCS605**, whose Adreno 615
is a far more capable part — the Pi 400 GPU number says nothing about the
target's GPU.

### 4.7 Stage 5 — correctness parity check

No qualitative evaluation of the predictions is needed: predictions should be
identical across hardware, and the results serve as a test that each model runs
*correctly* after export and on a different ISA. That is a **numerical
equivalence** check, not a re-evaluation, so it is designed as three tiers of
increasing tolerance:

| Tier | Check | Pass criterion |
|---|---|---|
| **T1 tensor** | Raw model outputs on one fixed preprocessed input, host-PyTorch-fp32 vs each Pi runtime | `max\|Δ\| ≤ 1e-3` (ORT / TorchScript); NCNN reported separately at a looser tolerance |
| **T2 detection** | Final detections over the 600-image subset, host reference vs Pi | ≥ 99 % of host detections matched by a Pi detection at IoU ≥ 0.99 and `\|Δscore\| ≤ 0.01` |
| **T3 metric** | Subset mAP scored from each side's predictions JSON | `\|Δ mAP\| ≤ 0.002` |

`3-bench_parity.py` on the Pi emits predictions in the **frozen JSON contract**
already defined by `eval_suite/predict.py` — `{checkpoint, annotations, eval:
{conf_thres, iou_thres, max_det, image_size}, num_images, predictions:
[{image_id, category_id, bbox, score}]}` — so T3 is a straight reuse of
`eval_suite/run_evaluation.py::evaluate_from_predictions` on the host. **No
scoring runs on the Pi**; this is exactly the predict-once / score-separately
split the eval suite was built around.

The host reference is generated **three ways** — GPU PyTorch, host-CPU PyTorch,
and host-CPU ONNX Runtime — so that any delta can be attributed to *device* vs
*runtime* vs *export* rather than lumped together.

A note on the T3 number: a 600-image subset cannot produce a meaningful 225-class
mAP in absolute terms, and it is not used as one. Only the **delta** between two
prediction sets over the same images is interpreted.

### 4.8 Stage 6 — reporting

`5-report.py` joins everything into `reports/embedded_benchmark/`:

| Artifact | Contents |
|---|---|
| `device_profile.json` | Full Pi fingerprint: uname, cpuinfo flags, governor, achieved freqs, mesa/Vulkan versions, every runtime version, link speed |
| `latency_raw.jsonl` | One record per (model, runtime, threads, window, invocation) with all raw per-iteration timings — the reproducibility backstop |
| `latency_summary.csv` | median / IQR / p95 / mean per cell, derived FPS, peak RSS MB, file size MB, params, GFLOPs |
| `thermal_summary.csv` | Per-cell max temp, freq stats, throttle bits, validity verdict, sustained-run drift |
| `monitor_<cell>.csv` | Raw 200 ms sampler traces |
| `parity_summary.csv` | T1/T2/T3 results per (model, runtime) with pass/fail and actual values |
| `embedded_benchmark.md` | The human-readable report: protocol, tables, the QCS605 translation, the verdict, and the explicit non-claims |
| `plots/embedded_latency_vs_map.png` | Accuracy-vs-latency Pareto scatter, 30 ms threshold as a vertical rule |
| `plots/embedded_latency_by_runtime.png` | Grouped bars, model × runtime × threads |

Plot generation reuses the established pattern in
`scripts/synthetic_model_comparison/6-compare_maxlen_cells.py` (matplotlib →
`reports/`, PNGs then copied by hand into `thesis/manuscript/figures/plots/`).
The Pareto plot is the direct structural analogue of the already-drafted
`fig:generator-cost-vs-map`, and is the single artifact that connects the
accuracy chapters to the title's "real-time on embedded hardware" claim.

mAP values for the Pareto plot come from the existing full-test-set eval reports
(`<run_dir>/eval_best/evaluation_report.json`), **not** from the benchmark
subset — the subset is for parity only.

---

## 5. What the thesis gets

### 5.1 The verdict

`4-Results.tex` §`sec:results_ax_visio_benchmark` demands: *"State plainly
whether the unoptimized model met a stated real-time threshold."* The threshold
has to be stated in the manuscript for the first time, with provenance:

- **≤30 ms per frame at 640×640** on the QCS605 — from
  `docs/2026-03-10_object-detection-models-for-embedded-systems.md` §1.
- **≤500 MB memory budget** — same source.
- Both are **project design targets, not vendor-certified figures**, and must be
  labelled as such per `claims.md`'s claim taxonomy.
- Translated to the measurement device: **Pi 400 ≤ ~33–35 ms** implies ≤30 ms on
  the QCS605 CPU.

Published reference numbers predict the FP32 verdict will be a clear miss —
YOLO26n is recorded at 67.7 ms / 14.8 FPS on a **Pi 5** with NCNN
(`docs/progress_notes/2026-03-18_speciesnet-pipeline-and-experiment-design.md`),
and the Pi 400 is the slower device. A measured miss, honestly reported with the
quantization headroom named, is the correct and expected output of a feasibility
check. This plan does not assume the result either way.

### 5.2 Text to write and to change

| File | Change |
|---|---|
| `4-Results.tex` §`sec:results_ax_visio_benchmark` | Write the subsection: export pipeline, protocol, latency table, Pareto figure, verdict, non-claims |
| `4-Results.tex` §`sec:results_general_observations` | Closing statement that the embedded result is preliminary/indicative |
| `2-Literature_Review.tex` §`sec:lit_embedded_inference` | **Rewrite the final paragraph.** It currently argues the Pi 5 as a +60 % ceiling with a 0.6× discount. Replace with the Pi 400 near-parity/floor argument of §3.1, keeping the software-maturity reasoning (which still applies) and adding the no-DSP-analogue caveat |
| `1-Introduction.tex` §`sec:objective` | Number an embedded-feasibility RQ alongside the distillation RQ — the title promises it, and this work is what backs it |
| `5-Discussion_and_Conclusion.tex` | Answer that RQ; Limitations (FP32 only, single device, no DSP proxy, small-n); Further Work (PTQ/QAT, the deferred SNPE/Hexagon path) |
| `36-Model_Training_and_Experiment_Tracking.tex` | **Resolve an inconsistency:** it stubs a `\subsection{Quantization-Aware Training}` that §2.3.2 explicitly states was never carried out. Drop or re-scope it |
| `4-Results.tex` benchmark stub wording | **Resolve a second inconsistency:** the stub names *YOLOv5s* as the exported model, while §3.5 and `docs/2026-09-07_model-comparison-teacher-and-students.md` treat YOLO26n as the deployment candidate. This plan benchmarks both; the text should say so explicitly rather than let the discrepancy stand |

Every number carries its artifact path in a `%` comment on the same line, per
`claims.md` §2 — e.g. `% reports/embedded_benchmark/latency_summary.csv`.

**Non-claims to state out loud** (per `claims.md` §1 and §6): FP32 unoptimized
only, no PTQ/QAT/pruning; a bounded measurement campaign on one device, not a
benchmarking study; **no power or thermal-envelope measurement** (the Pi has no
instrumented rail, and the AX Visio has no documented power budget anywhere in
the repo); and the QCS605's DSP/GPU path is entirely unproxied.

### 5.3 Docs to update

- `docs/2026-03-09_hardware-proxy-selection.md` — dated addendum: the device
  actually used is a Pi 400; the −10 % / −20 % delta replaces the Pi-5 60 % cap
  rule as the translation to the QCS605.
- `docs/<date>_embedded-benchmark-results.md` — the results writeup; row in
  `docs/README.md`.
- `CLAUDE.md` §"Running Code" — a second `uv run --script` exception for
  `scripts/benchmark/` Pi-side scripts, with the cu130-index reason.
- `TODO.md` §5.2 → done, following the dated-note convention used throughout
  that file. §5.1 (QAT deferral) gets a note that this plan deliberately
  excludes quantization.

---

## 6. Sequencing

**Key scheduling insight: latency is weight-independent.** It depends on
architecture and input shape, not on the values in the tensors. So Stages 1–4
can run **now**, against whatever checkpoint exists, and do not wait on the
in-flight Band-A retrain campaign (`yolo26n-bs32-20260910-212812` was still
training on the A40 host as of 2026-09-17; its KD counterpart is on
`gpu-server`). Only the **parity check (Stage 5)** wants final checkpoints, and
even it is really testing the export path rather than the weights.

| Step | Where | Blocked by |
|---|---|---|
| 1. Pi env bring-up + `device_profile.json` | Pi | — |
| 2. Export + host verification gate | host | — |
| 3. Subset build + rsync to Pi | host→Pi | — |
| 4. Latency sweep, CPU | Pi | 1–3 |
| 5. Vulkan probe | Pi | 4 |
| 6. Parity run + host scoring | Pi→host | 4; final ckpts preferred |
| 7. Report + plots | host | 4–6 |
| 8. Thesis §4.3.2 + §2.3.2 rewrite + docs | host | 7 |

Rough effort: ~2–3 days of implementation, plus unattended device run time (the
MegaDetector 1280 cells will be slow — plan for hours, not minutes).

---

## 7. Risks and fallbacks

| Risk | Mitigation |
|---|---|
| **Thermal throttling** on a passively-cooled Pi 400 corrupts the numbers | Hard validity gate (§4.5), automatic cooldown + retry, throttle state recorded per cell and reported |
| **MegaDetector v5 at 1280×1280 OOMs** on 3.7 GB (280 MB checkpoint + large activations) | Fall back to ORT with a tuned-down arena config, then to a 640 px MD run, and finally report the OOM itself — "the teacher's detector does not fit the device" is a valid and thesis-relevant finding |
| **YOLO26n's end-to-end head has ops NCNN/Vulkan cannot lower** (TopK the likely culprit) | Fallback: export the raw head output and do top-k in numpy, timing it as post-processing. Recorded as a format-portability finding, which §2.3.2 already flags as a real concern |
| **NCNN fp16 defaults silently change numerics** | Options forced off explicitly; the T1 parity check would catch it regardless |
| **Slow link (2.4 MB/s)** makes iteration painful | Transfer exports + subset once (~500 MB, ~4 min); everything after that is SSH command traffic only |
| **torch aarch64 wheel pulls a large dependency tree** onto the Pi | Pi scripts depend on no repo code; TorchScript-only usage means no `ultralytics`/`yolov5` install is needed on the device |
| Comparing YOLO26n (in-graph top-k) against YOLOv5s (external NMS) as if like-for-like | The four-window decomposition makes the asymmetry explicit, and it is called out in the results text |

---

## 8. Verification

The work is done when all of the following hold:

1. `uv run python scripts/benchmark/1-export_models.py` completes and its
   host-side gate reports `max|Δ| ≤ 1e-4` for every exported model.
2. `bash scripts/benchmark/pi/run_all.sh` completes on the Pi under tmux with
   **zero cells marked invalid** by the thermal gate.
3. `reports/embedded_benchmark/parity_summary.csv` shows T1/T2/T3 **pass** for
   TorchScript and ONNX Runtime on all three detectors — i.e. the exported
   models on ARM produce numerically the same predictions as the reference on
   the A40.
4. `reports/embedded_benchmark/latency_summary.csv` has a complete matrix with
   no missing cells (Vulkan cells may carry a recorded failure reason).
5. `reports/embedded_benchmark/embedded_benchmark.md` states the verdict against
   the ≤30 ms / ≤500 MB targets, in both Pi 400 and translated-QCS605 terms,
   with the three non-claims explicit.
6. Both plots render, are copied into `thesis/manuscript/figures/plots/`, and are
   referenced from `4-Results.tex` with full paths spelled out (per commit
   `81fa911`'s convention).
7. A spot-check re-run of one cell reproduces its median within the reported IQR.
