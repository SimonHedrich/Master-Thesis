# Embedded Benchmark Results — Raspberry Pi 400 as QCS605 Proxy

**Date:** 2026-09-18
**Closes:** `TODO.md` §5.2
**Plan:** [`plans/2026-09-17_on-device-benchmarking-plan.md`](plans/2026-09-17_on-device-benchmarking-plan.md)
**Artifacts:** `reports/embedded_benchmark/`
**Feeds:** `thesis/manuscript/chapters/4-Results.tex` §`sec:results_embedded_benchmark`

---

## 1. TL;DR

Every model this thesis evaluates was exported and timed on a **Raspberry Pi
400**. Before this, the thesis's title claim — real-time inference on embedded
hardware — had no measurement behind it anywhere in the repo.

| Model | Params | GFLOPs | $W_\text{infer}$ | $W_\text{e2e}$ | FPS | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| **YOLO26n direct-FT** | 2.71 M | 6.8 | **401 ms** | 423 ms | 2.36 | 284 MB |
| YOLO26n KD | 2.71 M | 6.8 | 419 ms | 442 ms | 2.26 | 284 MB |
| YOLOv5s | 7.63 M | 17.7 | 929 ms | 949 ms | 1.05 | 362 MB |
| SpeciesNet classifier | 52.71 M | 49.2 | 2,546 ms | 2,543 ms | 0.39 | 567 MB |
| MegaDetector v5a | 140.06 M | 831.6 | 30,998 ms | 30,963 ms | 0.03 | 1,570 MB |

ONNX Runtime, 4 threads, batch 1, FP32, `performance` governor. Median of three
independent process invocations.
Source: `reports/embedded_benchmark/latency_summary.csv`.

**Nothing meets the ≤30 ms / ≤500 MB design targets.** The fastest model is
~12× over the latency budget after translating to the QCS605 CPU. The
MD+SpeciesNet teacher, at a derived ≈35 s per frame, is over by three orders of
magnitude.

**No GPU number exists** — see §6.

## 2. Why the Pi 400, and how its numbers translate

The plan specified the Raspberry Pi 5; the available device was a Pi 400. This
turns out to be the better proxy and the stronger claim, and required no change
to the project's own analysis — only a change of direction in the translation
rule. `docs/2026-03-09_hardware-proxy-selection.md` already rates the Pi 4/400
at **−10 % CPU** against the QCS605, versus the Pi 5's **+60 %**; the Pi 5 was
recommended on software-maturity grounds, not fidelity. Full reasoning in that
document's 2026-09-17 addendum.

$$\text{QCS605} \approx \text{Pi 400} \times 0.85\text{–}0.90
\quad\Rightarrow\quad \text{pass band: Pi 400} \leq 33\text{–}35\,\text{ms}$$

| Model | $W_\text{e2e}$ Pi 400 | Implied QCS605 CPU | ≤30 ms? | Peak RSS | ≤500 MB? |
|---|---:|---:|:--:|---:|:--:|
| YOLO26n direct-FT | 423 ms | 360–381 ms | no (~12×) | 284 MB | yes |
| YOLO26n KD | 442 ms | 375–398 ms | no | 284 MB | yes |
| YOLOv5s | 949 ms | 807–854 ms | no (~28×) | 362 MB | yes |
| SpeciesNet classifier | 2,543 ms | 2,162–2,289 ms | no | 567 MB | **no** |
| MegaDetector v5a | 30,963 ms | 26,318–27,867 ms | no (~900×) | 1,570 MB | **no** |
| MD+SN ensemble (derived) | ≈35,290 ms | ≈30–32 s | no | — | **no** |

Both targets are **project design estimates, not vendor-certified figures**
(`docs/2026-03-10_object-detection-models-for-embedded-systems.md` §1). The
translation covers the **CPU path only**: the Pi 400 has no Hexagon 685
analogue, so the accelerated path a shipping product would use is unmeasured.

This outcome was expected. A published figure puts YOLO26n at 67.7 ms on the
*faster* Pi 5, already above budget, and the remaining gap was always meant to
be closed by the quantization pass this thesis did not carry out.

## 3. What the latency is made of

| Model | $W_\text{pre}$ median img | $W_\text{pre}$ AX Visio 4192×3120 | $W_\text{infer}$ | $W_\text{post}$ |
|---|---:|---:|---:|---:|
| YOLO26n direct-FT | 23.8 ms | 275.3 ms | 401.3 ms | 0.19 ms |
| YOLOv5s | 21.0 ms | 272.9 ms | 928.7 ms | 0.75 ms |
| SpeciesNet | 10.7 ms | 264.0 ms | 2,545.8 ms | 0.14 ms |
| MegaDetector | 34.5 ms | 255.7 ms | 30,998.4 ms | 2.06 ms |

Two things worth noting. **Post-processing is negligible** (≤2 ms everywhere),
including YOLOv5's external NMS — so YOLO26's in-graph NMS-free head buys
nothing measurable in latency terms on this workload. **Pre-processing is not
negligible at sensor resolution**: decoding a native AX Visio still costs
256–275 ms, comparable to a whole YOLO26n forward pass, and scales with capture
resolution rather than model size. A deployment reading frames at full sensor
resolution pays that on every frame.

Model load time is excluded from all windows but recorded: 309 ms for YOLO26n,
1.2 s for SpeciesNet, **24.4 s for MegaDetector**.

## 4. Thread scaling

| Model | Runtime | 1 thread | 2 threads | 4 threads | 1→4 |
|---|---|---:|---:|---:|---:|
| YOLO26n direct-FT | ORT | 925.4 | 542.3 | 401.3 | 2.31× |
| YOLO26n KD | ORT | 932.6 | 549.7 | 418.5 | 2.23× |
| YOLOv5s | ORT | 2,325.8 | 1,321.7 | 928.7 | 2.50× |
| SpeciesNet | ORT | 6,514.7 | 3,690.0 | 2,545.8 | 2.56× |
| MegaDetector | ORT | 94,852.9 | 49,224.2 | 30,998.4 | 3.06× |

Scaling improves with model size, as expected: the largest model is the most
compute-bound and the least dominated by memory traffic. **The 2-thread column
is the more transferable figure** — the QCS605 has 2 big A75 cores plus 6 little
A55s, so a 4×A72 number and a 2-thread number bracket it from opposite
directions.

## 5. Runtime choice is a correctness question here, not only a performance one

ONNX Runtime beat TorchScript on every model that ran under both — 1.75×
(YOLOv5s), 1.35× (MegaDetector), 1.16× (SpeciesNet) — and it was the only
runtime that completed all 45 CPU cells without aborting.

Multi-threaded TorchScript terminates with **SIGILL** on YOLO26n on this
ARMv8.0 core, at both 2 and 4 threads, reproducibly across three invocations,
while single-threaded TorchScript is fine and the other three models run
multi-threaded without issue. Disabling PyTorch's oneDNN backend was tried as a
remedy and **extended the failure to every model** rather than removing it.
Full evidence and controls: `reports/embedded_benchmark/torchscript_sigill_finding.md`.

## 6. No GPU number, and exactly why

The device's GPU stack is fine: `vulkaninfo` reports **V3D 4.2.14.0**, an
integrated GPU on Vulkan 1.3.354 via Mesa 26.2.2, and the `ncnn` aarch64 wheel
ships with Vulkan compiled in. NCNN-Vulkan was the only GPU compute path
available — VideoCore VI has no OpenCL driver, and ONNX Runtime has no Vulkan
execution provider.

The blocker is solely the **model converter**: `pnnx` aborts with
`std::out_of_range: map::at` at `pass_level2` on both detectors. Controls rule
out the torch version (it converts a torch-2.12 ResNet18), `jit.freeze`, and
YOLO26's end-to-end head (it also fails on the backbone with the head removed).
Full evidence: `reports/embedded_benchmark/ncnn_conversion_finding.md`.

Expected impact had it worked: V3D's ~13.5 GFLOPS fp32 is *below* four A72 NEON
cores, so the GPU would most likely have lost to the CPU — and that result would
**not** transfer to the QCS605, whose Adreno 615 is far more capable.

## 7. Measurement validity

**No cell was thermally compromised.** All 76 cells ran under the
`performance` governor with a 200 ms sampler recording core temperature, per-core
frequency and the SoC throttle register, plus a pre-gate that waits for the SoC
to drop below 55 °C before a cell starts.

- Cells failing the validity gate: **0 / 76**
- Peak temperature across the whole campaign: **64.8 °C** (gate trips at 80 °C)
- `vcgencmd get_throttled` read **0x0** for the entire campaign
- Sustained runs (up to 310 iterations): drift **+0.28 % to +2.33 %**

Reproducibility across independent process invocations is ~0.1–1 % (e.g.
YOLO26n at 4 threads: 401.3 / 399.1 / 405.5 ms).

## 8. Parity — the models compute the same thing on ARM

A numerical-equivalence check, not a re-evaluation: predictions should be
identical across hardware, so this tests that export and a different ISA did
not change the model. Host references come from this project's own
`eval_suite/predict.py`.

| Model | Side | T1 raw-tensor | T2 detection match | T3 subset mAP | Δ mAP |
|---|---|---:|---:|---:|---:|
| YOLO26n direct-FT | host CPU *(control)* | — | 99.69 % | 0.5915 | −0.0003 |
| YOLO26n direct-FT | Pi, ONNX Runtime | PASS 1.22e−04 | 98.97 % | 0.5898 | −0.0020 |
| YOLO26n direct-FT | Pi, TorchScript | PASS 3.05e−04 | 99.12 % | 0.5898 | −0.0020 |
| YOLO26n KD | host CPU *(control)* | — | 99.85 % | 0.5564 | −0.0001 |
| YOLO26n KD | Pi, ONNX Runtime | PASS 1.22e−04 | 99.47 % | 0.5562 | −0.0002 |
| YOLO26n KD | Pi, TorchScript | PASS 3.66e−04 | 99.47 % | 0.5562 | −0.0002 |

Raw tensor agreement passes everywhere. One of the six comparisons falls just
under the pre-set 99 % detection-match criterion (98.97 %); it is reported as a
miss rather than absorbed by moving the threshold afterwards, and its mAP delta
stays within tolerance.

**The baseline matters for reading these.** Host GPU vs host CPU — same machine,
same weights — already differs by ~0.15 % of detections, because fp32
accumulation order differs between kernels. A device result near that level is
indistinguishable from same-machine noise.

**One failure was caught and was not a hardware issue.** The first direct-FT run
failed T2 at 47 % with both runtimes agreeing exactly with each other. The cause
was that its checkpoint came from a training run still in flight: `best.pt` was
rewritten between the export and the host reference, so the two sides held
different weights (verified: `max_abs = 371.8` between the exported reference
and the later checkpoint). `1-export_models.py` now freezes a hashed snapshot of
the source checkpoint into the export directory, and `4-score_parity.py` builds
the host reference from that snapshot, so the failure mode cannot recur
silently. This is precisely what the parity tier exists to catch.

## 9. Non-claims

- **FP32, unoptimized.** No PTQ, QAT or pruning. ONNX Runtime applies its own
  graph optimizations at session creation (`ORT_ENABLE_ALL`), recorded per cell;
  the shipped ONNX graph is the unmodified export.
- **One device, a bounded campaign** — not a benchmarking study.
- **No power or thermal-envelope measurement.** The Pi has no instrumented rail
  and no AX Visio power budget is documented anywhere in this repo.
- **No AX Visio measurement.** That hardware was not available; the thesis
  section and §2.3.2 were corrected to stop implying one.
- **The QCS605's DSP path is entirely unproxied**, and INT8 measured on this
  ARMv8.0 core would understate the target's gains (no `asimddp`, no fp16
  arithmetic).
- **YOLOv5s is latency-only**: its post-Band-A-fix checkpoint is not on this
  machine, so it was exported architecture-only. Latency is weight-independent;
  its accuracy figures come from its own eval report.

## 10. What this changes

1. `TODO.md` §5.2 closed; §5.1 (QAT) annotated as the largest remaining lever —
   though note that even a 4× INT8 gain leaves YOLO26n near 100 ms/frame, still
   above budget on the CPU path alone.
2. The manuscript's proxy argument (§2.3.2) was rewritten from the Pi 5 ceiling
   to the Pi 400 floor, and the AX Visio claim removed.
3. §2.1's "fixed, tight per-frame latency budget" premise, and §2.1.3's reasoned
   claim that the ensemble cannot ship, are now **measured** rather than
   asserted.
4. The deployment-candidate argument for YOLO26n over YOLOv5s now holds on
   latency as well as accuracy: 2.2× faster *and* more accurate.
