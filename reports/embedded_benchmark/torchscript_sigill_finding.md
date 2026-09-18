# TorchScript multi-threading aborts with SIGILL on the Cortex-A72 (2026-09-18)

**Outcome: on this device ONNX Runtime is the only runtime that executes every
model reliably across every thread count.** PyTorch's multi-threaded CPU path
aborts with `SIGILL` (exit 132, illegal instruction) for one of the four models
under its default configuration, and for *all four* once its oneDNN backend is
disabled. Single-threaded TorchScript is unaffected throughout.

This is a runtime-portability result, not a gap in the data. The thesis's
Literature Review §2.3.2 already names the export format and the runtime's
threading behaviour as variables a benchmark protocol has to hold fixed, citing
*TensorFlow Lite Micro* on exactly this kind of embedded-ecosystem
fragmentation.

## What was measured

Raspberry Pi 400, Cortex-A72, **ARMv8.0**: `Features: fp asimd evtstrm crc32
cpuid` — no `asimddp` (8-bit dot product), no `asimdhp` (half-precision
arithmetic). `torch 2.9.1+cpu`, TorchScript traced and frozen on the host with
torch 2.12, batch size 1.

| Model | 1 thread | 2 threads | 4 threads |
|---|---|---|---|
| YOLO26n direct-FT | ran (1,271 ms) | **SIGILL** | **SIGILL** |
| YOLO26n KD | ran (1,275 ms) | **SIGILL** | **SIGILL** |
| YOLOv5s | ran (2,792 ms) | ran (2,240 ms) | ran (1,624 ms) |
| SpeciesNet classifier | ran | ran (4,444 ms) | ran (2,951 ms) |
| MegaDetector v5a | ran (107,855 ms) | ran (65,441 ms) | ran (41,925 ms) |

Each failing configuration was attempted three times and failed identically
every time.

## The mitigation attempt, and why it was rejected

`torch.backends.mkldnn.enabled = False` was the obvious hypothesis: PyTorch
dispatches multi-threaded convolutions on aarch64 to oneDNN/ACL, which could
plausibly select a kernel assuming ARMv8.2 extensions this core lacks.

It made the failure **worse, not better**. A controlled diagnostic on YOLO26n at
2 threads aborted with exit 132 both with and without the flag, and the
subsequent re-run of every multi-threaded TorchScript cell with `--no-mkldnn`
aborted in **all 24 cells** — including YOLOv5s, SpeciesNet and MegaDetector,
which complete normally with oneDNN enabled.

So the non-oneDNN multi-threaded fallback path is itself ISA-unsafe on this
core, and oneDNN is what makes three of the four models work at all. The
mitigation was rejected and the default-configuration numbers stand.

## What this rules out

- **Not "multi-threading is broken".** Three of four models run correctly on
  2 and 4 threads under the default configuration.
- **Not model size.** MegaDetector, at 140 M parameters and 831 GFLOPs, is the
  largest model here and runs fine multi-threaded; YOLO26n at 2.71 M does not.
- **Not the NMS-free head alone.** The same architecture runs correctly
  single-threaded, so the head's operators are all executable on this core —
  the fault appears only when a parallel kernel is selected for them.

The precise offending kernel was not isolated; doing so would mean bisecting
PyTorch's aarch64 dispatch, which is outside this thesis's scope.

## Consequence

- Every headline number in this benchmark is taken with **ONNX Runtime**, which
  completed all 5 models × 3 thread counts × 3 invocations without a single
  abort. ORT was also the faster runtime in every model where both ran
  (1.27×–1.75× faster than TorchScript).
- The TorchScript column is reported where it exists and marked `SIGILL` where
  it does not, rather than left blank.
- For a deployment on this class of hardware the practical reading is that
  runtime choice is not only a performance decision but a correctness one.
