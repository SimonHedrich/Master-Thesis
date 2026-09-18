# NCNN conversion: a reproducible pnnx failure (2026-09-17)

**Outcome: NCNN is not reachable for either detector with the available
tooling, so it is dropped from the runtime matrix and the NCNN-Vulkan GPU
probe cannot be run.** This is recorded as a format-portability finding, which
is the disposition the benchmarking plan (§4.6) authorises for exactly this
case, and which the thesis's Literature Review §2.3.2 already anticipates when
it names the export format as a variable the benchmark protocol must hold
fixed and cites TFLite Micro on embedded-ecosystem fragmentation.

## What was attempted

`pnnx 20260526` (the only version on PyPI; the `ncnn` wheel itself ships the
runtime only, no converter) was run against TorchScript and ONNX exports
produced by `scripts/benchmark/1-export_models.py` on `torch 2.12.0`.

## Evidence

| Input graph | Source | Result |
|---|---|---|
| ResNet18, traced | torchvision, torch 2.12.1 | **converted** — `r18.ncnn.param` + `.bin` |
| ResNet18, traced + `torch.jit.freeze` | torchvision, torch 2.12.1 | **converted** |
| YOLOv5s (225-class), traced + frozen | `yolov5s_model()` | `terminate called after throwing an instance of 'std::out_of_range' what(): map::at` at `pass_level2` |
| YOLO26n (225-class), traced + frozen | `yolo26n_model()` | same crash at `pass_level2` |
| YOLO26n via ONNX instead of TorchScript | opset 17 export | same crash at `pass_level2` (reaches `pass_level1`, emits `.pnnx.*`, then aborts) |
| **YOLO26n with the Detect head replaced by a pass-through** (backbone + neck only, outputs `(1,64,80,80)`, `(1,128,40,40)`, `(1,256,20,20)`) | `yolo26n_model()` | same crash at `pass_level2` |

## What this rules out

- **Not a torch-version mismatch.** pnnx converts a graph traced by the same
  torch 2.12 build without complaint.
- **Not `torch.jit.freeze`.** The frozen ResNet18 converts.
- **Not YOLO26's NMS-free end-to-end head.** The failure reproduces with the
  Detect head removed entirely, and reproduces identically on YOLOv5s, whose
  head is an ordinary anchor-based decode.
- **Not the TorchScript route specifically.** The ONNX route fails at the same
  pass.

The crash is inside pnnx's `pass_level2` graph pattern-matching on the
convolutional backbone itself, and is not something the export side can work
around.

## Consequence for the benchmark

- The CPU runtime matrix is **TorchScript + ONNX Runtime** (two independent
  runtimes, both passing the export gate), not three.
- **No GPU number is produced.** NCNN-Vulkan was the only GPU compute path on
  this device: VideoCore VI has no OpenCL driver (VC4CL covers the Pi 3's VC4
  only), and ONNX Runtime has no Vulkan execution provider. `Stage 0` still
  installs and verifies the Vulkan stack on the device, so the finding is
  precise: **the device exposes a working Vulkan 1.x/V3D stack and the NCNN
  runtime wheel ships with Vulkan compiled in — the blocker is solely the
  model converter.**

## Identified but not attempted

An ONNX → TFLite → LiteRT GPU-delegate route (OpenGL ES 3.1 compute shaders,
which mesa's v3d driver does support on VideoCore VI) would bypass pnnx
entirely. It needs a TensorFlow-based conversion chain (`onnx2tf`) on the host
and the LiteRT GPU delegate plus EGL/GLES libraries on the device. That is a
second full toolchain and was out of the plan's time-box; it belongs in Further
Work alongside the SNPE/Hexagon path.
