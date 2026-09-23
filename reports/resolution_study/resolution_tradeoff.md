# Input resolution: accuracy against measured on-device latency

One YOLO26n checkpoint (`yolo26n-bs32-20260910-212812`, trained at 640 px), 
scored at four inference resolutions and timed on the Raspberry Pi 400 
(ONNX Runtime, 4 threads, batch 1, FP32).

**Accuracy is on a fixed proportional subsample** (8,000 of 63,802 real + 
1,411 of 11,250 synthetic test images, seed 42), which runs ~0.015 mAP 
optimistic against the published full-test headline. The offset is constant 
across resolutions, so the shape of the curve is sound — but these values 
must not be quoted alongside full-test figures.

| Arm | Input | GFLOPs | W_infer (ms) | W_e2e (ms) | QCS605 est. (ms) | Peak RSS (MB) | mixed mAP | real mAP | Δ real vs 640 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no-retraining | 640 | 6.81 | 401.3 | 423.5 | 360–381 | 284 | 0.6134 | 0.5483 |  |
| no-retraining | 512 | 4.33 | 269.4 | 283.1 | 241–255 | 260 | 0.5799 | 0.5044 | -0.0439 (-8.0 %) |
| no-retraining | 416 | 2.85 | 172.1 | 179.8 | 153–162 | 241 | 0.5379 | 0.4529 | -0.0954 (-17.4 %) |
| no-retraining | 320 | 1.68 | 102.5 | 109.0 | 93–98 | 226 | 0.4609 | 0.3541 | -0.1942 (-35.4 %) |
| resolution-native | 320 | 1.68 | 102.5 | 109.0 | 93–98 | 226 | 0.5175 | 0.4375 | -0.1109 (-20.2 %) |

Design targets: **≤30 ms** on the QCS605 (Pi 400 pass band ≤33–35 ms) and **≤500 MB**. Every row meets the memory budget; none meets the latency budget.
