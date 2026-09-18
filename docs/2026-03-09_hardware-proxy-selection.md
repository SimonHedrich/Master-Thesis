# Hardware Proxy Selection for CV Model Research

This analysis details the selection of a hardware proxy for academic research into computer vision (CV) models. The primary goal is to find a platform that mimics the performance of the **Qualcomm Dragonwing™ QCS605** while avoiding its software stability and firmware limitations.

## The Use Case & Problem Statement

The research focuses on selecting and optimizing CV models for edge deployment. The final target is the **Qualcomm QCS605**, a 2018-era SoC designed for high-end smart cameras.

**The Problem:** While the QCS605 is hardware-capable, its **firmware is outdated**. This results in poor support for modern machine learning operations (Ops) on the Hexagon DSP and Adreno GPU. Researchers need a more **reliable, well-supported** alternative with a modern software stack to conduct their development without spending excessive time debugging hardware-specific driver issues.

---

## Baseline: Qualcomm QCS605 Architecture

To find a proxy, we must first establish the "performance dimension" of the QCS605:

* **CPU:** Octa-core Kryo 300 (2x "Gold" A75-based @ 2.5 GHz + 6x "Silver" A55-based @ 1.7 GHz).
* **GPU:** Adreno 615.
* **AI:** Hexagon 685 DSP (~2.1 TOPS).

---

## Comparative Hardware Analysis

The following table compares the QCS605 against the relevant Single Board Computer (SBC) options. Prices are based on the German market as of March 2026.

| Device | CPU Hierarchy | GPU Architecture | RAM | Price (DE) | Performance vs. QCS605 |
| --- | --- | --- | --- | --- | --- |
| **Orange Pi 4 LTS** | 2x A72 + 4x A53 | Mali-T860 MP4 | 4GB | ~€110–€150 | **-25% (Conservative)** |
| **Orange Pi 4 Pro** | 2x A76 + 6x A55 | Imagination BXM | 4–16GB | ~€95–€120 | **+20% (Comparable)** |
| **Raspberry Pi 4 / 400** | 4x A72 | VideoCore VI | 4–8GB | ~€55–€105 | **-10% (Comparable)** |
| **Qualcomm QCS605** | 2x A75 + 6x A55 | Adreno 615 | 2–8GB | N/A | **Baseline** |
| **Raspberry Pi 5** | 4x A76 | VideoCore VII | 4–8GB | ~€85–€125 | **+60% (Superior)** |
| **Orange Pi 5 Plus** | 4x A76 + 4x A55 | Mali-G610 MP4 | 8–32GB | ~€145–€210 | **+300% (Overkill)** |

### 1. The Best Hardware Mirror: Orange Pi 4 Pro

The Orange Pi 4 Pro is the closest architectural relative to the QCS605.

* **CPU:** Its core layout (2x A76 + 6x A55) perfectly mirrors the QCS605's dual-performance core philosophy, but with a newer generation (A76 vs A75).
* **Performance:** The CPU is roughly **20% faster** than the QCS605, providing a suitable safety margin for research.
* **Key Advantage:** Unlike the older LTS model, the Pro includes an **NPU (3 TOPS)**, providing a reliable alternative to the Qualcomm DSP.

### 2. The "Reliability & Budget" Choice: Raspberry Pi 4 / 400

The Raspberry Pi 4 (and the keyboard-integrated Pi 400) is a close match for raw compute power.

* **CPU:** Its quad-core A72 is a slightly older architecture than the A75, but having four cores instead of two "big" cores balances out the performance. The **Pi 400** is slightly faster (1.8 GHz vs 1.5 GHz).
* **Performance:** Estimated **10–15% slower** than the QCS605.
* **Pricing:** Currently the most affordable options in Germany, available via resellers like **BerryBase**.

### 3. The "Safe Margin" Choice: Raspberry Pi 5

The modern standard in the Raspberry Pi lineup.

* **CPU:** Its four A76 cores are roughly **60% faster** than the QCS605's CPU.
* **GPU:** The VideoCore VII is significantly more powerful (~40% faster).
* **Research Value:** Provides a performance ceiling — if a model struggles on a Pi 5, it will not run on the QCS605. Performance targets can be capped to 60% of the Pi 5's capability to match the QCS605.
* **Key Advantage:** Unbeatable community support and **modern software stack**.

### 4. The "Worst-Case" Mirror: Orange Pi 4 LTS

* **Performance:** Estimated **25% slower** than the QCS605.
* **Analysis:** AliExpress pricing is inflated (€110+). Given that the Raspberry Pi 5 is cheaper and significantly better, the 4 LTS is **not recommended** unless legacy RK3399 behavior is specifically required.

---

## Estimated Performance Delta

| Metric | Pi 4 vs. QCS605 | OPi 4 Pro vs. QCS605 | Pi 5 vs. QCS605 |
| --- | --- | --- | --- |
| **CPU (Inference)** | -10% | +20% | +60% |
| **GPU (Inference)** | -20% | +10% | +40% |
| **Software Support** | Excellent | Good | Excellent |

---

## Discarded Alternatives

* **Orange Pi 5 Plus:** Too powerful. Its 6 TOPS NPU and RK3588 chip are 3–5x faster than the QCS605, which would skew research results.
* **NVIDIA Jetson Orin Nano:** Too expensive (~€500+) and its GPU-heavy architecture (CUDA) is fundamentally different from the Adreno/Hexagon path of the QCS605.
* **LattePanda:** Uses x86 (Intel) architecture. While powerful, the optimization paths (OpenVINO) differ too much from ARM-based mobile chips to serve as a valid proxy.
* **NXP i.MX 8M Plus:** Too niche/industrial and expensive for general academic research.
* **Google Coral:** Abandoned software support; limited to highly specific INT8 quantized models.
* **ESP32:** A microcontroller; physically incapable of running standard OpenCV/TensorFlow desktop-class models.

---

## Final Recommendation

For academic research, use a **Raspberry Pi 5 (8GB)** (~€125 at BerryBase). While the **Orange Pi 4 Pro** is the better architectural twin, the Raspberry Pi 5's significantly more mature software ecosystem avoids driver debugging and ensures focus on CV model logic. To match the QCS605's performance, cap target inference times to 60% of the Pi 5's measured capability.

---

## Addendum, 2026-09-17: the device actually used is a Raspberry Pi 400

The benchmarking campaign specified in
[`plans/2026-09-17_on-device-benchmarking-plan.md`](plans/2026-09-17_on-device-benchmarking-plan.md)
was run on a **Raspberry Pi 400**, not the Raspberry Pi 5 recommended above.
This addendum records the substitution and its consequence for the
translation rule, rather than leaving the earlier recommendation to read as if
it had been followed.

**Nothing in the analysis above needed to change to accommodate it.** This
document already rates the Pi 4/400 family at **−10 % CPU / −20 % GPU** against
the QCS605 — closer to parity than any other option considered except the
Orange Pi 4 Pro — and already notes that the Pi 400 is the faster member of that
family at 1.8 GHz. The Pi 5 was recommended on **software-maturity** grounds,
not on fidelity. That reasoning still holds, and the Pi 400 turns out to share
the relevant part of it: Raspberry Pi OS (Debian 13 trixie) carries current
mesa and the full aarch64 Python wheel ecosystem, so no driver debugging was
needed.

### The translation rule changes direction

| | Pi 5 (recommended above) | Pi 400 (actually used) |
|---|---|---|
| CPU vs QCS605 | +60 % faster | 10–15 % slower |
| Role | performance **ceiling** | near-parity, mild **floor** |
| Translation | `QCS605 ≈ Pi5 × 1.6` → cap targets at 60 % | `QCS605 ≈ Pi400 × 0.85–0.90` |
| Device-side pass band for a ≤30 ms QCS605 budget | ≤ ~18–19 ms | **≤ ~33–35 ms** |

**The §"Final Recommendation" 60 %-cap rule does not apply to the measurements
that were taken.** Use the −10 %/−20 % deltas from the "Estimated Performance
Delta" table above instead.

This is the stronger of the two claims. The Pi 5 argument required trusting a
1.6× discount in the unfavourable direction — a model that passed on the Pi 5
only passed on the QCS605 if the 60 % estimate held. A model fast enough on the
Pi 400 is fast enough on the QCS605 almost regardless of whether the −10 %
figure is exactly right.

### Two caveats that the Pi 5 choice would not have avoided either

1. **No accelerator analogue.** The Pi 400 has nothing corresponding to the
   QCS605's Hexagon 685 DSP (~2.1 TOPS), and VideoCore VI is far weaker than
   the Adreno 615. The translation rule covers the **CPU path only**; the
   QCS605's accelerated path remains unmeasured. This is the "Hardware Proxy
   Fidelity" risk already flagged in
   `2026-03-10_object-detection-models-for-embedded-systems.md`, and it applies
   to the Pi 5 as well.
2. **An older ISA than either the Pi 5 or the target.** The Pi 400's Cortex-A72
   is ARMv8.0: measured `Features: fp asimd evtstrm crc32 cpuid`, with **no
   `asimddp` (INT8 dot product) and no `asimdhp` (fp16 arithmetic)**. The Pi 5's
   Cortex-A76 and the QCS605's Kryo 300 (A75/A55, ARMv8.2) both have them. Any
   future INT8 quantization measured on this device would therefore
   *understate* the speedup available on the target — roughly 1.3–2× here
   against the 3–4× an ARMv8.2 core would give, and far less than the ~10× the
   PTQ literature reports for Hexagon HVX. The FP32 results in this campaign
   are unaffected.

Measured device profile: `reports/embedded_benchmark/device_profile.json`.
