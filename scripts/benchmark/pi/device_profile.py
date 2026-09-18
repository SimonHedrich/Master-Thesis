#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
Capture the device fingerprint. RUNS ON THE PI.

Every claim in the benchmark report is conditional on the machine it was taken
on, so the machine is recorded as an artifact rather than described in prose.
Stdlib only.

Usage (on the Pi):
    uv run --script ~/benchmark/device_profile.py > ~/benchmark/results/device_profile.json
"""
from __future__ import annotations

import json
import platform
import re
import subprocess


def sh(cmd: str) -> str:
    try:
        return subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=25
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def main() -> None:
    cpuinfo = open("/proc/cpuinfo").read()
    osrel = open("/etc/os-release").read()
    feats = re.search(r"Features\s*:\s*(.+)", cpuinfo)
    pretty = re.search(r'PRETTY_NAME="([^"]+)"', osrel)

    prof = {
        "device_model": open("/proc/device-tree/model").read().strip("\x00").strip(),
        "cpu": sh("lscpu | grep -E '^Model name' | head -1 | sed 's/.*: *//'"),
        "cores": sh("nproc"),
        "cpu_max_mhz": sh("lscpu | grep -E 'CPU max MHz' | sed 's/.*: *//'"),
        "cpu_features": feats.group(1) if feats else "",
        "has_dotprod": "asimddp" in cpuinfo,
        "has_fp16_arith": "asimdhp" in cpuinfo,
        "l2_cache": sh("lscpu | grep -E 'L2 cache' | sed 's/.*: *//'"),
        "mem_total": sh("free -h | awk '/^Mem:/{print $2}'"),
        "swap_total": sh("free -h | awk '/^Swap:/{print $2}'"),
        "os": pretty.group(1) if pretty else "",
        "kernel": platform.release(),
        "arch": platform.machine(),
        "governor_at_capture": sh("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "available_governors": sh("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors"),
        "gpu_vulkan_device": sh(
            "vulkaninfo --summary 2>/dev/null | grep -A6 GPU0 | grep deviceName | sed 's/.*= *//'"
        ),
        "gpu_vulkan_type": sh(
            "vulkaninfo --summary 2>/dev/null | grep -A6 GPU0 | grep deviceType | sed 's/.*= *//'"
        ),
        "gpu_vulkan_api": sh(
            "vulkaninfo --summary 2>/dev/null | grep -A3 GPU0 | grep apiVersion | sed 's/.*= *//'"
        ),
        "mesa_driver": sh("vulkaninfo --summary 2>/dev/null | grep -m1 driverInfo | sed 's/.*= *//'"),
        "opencl": "none — VideoCore VI has no OpenCL driver (VC4CL covers the Pi 3's VC4 only)",
        "python": platform.python_version(),
        "uv": sh("uv --version"),
        "idle_temp_c": sh("vcgencmd measure_temp | sed 's/[^0-9.]//g'"),
        "throttled_at_capture": sh("vcgencmd get_throttled"),
        "uptime_days": sh("awk '{print int($1/86400)}' /proc/uptime"),
        "cooling": "passive — internal heatspreader only, no fan",
    }
    print(json.dumps(prof, indent=2))


if __name__ == "__main__":
    main()
