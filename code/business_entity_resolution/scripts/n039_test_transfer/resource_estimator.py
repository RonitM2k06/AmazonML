#!/usr/bin/env python3
"""Requirement D: inspects this machine and extrapolates measured smoke-test
throughput to full test scale. Does not run anything at full scale itself --
takes measured (records, seconds) pairs as input from the smoke test."""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path
from typing import Dict


def inspect_machine() -> Dict[str, object]:
    import torch
    info: Dict[str, object] = {}
    info["platform"] = platform.platform()
    info["processor"] = platform.processor()
    try:
        cpu_brand = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
        info["cpu_brand"] = cpu_brand or "unknown"
    except Exception:
        info["cpu_brand"] = "unknown"
    try:
        ncpu = subprocess.run(["sysctl", "-n", "hw.ncpu"], capture_output=True, text=True, timeout=5).stdout.strip()
        info["cpu_cores"] = int(ncpu) if ncpu else None
    except Exception:
        info["cpu_cores"] = None
    try:
        mem_bytes = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5).stdout.strip()
        info["ram_gb"] = round(int(mem_bytes) / (1024 ** 3), 1) if mem_bytes else None
    except Exception:
        info["ram_gb"] = None
    info["mps_available"] = bool(torch.backends.mps.is_available())
    info["cuda_available"] = bool(torch.cuda.is_available())
    return info


def extrapolate(
    measured_records: int,
    measured_seconds: float,
    target_total_records: int,
    label: str,
) -> Dict[str, object]:
    if measured_seconds <= 0 or measured_records <= 0:
        return {"label": label, "error": "no measurement / zero-duration, cannot extrapolate"}
    rate = measured_records / measured_seconds
    est_seconds = target_total_records / rate
    return {
        "label": label,
        "measured_records": measured_records,
        "measured_seconds": round(measured_seconds, 2),
        "measured_records_per_sec": round(rate, 2),
        "target_total_records": target_total_records,
        "extrapolated_seconds": round(est_seconds, 1),
        "extrapolated_hours": round(est_seconds / 3600, 2),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(inspect_machine(), indent=2))
