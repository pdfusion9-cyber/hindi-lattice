#!/usr/bin/env python3
"""Fail-closed hardware preflight for one NeuroCore GPU worker.

This utility only inventories hardware. It does not download datasets, load
model weights, allocate training tensors, or start training.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

GIB = 1024 ** 3
MIN_CPUS = 4
MIN_RAM_BYTES = 24 * GIB
MIN_GPU_MEMORY_MIB = 15_000
MIN_DISK_TOTAL_BYTES = 150 * 10**9
MIN_DISK_FREE_BYTES = 100 * 10**9


def parse_gpu_rows(raw: str) -> list[dict[str, Any]]:
    rows = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        fields = [item.strip() for item in line.split(",")]
        if len(fields) < 3:
            raise ValueError(f"unexpected nvidia-smi row: {line!r}")
        rows.append({
            "index": fields[0],
            "name": fields[1],
            "memory_total_mib": int(fields[2]),
        })
    return rows


def assess_profile(
    *,
    runner_id: int,
    cpu_count: int,
    ram_bytes: int,
    disk_total_bytes: int,
    disk_free_bytes: int,
    gpus: list[dict[str, Any]],
) -> dict[str, Any]:
    blockers = []
    if cpu_count < MIN_CPUS:
        blockers.append(f"CPU cores below minimum {MIN_CPUS}")
    if ram_bytes < MIN_RAM_BYTES:
        blockers.append(f"host RAM below minimum {MIN_RAM_BYTES // GIB} GiB")
    if disk_total_bytes < MIN_DISK_TOTAL_BYTES:
        blockers.append("runner disk capacity below 150 GB")
    if disk_free_bytes < MIN_DISK_FREE_BYTES:
        blockers.append("free disk space below 100 GB")
    qualifying = [
        gpu for gpu in gpus
        if isinstance(gpu.get("memory_total_mib"), int)
        and gpu["memory_total_mib"] >= MIN_GPU_MEMORY_MIB
    ]
    if not qualifying:
        blockers.append(
            f"no NVIDIA GPU with at least {MIN_GPU_MEMORY_MIB} MiB VRAM detected"
        )
    return {
        "status": "GPU_RUNNER_PREFLIGHT_PASS" if not blockers else "GPU_RUNNER_PREFLIGHT_BLOCKED",
        "runner_id": runner_id,
        "cpu_count": cpu_count,
        "host_ram_gib": round(ram_bytes / GIB, 2),
        "disk_total_gb": round(disk_total_bytes / 10**9, 2),
        "disk_free_gb": round(disk_free_bytes / 10**9, 2),
        "gpu_count": len(gpus),
        "gpus": gpus,
        "qualifying_gpu_count": len(qualifying),
        "minimums": {
            "cpu_count": MIN_CPUS,
            "host_ram_gib": MIN_RAM_BYTES // GIB,
            "gpu_vram_mib": MIN_GPU_MEMORY_MIB,
            "disk_total_gb": 150,
            "disk_free_gb": 100,
        },
        "blockers": blockers,
        "training_started": False,
        "external_weights_used": False,
        "open_source_model_integrated": False,
    }


def collect_profile(runner_id: int) -> dict[str, Any]:
    cpu_count = os.cpu_count() or 0
    ram_bytes = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    disk = shutil.disk_usage("/")
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return assess_profile(
            runner_id=runner_id,
            cpu_count=cpu_count,
            ram_bytes=ram_bytes,
            disk_total_bytes=disk.total,
            disk_free_bytes=disk.free,
            gpus=[],
        ) | {"nvidia_smi_error": str(exc)}
    gpus = parse_gpu_rows(result.stdout)
    return assess_profile(
        runner_id=runner_id,
        cpu_count=cpu_count,
        ram_bytes=ram_bytes,
        disk_total_bytes=disk.total,
        disk_free_bytes=disk.free,
        gpus=gpus,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-id", type=int, required=True, choices=range(20))
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = collect_profile(args.runner_id)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "GPU_RUNNER_PREFLIGHT_PASS":
        raise SystemExit("GPU runner preflight failed: " + "; ".join(report["blockers"]))


if __name__ == "__main__":
    main()
