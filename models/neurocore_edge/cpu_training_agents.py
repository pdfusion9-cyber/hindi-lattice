#!/usr/bin/env python3
"""CPU-first micro-agent planner for native NeuroCore expert training.

This module plans and gates CPU work; it does not load model weights or start
production training. It deliberately requires shard-only loading for expert
workers so a runner never materializes all 20 experts merely to update one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import socket
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

GIB = 1024 ** 3
DEFAULT_MEMORY_SAFETY_FRACTION = 0.80
ADAMW_FP32_BYTES_PER_TRAINABLE_PARAMETER = 16
FP32_BYTES_PER_FROZEN_PARAMETER = 4


@dataclass(frozen=True)
class CPUProfile:
    logical_cpus: int
    affinity_cpus: int
    quota_cpus: float | None
    effective_cpus: int
    available_memory_bytes: int
    free_disk_bytes: int
    hostname: str
    runner_name: str | None = None
    runner_os: str | None = None
    runner_arch: str | None = None
    github_job: str | None = None
    github_run_id: str | None = None
    machine_instance_fingerprint: str | None = None
    machine_fingerprint_source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["available_memory_gib"] = round(self.available_memory_bytes / GIB, 3)
        result["free_disk_gib"] = round(self.free_disk_bytes / GIB, 3)
        return result


def _read_int(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="ascii").strip()
        if value == "max":
            return None
        return int(value)
    except (OSError, ValueError):
        return None


def _memory_available_bytes() -> int:
    mem_available = None
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                mem_available = int(line.split()[1]) * 1024
                break
    except (OSError, ValueError, IndexError):
        pass
    if mem_available is None:
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            mem_available = page_size * os.sysconf("SC_AVPHYS_PAGES")
        except (AttributeError, OSError, ValueError):
            mem_available = 0

    limit = _read_int(Path("/sys/fs/cgroup/memory.max"))
    current = _read_int(Path("/sys/fs/cgroup/memory.current"))
    if limit and current is not None:
        mem_available = min(mem_available, max(0, limit - current))
    return max(0, int(mem_available))


def _cpu_quota() -> float | None:
    try:
        quota_text, period_text = Path("/sys/fs/cgroup/cpu.max").read_text(encoding="ascii").split()
        if quota_text == "max":
            return None
        quota, period = int(quota_text), int(period_text)
        if quota <= 0 or period <= 0:
            return None
        return quota / period
    except (OSError, ValueError):
        return None


def _machine_identity() -> tuple[str | None, str | None]:
    """Return a hashed VM identity without exposing raw machine identifiers."""
    candidates = (
        (Path("/sys/class/dmi/id/product_uuid"), "dmi_product_uuid"),
        (Path("/etc/machine-id"), "etc_machine_id"),
    )
    for path, source in candidates:
        try:
            raw = path.read_text(encoding="ascii").strip()
        except OSError:
            continue
        if raw:
            return hashlib.sha256(raw.encode("utf-8")).hexdigest(), source
    return None, None


def detect_cpu_profile() -> CPUProfile:
    logical = max(1, os.cpu_count() or 1)
    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = logical
    quota = _cpu_quota()
    effective = min(logical, max(1, affinity))
    if quota is not None:
        effective = min(effective, max(1, math.floor(quota)))
    try:
        free_disk = shutil.disk_usage(Path.cwd()).free
    except OSError:
        free_disk = 0
    machine_fingerprint, fingerprint_source = _machine_identity()
    return CPUProfile(
        logical_cpus=logical,
        affinity_cpus=max(1, affinity),
        quota_cpus=quota,
        effective_cpus=max(1, effective),
        available_memory_bytes=_memory_available_bytes(),
        free_disk_bytes=int(free_disk),
        hostname=socket.gethostname(),
        runner_name=os.environ.get("RUNNER_NAME"),
        runner_os=os.environ.get("RUNNER_OS"),
        runner_arch=os.environ.get("RUNNER_ARCH"),
        github_job=os.environ.get("GITHUB_JOB"),
        github_run_id=os.environ.get("GITHUB_RUN_ID"),
        machine_instance_fingerprint=machine_fingerprint,
        machine_fingerprint_source=fingerprint_source,
    )


def estimate_worker_memory_bytes(
    *,
    shared_frozen_parameters: int,
    active_trainable_parameters: int,
    activation_reserve_bytes: int,
    staging_reserve_bytes: int = 512 * 1024**2,
    trainable_bytes_per_parameter: int = ADAMW_FP32_BYTES_PER_TRAINABLE_PARAMETER,
    frozen_bytes_per_parameter: int = FP32_BYTES_PER_FROZEN_PARAMETER,
) -> int:
    """Conservative host-RAM estimate for shared-frozen + selected-expert training."""
    values = (
        shared_frozen_parameters, active_trainable_parameters,
        activation_reserve_bytes, staging_reserve_bytes,
        trainable_bytes_per_parameter, frozen_bytes_per_parameter,
    )
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in values):
        raise ValueError("memory estimate inputs must be non-negative integers")
    if trainable_bytes_per_parameter < 8 or frozen_bytes_per_parameter < 2:
        raise ValueError("memory bytes per parameter are implausibly small")
    return (
        shared_frozen_parameters * frozen_bytes_per_parameter
        + active_trainable_parameters * trainable_bytes_per_parameter
        + activation_reserve_bytes
        + staging_reserve_bytes
    )


def plan_cpu_threads(profile: CPUProfile) -> dict[str, Any]:
    """One training process per runner; avoid nested data-loader/BLAS oversubscription."""
    threads = max(1, min(profile.effective_cpus, 4))
    return {
        "intra_op_threads": threads,
        "interop_threads": 1,
        "dataloader_workers": 0,
        "parallel_training_processes_per_runner": 1,
        "environment": {
            "OMP_NUM_THREADS": str(threads),
            "MKL_NUM_THREADS": str(threads),
            "OPENBLAS_NUM_THREADS": str(threads),
            "TOKENIZERS_PARALLELISM": "false",
        },
        "reason": "CPU quota/affinity aware; a single expert process owns the runner CPU budget",
    }


def choose_micro_batch(
    *,
    effective_batch_size: int,
    activation_bytes_per_example: int,
    static_memory_bytes: int,
    safe_memory_bytes: int,
) -> dict[str, int]:
    if effective_batch_size < 1 or activation_bytes_per_example < 1:
        raise ValueError("batch and activation estimates must be positive")
    remaining = safe_memory_bytes - static_memory_bytes
    if remaining < activation_bytes_per_example:
        raise MemoryError("static model/optimizer estimate leaves no room for one example")
    capacity = min(effective_batch_size, remaining // activation_bytes_per_example)
    valid = [n for n in range(1, capacity + 1) if effective_batch_size % n == 0]
    if not valid:
        raise MemoryError("no exact micro-batch/accumulation plan fits memory")
    micro = max(valid)
    return {
        "effective_batch_size": effective_batch_size,
        "micro_batch_size": micro,
        "gradient_accumulation_steps": effective_batch_size // micro,
        "effective_batch_preserved": True,
    }


def checkpoint_interval(
    *,
    measured_step_seconds: float,
    measured_checkpoint_seconds: float,
    target_overhead_fraction: float = 0.01,
    max_recovery_seconds: float = 300.0,
    minimum_steps: int = 10,
    maximum_steps: int = 2000,
) -> int:
    if measured_step_seconds <= 0 or measured_checkpoint_seconds < 0:
        raise ValueError("measured step/checkpoint times are invalid")
    if not 0 < target_overhead_fraction <= 0.10:
        raise ValueError("target checkpoint overhead must be in (0, 0.10]")
    if max_recovery_seconds <= 0 or minimum_steps < 1 or maximum_steps < minimum_steps:
        raise ValueError("checkpoint bounds are invalid")
    if measured_checkpoint_seconds == 0:
        return maximum_steps
    overhead_bound = math.ceil(
        measured_checkpoint_seconds / (measured_step_seconds * target_overhead_fraction)
    )
    recovery_bound = max(1, int(max_recovery_seconds / measured_step_seconds))
    return max(minimum_steps, min(maximum_steps, overhead_bound, recovery_bound))


def _balanced_assignments(expert_costs: list[float], runner_count: int) -> list[dict[str, Any]]:
    if runner_count < 1:
        raise ValueError("runner_count must be positive")
    bins = [{"runner_slot": i, "expert_ids": [], "estimated_work": 0.0} for i in range(runner_count)]
    for expert_id, cost in sorted(enumerate(expert_costs), key=lambda item: (-item[1], item[0])):
        target = min(bins, key=lambda row: (row["estimated_work"], row["runner_slot"]))
        target["expert_ids"].append(expert_id)
        target["estimated_work"] += cost
    for row in bins:
        row["expert_ids"].sort()
    return bins


def create_cpu_training_plan(
    *,
    profile: CPUProfile,
    shared_backbone_parameters: int,
    expert_parameters: list[int],
    runner_count: int = 20,
    activation_reserve_bytes: int = 1536 * 1024**2,
    staging_reserve_bytes: int = 512 * 1024**2,
    memory_safety_fraction: float = DEFAULT_MEMORY_SAFETY_FRACTION,
    effective_batch_size: int = 8,
    activation_bytes_per_example: int = 128 * 1024**2,
    measured_step_seconds: float = 1.0,
    measured_checkpoint_seconds: float = 2.0,
    minimum_free_disk_bytes: int = 5 * GIB,
) -> dict[str, Any]:
    if shared_backbone_parameters < 1:
        raise ValueError("shared_backbone_parameters must be positive")
    if len(expert_parameters) != 20 or any(p < 1 for p in expert_parameters):
        raise ValueError("exactly 20 positive expert parameter counts are required")
    if not 0.5 <= memory_safety_fraction <= 0.9:
        raise ValueError("memory_safety_fraction must be between 0.5 and 0.9")

    safe_memory = int(profile.available_memory_bytes * memory_safety_fraction)
    thread_plan = plan_cpu_threads(profile)
    workers: list[dict[str, Any]] = []
    blockers: list[str] = []
    for expert_id, expert_count in enumerate(expert_parameters):
        estimated = estimate_worker_memory_bytes(
            shared_frozen_parameters=shared_backbone_parameters,
            active_trainable_parameters=expert_count,
            activation_reserve_bytes=activation_reserve_bytes,
            staging_reserve_bytes=staging_reserve_bytes,
        )
        workers.append({
            "expert_id": expert_id,
            "loaded_parameter_count_target": shared_backbone_parameters + expert_count,
            "other_expert_weights_loaded": False,
            "estimated_peak_memory_bytes": estimated,
            "memory_fits": estimated <= safe_memory,
        })

    if profile.effective_cpus < 1:
        blockers.append("no usable CPU capacity detected")
    if profile.available_memory_bytes <= 0:
        blockers.append("available host memory could not be measured")
    if profile.free_disk_bytes < minimum_free_disk_bytes:
        blockers.append("free disk is below the checkpoint/artifact safety minimum")
    memory_failures = [w["expert_id"] for w in workers if not w["memory_fits"]]
    if memory_failures:
        blockers.append(
            "selected-expert worker memory estimate exceeds safe RAM on experts "
            + ",".join(map(str, memory_failures))
        )

    assignments = _balanced_assignments([float(p) for p in expert_parameters], runner_count)
    batch = None
    try:
        worst_static = max(
            estimate_worker_memory_bytes(
                shared_frozen_parameters=shared_backbone_parameters,
                active_trainable_parameters=p,
                activation_reserve_bytes=activation_reserve_bytes,
                staging_reserve_bytes=staging_reserve_bytes,
            ) - activation_reserve_bytes
            for p in expert_parameters
        )
        batch = choose_micro_batch(
            effective_batch_size=effective_batch_size,
            activation_bytes_per_example=activation_bytes_per_example,
            static_memory_bytes=worst_static,
            safe_memory_bytes=safe_memory,
        )
    except MemoryError as exc:
        blockers.append(str(exc))

    interval = checkpoint_interval(
        measured_step_seconds=measured_step_seconds,
        measured_checkpoint_seconds=measured_checkpoint_seconds,
    )
    shared_base_training_bytes = estimate_worker_memory_bytes(
        shared_frozen_parameters=0,
        active_trainable_parameters=shared_backbone_parameters,
        activation_reserve_bytes=activation_reserve_bytes,
        staging_reserve_bytes=staging_reserve_bytes,
    )
    shared_base_fits = shared_base_training_bytes <= safe_memory
    if not shared_base_fits:
        blockers.append("shared-backbone AdamW state exceeds the safe single-runner RAM budget")

    return {
        "status": "CPU_TRAINING_PLAN_APPROVED" if not blockers else "CPU_TRAINING_PLAN_BLOCKED",
        "hardware_profile": profile.to_dict(),
        "cpu_thread_plan": thread_plan,
        "memory_policy": {
            "available_memory_bytes": profile.available_memory_bytes,
            "safe_memory_bytes": safe_memory,
            "memory_safety_fraction": memory_safety_fraction,
            "optimizer_assumption": "FP32 AdamW: 4-byte weights + 4-byte gradients + 8-byte moments per trainable parameter",
            "frozen_weight_assumption": "FP32 frozen weights: 4 bytes per parameter",
            "selected_expert_only_loading_required": True,
            "full_20_expert_materialization_per_worker_allowed": False,
        },
        "expert_workers": workers,
        "runner_assignments": assignments,
        "batch_policy": batch,
        "checkpoint_interval_steps": interval,
        "shared_base_stage": {
            "estimated_training_memory_bytes": shared_base_training_bytes,
            "fits_single_runner_safe_memory": shared_base_fits,
            "status": "PASS" if shared_base_fits else "BLOCKED",
            "note": "Shared-base AdamW must fit independently; 20 expert runners do not multiply one runner's RAM.",
        },
        "production_dispatch_allowed": False,
        "blockers": blockers,
        "required_next_implementation": [
            "native shared-backbone + selected-expert module that never instantiates unrelated experts",
            "mmap/streamed checkpoint loading with config/base fingerprints",
            "measured CPU throughput calibration before choosing precision or batch size",
            "shared-backbone optimizer/memory solution; refuse dispatch while its single-runner estimate fails",
        ],
        "provenance": {
            "native_neurocore_only": True,
            "external_model_weights_used": False,
            "open_source_model_integrated": False,
            "external_data_used": False,
            "user_data_used": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-id", type=int, default=0)
    parser.add_argument("--profile-report", type=Path)
    args = parser.parse_args()
    profile = detect_cpu_profile()
    result = {"runner_id": args.runner_id, "cpu_profile": profile.to_dict()}
    print(json.dumps(result, indent=2))
    if args.profile_report:
        args.profile_report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
