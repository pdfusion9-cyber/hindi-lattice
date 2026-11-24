#!/usr/bin/env python3
"""Verify 20 complete reports from distinct GitHub-hosted runner instances.

RUNNER_NAME is the platform-assigned runner identity. Hostnames and machine
fingerprints are recorded for diagnostics, but hostnames are not used as the
sole identity because hosted-runner containers can share host naming.
"""
from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    reports = sorted(Path(".").glob("cpu-runner-*.json"))
    if len(reports) != 20:
        raise SystemExit(f"expected 20 CPU runner reports, found {len(reports)}")
    ids: set[int] = set()
    runner_names: set[str] = set()
    run_ids: set[str] = set()
    rows = []
    for path in reports:
        payload = json.loads(path.read_text(encoding="utf-8"))
        runner_id = payload.get("runner_id")
        profile = payload.get("cpu_profile", {})
        runner_name = profile.get("runner_name")
        run_id = profile.get("github_run_id")
        if not isinstance(runner_id, int) or isinstance(runner_id, bool) or runner_id in ids:
            raise SystemExit(f"missing or duplicate runner id in {path}")
        if not isinstance(runner_name, str) or not runner_name:
            raise SystemExit(f"missing GitHub RUNNER_NAME in {path}")
        if runner_name in runner_names:
            raise SystemExit(f"duplicate GitHub runner identity {runner_name!r}")
        if not isinstance(run_id, str) or not run_id:
            raise SystemExit(f"missing GITHUB_RUN_ID in {path}")
        for key in ("effective_cpus", "available_memory_bytes", "free_disk_bytes"):
            if not isinstance(profile.get(key), int) or profile[key] <= 0:
                raise SystemExit(f"invalid {key} in {path}")
        ids.add(runner_id)
        runner_names.add(runner_name)
        run_ids.add(run_id)
        rows.append(profile)
    if ids != set(range(20)):
        raise SystemExit(f"runner IDs mismatch: {sorted(ids)}")
    if len(run_ids) != 1:
        raise SystemExit(f"reports came from multiple workflow runs: {sorted(run_ids)}")
    summary = {
        "status": "CPU_RUNNER_INVENTORY_PASS",
        "runner_count": len(reports),
        "distinct_github_runner_names": len(runner_names),
        "workflow_run_id": next(iter(run_ids)),
        "hostnames_are_diagnostic_only": True,
        "machine_fingerprints_available": sum(
            bool(row.get("machine_instance_fingerprint")) for row in rows
        ),
        "cpu_counts": sorted(row["effective_cpus"] for row in rows),
        "memory_gib": sorted(
            round(row["available_memory_bytes"] / 1024**3, 2) for row in rows
        ),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
