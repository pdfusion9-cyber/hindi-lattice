#!/usr/bin/env python3
"""Verify that all 20 CPU inventory jobs produced distinct runner reports."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    reports = sorted(Path(".").glob("cpu-runner-*.json"))
    if len(reports) != 20:
        raise SystemExit(f"expected 20 CPU runner reports, found {len(reports)}")
    ids: set[int] = set()
    hosts: set[str] = set()
    for path in reports:
        payload = json.loads(path.read_text(encoding="utf-8"))
        runner_id = payload.get("runner_id")
        profile = payload.get("cpu_profile", {})
        host = profile.get("hostname")
        if not isinstance(runner_id, int) or runner_id in ids:
            raise SystemExit(f"missing or duplicate runner id in {path}")
        if not isinstance(host, str) or not host or host in hosts:
            raise SystemExit(f"missing or duplicate runner hostname in {path}")
        for key in ("effective_cpus", "available_memory_bytes", "free_disk_bytes"):
            if not isinstance(profile.get(key), int) or profile[key] <= 0:
                raise SystemExit(f"invalid {key} in {path}")
        ids.add(runner_id)
        hosts.add(host)
    if ids != set(range(20)):
        raise SystemExit(f"runner IDs mismatch: {sorted(ids)}")
    summary = {
        "status": "CPU_RUNNER_INVENTORY_PASS",
        "runner_count": len(reports),
        "distinct_host_count": len(hosts),
        "cpu_counts": sorted(json.loads(p.read_text())["cpu_profile"]["effective_cpus"] for p in reports),
        "memory_gib": sorted(round(json.loads(p.read_text())["cpu_profile"]["available_memory_bytes"] / 1024**3, 2) for p in reports),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
