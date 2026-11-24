import json

import pytest

from verify_cpu_runner_inventory import main


def _write_reports(root, duplicate_runner=False):
    for runner_id in range(20):
        runner_name = "GitHub Actions 0" if duplicate_runner else f"GitHub Actions {runner_id}"
        payload = {
            "runner_id": runner_id,
            "cpu_profile": {
                "runner_name": runner_name,
                "github_run_id": "123456789",
                "hostname": "shared-hostname-is-allowed",
                "effective_cpus": 4,
                "available_memory_bytes": 8 * 1024**3,
                "free_disk_bytes": 50 * 1024**3,
                "machine_instance_fingerprint": f"fingerprint-{runner_id}",
            },
        }
        (root / f"cpu-runner-{runner_id}.json").write_text(json.dumps(payload))


def test_verifier_accepts_twenty_unique_runner_names_even_with_repeated_hostnames(tmp_path, monkeypatch, capsys):
    _write_reports(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["runner_count"] == 20
    assert report["distinct_github_runner_names"] == 20
    assert report["hostnames_are_diagnostic_only"] is True


def test_verifier_rejects_duplicate_github_runner_names(tmp_path, monkeypatch):
    _write_reports(tmp_path, duplicate_runner=True)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="duplicate GitHub runner identity"):
        main()


def test_verifier_rejects_mixed_workflow_run_ids(tmp_path, monkeypatch):
    _write_reports(tmp_path)
    report_path = tmp_path / "cpu-runner-19.json"
    payload = json.loads(report_path.read_text())
    payload["cpu_profile"]["github_run_id"] = "different-run"
    report_path.write_text(json.dumps(payload))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="multiple workflow runs"):
        main()


def test_verifier_rejects_missing_runner_name(tmp_path, monkeypatch):
    _write_reports(tmp_path)
    report_path = tmp_path / "cpu-runner-7.json"
    payload = json.loads(report_path.read_text())
    payload["cpu_profile"]["runner_name"] = None
    report_path.write_text(json.dumps(payload))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="missing GitHub RUNNER_NAME"):
        main()
