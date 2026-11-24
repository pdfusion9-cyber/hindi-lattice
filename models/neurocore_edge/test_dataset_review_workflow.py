import json
from pathlib import Path
import pytest

from dataset_review_workflow import build_packet

ROOT = Path(__file__).parent

def test_current_registry_builds_21_source_and_27_assignment_review_packet():
    registry = json.loads((ROOT / "dataset_approval.json").read_text(encoding="utf-8"))
    packet = build_packet(registry)
    assert packet["source_count"] == 21
    assert packet["assignment_count"] == 27
    assert packet["approved_source_count"] == 0
    assert packet["approved_assignment_count"] == 0
    assert packet["gate_status"] == {"corpus_preparation": "CLOSED", "production_training": "CLOSED"}
    assert all(x["decision"] == "PENDING" for x in packet["source_reviews"])
    assert all(x["decision"] == "PENDING" for x in packet["expert_assignments"])

def test_shared_source_is_reviewed_once_but_assignments_remain_separate():
    registry = {"datasets":[
        {"expert_id":1,"expert_name":"one","dataset_id":"org/shared"},
        {"expert_id":2,"expert_name":"two","dataset_id":"org/shared"},
    ]}
    packet = build_packet(registry | {"datasets": registry["datasets"] * 13 + registry["datasets"][:1]})
    assert packet["source_count"] == 1
    assert packet["assignment_count"] == 27

def test_wrong_assignment_count_fails_closed():
    with pytest.raises(ValueError, match="expected 27"):
        build_packet({"datasets":[{"expert_id":1,"expert_name":"one","dataset_id":"org/one"}]})


def test_cli_writes_valid_parseable_json(tmp_path):
    import subprocess
    import sys

    output = tmp_path / "dataset_human_review_packet.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "dataset_review_workflow.py"),
            "--output",
            str(output),
        ],
        check=True,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    packet = json.loads(output.read_text(encoding="utf-8"))
    assert packet["source_count"] == 21
    assert packet["assignment_count"] == 27
    assert packet["approved_source_count"] == 0
    assert packet["approved_assignment_count"] == 0
    assert packet["gate_status"]["corpus_preparation"] == "CLOSED"
    assert packet["gate_status"]["production_training"] == "CLOSED"
