import json
from pathlib import Path

import pytest

from expert_domain_synthetic_benchmark import build_corpora, run_benchmark

ROOT = Path(__file__).parent


def test_twenty_expert_synthetic_corpora_have_disjoint_splits():
    corpora = build_corpora(ROOT / "expert_manifest.json")
    assert set(corpora) == set(range(20))
    assert sum(len(x["train_records"]) for x in corpora.values()) == 60
    assert sum(len(x["validation_records"]) for x in corpora.values()) == 40
    for eid, corpus in corpora.items():
        train_ids = {x["record_id"] for x in corpus["train_records"]}
        valid_ids = {x["record_id"] for x in corpus["validation_records"]}
        assert train_ids.isdisjoint(valid_ids)
        assert all(x["expert_id"] == eid for x in corpus["train_records"] + corpus["validation_records"])


def test_corrupt_or_incomplete_expert_manifest_is_rejected(tmp_path):
    bad = tmp_path / "manifest.json"
    bad.write_text(json.dumps([{"expert_id": 0, "name": "doc_editing"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="unique expert IDs"):
        build_corpora(bad)


def test_baseline_and_managed_benchmark_match_native_model_state(tmp_path):
    report_path = tmp_path / "expert_domain_report.json"
    report = run_benchmark(report_path, ROOT / "expert_manifest.json", steps=20, seed=23)
    assert report["status"] == "EXPERT_DOMAIN_SYNTHETIC_BENCHMARK_PASS"
    assert report["corpus"]["external_data_used"] is False
    assert report["corpus"]["external_model_weights_used"] is False
    assert report["model_config"]["num_experts"] == 20
    assert report["comparison"]["max_abs_parameter_difference"] <= 1e-7
    assert report["comparison"]["baseline"]["steps"] == 20
    assert report["comparison"]["micro_agent_managed"]["steps"] == 20
    assert report["production_training_gate"] == "CLOSED"
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert len(saved["corpus"]["experts"]) == 20


def test_steps_must_cover_all_twenty_experts():
    with pytest.raises(ValueError, match="multiple of 20"):
        run_benchmark(ROOT / "unused.json", ROOT / "expert_manifest.json", steps=21)
