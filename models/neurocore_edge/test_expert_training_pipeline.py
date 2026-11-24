import json
from pathlib import Path
import pytest
from expert_training_pipeline import build_expert_plan

ROOT=Path(__file__).parent

def test_twenty_expert_synthetic_plan_is_complete_and_bounded():
    p=build_expert_plan(ROOT/"expert_manifest.json",ROOT/"dataset_approval.json",
                        ROOT/"config.json",synthetic_only=True,requested_steps=50,max_steps=30)
    assert p["expert_count"] == 20
    assert [x["expert_id"] for x in p["expert_assignments"]] == list(range(20))
    assert p["status"] == "SYNTHETIC_20_EXPERT_SMOKE_ALLOWED"
    assert p["micro_agent_gate"]["allocated_steps"] == 30
    assert p["resource_policy"]["max_parallel_experts"] == 4
    assert p["production_dispatch_allowed"] is False
    assert p["external_foundation_model_weights_used"] is False

def test_production_plan_is_refused_until_every_assignment_approved():
    p=build_expert_plan(ROOT/"expert_manifest.json",ROOT/"dataset_approval.json",
                        ROOT/"config.json",synthetic_only=False)
    assert p["status"] == "PRODUCTION_TRAINING_BLOCKED"
    assert p["unapproved_assignment_count"] > 0
    assert p["production_config_training_enabled"] is False
    assert p["production_dispatch_allowed"] is False

def test_manifest_must_cover_exactly_twenty_experts(tmp_path):
    bad=tmp_path/"manifest.json"
    bad.write_text(json.dumps([{"expert_id":0,"name":"only","datasets":[]}]))
    with pytest.raises(ValueError,match="exactly expert IDs"):
        build_expert_plan(bad,ROOT/"dataset_approval.json",ROOT/"config.json")
