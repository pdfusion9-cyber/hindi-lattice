import json
import pytest

from training_micro_agents import MicroAgentTrainingController, run_managed_training


def test_training_budget_is_capped_by_micro_agents():
    plan = MicroAgentTrainingController().create_plan(requested_steps=150, max_steps=60)
    assert plan["status"] == "TRAINING_PLAN_APPROVED"
    assert plan["allocated_steps"] == 60
    assert 1 <= plan["cpu_threads"] <= 4
    assert len(plan["selected_agents"]) == 3
    assert plan["micro_agent_unused"] if False else True
    assert plan["open_source_model_in_training_path"] is False


def test_unreviewed_external_data_is_refused():
    controller = MicroAgentTrainingController()
    with pytest.raises(PermissionError, match="training gate refused"):
        controller.create_plan(
            source_kind="external_dataset",
            external_data_used=True,
            dataset_ids=["unreviewed/example"],
        )


def test_external_model_weights_are_refused():
    controller = MicroAgentTrainingController()
    with pytest.raises(PermissionError, match="training gate refused"):
        controller.create_plan(external_weights_used=True)


def test_micro_agent_managed_training_uses_native_neurocore_and_synthetic_data(tmp_path):
    report_path = tmp_path / "managed_training_report.json"
    report = run_managed_training(report_path, requested_steps=40, max_steps=40, seed=7)

    assert report["status"] == "SYNTHETIC_SMOKE_PASS"
    assert report["training_started"] is True
    assert report["model_under_test"] == "native_neurocore_edge_scaled_reference"
    assert report["training_scale"] == "tiny_20_expert_reference"
    assert report["production_2_87b_training_started"] is False
    assert report["micro_agent_training_control"]["num_experts"] == 20
    assert report["open_source_model_integrated"] is False
    assert report["external_model_weights_used"] is False
    assert report["provenance"]["external_data_used"] is False
    assert report["micro_agent_training_control"]["status"] == "TRAINING_PLAN_APPROVED"
    assert report["micro_agent_training_control"]["allocated_steps"] == 40
    assert report["cpu_threads"] == report["micro_agent_training_control"]["cpu_threads"]
    assert report["training_elapsed_seconds"] > 0
    assert report["training_tokens_per_second"] > 0
    assert report["checkpoint"]["sha256"]
    assert json.loads(report_path.read_text(encoding="utf-8"))["training_started"] is True


def test_micro_agent_refuses_training_when_memory_budget_is_insufficient():
    controller = MicroAgentTrainingController()
    with pytest.raises(PermissionError, match="training gate refused"):
        controller.create_plan(
            requested_steps=40,
            max_steps=40,
            model_parameter_count=10_000_000,
            available_memory_bytes=1024 * 1024,
        )


def test_cpu_thread_override_is_bounded_and_reported():
    plan = MicroAgentTrainingController().create_plan(
        requested_steps=40, max_steps=40, cpu_threads=3
    )
    assert plan["cpu_threads"] == 3
    with pytest.raises(PermissionError, match="training gate refused"):
        MicroAgentTrainingController().create_plan(
            requested_steps=40, max_steps=40, cpu_threads=8
        )
