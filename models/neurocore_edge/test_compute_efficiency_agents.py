import pytest

from compute_efficiency_agents import (
    AdaptiveBatchPlanner,
    CheckpointScheduler,
    ComputeEfficiencyController,
    ExpertLoadBalancer,
)


def routing_profile():
    return [
        {"domain_id": domain, "route_counts": [domain + expert for expert in range(20)]}
        for domain in range(20)
    ]


def test_load_balancer_visits_all_twenty_domains_equally():
    plan = ExpertLoadBalancer().plan(routing_profile(), steps=100)
    assert len(plan["domain_schedule"]) == 100
    assert set(plan["domain_visits"].values()) == {5}
    assert set(map(int, plan["domain_visits"])) == set(range(20))


def test_load_balancer_rejects_missing_or_duplicate_experts():
    rows = routing_profile()
    with pytest.raises(ValueError):
        ExpertLoadBalancer().plan(rows[:-1], steps=100)
    rows[-1] = dict(rows[0])
    with pytest.raises(ValueError):
        ExpertLoadBalancer().plan(rows, steps=100)


def test_adaptive_batch_preserves_effective_batch():
    plan = AdaptiveBatchPlanner().plan(
        effective_batch_size=8,
        sequence_length=128,
        hidden_size=256,
        num_layers=2,
        activation_budget_bytes=64 * 1024 * 1024,
    )
    assert plan["effective_batch_size"] == 8
    assert plan["micro_batch_size"] * plan["gradient_accumulation_steps"] == 8
    assert plan["effective_batch_preserved"] is True


def test_adaptive_batch_refuses_when_even_one_sequence_does_not_fit():
    with pytest.raises(MemoryError):
        AdaptiveBatchPlanner().plan(
            effective_batch_size=8,
            sequence_length=4096,
            hidden_size=4096,
            num_layers=24,
            activation_budget_bytes=1024,
        )


def test_checkpoint_scheduler_bounds_interval():
    assert CheckpointScheduler().plan(40) == 10
    assert CheckpointScheduler().plan(1000) == 200


def test_controller_combines_cpu_compute_agents_without_external_weights():
    plan = ComputeEfficiencyController().create_plan(
        routing_profile=routing_profile(),
        steps=100,
        effective_batch_size=8,
        sequence_length=128,
        hidden_size=256,
        num_layers=2,
        activation_budget_bytes=64 * 1024 * 1024,
    )
    assert plan["status"] == "COMPUTE_POLICY_APPROVED"
    assert len(plan["domain_schedule"]) == 100
    assert plan["effective_batch_size"] == 8
    assert plan["provenance"]["open_source_model_integrated"] is False
