from compute_efficiency_agents import ComputeEfficiencyController


def _profile():
    return [
        {"domain_id": d, "route_counts": [100 if e == d else 10 for e in range(20)]}
        for d in range(20)
    ]


def test_compute_agents_create_bounded_equal_quota_plan():
    plan = ComputeEfficiencyController().create_plan(
        routing_profile=_profile(), steps=40, effective_batch_size=4,
        sequence_length=48, hidden_size=16, activation_budget_bytes=65536,
    )
    assert plan["status"] == "COMPUTE_POLICY_APPROVED"
    assert len(plan["domain_schedule"]) == 40
    assert set(plan["domain_schedule"]) == set(range(20))
    assert set(plan["decisions"]["expert_load_balancer"]["domain_visits"].values()) == {2}
    batch = plan["decisions"]["adaptive_batch_planner"]
    assert batch["micro_batch_size"] * batch["gradient_accumulation_steps"] == 4
    assert batch["effective_batch_preserved"] is True
    assert plan["checkpoint_interval"] >= 5


def test_compute_agents_reject_incomplete_routing_profile():
    try:
        ComputeEfficiencyController().create_plan(
            routing_profile=_profile()[:-1], steps=40,
        )
    except PermissionError:
        pass
    else:
        raise AssertionError("incomplete routing profile must fail closed")
