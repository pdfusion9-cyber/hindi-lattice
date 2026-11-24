from cpu_training_agents import (
    CPUProfile, checkpoint_interval, create_cpu_training_plan,
    estimate_worker_memory_bytes, choose_micro_batch, plan_cpu_threads,
)

GIB = 1024 ** 3


def profile(memory_gib=12, cpus=4, free_disk_gib=80):
    return CPUProfile(
        logical_cpus=cpus, affinity_cpus=cpus, quota_cpus=float(cpus),
        effective_cpus=cpus, available_memory_bytes=memory_gib * GIB,
        free_disk_bytes=free_disk_gib * GIB, hostname="ci-test-runner",
    )


def test_cpu_thread_plan_uses_available_quota_without_oversubscription():
    plan = plan_cpu_threads(profile(cpus=4))
    assert plan["intra_op_threads"] == 4
    assert plan["interop_threads"] == 1
    assert plan["dataloader_workers"] == 0
    assert plan["parallel_training_processes_per_runner"] == 1


def test_selected_expert_plan_fits_but_requires_other_experts_to_be_unloaded():
    plan = create_cpu_training_plan(
        profile=profile(), shared_backbone_parameters=872_000_000,
        expert_parameters=[110_000_000] * 20, runner_count=20,
    )
    assert plan["status"] == "CPU_TRAINING_PLAN_APPROVED"
    assert len(plan["expert_workers"]) == 20
    assert all(row["other_expert_weights_loaded"] is False for row in plan["expert_workers"])
    assert len({tuple(row["expert_ids"]) for row in plan["runner_assignments"]}) == 1
    assert all(row["memory_fits"] for row in plan["expert_workers"])
    assert plan["production_dispatch_allowed"] is False
    assert plan["provenance"]["open_source_model_integrated"] is False


def test_shared_backbone_adamw_is_reported_as_separate_single_runner_blocker():
    plan = create_cpu_training_plan(
        profile=profile(memory_gib=12), shared_backbone_parameters=872_000_000,
        expert_parameters=[110_000_000] * 20, runner_count=20,
    )
    assert plan["shared_base_stage"]["fits_single_runner_safe_memory"] is False
    assert plan["shared_base_stage"]["status"] == "BLOCKED"


def test_low_memory_expert_worker_is_fail_closed():
    plan = create_cpu_training_plan(
        profile=profile(memory_gib=4), shared_backbone_parameters=872_000_000,
        expert_parameters=[110_000_000] * 20, runner_count=20,
    )
    assert plan["status"] == "CPU_TRAINING_PLAN_BLOCKED"
    assert any("memory estimate" in item for item in plan["blockers"])


def test_micro_batch_preserves_effective_batch():
    plan = choose_micro_batch(
        effective_batch_size=8, activation_bytes_per_example=128 * 1024**2,
        static_memory_bytes=5 * GIB, safe_memory_bytes=8 * GIB,
    )
    assert plan["micro_batch_size"] == 8
    assert plan["gradient_accumulation_steps"] == 1
    assert plan["effective_batch_preserved"] is True


def test_checkpoint_interval_balances_overhead_and_recovery():
    interval = checkpoint_interval(
        measured_step_seconds=0.5, measured_checkpoint_seconds=2.0,
        target_overhead_fraction=0.01, max_recovery_seconds=300,
    )
    assert interval == 600


def test_memory_estimate_matches_fp32_adamw_assumptions():
    total = estimate_worker_memory_bytes(
        shared_frozen_parameters=100, active_trainable_parameters=10,
        activation_reserve_bytes=1000, staging_reserve_bytes=500,
    )
    assert total == 100 * 4 + 10 * 16 + 1000 + 500
