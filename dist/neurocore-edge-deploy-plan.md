# NeuroCore-Edge public CPU deployment

- Public GitHub-hosted `ubuntu-24.04` CPU runners only.
- 20 expert jobs, one expert ID per job, up to 280 minutes per cycle.
- Checkpoint every 250 steps; retain optimizer state.
- Release-backed checkpoints; exact expert tensor keys/shapes and zero parameter-count delta on merge.
- Training stays blocked while dataset approvals are pending.
- This plan is not a trained model; public runner CPU throughput and convergence remain unmeasured.
