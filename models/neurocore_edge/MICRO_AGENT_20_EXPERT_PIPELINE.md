# 20-Expert Micro-Agent Training Pipeline

The planner covers all 20 expert IDs from `expert_manifest.json`. Each expert
gets a deterministic assignment record, dataset-approval status, bounded
resource policy and isolated checkpoint namespace.

## Current mode: synthetic topology smoke

- Generated in-memory synthetic tokens only.
- Native NeuroCore architecture only; no external foundation-model weights.
- Up to four expert jobs in a bounded smoke plan.
- No external corpus is downloaded and no production checkpoint is created.
- Shared-backbone updates and expert merge are deliberately disabled.

Run:

```bash
python expert_training_pipeline.py --manifest expert_manifest.json --approvals dataset_approval.json --config config.json --output expert_training_plan.json
python synthetic_training_smoke.py --report-path expert20_smoke.json --steps 40 --num-experts 20
```

The synthetic model is a deliberately small reference configuration with 20
expert modules, not the full 2.87B-parameter model. The smoke verifies model
construction, forward/backward, finite gradients/loss, and checkpoint/resume
infrastructure. It is not a domain-quality or throughput benchmark.

## Production gate

Production mode refuses to dispatch unless every expert/dataset assignment is
approved, the registry status is `APPROVED_FOR_COMMERCIAL_TRAINING`, and the
production training config is explicitly enabled. Current repository state
has zero approved assignment rows and `training.enabled=false`; therefore
production training must remain blocked.

Qwen3 and all other open-source model weights/checkpoints remain outside this
training pipeline.
