# NeuroCore 20-GPU runner readiness

## Why this preflight exists

The successful Run #34 validated 20 separate runner jobs using a tiny synthetic
model. It did **not** validate GPU availability or production-scale training.
This workflow is deliberately a hardware preflight only.

## Configure GitHub

1. The repository must be accessible to an organization that can use GitHub
   larger runners. GitHub documents larger runners as available to organizations
   and enterprises on GitHub Team or GitHub Enterprise Cloud.
2. Configure a GPU larger-runner pool with repository access and concurrency
   capacity for 20 simultaneous jobs. The documented GitHub GPU larger runner
   has 4 vCPU, 28 GB RAM, one NVIDIA Tesla T4 with 16 GB VRAM, and 176 GB SSD.
3. Ensure billing and spending limits permit the run.
4. Run **NeuroCore 20-GPU Runner Preflight** from Actions and enter the exact
   configured larger-runner name/label as `gpu_runner_label`.

## Per-runner acceptance

Every one of the 20 jobs must report:
- at least 4 CPU cores;
- at least 24 GiB host RAM;
- at least one NVIDIA GPU with at least 15,000 MiB VRAM;
- at least 150 GB total disk and 100 GB free disk.

All 20 runner IDs (0 through 19) must pass. A single failure blocks the
20-runner inventory summary.

## Production gate (separate)

Passing the hardware preflight is not permission to train. The final job checks
the canonical model config and dataset registry and reports any blockers. It
never dispatches production training.

As of the current branch:
- `config.json` still describes architecture `0.2.0-pytorch-reference`
  and marks the model `UNTRAINED_INITIALIZATION_ONLY`;
- `training.enabled` is false;
- all 27 dataset-to-expert assignment rows remain pending approval;
- the codebase still lacks the production tokenizer, complete attention /
  sandwich architecture and a production full-scale optimizer/checkpoint path.

The old architecture audit also reports a conflict with a later v0.3.x design
candidate. The two snapshots must be reconciled into one canonical source
before any full-size checkpoint or data corpus is created.

## Safety boundary

- Native NeuroCore weights only. No external foundation model checkpoints,
  distillation or model-weight merging.
- Dataset download is blocked until immutable revision, license, commercial
  training rights, redistribution scope, PII, consent, schema and split reviews
  are explicitly approved.
- Tiny synthetic runs are engineering tests, not production training.
