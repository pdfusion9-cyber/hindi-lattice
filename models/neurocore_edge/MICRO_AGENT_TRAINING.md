# Micro-Agent Managed NeuroCore Training

This milestone integrates the existing hybrid micro-agent runtime with the
native NeuroCore training lifecycle. It does **not** integrate Qwen3 or any
other open-source foundation-model weights, tokenizer, checkpoint, or runtime
into NeuroCore.

## Training control plane

Three deterministic project-owned micro-agents run before a training session:

1. `training_data_gate` permits only generated in-memory synthetic data in
   this pilot. External datasets and user data are refused.
2. `training_budget` caps steps, selects one CPU thread, and records the
   balanced pilot budget.
3. `checkpoint_guard` requires the native NeuroCore reference model, no
   external weights, and an ephemeral synthetic-only checkpoint.

The controller uses the project's `MicroAgentOrchestrator`, bounded parallel
dispatch, task identity checks, and explicit resource accounting. These are
policy agents, not separately trained neural networks and not calls to Qwen.

## Training pilot

Run locally from this directory:

```bash
python training_micro_agents.py --report-path micro_agent_training_report.json --steps 40 --max-steps 40
```

The pilot trains a deliberately tiny configuration of the native NeuroCore
sparse-MoE reference implementation for 40 steps on generated in-memory token
cycles. It validates loss/gradient finiteness, checkpoint reload, and optimizer
resume. It is an engineering smoke experiment, not full-model convergence.

## Efficiency gains being tested

- The data gate prevents wasted compute on unapproved or invalid data.
- The budget agent caps work before model allocation and uses a small CPU thread
  budget.
- The checkpoint guard avoids incompatible/external weights entering the
  checkpoint path.
- The training report records agent calls, completion, elapsed planning time,
  allocated steps, and checkpoint verification.

These are bounded engineering controls; no throughput gain is claimed until
measured against a baseline with the same hardware and workload.

## Strict separation

- Qwen3 remains a separate external evaluation target only.
- No Qwen3 package, model download, model API, weights or checkpoints are
  required by the training controller.
- The production `config.json` training flag remains false.
- All external dataset approval rows remain pending; this pilot does not
  change any approval.
- Production training and 20-expert multi-runner training are not launched.

A successful pilot proves only that the native tiny reference model can be
trained under the micro-agent lifecycle controller on synthetic data.
