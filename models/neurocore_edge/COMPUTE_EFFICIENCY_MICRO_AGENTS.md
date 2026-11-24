# NeuroCore compute-efficiency Micro-Agents

This reference pilot adds three deterministic policy agents on top of the
project's existing `MicroAgentOrchestrator`:

1. **expert_load_balancer** profiles actual top-2 router assignments on the
   held-out synthetic shards, then creates a deterministic domain schedule
   that minimizes projected router-load coefficient of variation while
   preserving equal visit quotas for all 20 domains.
2. **adaptive_batch_planner** selects a physical micro-batch under a bounded
   activation-memory proxy and computes gradient accumulation so the effective
   batch size remains unchanged.
3. **checkpoint_scheduler** chooses a bounded periodic save interval. The
   benchmark saves model, optimizer, step and RNG state to an ephemeral
   runner-local file, retaining no production checkpoint.

## Run

```bash
cd models/neurocore_edge
python -m pytest -q test_compute_efficiency_agents.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python compute_efficiency_benchmark.py \
  --manifest expert_manifest.json --report-path compute_efficiency_report.json --steps 40
```

The report compares a baseline and managed run with the same native model,
seed, load-balanced schedule, examples per optimizer step and total token
budget. It records wall-clock training time, tokens/second, checkpoint time
and bytes, sparse-gradient steps, parameter parity, and the policy's estimated
activation memory.

## Measurement limits

The activation estimate is an analytical proxy, not measured peak RAM/VRAM.
The benchmark is a tiny CPU synthetic engineering test, not the 2.87B-parameter
model, not a real corpus, not domain quality evidence, and not production
convergence. Timing from one CI run is noisy; repeat on identical hardware
before claiming a speedup. The baseline also uses the same balanced schedule,
so this isolates batch/checkpoint control rather than claiming a scheduling
advantage over an unbalanced baseline.

No external data, user data, open-source model weights, Qwen checkpoints, or
external foundation-model runtime are used or integrated. Production training
remains blocked while dataset assignments are unapproved and the production
config has training disabled.
