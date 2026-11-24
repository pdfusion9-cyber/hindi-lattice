# CPU-first NeuroCore Micro-Agent Training Design

**Purpose:** make the native NeuroCore training pipeline use CPU-only runners efficiently, without introducing an external base model, external weights, or an external model runtime.

## The micro-agent team

| Agent | Responsibility | Measured signal / action |
|---|---|---|
| CPU Profiler | Detect effective CPU quota, affinity, available RAM and disk | cgroup-aware profile, not just os.cpu_count() |
| Thread Planner | Prevent nested oversubscription | One training process per runner; bounded intra-op threads; one inter-op thread; no DataLoader subprocesses for in-memory batches |
| Expert Load Balancer | Distribute 20 expert shards across available runners | Longest-processing-time-first assignment using observed tokens/second and estimated shard work |
| Memory Guard | Refuse allocations that exceed safe host RAM | FP32 AdamW estimate: 16 bytes/trainable parameter; FP32 frozen weights: 4 bytes/parameter; explicit activation and staging reserve |
| Adaptive Batch Agent | Select a micro-batch from the remaining memory budget | Keep the effective batch constant through gradient accumulation |
| Checkpoint Agent | Bound both checkpoint overhead and lost work after interruption | Use measured step/checkpoint seconds and a maximum recovery window |
| Throughput Calibrator | Run short warm-up trials before selecting threads, batch size or precision | Compare tokens/second and validation parity on the same runner class; never assume more threads are faster |

## CPU execution shape

- Twenty independent expert tasks remain twenty separate runner jobs when twenty runner slots are available.
- Each runner owns one expert-training process. It should not run a second competing training process or spawn an excessive number of data-loader workers.
- Expert workers must load only the frozen shared backbone plus the target expert shard. Other 19 expert weight tensors must not be materialized on that worker.
- Tokenization/packing should be performed once per immutable dataset revision, then re-used by all expert steps. The training loop should consume deterministic, pre-tokenized, length-bucketed batches.
- Use measured CPU kernels and a validated numerical-parity test before enabling BF16/oneDNN-specific paths. The planner must not assume BF16 is faster or supported on every runner.
- Periodic checkpoints must be atomic and include model/config fingerprints, optimizer state, completed step, RNG state, data cursor and provenance.

## Critical 2.87B memory constraint

A full FP32 AdamW update is approximately 16 bytes per trainable parameter before activations and temporary buffers. For 2,869,448,704 parameters that is about 45.9 GB (42.7 GiB), so a typical 16 GiB CPU runner cannot train the entire model in one process with this optimizer.

The expert-shard path can be smaller only if it truly loads the frozen shared backbone and the selected expert alone. The existing reference path constructs the complete model and loads the complete shared state before selecting one expert; that is a topology smoke test, not a memory-sharded production implementation.

The shared backbone is a separate bottleneck. With the candidate 872,876,056 shared parameters, a full FP32 AdamW update is roughly 14.0 GB before activations. It does not fit a conservative per-runner safe-memory budget on a 16 GiB host. The CPU planner therefore reports this stage separately and does not treat twenty expert runners as extra RAM for the single shared-base runner.

## What this first implementation does

cpu_training_agents.py provides:
- effective CPU/RAM/disk detection (including Linux cgroup constraints);
- deterministic CPU thread policy;
- conservative frozen/trainable memory estimator;
- adaptive micro-batch plus gradient-accumulation planner;
- checkpoint interval based on measured step and save time;
- load-balanced assignment of 20 expert IDs to runner slots;
- a fail-closed status when expert memory or disk requirements do not fit.

This is a planner/preflight, not a production trainer. It deliberately reports production_dispatch_allowed=false. No training run, dataset approval, model config, tokenizer, or checkpoint is changed by this module.

## Next engineering gates

1. Implement a true shared-backbone + selected-expert model slice and memory-mapped/sharded checkpoint reader; do not instantiate unrelated experts on expert workers.
2. Resolve the shared-backbone optimizer/partition strategy; current single-runner FP32 AdamW does not fit the conservative CPU memory budget.
3. Profile real CPU throughput on the target runner class and test thread counts, sequence lengths and micro-batch sizes on identical data.
4. Reconcile the canonical 2.87B configuration and source implementation; keep the dataset rights/approval registry fail-closed.
5. Only after memory, throughput, architecture and dataset gates pass, make a separately reviewed production dispatch workflow.

## Truthful success criteria

A pilot is successful only if repeatable same-hardware tests show higher tokens/second, lower wall-clock time per token at equal effective batch and comparable validation loss, without exceeding memory/disk limits. A planner decision alone is not evidence of a speedup.
