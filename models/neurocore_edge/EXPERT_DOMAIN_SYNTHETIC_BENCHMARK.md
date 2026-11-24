# 20-Expert Domain Synthetic Benchmark

This benchmark creates authored fictional task/answer examples for each of the
20 NeuroCore expert domains. It writes no raw external or user data and does not
use an open-source foundation model as a teacher, tokenizer, or checkpoint.

## Dataset layout

Each expert gets three training examples and two separate held-out validation
examples. Train and validation record IDs are disjoint. The benchmark reports
UTF-8 byte counts, example counts, per-expert training steps and per-expert
validation loss.

## Controlled comparison

Two trials use the same native 20-expert tiny NeuroCore configuration, seed,
synthetic corpus, optimizer and round-robin domain schedule:

- Single-agent baseline: training starts directly.
- Micro-Agent-managed: data gate, budget agent and checkpoint guard run first.

Both trials receive exactly the same number of optimizer steps; the final model
parameters must match to within 1e-7. Wall-clock results are one-run measurements,
not proof of a speedup. Repeat on the same runner before drawing efficiency
conclusions.

## Run

```bash
python expert_domain_synthetic_benchmark.py --manifest expert_manifest.json --report-path expert_domain_benchmark.json --steps 40
```

This is a tiny engineering benchmark, not training of the 2.87B production
configuration and not evidence of real domain accuracy. The production training
flag and all external dataset approval gates remain closed.
