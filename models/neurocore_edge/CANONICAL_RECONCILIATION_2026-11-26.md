# Canonical NeuroCore architecture reconciliation — 2026-11-26

## Source contract

The six-page NeuroCore-Edge blueprint specifies a sandwich-looped predictive core: 2 prelude layers, 4 shared core layers repeated T times, and 2 coda layers, with effective depth D_eff = 2 + 4T + 2. The canonical configuration sets T = 5, so the implementation must contain 8 unique parameterized blocks and execute 24 effective operations.

The same blueprint specifies a hybrid SSM/attention working-memory design combining Gated DeltaNet/Mamba with GQA, 128 meta-tokens, sliding-window attention for about 90% of layers, and 3 global-attention layers. The v0.3.2 config also records 16 query heads, 4 KV heads, a 2,048-token window, and 127 control codes.

## Budget contradiction found

The v0.3.3 package readiness audit records:
- total parameters: 3,074,885,656;
- shared parameters: 872,876,056;
- expert parameters: 2,202,009,600.

Given 8 unique parameterized blocks, 20 experts per block, and the current source's three bias-free expert matrices (gate, up, down), the expert budget implies:

2,202,009,600 / (8 x 20 x 3 x 2,048) = 2,240

The canonical v0.3.2 config currently declares expert_intermediate_size = 2,899. That value implies:

8 x 20 x 3 x 2,048 x 2,899 = 2,849,832,960

That is 647,823,360 more expert parameters than the audited expert budget. The code must not silently select either value. A canonical author must reconcile the intermediate width, expert topology, and parameter split against the intended model source before a checkpoint layout is frozen.

## Implementation gap found

The canonical branch's checked-in modeling_neurocore_edge.py still instantiates cfg.num_hidden_layers independent layers. It does not implement the 8-block sandwich reuse. The file also lacks a GQA/RoPE attention module and an explicit 128-token meta-token bank. It therefore does not implement several explicit architectural elements in the source blueprint.

The repository contains no NeuroCore tokenizer asset and no trained base checkpoint. Its config correctly remains UNTRAINED_INITIALIZATION_ONLY with training disabled.

## Fail-closed audit

canonical_architecture_audit.py reports the budget arithmetic and checks the canonical topology, model-source structure, GQA/RoPE, meta-token bank, tokenizer assets, and trained checkpoint presence without allocating the 3B model. A blocked audit is a successful diagnostic result, not permission to train. Use --strict in any future training launcher to get a non-zero exit whenever readiness blockers remain.

## Work order

1. Resolve the expert width/budget contradiction with an exact model-level parameter count.
2. Implement the 8-unique-block / 24-effective-operation sandwich and test parameter sharing.
3. Implement and test GQA/RoPE plus sliding/global attention semantics and the 128 meta-token bank according to the agreed source contract.
4. Produce the canonical 65,536-entry NeuroCore tokenizer and persist its immutable hash.
5. Complete shared-backbone pretraining and create a private, metadata-verified base checkpoint.
6. Only then begin teacher-to-student distillation of one fixed expert; keep dataset approval and teacher-license gates separate.

The audit does not approve any dataset, enable training, or represent a trained model.
