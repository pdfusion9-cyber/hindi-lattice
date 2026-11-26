# Canonical open-weight teacher assimilation

## Fixed assignment and first pilot

The first bounded pilot remains expert 11, `hindi_indic_docs`, assigned to the Dhīr 5B frozen teacher (`zenteiq-ai/Dhir-5B-A1.7B`). This does not alter the fixed 20-runner / 20-expert mapping. Each runner ID remains equal to its permanent expert ID; no worker rotation is permitted.

## What happens to the actual model weights

1. Load the external checkpoint at a pinned immutable revision, using its native Transformers architecture and its own tokenizer.
2. Freeze every teacher parameter. The teacher produces supervised target sequences; it is not a hosted inference dependency for the finished NeuroCore model.
3. Load a separately trained canonical NeuroCore v0.3.2 base checkpoint and the pinned NeuroCore tokenizer.
4. Force the assigned expert ID through the canonical sandwich forward path. Only that expert's tensors in each unique prelude/core/coda block receive gradients; shared layers, routers and the other 19 experts stay frozen.
5. Tokenize teacher outputs with the NeuroCore tokenizer and compute causal next-token loss. Save resumable expert-only private checkpoints with teacher revision and base/tokenizer/prompt fingerprints.

## Why arbitrary tensor transplant is prohibited

Dhīr 5B is documented as a 28-layer, width-2,560 MoE with 12 experts and top-2 routing. Canonical NeuroCore has 2,048 hidden width, 20 experts and an 8-unique-block sandwich reused across 24 effective operations. The model dimensions and parameter semantics are not interchangeable. The implementation therefore rejects unapproved tensor mappings and shape mismatches. Exact tensor copy is only permitted when an explicit key mapping, exact shape and separately recorded semantic compatibility all pass.

## Implemented files

- `teacher_weight_assimilation.py`: frozen teacher loader, exact-copy audit, fixed-expert sequence distiller, strict resumable checkpointing and canonical forward adapter.
- `test_teacher_weight_assimilation.py`: CPU tests using the real canonical NeuroCore model at tiny dimensions.
- `train_teacher_assimilated_expert.py`: one-expert CLI; requires pinned teacher and tokenizer revisions, an authorized prompt set, and an already-trained canonical base checkpoint.

## Launch gates still closed

This integration is code and test infrastructure, not proof of a production transfer. The repository's current NeuroCore config remains `UNTRAINED_INITIALIZATION_ONLY`; no trained shared-base checkpoint or NeuroCore tokenizer asset is present. The CLI deliberately rejects a random/uncertified student base and moving model revisions. Therefore no real Dhīr download or production expert update is claimed yet. The first real run requires:
- a private, checksum-verified trained NeuroCore v0.3.2 base checkpoint;
- the approved/pinned 65,536-token NeuroCore tokenizer;
- a pinned immutable Dhīr revision and verified license/access terms;
- a prompt set whose rights, privacy and consent status are reviewed;
- measured GPU/host memory sufficient for the teacher, student activations and expert optimizer state.

Weights and trained checkpoints must remain outside the public Git repository.
