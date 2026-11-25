# NeuroCore teacher-to-student pilot readiness (2026-11-25)

## Decision

The first candidate pilot is **expert 11 — `hindi_indic_docs`**, using its existing fixed assignment **`dhir_5b` → `zenteiq-ai/Dhir-5B-A1.7B`**. This preserves the current 20-runner/20-expert assignment; no worker is rotated and no teacher is made a permanent inference dependency.

Use sequence distillation: load Dhīr in its native architecture and tokenizer as a frozen teacher, generate Hindi/Indic document targets, tokenize those targets with the approved NeuroCore tokenizer, and train only NeuroCore expert 11. Do not transplant unrelated Dhīr tensors into NeuroCore. Do not train the BG-Gate router as if it were expert 18's MLP.

## Candidate evidence

The official ZenteiQ model page describes Dhīr 5B as a 5.09B-parameter MoE with 1.79B active parameters per token, English plus ten Indian languages, a 131k-entry tokenizer, 16k context, and Apache-2.0 licensing. ZenteiQ's technical overview gives 28 layers, width 2,560, and approximately 10.19 GB BF16 weights. The published model is useful as a frozen multilingual teacher; these values do not make its tensors shape-compatible with NeuroCore.

Candidate:
- Hugging Face repository: `zenteiq-ai/Dhir-5B-A1.7B`
- Model card: https://huggingface.co/zenteiq-ai/Dhir-5B-A1.7B
- Official license/model page: https://www.zenteiq.ai/dhir/
- Transfer mode: frozen teacher → NeuroCore expert 11 sequence distillation
- Immutable 40-character teacher commit SHA: **NOT YET PINNED**; do not download `main` for a reproducible training job.

## Repository audit — launch is currently blocked

The following checks were performed against the active adoption branch and the canonical v0.3.2 branch:

1. **No trained NeuroCore base checkpoint exists.** Both branches' configuration identifies the model as `UNTRAINED_INITIALIZATION_ONLY`. Their Git trees contain no `.safetensors`, `.pt`, `.pth`, or model weight blobs. Teacher-to-expert distillation with a random shared backbone, random token embeddings and random output head would not produce a capable student.
2. **The adoption branch is not the canonical v0.3.2 architecture.** The adoption branch uses `0.2.0-pytorch-reference`, 24 independently-instantiated layers, intermediate size 660 and 2,869,448,704 parameter estimate. The canonical branch declares `0.3.2-canonical`, a 2-prelude + 4-core-blocks-repeated-5-times + 2-coda topology, intermediate size 2,899 and 3,074,885,656 parameters. The current adoption trainer imports the former implementation. These must not be treated as interchangeable checkpoint layouts.
3. **The canonical model implementation still needs architecture reconciliation.** Its checked-in Python reference uses 24 independently-instantiated layers and does not yet implement the config's 8 unique parameterized blocks repeated across 24 effective operations. The config's parameter count therefore must be reproduced by an executable meta-device count before a base checkpoint is accepted.
4. **No NeuroCore student tokenizer asset is present** in either branch (no `tokenizer.json`, vocabulary, merges or SentencePiece model). The config declares 65,536 tokens. Dhīr's own tokenizer has about 131k entries; it cannot be substituted as the NeuroCore tokenizer without a deliberate model/config redesign.
5. **Dataset approval is still default-deny.** The approval manifest remains pending; no exact dataset revision/config/splits or commercial-training/PII/consent approvals are recorded for expert 11.
6. **The teacher revision is not pinned.** A model ID or moving `main` ref is not a reproducible immutable revision.
7. **The previous CPU CI success is code/test evidence only.** It does not download Dhīr, validate a full teacher forward pass, train a production NeuroCore checkpoint, or prove that GitHub-hosted CPU runners can train a multi-billion-parameter model.

## Required order before the pilot

1. Reconcile and freeze the **canonical v0.3.2 executable architecture** (including the repeated-block topology, parameter count and forced-expert forward path).
2. Commit and test the canonical tokenizer with exactly the declared vocabulary and a SHA-256 fingerprint.
3. Complete the shared-backbone pretraining stage and produce a private checkpoint whose metadata proves the exact architecture/config/tokenizer and completed shared stage. Never use a random initialization as the distillation base.
4. Approve a specific expert-11 prompt/target dataset with immutable revision, splits, provenance, license, commercial-training permission, PII/consent/schema review and held-out split.
5. Pin the Dhīr teacher repository to a 40-character immutable revision and record hashes for config, tokenizer and every downloaded weight shard. Confirm download/access terms for the chosen source.
6. Verify a GPU/host memory profile sufficient for the BF16 teacher, student expert training and optimizer state; GitHub standard CPU runners are not an acceptable production teacher-training target.
7. Run a bounded expert-11 pilot; save resumable private checkpoints; compare held-out Hindi/Indic document metrics against the base checkpoint before unlocking the other 19 fixed assignments.

## Safety rule

Until all requirements above pass, the trainer must remain unlaunched for real model weights. A tiny synthetic unit test may verify mechanics, but it must never be labeled as pretrained knowledge transfer or production training.
