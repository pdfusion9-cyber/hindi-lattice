# Dhīr-2B adoption in NeuroCore

The 0.4.0-dhir2b-adapters path uses the native Dhīr-2B Transformers implementation as the NeuroCore dense backbone. Pretrained tensor names and shapes are not rewritten or approximated. The public architecture description identifies 28 decoder layers, hidden width 2,048 and 16,384 context. The exact vocabulary and feed-forward intermediate size are read from the immutable checkpoint config and tokenizer at import time, never guessed.

## NeuroCore experts

Twenty independent residual bottleneck experts are attached after every one of the 28 decoder blocks. Expert ID 0–19 keeps the same identity at every layer. Zero-initialized adapter output projections preserve the original Dhīr logits before training. The backbone and router are frozen during fixed-expert training.

A top-2 router exists, but it is untrained and is not ready for automatic production routing. Train each specialist with an explicit expert ID, then perform separate router calibration and independent validation.

## CPU workflow

Install requirements-dhir2b.txt and pin the source to a 40-character Hugging Face commit SHA:

~~~bash
python models/neurocore_edge/import_dhir2b_weights.py \
  --revision <40-character-commit-sha> \
  --output /private/neurocore-dhir2b
~~~

The importer stores the complete backbone, tokenizer, adapter bank and manifest locally. BF16 is the default; float32 is available. Base weights alone require about 4.06 GB decimal before runtime memory, activations and optimizer state.

Expert training requires exact approval records for both the JSONL corpus SHA-256 and the local tokenizer directory SHA-256 in dataset_approval.json:

~~~bash
python models/neurocore_edge/train_dhir2b_expert.py \
  --model-dir /private/neurocore-dhir2b \
  --data /private/approved-expert-11.jsonl \
  --approval models/neurocore_edge/dataset_approval.json \
  --expert-id 11 --steps 100 \
  --output /private/expert-11-dhir2b.pt
~~~

Output checkpoints remain provisional: EXPERT_ADAPTER_TRAINED_UNVALIDATED. They are not automatically certified or merged into a production model.

## Scope of CI

The generic NeuroCore reference tests use a tiny synthetic 28-layer backbone to check layer attachment, exact vocabulary validation, initial output preservation and fixed-expert gradient isolation. Dedicated CPU workflows separately import the real pinned Dhīr-2B checkpoint and verify checkpoint save/reload. Neither kind of CI run performs expert training.

## Verified real Dhīr-2B import and checkpoint

On 2026-11-26, the dedicated CPU import workflow completed successfully:
- Model: `zenteiq-ai/Dhir-2B`
- Immutable revision: `7cbe444cd1e45e25197cfa0f683b2cb575858169`
- Device: CPU, BF16, GPU not used
- Exact tensor load audit: clean; no missing, unexpected, mismatched or error tensors
- Native Dhīr backbone: 28 decoder blocks, hidden width 2,048, vocabulary 131,072
- NeuroCore overlay: 20 experts, top-2 routing, rank-8 bottleneck adapters after every decoder block

The follow-up checkpoint roundtrip also passed: logits were finite and exactly identical before and after save/reload. The saved checkpoint tree SHA-256 is `90c3b46d38f52f03a41ed6d5b07a530dd864f107176d3c98e64c808452710148`, total size 4,150,375,510 bytes. It was persisted as Actions artifact `dhir2b-neurocore-cpu-checkpoint` from [run 45780741273](https://github.com/pdfusion9-cyber/hindi-lattice/actions/runs/45780741273), artifact ID `15098526765`, expiring 2026-12-26.

This is the Dhīr-compatible NeuroCore path to use for the first specialist run. The separate 0.3.2-canonical 3.075B sandwich architecture is not the weight-compatible target and must not be substituted. The artifact is attached to a public repository's Actions run; it is not private storage.

## First specialist training state

Expert 11 (`hindi_indic_docs`) is the proposed first fixed specialist, using the pinned IndicCorpV2 Hindi shard. The Dhīr tokenizer is already bundled in the verified checkpoint; no new 65,536-token tokenizer should be built for this path. The trainer still requires exact SHA-256 approval for the cleaned JSONL corpus and the local tokenizer directory. Those approvals are pending, and no expert optimizer step has been run.


## Resuming a CPU expert run

A provisional expert checkpoint can be resumed with \`--resume-from\`. The trainer verifies the exact model ID, immutable base revision, expert ID, architecture, tokenizer SHA-256 and dataset SHA-256 before restoring expert weights, optimizer state and PyTorch RNG state. A mismatch aborts the resume.


## Verified source and controlled custom-code loading

The metadata audit passed at immutable revision \`7cbe444cd1e45e25197cfa0f683b2cb575858169\`. Exact config: 28 layers, width 2,048, intermediate width 8,192, 16 query heads, 8 KV heads, head dimension 128, vocabulary 131,072, context 16,384, RoPE theta 1,000,000, RMSNorm epsilon 1e-6, SiLU, tied embeddings.

The upstream repository uses \`auto_map\` custom Transformers code. Static source was inspected and SHA-256 pinned:
- \`configuration_dhir.py\`: \`9ddceca88e4d7780ccab5d194d93bb1bf557238e4f71b5b7abe4d2b540ba5cee\`
- \`modeling_dhir.py\`: \`9836ca6d3996428bbcd7dc768e885268707dbb7d3dc04ca21df494db893bfc45\`

The importer verifies both files against these exact hashes before using Transformers' custom-code loading at the immutable model revision. When saving a local NeuroCore checkpoint, the verified source files are copied alongside the backbone; local reload verifies the hashes again before enabling custom code. No unpinned remote code is accepted. This is a pinned-source control, not a claim that arbitrary custom code is generally safe.

The metadata audit does not download the two model weight shards (approximately 4.06 GB combined). Real weight import and CPU pilot training remain pending.
