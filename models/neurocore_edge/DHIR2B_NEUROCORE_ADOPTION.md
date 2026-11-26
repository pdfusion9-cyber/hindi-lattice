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

CI uses a tiny synthetic 28-layer backbone. It checks layer attachment, exact vocabulary validation, initial output preservation and fixed-expert gradient isolation. It does not download, import or train the real 2B checkpoint. The v0.3.2 reference files and parameter audit remain unchanged.


## Resuming a CPU expert run

A provisional expert checkpoint can be resumed with \`--resume-from\`. The trainer verifies the exact model ID, immutable base revision, expert ID, architecture, tokenizer SHA-256 and dataset SHA-256 before restoring expert weights, optimizer state and PyTorch RNG state. A mismatch aborts the resume.


## Verified source and controlled custom-code loading

The metadata audit passed at immutable revision \`7cbe444cd1e45e25197cfa0f683b2cb575858169\`. Exact config: 28 layers, width 2,048, intermediate width 8,192, 16 query heads, 8 KV heads, head dimension 128, vocabulary 131,072, context 16,384, RoPE theta 1,000,000, RMSNorm epsilon 1e-6, SiLU, tied embeddings.

The upstream repository uses \`auto_map\` custom Transformers code. Static source was inspected and SHA-256 pinned:
- \`configuration_dhir.py\`: \`9ddceca88e4d7780ccab5d194d93bb1bf557238e4f71b5b7abe4d2b540ba5cee\`
- \`modeling_dhir.py\`: \`9836ca6d3996428bbcd7dc768e885268707dbb7d3dc04ca21df494db893bfc45\`

The importer verifies both files against these exact hashes before using Transformers' custom-code loading at the immutable model revision. When saving a local NeuroCore checkpoint, the verified source files are copied alongside the backbone; local reload verifies the hashes again before enabling custom code. No unpinned remote code is accepted. This is a pinned-source control, not a claim that arbitrary custom code is generally safe.

The metadata audit does not download the two model weight shards (approximately 4.06 GB combined). Real weight import and CPU pilot training remain pending.
