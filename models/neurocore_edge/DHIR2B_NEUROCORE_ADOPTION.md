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
