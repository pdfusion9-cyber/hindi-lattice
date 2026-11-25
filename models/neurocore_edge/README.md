# NeuroCore-Edge v0.3.2 — canonical reference source

**State: untrained architecture/reference implementation. No trained checkpoint exists. Training, dataset approval, continuous training and quantization remain disabled.**

## Canonical architecture contract

- 3,074,921,472 unique parameters, verified by PyTorch meta-device counting in GitHub Actions run #124 (shared: 225,088,512; experts: 2,849,832,960), with tied embedding/output weights counted once.
- 20 distinct sparse experts per unique block; normal inference/training routes Top-2.
- Sandwich execution: 2 prelude + 4 unique core blocks repeated 5 times + 2 coda = 24 effective operations over 8 unique parameterized blocks.
- RoPE grouped-query causal attention (16 query heads / 4 KV heads), sliding-window mask, recurrent reference memory cell, gated ReLU-squared expert MLPs.

## Training pipeline contract

- Shared stage trains shared backbone while expert tensors are frozen.
- Expert stage starts from the exact completed shared checkpoint and trains only one selected expert ID across all unique blocks.
- Checkpoints must preserve model, optimizer, scheduler, RNG and data cursor state and reject mismatched config/tokenizer/dataset/base fingerprints.
- Strict merge requires exactly one completed shard for every expert ID 0–19 and exact tensor keys/shapes/dtypes/finiteness.
- Held-out evaluation reports Hindi and English next-token Top-1 accuracy, NLL and perplexity separately.

## Fail-closed gates

- config.json keeps training disabled; explicit CLI opt-in alone does not approve data.
- Dataset approval manifest defaults every dataset to pending and has empty exact-SHA approval lists for tokenized corpora and tokenizer.
- Tokenizer construction requires exact source-dataset revisions/splits and explicit commercial-training, redistribution, PII, consent, schema and split approvals; generated tokenizer SHA-256 must then be approved before model training.
- Training must reject any tokenized corpus or tokenizer whose exact SHA-256 is not uniquely approved for its stage/expert.
- CI never launches training. CPU preflight is not evidence of 3B training capacity.
- Quantization and mobile export remain disabled until packing kernels, calibration, parity, memory, latency, energy and thermal tests pass.

## Evidence boundary

A successful syntax check, tiny CPU test or meta-device parameter count is architecture/test evidence only. It is not evidence of convergence, language accuracy, trained checkpoints or mobile readiness.
