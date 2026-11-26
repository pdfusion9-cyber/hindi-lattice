# Dhīr–NeuroCore training data formats

The fixed-expert trainer accepts three JSONL row shapes:

- Instruction SFT (preferred for prompt/answer tasks): {"prompt":"...","response":"..."}
- Legacy text language modelling: {"text":"..."}
- Pre-tokenized legacy language modelling: {"input_ids":[...]}

For instruction SFT, the trainer serializes the user prompt and assistant response with the pinned Dhīr tokenizer's native chat_template. It checks that the prompt plus generation prefix is an exact prefix of the full conversation, then masks prompt labels with -100. The loss therefore applies only to assistant completion tokens. It will fail closed if the native template is missing or does not preserve that prefix.

Legacy text and input_ids rows retain full-token causal-LM loss for compatibility with existing corpus preparation. Avoid mixing objectives in one corpus unless that mix is intentional; the checkpoint records the objective(s) used.

All inputs continue through the existing exact dataset revision, PII/privacy, consent, license, corpus SHA-256 and tokenizer SHA-256 approval checks. This document does not approve any dataset or authorize a training run. Keep raw/private corpora outside the public Git checkout.
