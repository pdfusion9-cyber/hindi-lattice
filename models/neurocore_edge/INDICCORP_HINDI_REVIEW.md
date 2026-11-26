# IndicCorpV2 Hindi candidate review

## Pinned source identity

- Repository: ai4bharat/IndicCorpV2
- Immutable source revision: 2d7285e6ce14fdb3fb2449c9f89427b9f582ac3f
- Dataset config: indiccorp_v2
- Hindi split: hin_Deva
- Text field: text
- Source pattern: data/hi-*.txt
- Hindi shards at the pinned revision: hi-1.txt, hi-2.txt, hi-3.txt
- Each shard is 26,661,096,882 bytes (about 26.7 GB); combined raw size is about 80.0 GB.
- Dataset card and AI4Bharat IndicBERT README declare CC0-1.0 for datasets created in the work.

## Review boundary

The dataset card calls this a pretraining corpus and declares CC0-1.0. The accompanying ACL 2023 paper describes a 20.9B-token corpus across 24 languages, but the reviewed card/paper summary does not provide a per-row upstream-source manifest for the Hindi text. The presence of a CC0 declaration is relevant evidence, not a substitute for a source-level provenance/privacy review.

A bounded audit streams at most 500 Hindi rows, computes aggregate language, length, duplicate and PII-pattern indicators, and never prints or persists sample text. It is review-only and must not be used to bypass human approval.

## Status

PENDING_REVIEW. No train/validation artifact has been created, no corpus text is committed to this public repository, and no training is permitted until:
1. A human reviewer accepts the source-rights and privacy/consent review.
2. The exact tokenized pilot file is produced and its SHA-256 approved for Expert 11.
3. The local Dhīr tokenizer directory SHA-256 is explicitly approved.
4. Training settings and the use of the resulting expert adapter are approved.

The first approved pilot should be bounded (small sample, short sequence length, low step count) and stored outside the public Git repository. A larger corpus run is a separate decision.
