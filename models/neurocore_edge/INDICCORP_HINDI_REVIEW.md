# IndicCorpV2 Hindi candidate review

## Pinned source identity
- Repository: ai4bharat/IndicCorpV2
- Immutable revision: 2d7285e6ce14fdb3fb2449c9f89427b9f582ac3f
- Config: indiccorp_v2
- Review shard: data/hi-1.txt
- Hugging Face text-builder column: text
- Shard size: 26,661,096,882 bytes
- Shard SHA-256: 9c9dfed015247004176c12b831bc7a4720bd79ebd8ac424e46c13997b57f8a73

## Public evidence checked 2026-11-26
- Dataset card text declares CC0-1.0 for datasets created as part of the work.
- Hub metadata has no separate machine-readable license tag in the inspected card.
- ACL 2023 paper describes IndicCorp as 20.9B tokens across 24 languages.
- AI4Bharat/IndicBERT repository identifies IndicCorpV2 as pretraining data.
- Checked public documentation does not supply a per-record source manifest, domain-level rights ledger, or PII/consent certification for the Hindi shard.

Sources:
- https://huggingface.co/datasets/ai4bharat/IndicCorpV2
- https://huggingface.co/datasets/ai4bharat/IndicCorpV2/blob/2d7285e6ce14fdb3fb2449c9f89427b9f582ac3f/README.md
- https://aclanthology.org/2023.acl-long.693/
- https://github.com/AI4Bharat/IndicBERT

## Review-only boundary
CC0 is the publisher's stated release license, but the missing row-level provenance ledger means this audit cannot certify rights for every upstream source. A responsible human must review the intended commercial and redistribution use.

The initial 500-nonempty review found one phone-like record. Expanded audit checks up to 5,000 non-empty records with a separate 100,000 raw-line ceiling. It reports aggregate counts for email, India-mobile, Aadhaar-like and PAN-like patterns, including unique rows with any such pattern. It never prints or persists sample text. These heuristics are not complete PII detection.

## Status
PENDING_REVIEW. This is not training approval. No training corpus is authorized until a human reviewer accepts rights/privacy risks and explicitly approves exact SHA-256 values for the private cleaned corpus and local tokenizer directory.


## Expanded bounded audit result (2026-11-26)
Workflow run: https://github.com/pdfusion9-cyber/hindi-lattice/actions/runs/45815102550

| Measure | Result |
|---|---:|
| Target non-empty records | 5,000 |
| Raw rows seen | 9,999 |
| Non-empty text records | 5,000 |
| Empty separator rows | 4,999 |
| Duplicate text rows | 1 |
| Contains Devanagari | 5,000 |
| Contains Latin script | 424 |
| Phone-like rows | 3 |
| Email-like rows | 1 |
| Aadhaar-like rows | 0 |
| PAN-like rows | 0 |
| Unique rows with any PII-pattern flag | 4 |

No sample text was persisted. The report is a pattern-based screen, not a complete privacy review. All four candidate rows must be excluded or reviewed before any training-corpus construction. The dataset remains PENDING_REVIEW; this audit did not perform training.

Aggregate-only artifact: ID 15116065188, SHA-256 `909b2760e7b303ed6382c0622efb98e76a68ce54a1156ecd68b2b17e6544b783`.
