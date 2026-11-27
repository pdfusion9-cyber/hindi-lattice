# Preliminary quality audit — synthetic Hindi SFT pilot

**Status: MODEL-ASSISTED REVIEW NOTES ONLY — NOT HUMAN APPROVAL AND NOT TRAINING AUTHORIZATION.**

Audit basis: all four synthetic JSONL files were re-reviewed row-by-row after the latest content revision.

## Inventory
- Expert 16 / self-correction: 14 train, 4 held-out validation.
- Expert 17 / planning: 15 train, 4 held-out validation.
- Total: 29 training and 8 held-out validation examples.
- Exact train/validation prompt strings are disjoint.
- Every row uses the two-field \`prompt\` / \`response\` schema.

## Expert 16 — self-correction
- All 14 training prompts now explicitly refer to a previous assistant answer, rather than merely asking to fix an unrelated statement.
- Training topics include arithmetic order, rectangle area, discount, weather versus climate, seasons, photosynthesis, magnetism, common-denominator addition, speed/time, mass conversion, successive percentage change, mean/median and triangle area.
- Validation includes two correction cases, one case where the earlier answer is correct and should be retained, and one case where the earlier answer is unsupported because the map scale is missing.
- This improves task identity and calibration, but four validation examples remain too few for a reliable generalization estimate. The held-out skills still overlap broad training concepts.

## Expert 17 — planning
- All 15 training and all 4 validation responses now include goal, duration, accountable role, resources/dependencies, indicator, risk and fallback fields.
- Advanced scenarios defer emergency, technical, clinical, accessibility and water-quality decisions to qualified/authorized roles.
- Planting timeline corrected: site preparation days 1–10, planting days 11–15, and survival checks 30/60 days after planting, within the 90-day project period.
- School rainwater planning records flow and leak checks against the approved design without inventing technical thresholds.
- Pond-cleaning outcome is a logged quantity of waste removed and lawfully disposed; ecological work remains subject to expert review and approval.
- Four validation examples are still a small smoke-test set, not a sound statistical evaluation.

## Automated checks and limits
CI checks JSONL schema and exact inventories, Devanagari presence, email/phone/Aadhaar/PAN-like patterns, train/validation prompt disjointness, all training prompts referencing prior answers, validation retention/uncertainty calibration examples, all planning rubric fields and the planting timeline. These checks do not prove factual correctness, comprehensive PII absence, rights, consent, bias, local feasibility or production readiness.

The tokenizer dry-run previously passed using the pinned Dhīr-2B tokenizer and production trainer parser. That dry-run did not load model weights, execute a forward pass, compute model loss or perform training. The revised text requires a fresh tokenizer dry-run because source-file hashes changed.

## Fresh Dhīr tokenizer dry-run — 2026-11-27

The revised four-file dataset was re-run through the pinned Dhīr tokenizer and the production trainer parser in GitHub Actions run [#290](https://github.com/pdfusion9-cyber/hindi-lattice/actions/runs/45934026261).

- Result: `SYNTHETIC_SFT_TOKENIZER_DRY_RUN_PASSED_NO_TRAINING`.
- Training / held-out validation rows: 29 / 8.
- Dhīr vocabulary size: 131,072.
- Tokenizer vocabulary SHA-256: `cff9e969e05108763f796df6dafd614ee8159ebbf3cc165594265fa0ea02ee9c`.
- Trainer-compatible tokenizer directory SHA-256: `84dba4df7e54dddadd907939abbee248b181753ae97523a5407b48db7eabf990`.
- `training_performed=false`; `optimizer_steps=0`.

This confirms parsing, native chat-template prefix handling, completion-only masking and tokenizer compatibility for the current pilot rows. It does not load the 4.15 GB Dhīr checkpoint, run a model forward/backward pass, authorize training, or validate answer quality beyond the existing content checks. The exact tokenizer hash is evidence for this dry-run environment; it must still be entered into the approval registry only after the required independent human and rights review.

## Rights, privacy and approval boundary
- Synthetic examples are newly authored, but the repository has no root LICENSE file and synthetic registry rows have no license ID or terms URL.
- Commercial-training and public-redistribution permissions remain false. A yes in chat does not itself supply the missing rights-holder/license record, reviewer identity/date, or exact-hash approvals.
- No dataset, privacy, consent, corpus or tokenizer approval is granted by this report.
- Registry status must remain \`ALL_DATASETS_PENDING_REVIEW\`; the trainer's default-deny gate remains in force.
- No model training or optimizer step has been performed.
