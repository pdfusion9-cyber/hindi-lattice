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
## Machine-assisted content review — 2026-11-27 (not independent sign-off)

All 37 JSONL records were inspected at the pinned data revision. The review is limited to the visible examples and repository checks; it is not a factual/legal/privacy certification.

### Expert 16 calibration gap
- All 14 current training prompts contain an earlier answer that needs correction. The training split therefore shows correction, but does not directly teach the complementary behavior of retaining a correct prior answer.
- One of the four held-out validation examples asks the model to retain a correct average. That is useful as an evaluation case, but it is not training exposure.
- **Recommended before any Expert 16 training:** add several new training examples with a correct earlier answer, an insufficient-information case, and a partly-correct answer. Keep all four existing validation prompts held out. This is a dataset-quality recommendation, not an approval.
- The remaining correction examples cover arithmetic, geometry, discount, weather/climate, seasons, photosynthesis, magnetism, fractions, speed/time, unit conversion, percentages, mean/median and triangle area. Their answers include correction steps and an explanation of the prior mistake.

### Expert 17 planning quality
- All 19 plans (15 train, 4 validation) contain the seven required fields and end with a fallback action.
- The advanced plans consistently delegate flood response, electrical/solar installation, road traffic control, health screening, accessibility changes and water testing to named/qualified or authorized roles.
- Safety boundaries are explicit in the relevant examples: do not enter floodwater, do not use children/untrained volunteers to stop traffic, do not let organizers diagnose/treat, and do not label water safe/unsafe without an authorized report.
- Some indicators are counts, proportions or recorded checks without an invented numerical success threshold. A human reviewer should confirm whether each indicator can actually be measured with the resources and baseline available in the intended setting; do not insert arbitrary targets just to make the rubric look numeric.
- These are planning exercises, not executable local project instructions. Actual permits, budgets, technical designs, staffing and local conditions need independent confirmation.

### Human reviewer checklist (all fields remain pending)
For each expert, a reviewer must inspect every training row and every held-out row; confirm factual correctness and Hindi quality; review bias, safety, duplication and privacy; confirm authorship/rights; decide the allowed training and redistribution scope; and sign/date the exact file hashes below. A separate reviewer should verify the prepared model-input file and the tokenizer directory used by the actual checkpoint. Record a distinct approval ID for each decision. Do not move validation rows into training.

- All 15 training and all 4 validation responses now include goal, duration, accountable role, resources/dependencies, indicator, risk and fallback fields.
- Advanced scenarios defer emergency, technical, clinical, accessibility and water-quality decisions to qualified/authorized roles.
- Planting timeline corrected: site preparation days 1–10, planting days 11–15, and survival checks 30/60 days after planting, within the 90-day project period.
- School rainwater planning records flow and leak checks against the approved design without inventing technical thresholds.
- Pond-cleaning outcome is a logged quantity of waste removed and lawfully disposed; ecological work remains subject to expert review and approval.
- Four validation examples are still a small smoke-test set, not a sound statistical evaluation.

## Automated checks and limits
CI checks JSONL schema and exact inventories, Devanagari presence, email/phone/Aadhaar/PAN-like patterns, train/validation prompt disjointness, all training prompts referencing prior answers, validation retention/uncertainty calibration examples, all planning rubric fields and the planting timeline. These checks do not prove factual correctness, comprehensive PII absence, rights, consent, bias, local feasibility or production readiness.

The fresh dry-run did not load model weights, execute a model forward pass, compute model loss or perform training. It verified the native Dhīr chat-template prefix, completion-only labels, and that every source row was retained without truncation.

## Fresh Dhīr tokenizer dry-run — 2026-11-27

The revised four-file dataset was re-run through the pinned Dhīr tokenizer and the production trainer parser in GitHub Actions run [#290](https://github.com/pdfusion9-cyber/hindi-lattice/actions/runs/45934026261).

- Result: `SYNTHETIC_SFT_TOKENIZER_DRY_RUN_PASSED_NO_TRAINING`.
- Training / held-out validation rows: 29 / 8.
- Dhīr vocabulary size: 131,072.
- Tokenizer vocabulary SHA-256: `cff9e969e05108763f796df6dafd614ee8159ebbf3cc165594265fa0ea02ee9c`.
- Trainer-compatible tokenizer directory SHA-256: `84dba4df7e54dddadd907939abbee248b181753ae97523a5407b48db7eabf990`.
- Tokenizer saved-file SHA-256: `cbbca2da083dfc8a5975263bcedf6a580b9db01ba26b5dd622ac7b8f5a5b8056`.
- Native chat-template SHA-256: `4bc9a0aac3faabc9a7399b1df253929f73d0448329f55d6700a73a0fff7c2305`.
- Combined hash of the two Expert 16/17 training source files: `9dfc9d7dc5edb903a7f7002625221d14e7dafadf213f146eda1c5c09b6cce704`.
- Pinned source-data commit for all four JSONL files: `0e3c5c9a19490e6736888c8742b9397a6146ccbb`.

| Dataset split | Rows retained | Source-file SHA-256 | Max / mean sequence length | Assistant target tokens |
|---|---:|---|---:|---:|
| Expert 16 train | 14 / 14 | `4990d5b46093ccd2d04afda1dfe76b5eee023f904aeb640945ca114ae4d0ae5c` | 106 / 90.500 | 801 |
| Expert 16 validation | 4 / 4 | `86e4ded58b5f97381843f15fe68cd74765de3f0b5c6a8206d73c5a966867244b` | 102 / 83.250 | 206 |
| Expert 17 train | 15 / 15 | `0ad7f459e02ff5fa43c26b66db973dde3a5fdab31353876e18cb07f508c7954d` | 239 / 190.867 | 2,422 |
| Expert 17 validation | 4 / 4 | `7595b4b6946bb34a818d7b31d67b01adb9db0c730b7d976fc820bc40b318c2c4` | 214 / 179.000 | 609 |

- `training_performed=false`; `optimizer_steps=0`.

This confirms parsing, native chat-template prefix handling, completion-only masking and tokenizer compatibility for the current pilot rows. It does not load the 4.15 GB Dhīr checkpoint, run a model forward/backward pass, authorize training, or validate answer quality beyond the existing content checks. The exact tokenizer hash is evidence for this dry-run environment; it must still be entered into the approval registry only after the required independent human and rights review.

## Rights, privacy and approval boundary
- Synthetic examples are newly authored, but the repository has no root LICENSE file and synthetic registry rows have no license ID or terms URL.
- Commercial-training and public-redistribution permissions remain false. A yes in chat does not itself supply the missing rights-holder/license record, reviewer identity/date, or exact-hash approvals.
- No dataset, privacy, consent, corpus or tokenizer approval is granted by this report.
- Registry status must remain \`ALL_DATASETS_PENDING_REVIEW\`; the trainer's default-deny gate remains in force.
- No model training or optimizer step has been performed.
