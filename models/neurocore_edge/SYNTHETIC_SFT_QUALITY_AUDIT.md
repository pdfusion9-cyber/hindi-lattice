# Preliminary quality audit — synthetic Hindi SFT pilot

**Status: MODEL-ASSISTED REVIEW NOTES ONLY — NOT HUMAN APPROVAL AND NOT TRAINING AUTHORIZATION.**

Audit basis: all four synthetic JSONL files were re-reviewed row-by-row after the latest content revision.

## Inventory
- Expert 16 / self-correction: 20 train, 4 held-out validation.
- Expert 17 / planning: 15 train, 4 held-out validation.
- Total: 35 training and 8 held-out validation examples.
- Exact train/validation prompt strings are disjoint.
- Every row uses the two-field \`prompt\` / \`response\` schema.

## Expert 16 — self-correction
- All 20 training prompts now explicitly refer to a previous assistant answer, rather than merely asking to fix an unrelated statement.
- Training topics include arithmetic order, rectangle area, discount, weather versus climate, seasons, photosynthesis, magnetism, common-denominator addition, speed/time, mass conversion, successive percentage change, mean/median and triangle area.
- Validation includes two correction cases, one case where the earlier answer is correct and should be retained, and one case where the earlier answer is unsupported because the map scale is missing.
- This improves task identity and calibration, but four validation examples remain too few for a reliable generalization estimate. The held-out skills still overlap broad training concepts.

## Machine-assisted content review — 2026-11-27 (not independent sign-off)

All 43 JSONL records were inspected at the source revision before the six calibration rows were appended; the six new rows received schema/PII-pattern checks in the updated CI, but still require human review. The review is limited to the visible examples and repository checks; it is not a factual/legal/privacy certification.

### Expert 16 calibration gap
- The expanded 20-row training split now includes examples that retain a correct prior answer, reject unsupported claims when information is missing, and preserve a correct part while correcting only the faulty part. The earlier 14 correction examples remain intact.
- One of the four held-out validation examples asks the model to retain a correct average. That is useful as an evaluation case, but it is not training exposure.
- The calibration gap identified in the earlier review has been addressed at the example-design level by six new training rows (two correct-answer retention, two insufficient-information, and two partial-correction examples). The four existing validation prompts remain held out. This is a design update, not a human quality or training approval.
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


## Expanded Expert 16 calibration — 2026-11-27

Six repository-authored training examples were added: two examples retaining a correct prior answer, two declining to assert unsupported information, and two preserving a correct part while correcting only the faulty part. The Expert 16 training split is now 20 rows; the existing four validation rows remain unchanged and held out. The Expert 17 split remains 15 train / 4 validation. Dataset registry and dry-run provenance pins were updated to source commit `e1b7824cccdc32004f4cfcd5acf91a2c76acfec9`; registry approvals remain false. A fresh Dhīr tokenizer dry-run is required because prior run #290 hashes apply to the old data.

## Automated checks and limits
CI checks JSONL schema and exact inventories (20/4 and 15/4), Devanagari presence, email/phone/Aadhaar/PAN-like patterns, train/validation prompt disjointness, all 20 Expert 16 training prompts referencing prior answers, calibration coverage, validation retention/uncertainty calibration examples, all planning rubric fields and the planting timeline. These checks do not prove factual correctness, comprehensive PII absence, rights, consent, bias, local feasibility or production readiness.

The fresh dry-run did not load model weights, execute a model forward pass, compute model loss or perform training. It verified the native Dhīr chat-template prefix, completion-only labels, and that every source row was retained without truncation.

## Fresh Dhīr tokenizer dry-run — 2026-11-27

Run #290 was a historical tokenizer dry-run for the earlier 14/4 + 15/4 dataset revision. After adding six Expert 16 training examples, the dry-run must be repeated against the new pinned revision before its hashes and token counts can be used for approval.

- Result: `SYNTHETIC_SFT_TOKENIZER_DRY_RUN_PASSED_NO_TRAINING`.
- Historical training / held-out validation rows: 29 / 8 (before Expert 16 expansion).
- Dhīr vocabulary size: 131,072.
- Tokenizer vocabulary SHA-256: `cff9e969e05108763f796df6dafd614ee8159ebbf3cc165594265fa0ea02ee9c`.
- Trainer-compatible tokenizer directory SHA-256: `84dba4df7e54dddadd907939abbee248b181753ae97523a5407b48db7eabf990`.
- Tokenizer saved-file SHA-256: `cbbca2da083dfc8a5975263bcedf6a580b9db01ba26b5dd622ac7b8f5a5b8056`.
- Native chat-template SHA-256: `4bc9a0aac3faabc9a7399b1df253929f73d0448329f55d6700a73a0fff7c2305`.
- The historical combined source hash above applies only to the pre-expansion corpus and is not valid for the new 35-row training corpus.
- New source-data commit for the expanded Expert 16 train file: `e1b7824cccdc32004f4cfcd5acf91a2c76acfec9`; both synthetic dataset registry entries now point to this revision for provenance.

| Dataset split | Rows retained | Source-file SHA-256 | Max / mean sequence length | Assistant target tokens |
|---|---:|---|---:|---:|
| Expert 16 train (run #300) | 20 / 20 | `ce148fd31b2850b18f0b398dd55cda12f30a33844b315c98dc8a16847ed23e66` | 106 / 89.450 | 1,093 |
| Expert 16 validation | 4 / 4 | `86e4ded58b5f97381843f15fe68cd74765de3f0b5c6a8206d73c5a966867244b` | 102 / 83.250 | 206 |
| Expert 17 train | 15 / 15 | `0ad7f459e02ff5fa43c26b66db973dde3a5fdab31353876e18cb07f508c7954d` | 239 / 190.867 | 2,422 |
| Expert 17 validation | 4 / 4 | `7595b4b6946bb34a818d7b31d67b01adb9db0c730b7d976fc820bc40b318c2c4` | 214 / 179.000 | 609 |

- `training_performed=false`; `optimizer_steps=0`.

This confirms parsing, native chat-template prefix handling, completion-only masking and tokenizer compatibility for the current pilot rows. It does not load the 4.15 GB Dhīr checkpoint, run a model forward/backward pass, authorize training, or validate answer quality beyond the existing content checks. The exact tokenizer hash is evidence for this dry-run environment; it must still be entered into the approval registry only after the required independent human and rights review.

## Current Dhīr tokenizer dry-run — Run #300 — 2026-11-27

The expanded corpus was tested with the immutable native Dhīr-2B tokenizer and the production SFT parser.

- Result: `SYNTHETIC_SFT_TOKENIZER_DRY_RUN_PASSED_NO_TRAINING`.
- Training / held-out validation rows: 35 / 8; all 43 rows retained.
- Expert 16 train: 20/20 rows, 1,093 supervised assistant tokens; max / mean sequence length 106 / 89.450.
- Expert 17 train: 15/15 rows, 2,422 supervised assistant tokens; max / mean sequence length 239 / 190.867.
- Expert 16 validation: 4/4 rows, 206 assistant tokens; Expert 17 validation: 4/4 rows, 609 assistant tokens. Neither validation split was used for training.
- Current Expert 16 training-file SHA-256: `ce148fd31b2850b18f0b398dd55cda12f30a33844b315c98dc8a16847ed23e66`.
- Expert 17 training-file SHA-256: `0ad7f459e02ff5fa43c26b66db973dde3a5fdab31353876e18cb07f508c7954d`.
- Combined training-file SHA-256: `4a1c15d5a38c2901a74e44291bd9f879665a73cf02071e51c483169730ecc21c`.
- Dhīr vocabulary SHA-256: `cff9e969e05108763f796df6dafd614ee8159ebbf3cc165594265fa0ea02ee9c`.
- Tokenizer native chat-template SHA-256: `4bc9a0aac3faabc9a7399b1df253929f73d0448329f55d6700a73a0fff7c2305`.
- Tokenizer saved-file SHA-256: `cbbca2da083dfc8a5975263bcedf6a580b9db01ba26b5dd622ac7b8f5a5b8056`.
- Trainer-compatible tokenizer directory SHA-256: `84dba4df7e54dddadd907939abbee248b181753ae97523a5407b48db7eabf990`.
- Registry/source revision: `e1b7824cccdc32004f4cfcd5acf91a2c76acfec9`.

The run report explicitly records `training_performed=false`, `optimizer_steps=0`, `model_weights_loaded=false`, and `model_forward_pass_performed=false`. This is a tokenizer/parser and loss-mask check, not a real-model training or quality result. Registry approvals remain pending and no approval gate was bypassed.

## Dhīr-2B Expert 16 no-update gradient preflight — 2026-11-27

### Preparation and CI status
- Preflight entrypoint: `models/neurocore_edge/preflight_dhir2b_expert.py`.
- Gate tests: `models/neurocore_edge/test_preflight_dhir2b_expert.py`.
- Manual workflow: `.github/workflows/dhir2b_expert16_no_update_preflight.yml`.
- The integrated canonical CI run [#306](https://github.com/pdfusion9-cyber/hindi-lattice/actions/runs/45955972685) passed, including preflight syntax and unit tests.
- The preflight is intentionally not part of ordinary push CI because it needs to download and load the multi-gigabyte checkpoint.

### Exact checkpoint to select for manual preflight
- Source workflow run ID: `45780741273`.
- Artifact ID: `15098526765`.
- Artifact name: `dhir2b-neurocore-cpu-checkpoint`.
- Artifact ZIP SHA-256: `720ffdf4f59c95b1ee1b3265a5e3f857173c2058d20c787f13ffff1dce819b33`.
- Checkpoint tree SHA-256: `90c3b46d38f52f03a41ed6d5b07a530dd864f107176d3c98e64c808452710148`.
- Checkpoint size: `4,150,375,510` bytes.
- Artifact was checked on 2026-11-27 and was not expired; listed expiry is 2026-12-26 08:10:31 UTC.

### How to trigger the no-update preflight
Open the repository Actions page, select **Dhīr-2B Expert 16 No-Update Gradient Preflight**, choose **Run workflow**, and provide checkpoint_run_id `45780741273`.

Expected outcome is one real CPU forward/backward pass on the shortest retained Expert 16 training row, with:
- finite loss and gradients;
- gradients reaching only Expert 16 adapter parameters;
- frozen Dhīr backbone, router weights and all other experts;
- no optimizer creation, no optimizer step and no parameter value change;
- no validation rows loaded into training;
- registry remains `ALL_DATASETS_PENDING_REVIEW`, with zero approved training artifacts/tokenizers.

### Successful real-checkpoint execution — 2026-11-27
- Preflight workflow run [#6](https://github.com/pdfusion9-cyber/hindi-lattice/actions/runs/45958672542) completed successfully.
- Exact checkpoint artifact: source run `45780741273`, artifact `15098526765`; the preflight loaded all three checkpoint shards.
- Machine report artifact: `dhir2b-expert16-no-update-preflight`, artifact ID `15179744438`, ZIP SHA-256 `8b9d6412453ac53b32917a85afba5f813f7f0a2a06e72594d6dc1e0a6472e32e`.
- Model identity: `zenteiq-ai/Dhir-2B`, pinned revision `7cbe444cd1e45e25197cfa0f683b2cb575858169`; architecture `0.4.0-dhir2b-adapters`; 28 layers, hidden size 2048, vocab 131072, 20 experts, adapter rank 8.
- Actual forward logits shape: `[1, 70, 131072]`; 42 supervised next-token targets; finite loss `4.448416709899902`; backward completed.
- Expert 16 trainable tensors: 56; finite gradient tensors: 56; non-zero gradient tensors: 28.
- Backbone, all routers and all other experts were frozen; frozen parameters with gradients: 0.
- Selected expert weights unchanged; all model weights unchanged; optimizer was not created; optimizer steps: 0.
- Dataset: 20 training rows; validation rows loaded: 0; validation not used for training. Registry remains `ALL_DATASETS_PENDING_REVIEW`; dataset not approved.
- Tokenizer raw files SHA-256: `cbbca2da083dfc8a5975263bcedf6a580b9db01ba26b5dd622ac7b8f5a5b8056`; vocabulary and chat-template hashes matched; the missing metadata manifest was rebuilt only after raw-file hash verification.
- `approval_gate_bypassed=false`; `training_performed=false`.

**Interpretation:** The real Dhīr-2B Expert 16 forward/backward gradient path is verified. This is an execution/gradient-path result, not evidence of learning quality, validation improvement, or training completion. No optimizer update or model training has occurred. The human approval gate remains closed.

## Rights, privacy and approval boundary
- Synthetic examples are newly authored, but the repository has no root LICENSE file and synthetic registry rows have no license ID or terms URL.
- Commercial-training and public-redistribution permissions remain false. A yes in chat does not itself supply the missing rights-holder/license record, reviewer identity/date, or exact-hash approvals.
- No dataset, privacy, consent, corpus or tokenizer approval is granted by this report.
- Registry status must remain \`ALL_DATASETS_PENDING_REVIEW\`; the trainer's default-deny gate remains in force.
- No model training or optimizer step has been performed.
