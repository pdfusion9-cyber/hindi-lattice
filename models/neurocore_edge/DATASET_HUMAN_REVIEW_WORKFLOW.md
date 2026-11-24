# Dataset human review workflow

## Scope
This workflow covers 21 distinct external dataset sources and 27 expert-to-dataset assignments in `dataset_approval.json`. The generated packet is `dataset_human_review_packet.json`.

## Required order

1. **Source rights review (one record per source):** establish canonical source, rights holder, exact license and scope, commercial ML-training permission, separate redistribution permission, upstream components, privacy/PII, consent/lawful basis, opt-out/takedown process, immutable revision, config and splits.
2. **Independent rights sign-off:** a qualified rights/legal reviewer records identity, ISO date, evidence URLs and review reference.
3. **Independent domain sign-off:** a different qualified reviewer records schema/modality/sample quality and source suitability. Rights and domain reviewers must be different people.
4. **Expert assignment review (one record per expert/source row):** assigned domain expert checks task relevance, schema adapter, split integrity, language/modality fit, sample quality, privacy handling and limitations.
5. **Decision:** use APPROVED only when every required check and evidence item is complete. Otherwise use HOLD or REJECT and explain why. Missing or ambiguous evidence is never treated as permission.
6. **Audit:** preserve reviewer, date, decision reference, evidence URLs and notes. Any source revision, license, split, schema or upstream-source change invalidates the prior decision and requires re-review.

## Decision semantics

- `PENDING`: not reviewed.
- `HOLD`: evidence incomplete, conflicting, inaccessible or requires legal clarification.
- `REJECT`: rights or suitability fail; do not use.
- `APPROVED`: only after independent source-level sign-offs and assignment-level expert approval are recorded and the approval validator passes.

## Separation of permissions
Permission to use data for model training is not the same as permission to redistribute raw examples, datasets, checkpoints, or derived artifacts. Record both independently.

## Gate policy
No automated approval. No training-data fetch or corpus preparation while required source or assignment approvals remain pending/held/rejected. Production model training is a separate gate and remains disabled even if corpus preparation is later approved. The public repository may expose reviewer metadata; do not upload personal data, confidential legal advice, private contracts or sensitive evidence. Store only approved public evidence references and non-sensitive review references.

## Current baseline
- 21 external source review records.
- 27 expert assignment review records.
- 0 source approvals and 0 assignment approvals.
- Corpus preparation: CLOSED.
- Production training: CLOSED.
