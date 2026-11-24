# External dataset rights review — primary-source findings

Review date: 2026-11-23
Scope: four datasets whose Hugging Face metadata audit exposed no SPDX/license field.
Disposition: discovery and risk triage only. This is not legal advice, does not approve any dataset, and does not establish commercial-training clearance.

## Decision summary

| Dataset ID | Public-source finding | Current decision |
|---|---|---|
| HuggingFaceM4/DocumentVQA | Dataset schema contains document images and answers; audit exposed no license field. Public benchmark description points to DocVQA and UCSF Industry Documents Library-derived documents. Item-level rights, privacy, and redistribution permissions are not established by mirror metadata. | **HOLD — do not ingest for training** until document-level provenance, rights, privacy/PII and redistribution permissions are documented. |
| ai4bharat/IndicCorpV2 | IndicCorp v2 paper reports CC0 for the released corpus. The corpus includes web/news and other sources; a corpus-level statement alone does not resolve upstream terms, takedowns/opt-outs, or every source component. HF mirror has no SPDX field and a mutable branch must not be used as a training pin. | **CONDITIONAL HOLD — human legal review only.** Obtain exact CC0 grant and scope, source inventory, upstream exclusions/permissions, current revision, and privacy/takedown handling. |
| ccdv/govreport-summarization | Canonical GovReport project/release materials are associated with LAUNCH and describe CC BY 4.0. The ccdv mirror's audit metadata has no license field. A canonical-release license cannot automatically be imputed to a third-party mirror or every underlying report. | **CONDITIONAL CANDIDATE — prefer canonical release.** Verify archive, attribution/notice requirements, source-document rights, and mirror fidelity before approval. |
| teknium/OpenHermes-2.5 | Public card describes a mixture of source datasets and generated assistant conversations. No SPDX/license field was exposed. Upstream terms, generated-output provenance, and possible personal/sensitive data are not cleared by the card. | **EXCLUDE from commercial training unless independently cleared source-by-source.** Missing license metadata is not permission. |

## Evidence reviewed

1. Hugging Face metadata and dataset-server split/schema audit from CI run 45192050686 (artifact neurocore-dataset-verification):
   - HuggingFaceM4/DocumentVQA: observed revision `a44195f8f989c10d06ab4ccc98ffe9cdb6d928e4`; fields include question, image, docId, ucsf_document_id and answers; no declared license field.
   - ai4bharat/IndicCorpV2: observed revision `2d7285e6ce14fdb3fb2449c9f89427b9f582ac3f`; text feature; no declared license field.
   - ccdv/govreport-summarization: observed revision `4e21184e01ae8017e2c036e180fe5e541fef60a0`; report and summary fields; no declared license field.
   - teknium/OpenHermes-2.5: observed revision `b82037821055c377bed0d495e72e46de3bc72e84`; conversations/source/model fields; no declared license field.
   These SHAs are audit observations only, not approval pins.
2. DocVQA benchmark paper: https://arxiv.org/abs/2007.00398
3. UCSF Industry Documents Library: https://www.industrydocuments.ucsf.edu/
4. IndicCorp v2 / IndicBERT v2 paper: https://arxiv.org/abs/2212.05409
5. LAUNCH GovReport project/release: https://launch.cs.illinois.edu/govreport/
6. OpenHermes-2.5 dataset card: https://huggingface.co/datasets/teknium/OpenHermes-2.5

## Mandatory human review before any status change

- Immutable revision SHA and exact config/split names.
- Canonical upstream source/version and source-by-source provenance inventory.
- Full license text and evidence that licensor controls relevant rights.
- Explicit commercial ML-training permission, separate from redistribution permission.
- Attribution/share-alike/notice obligations and compliance plan.
- PII/sensitive data/minor/confidentiality screening, lawful basis/consent where required, opt-out/takedown process.
- Schema/modality and sample inspection, redaction policy, reviewer decision.
- Legal review reference, named reviewer, date, durable evidence URLs.

## Gate status

- 27 expert-to-dataset approval rows remain pending; 0 approved.
- This review does not alter dataset_approval.json.
- No external dataset has been downloaded for training.
- Corpus-preparation and production-training gates remain CLOSED.
- Metadata, paper statements, and public cards are leads for review, not legal opinions or proof of commercial rights.
