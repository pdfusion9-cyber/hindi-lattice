# Preliminary source-rights review — 2026-11-23

This is a research triage note, not legal advice or an approval. It records public-source evidence found during a first-pass review of the four entries whose Hugging Face metadata audit returned no SPDX/license field. No dataset is approved by this document. All four remain blocked from corpus preparation and commercial training until a qualified reviewer records a source-specific decision in `dataset_approval.json`.

## Decision summary

| Dataset ID | Preliminary disposition | What public evidence supports | What remains unresolved |
|---|---|---|---|
| `HuggingFaceM4/DocumentVQA` | **HOLD — do not train** | The HF card exposes document images, questions, answers and UCSF document identifiers; metadata audit found no license field. | Rights in each underlying document/image, UCSF Industry Documents Library terms, privacy/confidentiality, redaction, redistribution and commercial-training permission. |
| `ai4bharat/IndicCorpV2` | **HOLD — conditional candidate only** | AI4Bharat's corpus/paper materials describe a large multilingual web corpus and state a CC0 release for the corpus. HF metadata audit found no SPDX field and lists multiple language splits. | Whether the CC0 statement covers every included upstream work and every component/source; source-specific restrictions, personal data, crawled-news and Wikipedia provenance, opt-outs/takedowns, and commercial model-training interpretation. Do not assume a wrapper/release statement extinguishes third-party rights. |
| `ccdv/govreport-summarization` | **HOLD mirror; investigate canonical source** | The original GovReport research/release is associated with the LAUNCH project; public project materials identify CC BY 4.0. The HF mirror exposes report/summary fields but no SPDX field in the audit. | Confirm that this exact mirror is byte-for-byte or record-for-record the canonical release, identify its exact revision and split, check notices/attribution requirements and underlying report rights, and document commercial training plus redistribution permissions. Prefer the canonical source over an unverified mirror. |
| `teknium/OpenHermes-2.5` | **EXCLUDE from commercial training pending new evidence** | The HF card describes a mixture of multiple source datasets and model-generated conversations; metadata audit found no SPDX field. | Per-component license/terms and provenance, upstream restrictions, rights to model-generated outputs, and evidence that commercial training and redistribution are permitted for every included component. A single aggregate license cannot be inferred. |

## Primary discovery links

### DocumentVQA
- Dataset card: https://huggingface.co/datasets/HuggingFaceM4/DocumentVQA
- Dataset paper: https://arxiv.org/abs/2007.00398
- UCSF Industry Documents Library: https://www.industrydocuments.ucsf.edu/

### IndicCorpV2
- Dataset card: https://huggingface.co/datasets/ai4bharat/IndicCorpV2
- AI4Bharat organization: https://github.com/AI4Bharat
- IndicCorp/IndicBERT research discovery: https://arxiv.org/search/?query=IndicCorp&searchtype=all

### GovReport
- HF mirror: https://huggingface.co/datasets/ccdv/govreport-summarization
- GovReport paper: https://arxiv.org/abs/2104.08643
- LAUNCH project: https://github.com/launch

### OpenHermes-2.5
- Dataset card: https://huggingface.co/datasets/teknium/OpenHermes-2.5

## Required human-review evidence before changing any row to approved

1. Exact immutable source revision, config and split.
2. Provenance for each component and upstream work; identify the actual rights holder.
3. Exact license text and applicable terms for the pinned revision and all component sources.
4. Explicit written basis for commercial model training; do not infer this from a metadata tag or public availability.
5. Separate decision on redistribution of raw data, examples, checkpoints and derived artifacts.
6. Privacy/PII, sensitive data, minors, confidentiality, consent/notice, opt-outs and takedown review.
7. Qualified reviewer, date, evidence URLs and legal-review reference recorded in the registry.

## Current project gates

- The approval registry remains 27 rows with 0 approved rows.
- This review changes no approval flags and does not download training examples.
- Corpus-preparation gate remains **CLOSED**.
- Production-training gate remains **CLOSED**.

## Limits

This first-pass desk review is not a legal opinion. Public papers, repository cards and license labels are discovery evidence only. In particular, the IndicCorpV2 corpus-level CC0 statement and the canonical GovReport release license must not be treated as automatic proof that every mirrored/upstream record is cleared for commercial training.