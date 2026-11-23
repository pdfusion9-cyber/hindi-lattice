# NeuroCore Edge dataset rights and modality audit — 2026-11-23

## Scope and counting

The current approval registry contains 27 expert-to-dataset assignment rows, 22 distinct dataset IDs, and 21 distinct external source IDs. The 22nd ID, `curated_mixture_from_all_domains`, is a synthetic aggregate name rather than an independently inspectable upstream source. Repeated source IDs across experts are intentional assignments; they are not duplicate rows for the same expert/source pair.

This audit records a conservative intake decision. It does not provide legal advice and does not grant any license. A Hugging Face dataset being public or downloadable is not evidence that all underlying works can be used for commercial model training or redistribution.

## Current gate

- Registry status remains `ALL_DATASETS_PENDING_REVIEW`.
- All 27 assignment rows remain `approved: false`.
- Training remains disabled.
- No dataset is approved by this document.
- Before an approval row can be changed, record an immutable revision, config and splits, exact license/terms URL, evidence for commercial model training, redistribution scope, PII/privacy review, consent review where applicable, schema fingerprint, reviewer and review date.

## Source-by-source disposition

| Dataset ID | Intake disposition | Required evidence / limitation |
|---|---|---|
| `grammarly/coedit` | Pending | Verify the exact Hub revision, card and upstream CoEdIT terms; check source/target schema and any source-text rights. |
| `teknium/OpenHermes-2.5` | Pending | Dataset is an aggregation. Audit component-level provenance and licenses; a single permissive label cannot clear all included records. |
| `ccdv/govreport-summarization` | Pending | Verify GovReport upstream terms, document redistribution scope, and exact Hub revision/config/splits. |
| `FiscalNote/billsum` | Pending | Verify upstream BillSum license and rights in source bill text and summaries; pin revision and splits. |
| `HuggingFaceM4/DocumentVQA` | Sidecar required; pending | Image/document VQA requires image payload, page provenance and a vision adapter. Text-only adapter must refuse it. Verify underlying document/image rights. |
| `ai4bharat/indicvoices_r` | Sidecar required; pending | Audio and speaker data require a separate audio pipeline plus speaker consent/rights and privacy review. |
| `mythicinfinity/libritts` | Sidecar required; pending | Audio/speaker sidecar required; verify LibriTTS/LibriSpeech source terms and speaker rights before any speech training. |
| `HuggingFaceFW/fineweb-edu` | Pending | Review dataset card and Common Crawl/source-content terms; establish permitted commercial training and privacy/PII policy. |
| `awinml/legalbench-rag` | Evaluation-only pending | Identify exact MAUD/LegalBench source and license; do not infer training rights from benchmark availability. |
| `kaushik-harsh-99/Indian-legal-data-v3` | Pending | Verify source court/legal materials, collection provenance, redistribution and personal-data handling. |
| `opennyaiorg/InJudgements_dataset` | Pending | Verify source-court terms, collection license and privacy/redaction policy for judgments. |
| `ajimeno/PubTabNet` | Sidecar required; pending | Table images and cell geometry require a vision/table sidecar; verify PubTabNet source-image rights. |
| `allenai/olmOCR-mix-0225` | Sidecar required; pending | OCR/page-image/layout sidecar required. Mixed PDF sources carry third-party rights; dataset card visibility alone is insufficient. |
| `ai4bharat/sangraha` | Pending | Verify the source-specific license and provenance for each included corpus/language; record exact revision and schema. |
| `ai4bharat/IndicCorpV2` | Pending | Verify official source terms and rights for all underlying corpora; do not transfer a repository-level license to third-party text without evidence. |
| `facebook/flores` | Evaluation-only pending | Benchmark/evaluation use only until benchmark terms and training permission are explicitly established. |
| `openlanguagedata/flores_plus` | Evaluation-only pending | Benchmark/evaluation use only until all component source licenses and training permission are verified. |
| `sarthak-wiz01/legalbench` | Evaluation-only pending | Treat as benchmark data, not training data, until task-by-task provenance and licenses are established. |
| `pszemraj/govreport-summarization-8192` | Pending | Derivative/extended GovReport corpus; verify inherited source rights and any modifications, pin revision and split. |
| `launch/gov_report_qs` | Pending | Verify upstream terms and exact schema/config. The adapter now handles dictionary-of-lists section and question/summary structures; regression tests cover this shape. |
| `hiyouga/glaive-function-calling-v2-sharegpt` | Pending | Audit each source/component license and tool-call data provenance; preserve ordered conversation/tool schemas. |

## Adapter coverage and fail-closed policy

- Text adapters support reviewed shapes for editing pairs, ordered chat/tool conversations, document summaries, hierarchical GovReport-QS, translation, plain text and legal context/QA.
- GovReport-QS accepts both list-of-records and dictionary-of-lists forms. Section title/paragraph counts, question/summary counts and optional parent-index counts are validated.
- Legal QA accepts `input` as a question field, preserving `context`/facts/document and answer/output fields.
- DocumentVQA, IndicVoices-R, LibriTTS, PubTabNet and olmOCR remain sidecar-required; they are not silently flattened into text.
- Unknown schemas and empty adapter output remain hard refusals.

## Next approval sequence

1. Obtain source-owner or authoritative terms evidence for each dataset.
2. Pin immutable revision, config and exact train/validation split.
3. Fingerprint the actual feature schema and verify the adapter against representative rows.
4. Complete privacy/PII and consent review.
5. Separate training-eligible sources from evaluation-only sources.
6. Have a human reviewer complete the approval record. Only then may the corpus builder prepare a local corpus; training remains a separate gate.
