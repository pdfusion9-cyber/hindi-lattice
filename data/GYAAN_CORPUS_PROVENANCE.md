# Gyaan Base Corpus: Provenance and Rights Review

## Current decision

**Status: NOT CLEARED FOR TRAINING OR REDISTRIBUTION**

The repository contains `data/gyaan_base.txt`, but the repository evidence inspected so far does not establish who authored or compiled the facts, which upstream sources were used, which license terms apply, or whether privacy/consent review was performed. Do not treat the file's public availability in this repository as permission to train on it or redistribute derived weights.

The bounded internal training path must remain blocked until the data owner or responsible reviewer supplies the evidence listed below. This review does not change the production training gate.

## Evidence recorded from the repository

| Item | Observed evidence |
|---|---|
| Repository path | `data/gyaan_base.txt` |
| Git blob SHA-1 | `3018157a78e6a35877e114b9297f8ae9e16a76ae` |
| File size at inspected branch | 486,927 bytes |
| First recorded addition | Commit `b76aea33ee4281f85ca8ae5b4686fd0e2e80c069` |
| Commit date | 2026-11-12 01:41:53 UTC |
| Commit message | `feat: base facts (7162 triples) so every run starts with knowledge` |
| Change in that commit | File added with 7,162 lines; no earlier file version in that commit |
| Commit signature | GitHub reports the commit as unsigned / not verified |
| Source citation fields | The observed line format is `subject | relation | object`; it has no per-row source URL or citation column |
| Repository-level license at that commit | The inspected root tree did not contain `LICENSE`, `LICENSE.md`, `COPYING`, or `NOTICE` |
| Data-specific provenance document | No `data/README.md` or `data/SOURCES.md` was present in the inspected branch |
| Training approval | `dataset_approval.json` is `ALL_DATASETS_PENDING_REVIEW`; 27 listed assignments and 0 approved |
| Model production setting | `config.json` has `training.enabled: false` |

The root README mentions a Hindi Wikipedia dump for the repository's separate character-model training workflow. That statement does **not** identify the sources of `gyaan_base.txt`; this review therefore makes no Wikipedia-origin claim for this corpus.

## Required evidence before any corpus training

The data owner/reviewer must attach or fill in all of the following:

1. **Ownership and compiler** — identify the person/organization that compiled the corpus and confirm authority to grant the proposed training rights.
2. **Upstream sources** — provide stable URLs or source identifiers, retrieval/creation dates, versions, and a mapping from corpus rows or row ranges to sources.
3. **License and permissions** — record each applicable license, its URL/version, attribution obligations, share-alike or other conditions, and explicit permission for model training and the intended commercial/non-commercial use.
4. **Privacy and consent** — document personal-data screening, sensitive-data handling, consent or other valid basis where required, and any exclusions/remediation.
5. **Quality and temporal review** — identify factual reviewers, disputed/ambiguous facts, time-sensitive fields, and corrections/removals.
6. **Approval record** — record reviewer identity/role, review date, scope, decision, and evidence location; only then update the relevant dataset approval entry through the repository's governance process.

A missing item means **not approved**. Do not infer permission from the fact that a file is publicly downloadable or that a statement appears factual.

## Training boundary while review is pending

- Keep `models/neurocore_edge/config.json: training.enabled` false.
- Do not run this corpus through GitHub-hosted/public runners.
- Do not create or publish checkpoints, reports, tokenizer files, or derived weights from the unreviewed corpus.
- Synthetic-fixture CI tests may continue; they are not corpus training.
- Do not represent the 64-hidden, 3-layer bounded test configuration as the full 2.87B-parameter model.

## Review record

- Current decision: `UNVERIFIED_PENDING_OWNER_AND_LICENSE_REVIEW`
- Privacy decision: `NOT_FORMALLY_CLEARED`
- Production approval: `false`
- Reviewer / date / evidence: `PENDING`

## Additional lineage check: separate Wikidata acquisition pipelines

A later history review found Wikidata acquisition code in this repository. Its chronology and outputs matter:

| Item | Repository evidence | What it establishes |
|---|---|---|
| `data/gyaan_base.txt` | First added 2026-11-12 01:41:53 UTC by commit `b76aea33ee4281f85ca8ae5b4686fd0e2e80c069` | The base file existed at this time; the commit only adds the 7,162-line file and contains no source mapping. |
| `ghana_banao.py` | Added 2026-11-12 04:38:20 UTC by commit `6b2ac8d85f8b65b94381652772e0fb1f8c35c94b` | A subsequent harvester queries `https://query.wikidata.org/sparql`, selects Wikidata item classes and properties, and writes Hindi subject-relation-value triples. |
| `.github/workflows/facts.yml` | Added 2026-11-12 04:38:27 UTC | A scheduled/manual workflow runs `ghana_banao.py`, writes `data/gyaan_ghana.txt`, then merges it with `data/gyaan_base.txt` into `data/gyaan_sab.txt` on a separate orphan `facts` branch. |
| `wd_jama.py` | Added 2026-11-13 13:57:34 UTC by commit `1b74b4599fecb4f72eb220949bc8dfcb8590d313` | A later script queries Wikidata labels and selected properties, writing `gyaan_wd.txt` and `gyaan_wd.txt.kism.txt`. |

**Important boundary:** these tools prove that the repository later acquired independent Wikidata-based data-generation pipelines. They do not prove that `data/gyaan_base.txt`, added roughly three hours before the `ghana_banao.py` commit, was generated by Wikidata or by either script. The observed `facts.yml` pipeline explicitly treats `gyaan_base.txt` as an existing input and produces a separate merged output. The current `gyaan_wd.txt` in the inspected branch contains only its header.

The `ghana_banao.py` and `wd_jama.py` scripts also do not, in their current plain-triple outputs, preserve complete row-level source identifiers, property IDs, exact query hashes, or retrieval timestamps. Their existence is useful provenance for the process, but it is not a complete row-level audit trail.

## Wikidata license note for a future, newly generated corpus

Wikidata's official licensing and data-access pages state that its structured data in the main, Property, Lexeme and EntitySchema namespaces is released under CC0. Wikidata's text in other namespaces is not covered by that same statement; it is licensed separately under CC BY-SA 4.0. See:

- https://www.wikidata.org/wiki/Wikidata:Licensing
- https://www.wikidata.org/wiki/Wikidata:Data_access
- https://www.wikidata.org/wiki/Wikidata:Copyright

This is a potential basis for a **new, independently generated and traceable structured-data corpus** from Wikidata. It is not retroactive proof or clearance for `data/gyaan_base.txt`. A future export should retain, for every row, at minimum the Wikidata subject QID, property ID, object QID or literal value, Hindi/English labels, endpoint, exact query or query hash, retrieval timestamp, and the corresponding source record. It should also keep facts separate from Wikipedia article text and other non-CC0 materials.

## Updated decision

- Existing `data/gyaan_base.txt`: provenance still unverified; do not train on it until its own origin and rights are established.
- Future Wikidata-derived corpus: feasible as a separately generated dataset if the export is restricted to Wikidata structured data and preserves source-level traceability.
- Production training: remains disabled.
- Privacy review: remains not formally cleared; CC0 status does not itself answer privacy, factual quality, or safety questions.

## New traceable exporter (candidate corpus only)

A separate exporter is available at `tools/export_wikidata_structured_corpus.py`. It queries a small reviewed allow-list of Wikidata structured-data classes and properties. It does not read, overwrite, or merge `data/gyaan_base.txt`.

Run a no-network plan first:

```bash
python tools/export_wikidata_structured_corpus.py --dry-run
```

A bounded live export (requires network access) can be run explicitly:

```bash
python tools/export_wikidata_structured_corpus.py \
  --scopes country city river mountain \
  --limit-per-property 200 \
  --page-size 100 \
  --request-interval 1.25 \
  --jsonl-out data/wikidata_structured.jsonl \
  --triples-out data/wikidata_structured_triples.txt \
  --manifest-out data/wikidata_structured_manifest.json
```

The canonical JSONL preserves subject QID, property ID, statement URI, object QID or literal datatype/language, Hindi/English labels, any statement-level reference URL/item, query SHA-256, endpoint, retrieval timestamp, and CC0 policy metadata. Records with no statement-level references are explicitly flagged; no citation is fabricated. The plain triple file is only a compatibility projection and must not replace the JSONL source of record.

The exporter defaults to a conservative request interval, caps page size and per-property row limits, sends a descriptive User-Agent, and retries transient HTTP failures. CI uses synthetic fixtures and does not contact Wikidata. A live export is not automatically approved for training: it still requires factual-quality, privacy, safety, scope, and dataset-owner review. The exporter does not modify dataset approvals or enable production training.

### Exporter integrity follow-up

A code review identified that a statement can have more than one Wikidata reference. The exporter now merges distinct statement-level reference URLs and reference-item IDs when the same statement appears in multiple query rows, instead of allowing deduplication to overwrite earlier references. The references retain the retrieval timestamp, endpoint, and query hash. A synthetic regression test covers this case. This is a metadata-preservation improvement; it does not constitute a live endpoint validation or dataset approval.

### Statement-rank and offline-plan safeguards

The exporter query now excludes Wikidata statements whose rank is `wikibase:DeprecatedRank` and records the returned statement rank in each JSONL record. Preferred-rank and normal-rank statements remain available; this preserves the rank metadata rather than silently flattening it. The test suite checks deprecated-rank exclusion and rank capture.

The offline `--dry-run` plan is also covered by a synthetic subprocess test. It verifies the planned query count and endpoint without making a network request. Passing this test does not verify that the live SPARQL endpoint accepts or executes every query.

## Live Wikidata smoke test result — 2026-11-25

The bounded live smoke workflow completed successfully on commit `95d27976b7414544a56fac86e4e231146720034f`.

- Workflow: `Wikidata Structured Data Smoke Test`
- Run: `45581169456`
- Result: `LIVE_WIKIDATA_SMOKE_PASS`
- Scope: country; six allow-listed properties; per-property limit 5
- Records exported: 28 (within the 30-record upper bound)
- Compatibility triples: 28
- Queries executed: 6
- Statements with statement-level references: 16
- Statements without statement-level references: 12
- JSONL and triples manifest SHA-256 checks: passed
- Uploaded workflow artifacts: none
- Temporary output remained in the runner's `/tmp` directory.

The 12 records without statement-level references remain explicitly flagged in the generated manifest; do not silently describe them as having external citations. The smoke test demonstrates that the live endpoint request, parsing, serialization, and local hash verification work for this bounded sample. It does not establish factual correctness, privacy clearance, full-corpus completeness, or approval for model training. Production training remains disabled, and no model training was run.

## Wikidata 200-record structural audit — 2026-11-25

A bounded live export and structural audit completed successfully.

- Workflow run: [45599526067](https://github.com/pdfusion9-cyber/hindi-lattice/actions/runs/45599526067)
- Commit tested: `12b7c98ec490e4d8d0e49ddeb2536ead9bad2e57`
- Scope: country, city, river, mountain; 16 queries
- JSONL records: 200
- Compatibility triple lines: 198 (JSONL is canonical; triple projection is not one-to-one)
- Property distribution: P1082 16, P131 16, P17 16, P2043 16, P2044 16, P2046 15, P30 15, P36 15, P37 15, P38 15, P403 15, P4552 15, P571 15
- Statement-level references present: 56/200 (28%)
- Statement-level references absent: 144/200 (72%)
- Multi-value subject-property groups flagged for human review: 18
- Structural issue count: 144 missing statement-level references
- Manifest SHA-256 checks: passed
- Audit input SHA-256: `bf4e42a38a762ef13732a0f269867cfd09286448e581fb2ffa13b9a9ec8c5d4b`
- Training-approved records: 0
- Public workflow artifacts: none

Interpretation: missing statement-level references are provenance gaps in this export, not proof that statements are false. Multiple values may be legitimate for multi-valued properties; the 18 groups are review flags, not automatic contradictions. Factual accuracy, freshness, privacy clearance, and training suitability were not assessed. Do not use this sample for model training until source coverage and human review are completed.

## Statement-reference enrichment results — 2026-11-25

The 200-record live enrichment workflow completed successfully:
[run 45602033164](https://github.com/pdfusion9-cyber/hindi-lattice/actions/runs/45602033164).

Observed results from its logs:
- Exported records: 200
- Distinct subject entities fetched in two Action API batches
- Statement IDs matched: 140
- Statement IDs unmatched: 60
- Records with any Wikidata reference record: 102
- Records without any Wikidata reference record: 98
- Records with URL or named-source external citations: 37
- Records without an external citation: 163
- Automated training approvals: 0

These are distinct measures. A Wikidata reference record can contain only an import marker such as P143; that is not counted as an external citation. The 60 unmatched statement IDs are an enrichment limitation and are not evidence that the statements are false. The run's first structural report used the old broad `missing_statement_reference` label; the audit schema is being clarified to separately report reference-record presence, external citation presence, import-only markers, and statement match status.

The integrated test workflow passed 46 tests, including the five reference-enrichment tests:
[run 45602086294](https://github.com/pdfusion9-cyber/hindi-lattice/actions/runs/45602086294).

No factual accuracy or privacy clearance is inferred. Training approval remains zero; this dataset is not approved for training.

