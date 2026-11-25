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

