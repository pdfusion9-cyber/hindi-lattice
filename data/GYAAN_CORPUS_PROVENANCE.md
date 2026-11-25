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
