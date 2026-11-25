# Wikidata Human Review Protocol

Status: **protocol defined; no individual record approved**

This protocol is for human review of the 200-record citation audit and its 18 multi-value groups. The aggregate report does not expose the per-record queue, so no individual QID/value pair is adjudicated by this document.

## 1. Review outcomes

Every record or group must receive exactly one human decision:

- `ACCEPT_SUPPORTED`: the claim is supported by an eligible, relevant source.
- `ACCEPT_WITH_QUALIFIER`: the claim is supported only with a stated time, place, scope, unit, or other qualifier.
- `NEEDS_BETTER_SOURCE`: a source exists but is not sufficiently authoritative, direct, independent, or relevant.
- `CONFLICTING_EVIDENCE`: eligible sources disagree; retain both claims and escalate.
- `VALID_MULTI_VALUE`: multiple values are independently supported and allowed by the property's semantics.
- `INVALID_OR_STALE_VALUE`: a value is demonstrably invalid, superseded, or out of scope; record the evidence and effective date.
- `INSUFFICIENT_EVIDENCE`: evidence is absent or inconclusive.
- `NOT_REVIEWED`: no reviewer has assessed the item.

No automated citation coverage score can assign one of these decisions.

## 2. Citation checks (apply to each statement)

| Check | Reviewer question | Required evidence / action |
|---|---|---|
| Claim match | Does the source directly support this exact subject–property–value? | Record the source passage, table, or dataset field |
| Source identity | Is the publisher/author identifiable? | Record publisher and source URL or source item |
| Authority | Is the source appropriate for this property and jurisdiction? | Explain why the publisher is competent for the claim |
| Independence | Is the source independent of the claim's origin? | Mark copied/imported sources as non-independent |
| Time validity | Does the source cover the statement's relevant date? | Record publication/update date and claim's point-in-time |
| Scope | Does geography, administrative boundary, population, or definition match? | Record scope and any mismatch |
| Units and precision | Do units, conversions, rounding and precision agree? | Preserve original units and conversion method |
| Qualifiers | Are start/end dates, applies-to, determination method, or other qualifiers needed? | Capture qualifiers before comparing values |
| Source accessibility | Can a reviewer inspect the source? | Record access date and inaccessible/dead-link status |

A URL's presence or a domain's frequency is not proof of claim correctness. P143 import markers do not count as external citations in this audit.

## 3. Multi-value group checks

For each subject + property group:

1. Confirm that the property permits more than one simultaneous value.
2. Check whether the values refer to different times, jurisdictions, definitions, entities, or measurement methods.
3. Compare qualifiers and units before calling values contradictory.
4. Seek independent, property-appropriate sources for each value.
5. If both values are supported and coexist legitimately, use `VALID_MULTI_VALUE`.
6. If they disagree for the same scope and time, use `CONFLICTING_EVIDENCE`; do not silently delete either value.
7. If evidence cannot resolve the issue, use `INSUFFICIENT_EVIDENCE`.

A multi-value group is a review trigger, not an error label.

## 4. Property-specific checks for this sample

| Property | Sample coverage | Review focus |
|---|---:|---|
| P2043 | 0.0% (16 records) | Length: unit, object type, measurement basis, and source |
| P403 | 0.0% (15 records) | Mouth of the watercourse: downstream endpoint, watercourse identity, and geographic scope |
| P2044 | 6.2% (16 records) | Height: reference datum, elevation vs. object height, units, and measurement date |
| P36 | 13.3% (15 records) | Capital: legal/administrative status, effective dates, and multiple capital functions |
| P4552 | 13.3% (15 records) | Mountain range: geographic definition and membership evidence |
| P131 | 18.8% (16 records) | Administrative location: boundary level, historical period, and parent hierarchy |
| P17 | 25.0% (16 records) | Country: sovereignty, disputed status, and time scope |
| P38 | 20.0% (15 records) | Currency: legal tender vs. official currency and effective dates |

These are property-level review prompts, not adjudications of any row.

## 5. Required review ledger fields

The reviewer ledger must retain:

- `record_id`, subject QID, property ID, object QID/value
- source URL or source item and exact supporting passage/table
- source publisher, publication/update date, access date
- claim time scope, geographic scope, units and qualifiers
- independence/authority rationale
- review outcome, reviewer identifier, review timestamp
- short rationale and any unresolved follow-up

Do not store personal data or credentials in the ledger. Do not overwrite source records during review.

## 6. Current queue and release gates

The audit of 200 records reported:

- 135 records without an external citation
- 71 records with import-marker-only references
- 62 records without reference records
- 18 multi-value groups requiring human review
- 0 training-approved records

The 18 group identities and 135 record-level gaps are in the CI workspace output, not in the published aggregate report. Re-run the audit and export its review JSON to a restricted review workspace before adjudicating individual items.

Release gates remain closed until source rights/licensing, privacy clearance, factual review, and dataset approval are completed by authorized reviewers. This protocol itself grants no dataset or training approval.
