# Wikidata Citation Coverage Review — 200-record sample

**Audit:** [Wikidata 200 Record Quality Audit #45626683159](https://github.com/pdfusion9-cyber/hindi-lattice/actions/runs/45626683159)  
**Source commit:** `9bf3e5752e68e6b46c1ea5673760a9bec22ccf1b`

## Overall results

| Measure | Result |
|---|---:|
| Records sampled | 200 |
| Statements matched | 200 / 200 |
| Records with reference records | 138 |
| Records without reference records | 62 |
| Records with external citations | 65 |
| Records without external citations | 135 |
| External-citation coverage | 32.5% |
| Import-marker-only records | 71 |
| URL citation values | 38 |
| Named source items | 33 |
| Multi-value groups needing human review | 18 |
| Training-approved records | **0** |

External citation coverage here counts a P854/P4656 URL or a P248 named source item. P143 imported-from markers are not counted as external citations.

## Coverage by property

| Property | Records | With external citation | Coverage | Import-only | No reference |
|---|---:|---:|---:|---:|---:|
| P1082 | 16 | 16 | 100.0% | 0 | 0 |
| P131 | 16 | 3 | 18.8% | 7 | 6 |
| P17 | 16 | 4 | 25.0% | 7 | 5 |
| P2043 | 16 | 0 | 0.0% | 16 | 0 |
| P2044 | 16 | 1 | 6.2% | 13 | 2 |
| P2046 | 15 | 4 | 26.7% | 11 | 0 |
| P30 | 15 | 15 | 100.0% | 0 | 0 |
| P36 | 15 | 2 | 13.3% | 0 | 13 |
| P37 | 15 | 9 | 60.0% | 1 | 5 |
| P38 | 15 | 3 | 20.0% | 0 | 12 |
| P403 | 15 | 0 | 0.0% | 4 | 11 |
| P4552 | 15 | 2 | 13.3% | 10 | 3 |
| P571 | 15 | 6 | 40.0% | 2 | 5 |

## Review priorities

1. **P2043 and P403:** 0% external citation coverage in this sample.
2. **P2044:** 6.2%.
3. **P36 and P4552:** 13.3%.
4. **P131, P17 and P38:** 18.8%–25.0%.

These are triage priorities, not proof that statements are false. The 18 multi-value groups may be legitimate multi-valued properties or conflicts and require human review.

## Source-domain observations

The most frequent domain was workwithdata.com (14 URL values). Other observed domains include English, German and Russian Wikipedia (3 each), World Bank data (2), Investopedia (2), and several domains with one each. Domain frequency is descriptive only; it does not establish source quality, factual accuracy, or licensing.

## Safety and limitations

- Production training remains **DISABLED**; training-approved records: **0**.
- Factual accuracy, privacy clearance and source rights have not been assessed by this citation report.
- No citations were automatically added.
- The 198 compatibility triple lines are a lossy projection; the 200-record JSONL is canonical.
- The per-record citation-gap and multi-value review lists are generated in the CI workspace and were not published as raw records.
