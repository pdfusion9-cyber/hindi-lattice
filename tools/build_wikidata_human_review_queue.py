#!/usr/bin/env python3
"""Generate a private human-review queue; never use public CI artifacts for row-level data."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

PRIORITY = ("P2043", "P403", "P2044", "P36", "P4552", "P131", "P17", "P38")
HINTS = {
    "P2043": "Check length unit, measured object, method and date",
    "P403": "Check watercourse identity, downstream endpoint and geography",
    "P2044": "Check vertical datum, units and measurement date",
    "P36": "Check capital role and effective dates",
    "P4552": "Check mountain-range definition and membership",
    "P131": "Check administrative level, boundary and historical period",
    "P17": "Check sovereignty, disputed status and time scope",
    "P38": "Check legal tender vs official currency and effective dates",
}
CHECKS = ["claim_source_match", "source_authority_independence", "time_scope",
          "geographic_scope", "units_qualifiers", "source_accessibility"]

def stable_id(kind, obj):
    raw=json.dumps(obj,sort_keys=True,ensure_ascii=False,separators=(",",":"))
    return kind+"-"+hashlib.sha256(raw.encode()).hexdigest()[:16]

def build_queue(rows):
    gaps=[]; buckets=defaultdict(list); counts=defaultdict(lambda:{"records":0,"citation_gaps":0})
    for row in rows:
        prop=str(row.get("property_id","UNKNOWN")); counts[prop]["records"]+=1
        buckets[(str(row.get("subject_qid","")),prop)].append(row)
        refs=row.get("wikidata_reference_records")
        refs=refs if isinstance(refs,list) else []
        if prop not in PRIORITY or row.get("has_external_citation"): continue
        imported=any(isinstance(x,dict) and x.get("imported_from_items") for x in refs)
        reason="NO_REFERENCE_RECORD" if not refs else ("IMPORT_MARKER_ONLY" if imported else "REFERENCE_WITHOUT_EXTERNAL_CITATION")
        ident={"record_id":row.get("record_id"),"subject_qid":row.get("subject_qid"),
               "property_id":prop,"statement_uri":row.get("statement_uri")}
        gaps.append({"review_id":stable_id("citation",ident),**ident,
            "subject_label_hi":row.get("subject_label_hi"),"relation_label_hi":row.get("relation_label_hi"),
            "object_qid":row.get("object_qid"),"display_value_hi":row.get("display_value_hi"),
            "review_reason":reason,"property_review_hint":HINTS[prop],"review_checks":CHECKS,
            "decision":"NOT_REVIEWED","reviewer":"","reviewed_at_utc":"",
            "evidence_url":"","evidence_passage_or_table":"","rationale":""})
        counts[prop]["citation_gaps"]+=1
    multi=[]
    for (subject,prop), group in sorted(buckets.items()):
        vals={}
        for row in group:
            key=str(row.get("object_qid") or row.get("object_lexical") or row.get("display_value_hi") or "")
            vals.setdefault(key,{"object_qid":row.get("object_qid"),"display_value_hi":row.get("display_value_hi"),
                "record_id":row.get("record_id"),"statement_uri":row.get("statement_uri"),
                "has_external_citation":bool(row.get("has_external_citation"))})
        if len(vals)<2: continue
        ident={"subject_qid":subject,"property_id":prop,"values":sorted(vals)}
        first=group[0]
        multi.append({"review_id":stable_id("multi",ident),"subject_qid":subject,
            "subject_label_hi":first.get("subject_label_hi"),"property_id":prop,
            "relation_label_hi":first.get("relation_label_hi"),"property_review_hint":HINTS.get(prop,"Check property semantics"),
            "values":[vals[k] for k in sorted(vals)],
            "review_checks":["property_allows_multiple_values","same_time_and_scope",
                "qualifiers_and_units","independent_support","valid_multi_value_or_conflict"],
            "decision":"NOT_REVIEWED","reviewer":"","reviewed_at_utc":"","evidence_summary":"","rationale":""})
    return {"schema_version":"1.0","status":"REVIEW_QUEUE_CREATED_NOT_ADJUDICATED",
        "record_count":len(rows),"priority_properties":list(PRIORITY),
        "citation_gap_review_count":len(gaps),"citation_gap_by_property":dict(sorted(counts.items())),
        "multi_value_group_count":len(multi),"citation_gap_review_list":gaps,
        "multi_value_review_list":multi,"training_approved_record_count":0,
        "training_approval":"NOT_GRANTED_BY_REVIEW_QUEUE"}

def write_ledger(path, queue):
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=["review_id","review_type","subject_qid","property_id","record_id","object_value",
            "review_reason","decision","reviewer","reviewed_at_utc","evidence_url","evidence_passage_or_table","rationale"]
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for x in queue["citation_gap_review_list"]:
            w.writerow({"review_id":x["review_id"],"review_type":"CITATION_GAP","subject_qid":x.get("subject_qid"),
                "property_id":x.get("property_id"),"record_id":x.get("record_id"),
                "object_value":x.get("display_value_hi") or x.get("object_qid"),
                "review_reason":x["review_reason"],"decision":"NOT_REVIEWED"})
        for x in queue["multi_value_review_list"]:
            w.writerow({"review_id":x["review_id"],"review_type":"MULTI_VALUE","subject_qid":x["subject_qid"],
                "property_id":x["property_id"],"object_value":" | ".join(str(v.get("display_value_hi") or v.get("object_qid") or "") for v in x["values"]),
                "decision":"NOT_REVIEWED"})

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,required=True); p.add_argument("--queue-json",type=Path,required=True)
    p.add_argument("--ledger-csv",type=Path,required=True); a=p.parse_args()
    rows=[json.loads(s) for s in a.input.read_text(encoding="utf-8").splitlines() if s.strip()]
    if not all(isinstance(x,dict) for x in rows): raise ValueError("each JSONL row must be an object")
    q=build_queue(rows); a.queue_json.parent.mkdir(parents=True,exist_ok=True)
    a.queue_json.write_text(json.dumps(q,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    write_ledger(a.ledger_csv,q)
    print(json.dumps({k:q[k] for k in ("status","record_count","citation_gap_review_count",
        "citation_gap_by_property","multi_value_group_count","training_approved_record_count")},ensure_ascii=False))
if __name__=="__main__": main()
