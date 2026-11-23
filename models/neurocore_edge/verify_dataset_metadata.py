#!/usr/bin/env python3
"""Fail-closed Hugging Face dataset metadata/schema audit. Never approves training."""
from __future__ import annotations
import json, os, time, urllib.parse, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REGISTRY = ROOT / "dataset_approval.json"
OUT = Path(os.environ.get("NEUROCORE_AUDIT_OUT", "dataset_verification_report.json"))
EVAL_ONLY = {"facebook/flores", "openlanguagedata/flores_plus", "sarthak-wiz01/legalbench", "awinml/legalbench-rag"}
SIDECAR = {"HuggingFaceM4/DocumentVQA", "ai4bharat/indicvoices_r", "mythicinfinity/libritts", "ajimeno/PubTabNet", "allenai/olmOCR-mix-0225"}

def get_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "NeuroCore-Dataset-Audit/1.0"})
    with urllib.request.urlopen(req, timeout=25) as r:
        return json.loads(r.read().decode("utf-8"))

def audit(dataset_id: str):
    row = {"dataset_id": dataset_id, "metadata_status": "unknown", "license_status": "unknown",
           "revision_status": "not_pinned", "splits_status": "unknown", "schema_status": "unknown",
           "training_disposition": "evaluation_only_pending" if dataset_id in EVAL_ONLY else
             ("sidecar_required_pending" if dataset_id in SIDECAR else "pending_human_rights_review"),
           "evidence": []}
    try:
        meta = get_json("https://huggingface.co/api/datasets/" + urllib.parse.quote(dataset_id, safe="/"))
        row["metadata_status"] = "available"
        row["resolved_revision"] = meta.get("sha")
        row["gated"] = meta.get("gated")
        row["last_modified"] = meta.get("lastModified")
        card = meta.get("cardData") or {}
        license_id = card.get("license")
        row["declared_license"] = license_id
        row["license_status"] = "declared_requires_terms_review" if license_id else "missing"
        row["card_tags"] = [x for x in (meta.get("tags") or []) if str(x).startswith("license:")]
        if not license_id:
            row["evidence"].append("No SPDX/license field exposed in dataset card metadata.")
        row["evidence"].append("A declared license is metadata only; upstream work rights, commercial training and redistribution require separate review.")
        try:
            q = urllib.parse.urlencode({"dataset": dataset_id})
            splits = get_json("https://datasets-server.huggingface.co/splits?" + q)
            split_rows = splits.get("splits", [])
            row["splits"] = [{"config": s.get("config"), "split": s.get("split")} for s in split_rows]
            row["splits_status"] = "listed" if split_rows else "not_listed"
            if split_rows:
                first = split_rows[0]
                params = urllib.parse.urlencode({"dataset": dataset_id, "config": first["config"], "split": first["split"], "offset": 0, "length": 5})
                try:
                    sample = get_json("https://datasets-server.huggingface.co/first-rows?" + params)
                    features = sample.get("features")
                    rows = sample.get("rows") or []
                    row["feature_names"] = [f.get("name") for f in features or []]
                    row["sample_rows_returned"] = len(rows)
                    row["schema_status"] = "sampled" if features is not None and rows else "empty_or_unavailable"
                except Exception as e:
                    row["schema_status"] = "unavailable"
                    row["evidence"].append("First-rows API: " + str(e)[:300])
        except Exception as e:
            row["splits_status"] = "unavailable"
            row["evidence"].append("Splits API: " + str(e)[:300])
    except Exception as e:
        row["metadata_status"] = "unavailable"
        row["license_status"] = "unverified"
        row["splits_status"] = "unverified"
        row["schema_status"] = "unverified"
        row["evidence"].append("Dataset metadata API: " + str(e)[:300])
    row["approval_granted"] = False
    return row

def main():
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    ids = sorted({r["dataset_id"] for r in registry["datasets"] if r["dataset_id"] != "curated_mixture_from_all_domains"})
    results = [audit(ds) for ds in ids]
    report = {
        "report_version": "1.0",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_registry_status": registry.get("status"),
        "distinct_external_dataset_count": len(ids),
        "datasets": results,
        "training_gate": "CLOSED",
        "approval_granted_count": 0,
        "note": "Automated public metadata, split and sample-schema discovery is not a legal determination. No dataset is approved. Human rights, provenance, privacy/consent, exact revision/config/split and adapter review remain required."
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    counts = {}
    for r in results:
        counts[r["metadata_status"]] = counts.get(r["metadata_status"], 0) + 1
    print(json.dumps({"report": str(OUT), "datasets": len(ids), "metadata_status_counts": counts,
                      "license_declared": sum(bool(r.get("declared_license")) for r in results),
                      "sampled_schemas": sum(r["schema_status"] == "sampled" for r in results),
                      "training_gate": "CLOSED"}, indent=2))
if __name__ == "__main__":
    main()
