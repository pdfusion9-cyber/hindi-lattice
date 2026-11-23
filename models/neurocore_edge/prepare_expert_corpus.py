#!/usr/bin/env python3
"""Approved streaming corpus builder using dataset_schema_adapters.

This is an intake/preparation stage, not a training launcher. It downloads no
source until all approval checks pass, refuses unknown/multimodal schemas, and
publishes files only after each source has produced a valid deterministic split.
"""
from __future__ import annotations
import argparse, hashlib, json, os, tempfile
from pathlib import Path
from dataset_schema_adapters import ADAPTER_VERSION, SchemaAdapterError, adapt_row, adapter_family

REQUIRED_FIELDS=("license_id","license_terms_url","reviewer","review_date","split_name","revision","schema_adapter_version","schema_fingerprint")
REQUIRED_FLAGS=("approved","commercial_training_allowed","redistribution_allowed","text_model_compatible","pii_reviewed","consent_reviewed","schema_reviewed","split_reviewed")

def sha256(path:Path)->str:
 h=hashlib.sha256()
 with path.open("rb") as f:
  for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
 return h.hexdigest()

def approved_rows(path:Path)->dict:
 p=json.loads(path.read_text(encoding="utf-8"))
 if p.get("status")!="APPROVED_FOR_COMMERCIAL_TRAINING":
  raise SystemExit("REFUSED: approval status is not APPROVED_FOR_COMMERCIAL_TRAINING")
 out={}
 for row in p.get("datasets",[]):
  did=row.get("dataset_id")
  if not did or did in out:raise ValueError("approval file has missing/duplicate dataset_id")
  missing=[k for k in REQUIRED_FIELDS if not str(row.get(k,"")).strip()]
  missing += [k for k in REQUIRED_FLAGS if row.get(k) is not True]
  if not missing and row.get("schema_adapter_version")==ADAPTER_VERSION and str(row.get("license_terms_url","")).startswith("https://"):
   out[did]=row
 return out

def is_valid(dataset_id:str,index:int,examples:list[str])->bool:
 key=(dataset_id+"\0"+str(index)+"\0"+"\0".join(examples)).encode("utf-8")
 return int.from_bytes(hashlib.sha256(key).digest()[:4],"big")%100<2

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--manifest",required=True);p.add_argument("--approval-file",required=True)
 p.add_argument("--expert-id",required=True,type=int);p.add_argument("--out-dir",required=True)
 p.add_argument("--max-examples-per-dataset",type=int,default=50000)
 a=p.parse_args()
 if a.max_examples_per_dataset<50:p.error("max-examples-per-dataset must be at least 50 for a 2% holdout")
 manifest=json.loads(Path(a.manifest).read_text(encoding="utf-8"))
 matches=[x for x in manifest.get("experts",[]) if int(x["expert_id"])==a.expert_id]
 if len(matches)!=1:raise SystemExit("expert id missing or duplicated in manifest")
 expert=matches[0];approved=approved_rows(Path(a.approval_file))
 ids=sorted(d for d in approved if adapter_family(d) not in {"sidecar_required","unregistered"}) if a.expert_id==18 else list(expert.get("datasets",[]))
 if not ids:raise SystemExit("REFUSED: no approved text datasets with registered adapters")
 refused=[d for d in ids if d not in approved]
 if refused:raise SystemExit("REFUSED: incomplete approval: "+", ".join(refused))
 for d in ids:
  if adapter_family(d) in {"sidecar_required","unregistered"}:raise SystemExit(f"REFUSED: {d} needs a sidecar or adapter")
 # Import only after rights/provenance/schema gate is satisfied.
 from datasets import load_dataset
 out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
 fd1,n1=tempfile.mkstemp(prefix=".train.",dir=out);fd2,n2=tempfile.mkstemp(prefix=".valid.",dir=out)
 tpath,vpath=Path(n1),Path(n2);total_train=total_valid=0;details=[]
 try:
  with os.fdopen(fd1,"w",encoding="utf-8") as train,os.fdopen(fd2,"w",encoding="utf-8") as valid:
   for did in ids:
    row=approved[did];split=str(row["split_name"]);rev=str(row["revision"])
    ds=load_dataset(did,name=str(row.get("config_name","")).strip() or None,split=split,revision=rev,streaming=True)
    seen=dtrain=dvalid=0
    for index,item in enumerate(ds):
     try:examples=[x.strip().replace("\x00"," ") for x in adapt_row(did,item,a.expert_id) if x.strip()]
     except SchemaAdapterError as e:raise SystemExit(f"SCHEMA_REFUSED {did} row {index}: {e}") from e
     if not examples:raise SystemExit(f"SCHEMA_REFUSED {did} row {index}: empty adapter output")
     is_holdout=is_valid(did,index,examples);dest=valid if is_holdout else train
     dest.write("\n\n".join(examples)+"\n\n")
     n=len(examples)
     if is_holdout:dvalid+=n;total_valid+=n
     else:dtrain+=n;total_train+=n
     seen+=1
     if seen>=a.max_examples_per_dataset:break
    if seen<2:raise SystemExit(f"SCHEMA_REFUSED {did}: fewer than two accepted source rows")
    details.append({"dataset_id":did,"config_name":row.get("config_name",""),"revision":rev,"split":split,
      "adapter_family":adapter_family(did),"adapter_version":ADAPTER_VERSION,"schema_fingerprint":row["schema_fingerprint"],
      "source_rows":seen,"train_examples":dtrain,"valid_examples":dvalid,"license_id":row["license_id"],
      "reviewer":row["reviewer"],"review_date":row["review_date"]})
  if not total_train or not total_valid:raise SystemExit("REFUSED: deterministic train/validation split is empty")
  os.replace(tpath,out/"train.txt");os.replace(vpath,out/"valid.txt")
  report={"status":"REVIEWED_EXPERT_CORPUS_READY","expert_id":a.expert_id,"expert_name":expert["name"],
   "adapter_version":ADAPTER_VERSION,"holdout_policy":"deterministic row-level 2%; dataset revision and split pinned",
   "train_examples":total_train,"valid_examples":total_valid,"datasets":details,
   "train_sha256":sha256(out/"train.txt"),"valid_sha256":sha256(out/"valid.txt"),
   "note":"Corpus preparation only; no training or checkpoint writes are performed."}
  (out/"corpus_manifest.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
  print(json.dumps(report,ensure_ascii=False,indent=2))
 finally:
  tpath.unlink(missing_ok=True);vpath.unlink(missing_ok=True)

if __name__=="__main__":main()
