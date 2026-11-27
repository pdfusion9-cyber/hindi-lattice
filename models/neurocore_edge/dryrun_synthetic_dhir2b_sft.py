"""Measurement-only SFT dry-run with the immutable native Dhīr-2B tokenizer."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from statistics import mean
from neurocore_tokenizer import NeuroCoreTokenizer, tokenizer_files_sha256
from train_dhir2b_expert import _as_token_ids, contains_pii_pattern, read_rows, sha_dir, sha_file

MODEL = "zenteiq-ai/Dhir-2B"
REVISION = "7cbe444cd1e45e25197cfa0f683b2cb575858169"
VOCAB_SIZE = 131072
VOCAB_SHA = "cff9e969e05108763f796df6dafd614ee8159ebbf3cc165594265fa0ea02ee9c"
MAX_LENGTH = 256
DATA_REVISION = "e1b7824cccdc32004f4cfcd5acf91a2c76acfec9"
SPECS = (
    (16, "self_correction_mnl", "local/neurocore_synthetic_sft/expert16_self_correction",
     "synthetic_sft/expert16_self_correction_train.jsonl",
     "synthetic_sft/expert16_self_correction_validation.jsonl"),
    (17, "coconut_planning", "local/neurocore_synthetic_sft/expert17_planning",
     "synthetic_sft/expert17_planning_train.jsonl",
     "synthetic_sft/expert17_planning_validation.jsonl"),
)

def load_rows(path):
    rows=[]
    for n,line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(),1):
        if not line.strip(): continue
        row=json.loads(line)
        if not isinstance(row,dict) or set(row)!={"prompt","response"}:
            raise ValueError(f"{path}:{n}: expected prompt/response")
        if not all(isinstance(row[k],str) and row[k].strip() for k in ("prompt","response")):
            raise ValueError(f"{path}:{n}: empty prompt/response")
        if contains_pii_pattern(row["prompt"]) or contains_pii_pattern(row["response"]):
            raise ValueError(f"{path}:{n}: PII pattern; refusing dry-run")
        rows.append(row)
    if not rows: raise ValueError(f"{path}: no records")
    return rows

def verify_mask(prompt_ids, full_ids, prepared, max_length=MAX_LENGTH):
    if len(full_ids)>max_length: raise ValueError(f"sequence exceeds max_length={max_length}; trainer would truncate")
    if not prompt_ids or len(full_ids)<=len(prompt_ids):
        raise ValueError("assistant completion is empty after native prompt prefix")
    if full_ids[:len(prompt_ids)]!=prompt_ids:
        raise ValueError("native assistant-generation prefix mismatch")
    ids=[int(x) for x in prepared["input_ids"]]
    labels=[int(x) for x in prepared["labels"]]
    expected=[-100]*len(prompt_ids)+full_ids[len(prompt_ids):]
    if ids!=full_ids: raise ValueError("trainer input differs from native full chat")
    if labels!=expected: raise ValueError("assistant-only loss mask mismatch")
    targets=sum(x!=-100 for x in labels[1:])
    if targets<1: raise ValueError("no assistant next-token targets")
    return len(prompt_ids),len(full_ids)-len(prompt_ids),targets,len(full_ids)

def inspect_split(tokenizer,path,split):
    source=load_rows(path)
    prepared=read_rows(path,tokenizer,max_length=MAX_LENGTH,max_rows=5000)
    if len(prepared)!=len(source):
        raise ValueError(f"{split}: trainer retained {len(prepared)}/{len(source)} rows")
    lengths=[]; masked=assistant=targets=0
    for row,encoded in zip(source,prepared,strict=True):
        p=_as_token_ids(tokenizer.apply_chat_template(
            [{"role":"user","content":row["prompt"]}],tokenize=True,add_generation_prompt=True))
        f=_as_token_ids(tokenizer.apply_chat_template(
            [{"role":"user","content":row["prompt"]},
             {"role":"assistant","content":row["response"]}],
            tokenize=True,add_generation_prompt=False))
        a,b,c,d=verify_mask(p,f,encoded)
        masked+=a; assistant+=b; targets+=c; lengths.append(d)
    return {
        "split":split,"file":Path(path).name,"source_file_sha256":sha_file(path),
        "source_rows":len(source),"trainer_rows_retained":len(prepared),
        "all_rows_retained":True,"tokenized_records_persisted":False,
        "max_sequence_length":max(lengths),"mean_sequence_length":round(mean(lengths),3),
        "total_sequence_tokens":sum(lengths),"prompt_tokens_masked":masked,
        "assistant_label_tokens":assistant,"supervised_next_token_targets":targets,
        "assistant_only_loss_mask_verified":True,
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--tokenizer-dir",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    root=Path(__file__).resolve().parent
    registry=json.loads((root/"dataset_approval.json").read_text(encoding="utf-8"))
    if registry.get("status")!="ALL_DATASETS_PENDING_REVIEW":
        raise PermissionError("dataset registry is not default-deny")
    if registry.get("approved_training_artifacts") or registry.get("approved_tokenizers"):
        raise PermissionError("expected zero approved artifacts/tokenizers")
    pending={}
    for eid,name,did,_,_ in SPECS:
        row=next((x for x in registry["datasets"] if x.get("expert_id")==eid and x.get("dataset_id")==did),None)
        if not row or row.get("revision")!=DATA_REVISION or row.get("approved") is not False:
            raise PermissionError(f"expert {eid} provenance or pending status mismatch")
        for key in ("commercial_training_allowed","pii_reviewed","consent_reviewed","schema_reviewed","split_reviewed"):
            if row.get(key) is not False: raise PermissionError(f"expert {eid}: expected {key}=false")
        pending[str(eid)]={"dataset_id":did,"dataset_revision":row["revision"],"approved":False}
    from transformers import PreTrainedTokenizerFast
    raw=PreTrainedTokenizerFast.from_pretrained(MODEL,revision=REVISION,trust_remote_code=False)
    wrapped=NeuroCoreTokenizer.from_tokenizer(raw,base_model_id=MODEL,base_revision=REVISION,expected_vocab_size=VOCAB_SIZE)
    if wrapped.neurocore_manifest["tokenizer_vocab_sha256"]!=VOCAB_SHA:
        raise ValueError("pinned Dhīr vocabulary SHA differs from the previously verified benchmark")
    if not wrapped.neurocore_manifest["chat_template_preserved"]:
        raise ValueError("native Dhīr chat template missing")
    args.tokenizer_dir.mkdir(parents=True,exist_ok=True)
    wrapped.save_pretrained(args.tokenizer_dir)
    tok=NeuroCoreTokenizer.from_pretrained(args.tokenizer_dir,base_model_id=MODEL,base_revision=REVISION,expected_vocab_size=VOCAB_SIZE)
    experts={}; combined=hashlib.sha256(); train_total=val_total=0
    for eid,name,did,train_file,val_file in SPECS:
        train=inspect_split(tok,root/train_file,"train_only")
        val=inspect_split(tok,root/val_file,"held_out_validation_not_for_training")
        train.update(dataset_id=did,dataset_revision=DATA_REVISION)
        val.update(dataset_id=did,dataset_revision=DATA_REVISION)
        combined.update(did.encode()+b"\0"+bytes.fromhex(train["source_file_sha256"]))
        experts[str(eid)]={"expert_name":name,"training":train,"validation":val,"validation_used_for_training":False}
        train_total+=train["source_rows"]; val_total+=val["source_rows"]
    tokdir_sha=sha_dir(args.tokenizer_dir)
    report={
      "status":"SYNTHETIC_SFT_TOKENIZER_DRY_RUN_PASSED_NO_TRAINING",
      "training_performed":False,"optimizer_steps":0,"model_weights_loaded":False,
      "model_forward_pass_performed":False,"tokenized_records_persisted":False,
      "raw_text_in_report":False,"base_model_id":MODEL,"base_revision":REVISION,
      "vocab_size":len(tok),"tokenizer_vocab_sha256":tok.neurocore_manifest["tokenizer_vocab_sha256"],
      "tokenizer_chat_template_sha256":tok.neurocore_manifest["chat_template_sha256"],
      "tokenizer_files_sha256":tokenizer_files_sha256(args.tokenizer_dir),
      "trainer_compatible_tokenizer_dir_sha256":tokdir_sha,"max_length":MAX_LENGTH,
      "experts":experts,
      "totals":{"training_examples":train_total,"held_out_validation_examples":val_total,
        "all_training_rows_retained":True,"all_validation_rows_retained":True,
        "validation_used_for_training":False,"training_source_files_combined_sha256":combined.hexdigest()},
      "approval_status":{"registry_status":"ALL_DATASETS_PENDING_REVIEW",
        "expert_16_and_17_approved":False,"approved_training_artifacts":0,
        "approved_tokenizers":0,"approval_gate_bypassed":False},
      "interpretation":"Real pinned Dhīr tokenizer and production trainer parser exercised; measurement only, not approval or training."
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":report["status"],"training_examples":train_total,
      "held_out_validation_examples":val_total,"vocab_size":len(tok),
      "tokenizer_vocab_sha256":report["tokenizer_vocab_sha256"],
      "trainer_compatible_tokenizer_dir_sha256":tokdir_sha,
      "training_performed":False,"optimizer_steps":0,"report":str(args.output)},ensure_ascii=False))

if __name__=="__main__": main()
