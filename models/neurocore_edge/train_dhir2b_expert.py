"""CPU-only fixed-expert adapter trainer with exact SHA-256 approval gates."""
from __future__ import annotations
import argparse, hashlib, json, os, tempfile
from pathlib import Path
import torch
import torch.nn.functional as F
from dhir2b_neurocore import Dhir2BNeuroCore

def sha_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def sha_dir(path):
    root=Path(path); h=hashlib.sha256()
    for p in sorted(x for x in root.rglob("*") if x.is_file()):
        h.update(p.relative_to(root).as_posix().encode()); h.update(b"\0")
        h.update(bytes.fromhex(sha_file(p)))
    return h.hexdigest()

def require_approval(path,expert_id,data_sha,tokenizer_sha):
    doc=json.loads(Path(path).read_text())
    data_ok=any(x.get("stage")=="dhir2b_expert" and x.get("expert_id")==expert_id
      and x.get("sha256")==data_sha and x.get("approved") is True and x.get("approval_id")
      for x in doc.get("approved_training_artifacts",[]))
    tok_ok=any(x.get("sha256")==tokenizer_sha and x.get("approved") is True and x.get("approval_id")
      for x in doc.get("approved_tokenizers",[]))
    if not (data_ok and tok_ok):
        raise PermissionError("training blocked: exact data and tokenizer SHA-256 approvals are required")
    return doc

def read_rows(path,tokenizer,max_length):
    rows=[]
    with open(path,encoding="utf-8") as f:
        for n,line in enumerate(f,1):
            if not line.strip(): continue
            item=json.loads(line)
            if "input_ids" in item: ids=[int(x) for x in item["input_ids"]]
            elif isinstance(item.get("text"),str):
                ids=tokenizer(item["text"],add_special_tokens=True,truncation=True,
                              max_length=max_length)["input_ids"]
            else: raise ValueError(f"row {n} requires text or input_ids")
            ids=ids[:max_length]
            if len(ids)<2: continue
            if min(ids)<0 or max(ids)>=len(tokenizer): raise ValueError(f"token id outside vocab at row {n}")
            rows.append(ids)
    if not rows: raise ValueError("no usable training sequences")
    return rows

def batch_rows(rows,indices,pad_id):
    length=max(len(rows[i]) for i in indices)
    ids=torch.full((len(indices),length),pad_id,dtype=torch.long)
    labels=torch.full_like(ids,-100)
    mask=torch.zeros_like(ids)
    for j,i in enumerate(indices):
        seq=rows[i]; n=len(seq)
        ids[j,:n]=torch.tensor(seq); labels[j,:n]=torch.tensor(seq); mask[j,:n]=1
    labels[mask==0]=-100
    return ids,labels,mask

def save_checkpoint(path,model,expert_id,optimizer,step,data_sha,tokenizer_sha):
    target=Path(path); target.parent.mkdir(parents=True,exist_ok=True)
    payload={"schema_version":1,"status":"EXPERT_ADAPTER_TRAINED_UNVALIDATED",
      "architecture_version":"0.4.0-dhir2b-adapters","expert_id":expert_id,
      "base_model_id":model.base_model_id,"base_revision":model.base_revision,
      "vocab_size":model.vocab_size,"num_hidden_layers":28,"adapter_rank":model.rank,
      "data_sha256":data_sha,"tokenizer_sha256":tokenizer_sha,"steps_completed":step,
      "expert_state_dict":{k:v.detach().cpu() for k,v in expert_state(model,expert_id).items()},
      "optimizer_state_dict":optimizer.state_dict(),"torch_rng_state":torch.get_rng_state()}
    fd,tmp=tempfile.mkstemp(prefix=target.name+".",suffix=".tmp",dir=str(target.parent)); os.close(fd)
    try: torch.save(payload,tmp); os.replace(tmp,target)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def expert_state(model,expert_id):
    return {f"layer_{i}.{k}":v.detach().cpu().clone()
      for i,layer in enumerate(model.adapter_layers)
      for k,v in layer.experts[expert_id].state_dict().items()}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir",required=True); p.add_argument("--data",required=True)
    p.add_argument("--approval",default="dataset_approval.json")
    p.add_argument("--expert-id",type=int,required=True); p.add_argument("--steps",type=int,default=100)
    p.add_argument("--batch-size",type=int,default=1); p.add_argument("--max-length",type=int,default=256)
    p.add_argument("--lr",type=float,default=2e-4); p.add_argument("--output",required=True)
    p.add_argument("--threads",type=int,default=2); a=p.parse_args()
    if not 0<=a.expert_id<20: p.error("--expert-id must be 0..19")
    if min(a.steps,a.batch_size,a.max_length,a.threads)<1 or a.lr<=0: p.error("invalid training arguments")
    torch.set_num_threads(a.threads)
    model,tokenizer,manifest=Dhir2BNeuroCore.load_checkpoint(a.model_dir)
    data_sha=sha_file(a.data); tok_sha=sha_dir(Path(a.model_dir)/"tokenizer")
    require_approval(a.approval,a.expert_id,data_sha,tok_sha)
    rows=read_rows(a.data,tokenizer,a.max_length)
    trainable=model.prepare_expert_training(a.expert_id); model.train()
    opt=torch.optim.AdamW(trainable,lr=a.lr)
    pad=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad is None: raise ValueError("tokenizer must define pad or EOS token")
    for step in range(1,a.steps+1):
        start=((step-1)*a.batch_size)%len(rows)
        indices=[(start+j)%len(rows) for j in range(a.batch_size)]
        ids,labels,mask=batch_rows(rows,indices,int(pad))
        opt.zero_grad(set_to_none=True)
        output=model(input_ids=ids,attention_mask=mask,expert_id=a.expert_id,use_cache=False)
        logits=output.logits
        loss=F.cross_entropy(logits[:,:-1,:].float().reshape(-1,model.vocab_size),
          labels[:,1:].reshape(-1),ignore_index=-100)
        if not torch.isfinite(loss): raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward(); torch.nn.utils.clip_grad_norm_(trainable,1.0); opt.step()
        save_checkpoint(a.output,model,a.expert_id,opt,step,data_sha,tok_sha)
        if step==1 or step%10==0 or step==a.steps:
            print(json.dumps({"step":step,"loss":float(loss.detach()),"expert_id":a.expert_id,
              "device":"cpu","status":"EXPERT_ADAPTER_TRAINING"},ensure_ascii=False))
    print(json.dumps({"status":"EXPERT_ADAPTER_TRAINED_UNVALIDATED","expert_id":a.expert_id,
      "steps_completed":a.steps,"checkpoint":a.output,"base_revision":manifest["base_revision"],
      "gpu_used":False},ensure_ascii=False,indent=2))

if __name__=="__main__": main()
