#!/usr/bin/env python3
"""Synthetic-only Hindi byte-level language-model baseline.

All training/evaluation sentences are original toy examples embedded below.
No repository corpus, external dataset, web text, or user data is read.
This is a tokenizer/model pipeline smoke test, not a useful Hindi assistant.
"""
from __future__ import annotations
import argparse, hashlib, json, random
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

TRAIN = [
"नमस्ते। यह हिंदी भाषा का छोटा अभ्यास है।",
"भारत की राजधानी नई दिल्ली है।",
"सूरज पूर्व दिशा से निकलता है और पश्चिम में अस्त होता है।",
"पानी जीवन के लिए आवश्यक है।",
"किसी प्रश्न का उत्तर सोचकर और स्पष्ट भाषा में देना चाहिए।",
"किताबें पढ़ने से नई जानकारी मिलती है।",
"किसान खेत में फसल उगाते हैं।",
"विद्यालय में विद्यार्थी अलग-अलग विषय पढ़ते हैं।",
"विज्ञान में प्रयोग और प्रमाण का महत्व है।",
"पेड़ हमें छाया, फल और स्वच्छ हवा देते हैं।",
"आज मौसम सुहावना है और आकाश साफ है।",
"समय का सही उपयोग सफलता में मदद करता है।",
"कंप्यूटर निर्देशों के अनुसार गणना करता है।",
"भाषा विचारों को साझा करने का माध्यम है।",
"संगीत मन को प्रसन्न कर सकता है।",
"नदी पहाड़ों से निकलकर मैदानों की ओर बहती है।",
"स्वस्थ रहने के लिए संतुलित भोजन और व्यायाम उपयोगी हैं।",
"छोटे कदम लगातार उठाने से बड़ा काम पूरा होता है।",
"हम नई तकनीक का उपयोग जिम्मेदारी से कर सकते हैं।",
"प्रकृति और पर्यावरण की रक्षा करना सबकी जिम्मेदारी है।",
]
EVAL = [
"दिल्ली भारत का एक बड़ा शहर है।",
"अच्छे प्रश्न सीखने की प्रक्रिया को बेहतर बनाते हैं।",
"पुस्तकालय में अनेक विषयों की पुस्तकें उपलब्ध हैं।",
"विचार स्पष्ट हों तो संवाद सरल बनता है।",
"बारिश के बाद पेड़-पौधे अधिक हरे दिखाई देते हैं।",
]

def encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))
def decode(ids: list[int]) -> str:
    return bytes(ids).decode("utf-8")

class HindiByteLM(nn.Module):
    def __init__(self, embedding=48, hidden=96):
        super().__init__()
        self.embedding = nn.Embedding(256, embedding)
        self.gru = nn.GRU(embedding, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 256)
    def forward(self, x):
        h, _ = self.gru(self.embedding(x))
        return self.head(h)

def evaluate(model, data, seed, steps=12, seq=48, batch=16):
    model.eval()
    g = torch.Generator().manual_seed(seed)
    losses=[]
    with torch.no_grad():
        for _ in range(steps):
            starts=torch.randint(0, len(data)-seq-1, (batch,), generator=g)
            x=torch.stack([data[int(i):int(i)+seq] for i in starts])
            y=torch.stack([data[int(i)+1:int(i)+seq+1] for i in starts])
            losses.append(float(F.cross_entropy(model(x).reshape(-1,256), y.reshape(-1))))
    return sum(losses)/len(losses)

def run(report_path: Path, steps=160, seed=7):
    if steps < 40: raise ValueError("steps must be >=40")
    torch.manual_seed(seed); random.seed(seed); torch.set_num_threads(1)
    train_text="\n".join(TRAIN*4)
    eval_text="\n".join(EVAL*2)
    # UTF-8 byte vocabulary is fixed at 256 and round-trips Hindi Unicode exactly.
    if decode(encode(train_text)) != train_text or decode(encode(eval_text)) != eval_text:
        raise RuntimeError("UTF-8 byte tokenizer round-trip failed")
    train=torch.tensor(encode(train_text), dtype=torch.long)
    heldout=torch.tensor(encode(eval_text), dtype=torch.long)
    model=HindiByteLM()
    optimizer=torch.optim.AdamW(model.parameters(), lr=0.003)
    initial_train=evaluate(model,train,seed+1)
    initial_eval=evaluate(model,heldout,seed+2)
    seq,batch=48,16
    for _ in range(steps):
        starts=torch.randint(0,len(train)-seq-1,(batch,))
        x=torch.stack([train[int(i):int(i)+seq] for i in starts])
        y=torch.stack([train[int(i)+1:int(i)+seq+1] for i in starts])
        optimizer.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model(x).reshape(-1,256),y.reshape(-1))
        if not torch.isfinite(loss): raise RuntimeError("non-finite loss")
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        if not torch.isfinite(grad): raise RuntimeError("non-finite gradients")
        optimizer.step()
    final_train=evaluate(model,train,seed+1)
    final_eval=evaluate(model,heldout,seed+2)
    if final_train >= initial_train: raise RuntimeError("training loss did not decrease")
    # Held-out score is reported, never used for optimizer updates.
    with torch.no_grad(): reference=model(heldout[:64].unsqueeze(0)).clone()
    state={"model":model.state_dict(),"optimizer":optimizer.state_dict(),"steps":steps,"seed":seed}
    report_path.parent.mkdir(parents=True,exist_ok=True)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        ckpt=Path(td)/"hindi_synthetic.pt"
        torch.save(state,ckpt)
        digest=hashlib.sha256(ckpt.read_bytes()).hexdigest()
        restored=torch.load(ckpt,map_location="cpu",weights_only=True)
        clone=HindiByteLM(); clone.load_state_dict(restored["model"],strict=True); clone.eval()
        with torch.no_grad(): reloaded=clone(heldout[:64].unsqueeze(0))
        diff=float((reference-reloaded).abs().max())
        if diff != 0.0: raise RuntimeError(f"checkpoint reload mismatch: {diff}")
        # Optimizer state must be present and recoverable.
        clone_opt=torch.optim.AdamW(clone.parameters(),lr=0.003)
        clone_opt.load_state_dict(restored["optimizer"])
    result={
      "status":"HINDI_SYNTHETIC_BASELINE_PASS",
      "experiment":"hindi_utf8_byte_gru_baseline_v1",
      "device":"cpu","seed":seed,"training_steps":steps,
      "parameter_count":sum(p.numel() for p in model.parameters()),
      "tokenizer":{"type":"UTF-8 byte-level","vocabulary_size":256,"unicode_round_trip":True},
      "train_examples":len(TRAIN),"heldout_examples":len(EVAL),
      "initial_train_loss":initial_train,"final_train_loss":final_train,
      "initial_heldout_loss":initial_eval,"final_heldout_loss":final_eval,
      "heldout_used_for_gradient_updates":False,
      "checkpoint":{"sha256":digest,"reload_max_abs_logit_difference":diff,"optimizer_state_restored":True},
      "provenance":{"data":"original toy sentences embedded in script","external_data_used":False,"repository_corpus_used":False,"user_data_used":False},
      "production_training_gate":"CLOSED",
      "note":"Tiny synthetic baseline only; not evidence of broad Hindi competence, factual accuracy, or production readiness."
    }
    report_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return result

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--report-path",type=Path,required=True)
    ap.add_argument("--steps",type=int,default=160)
    ap.add_argument("--seed",type=int,default=7)
    a=ap.parse_args()
    print(json.dumps(run(a.report_path,a.steps,a.seed),ensure_ascii=False,indent=2))
