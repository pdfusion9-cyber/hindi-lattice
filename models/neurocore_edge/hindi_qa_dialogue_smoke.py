#!/usr/bin/env python3
"""Synthetic-only supervised Hindi QA/dialogue smoke test.

Uses original toy prompts and answers embedded here. No external, repository,
or user data is read. This is a small pipeline test, not a Hindi assistant.
"""
from __future__ import annotations
import argparse, hashlib, json, random, tempfile
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

BYTE_VOCAB, EOS, BOS, VOCAB_SIZE = 256, 256, 257, 258
SEED = 11
TRAIN_PAIRS = [
("प्रश्न: भारत की राजधानी क्या है? उत्तर:", "नई दिल्ली"),
("प्रश्न: भारत का राजधानी शहर बताइए। उत्तर:", "नई दिल्ली"),
("प्रश्न: भारत की राजधानी कौन-सी है? उत्तर:", "नई दिल्ली"),
("प्रश्न: नेपाल की राजधानी क्या है? उत्तर:", "काठमांडू"),
("प्रश्न: नेपाल का राजधानी शहर बताइए। उत्तर:", "काठमांडू"),
("प्रश्न: नेपाल की राजधानी कौन-सी है? उत्तर:", "काठमांडू"),
("प्रश्न: जापान की राजधानी क्या है? उत्तर:", "टोक्यो"),
("प्रश्न: जापान का राजधानी शहर बताइए। उत्तर:", "टोक्यो"),
("प्रश्न: जापान की राजधानी कौन-सी है? उत्तर:", "टोक्यो"),
("प्रश्न: पानी का रासायनिक सूत्र क्या है? उत्तर:", "H2O"),
("प्रश्न: पानी का रासायनिक संकेत बताइए। उत्तर:", "H2O"),
("प्रश्न: जल का रासायनिक सूत्र क्या होता है? उत्तर:", "H2O"),
("प्रश्न: एक सप्ताह में कितने दिन होते हैं? उत्तर:", "सात दिन"),
("प्रश्न: सप्ताह के कुल दिनों की संख्या बताइए। उत्तर:", "सात दिन"),
("प्रश्न: सप्ताह में कुल कितने दिन गिने जाते हैं? उत्तर:", "सात दिन"),
("संवाद: नमस्ते। उत्तर:", "नमस्ते! मैं आपकी सहायता के लिए तैयार हूँ।"),
("संवाद: नमस्ते जी। उत्तर:", "नमस्ते! मैं आपकी सहायता के लिए तैयार हूँ।"),
("संवाद: किसी का अभिवादन कैसे करेंगे? उत्तर:", "नमस्ते! मैं आपकी सहायता के लिए तैयार हूँ।"),
("संवाद: धन्यवाद। उत्तर:", "आपका स्वागत है।"),
("संवाद: बहुत शुक्रिया। उत्तर:", "आपका स्वागत है।"),
("संवाद: आभार व्यक्त करने पर क्या कहेंगे? उत्तर:", "आपका स्वागत है।"),
("संवाद: फिर मिलेंगे। उत्तर:", "अलविदा! आपका दिन शुभ हो।"),
("संवाद: अभी विदा लेता हूँ। उत्तर:", "अलविदा! आपका दिन शुभ हो।"),
("संवाद: बातचीत समाप्त करते समय क्या कहेंगे? उत्तर:", "अलविदा! आपका दिन शुभ हो।"),
]
HELDOUT_PAIRS = [
("प्रश्न: भारत की राजधानी किस नगर को कहा जाता है? उत्तर:", "नई दिल्ली"),
("प्रश्न: नेपाल की राजधानी किस शहर में स्थित है? उत्तर:", "काठमांडू"),
("प्रश्न: जापान का मुख्य राजधानी नगर कौन-सा है? उत्तर:", "टोक्यो"),
("प्रश्न: H2O किस पदार्थ का सूत्र है? उत्तर:", "पानी"),
("प्रश्न: सात दिनों के समूह को क्या कहते हैं? उत्तर:", "एक सप्ताह"),
("संवाद: आपसे मिलकर खुशी हुई। उत्तर:", "मुझे भी आपसे मिलकर खुशी हुई।"),
("संवाद: कृपया सहायता कीजिए। उत्तर:", "ज़रूर, बताइए आपको किस विषय में सहायता चाहिए।"),
]

def encode(s: str) -> list[int]:
    return list(s.encode("utf-8"))

class HindiQADialogueModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, 32)
        self.encoder = nn.GRU(32, 64, batch_first=True)
        self.decoder = nn.GRU(32, 64, batch_first=True)
        self.output = nn.Linear(64, VOCAB_SIZE)
    def encode_prompt(self, ids):
        _, hidden = self.encoder(self.embedding(ids.view(1, -1)))
        return hidden
    def forward_teacher(self, prompt_ids, answer_ids):
        hidden = self.encode_prompt(prompt_ids)
        din = torch.cat([torch.tensor([BOS]), answer_ids[:-1]]).view(1, -1)
        decoded, _ = self.decoder(self.embedding(din), hidden)
        return self.output(decoded).squeeze(0)
    @torch.no_grad()
    def generate(self, prompt: str, max_bytes: int = 96) -> str:
        self.eval()
        ids = torch.tensor(encode(prompt) + [EOS])
        hidden = self.encode_prompt(ids)
        token = torch.tensor([[BOS]])
        generated = []
        for _ in range(max_bytes):
            out, hidden = self.decoder(self.embedding(token), hidden)
            next_id = int(self.output(out[:, -1]).argmax(-1).item())
            if next_id == EOS:
                break
            if 0 <= next_id < BYTE_VOCAB:
                generated.append(next_id)
            token = torch.tensor([[next_id]])
        return bytes(generated).decode("utf-8", errors="replace").strip()

def pair_loss(model, prompt, answer):
    p = torch.tensor(encode(prompt) + [EOS], dtype=torch.long)
    a = torch.tensor(encode(answer) + [EOS], dtype=torch.long)
    return F.cross_entropy(model.forward_teacher(p, a), a)

def evaluate(model, pairs):
    model.eval(); losses=[]; exact=0; samples=[]
    for prompt, expected in pairs:
        with torch.no_grad():
            losses.append(float(pair_loss(model, prompt, expected).item()))
        predicted=model.generate(prompt)
        match=predicted.strip()==expected.strip()
        exact += int(match)
        samples.append({"prompt":prompt,"expected":expected,"predicted":predicted,"exact_match":match})
    return {"mean_teacher_forced_loss":sum(losses)/len(losses),
            "exact_match_count":exact,"example_count":len(pairs),
            "exact_match_accuracy":exact/len(pairs),"samples":samples}

def run_experiment(report_path: Path, steps: int=500, seed: int=SEED):
    if steps < 100: raise ValueError("steps must be >=100")
    torch.manual_seed(seed); random.seed(seed); torch.set_num_threads(1)
    for p,a in TRAIN_PAIRS+HELDOUT_PAIRS:
        if bytes(encode(p)).decode("utf-8") != p or bytes(encode(a)).decode("utf-8") != a:
            raise RuntimeError("UTF-8 round-trip failed")
    model=HindiQADialogueModel()
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.004)
    initial_train=evaluate(model,TRAIN_PAIRS)
    initial_heldout=evaluate(model,HELDOUT_PAIRS)
    rng=random.Random(seed)
    last_grad=0.0
    for _ in range(steps):
        prompt,answer=TRAIN_PAIRS[rng.randrange(len(TRAIN_PAIRS))]
        optimizer.zero_grad(set_to_none=True)
        loss=pair_loss(model,prompt,answer)
        if not torch.isfinite(loss): raise RuntimeError("non-finite loss")
        loss.backward()
        grads=[p.grad for p in model.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):
            raise RuntimeError("missing/non-finite gradients")
        last_grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.0))
        if not torch.isfinite(torch.tensor(last_grad)): raise RuntimeError("non-finite grad norm")
        optimizer.step()
    final_train=evaluate(model,TRAIN_PAIRS)
    final_heldout=evaluate(model,HELDOUT_PAIRS)
    if final_train["mean_teacher_forced_loss"] >= initial_train["mean_teacher_forced_loss"]:
        raise RuntimeError("training loss did not decrease")
    ref_prompt=TRAIN_PAIRS[0][0]
    ref_output=model.generate(ref_prompt)
    state={"model_state":model.state_dict(),"optimizer_state":optimizer.state_dict(),
           "steps":steps,"seed":seed,"vocab_size":VOCAB_SIZE,"eos_id":EOS,"bos_id":BOS}
    report_path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        ckpt=Path(td)/"hindi_qa_dialogue.pt"
        torch.save(state,ckpt)
        digest=hashlib.sha256(ckpt.read_bytes()).hexdigest()
        restored=torch.load(ckpt,map_location="cpu",weights_only=True)
        clone=HindiQADialogueModel()
        clone.load_state_dict(restored["model_state"],strict=True)
        clone_opt=torch.optim.AdamW(clone.parameters(),lr=0.004)
        clone_opt.load_state_dict(restored["optimizer_state"])
        if clone.generate(ref_prompt)!=ref_output: raise RuntimeError("checkpoint generation mismatch")
    report={
      "status":"HINDI_QA_DIALOGUE_SMOKE_PASS","experiment":"supervised_hindi_qa_dialogue_gru_v1",
      "device":"cpu","seed":seed,"training_steps":steps,
      "parameter_count":sum(p.numel() for p in model.parameters()),
      "tokenizer":{"type":"UTF-8 byte-level","vocabulary_size":VOCAB_SIZE,"unicode_round_trip":True},
      "train_pairs":len(TRAIN_PAIRS),"heldout_pairs":len(HELDOUT_PAIRS),
      "heldout_used_for_gradient_updates":False,
      "initial_train":initial_train,"final_train":final_train,
      "initial_heldout":initial_heldout,"final_heldout":final_heldout,
      "last_gradient_norm":last_grad,
      "checkpoint":{"sha256":digest,"optimizer_state_restored":True,"generation_equal_after_reload":True},
      "provenance":{"kind":"synthetic_only","external_data_used":False,
                    "repository_corpus_used":False,"user_data_used":False},
      "production_training_gate":"CLOSED",
      "note":"Tiny synthetic smoke test only; held-out exact match measures toy prompt generalization, not broad Hindi competence."
    }
    report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return report

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--report-path",type=Path,required=True)
    ap.add_argument("--steps",type=int,default=500)
    ap.add_argument("--seed",type=int,default=SEED)
    a=ap.parse_args()
    print(json.dumps(run_experiment(a.report_path,a.steps,a.seed),ensure_ascii=False,indent=2))
