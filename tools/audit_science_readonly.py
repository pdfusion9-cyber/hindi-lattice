#!/usr/bin/env python3
"""Read-only Science checkpoint audit. Never trains, saves, or mutates weights."""
from __future__ import annotations
import argparse, collections, hashlib, json, re, sys
from pathlib import Path
import torch

# Make repository-root imports work when this file is executed as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.expert_inference import ExpertRunner, TOKEN_RE, pattern_id_for

SOURCE_RUN = "44795411621"
DOMAIN = "science"
MAX_SENTENCES = 400_000
SEQ_LEN = 16

def sha256_text(lines):
    h = hashlib.sha256()
    for line in lines:
        h.update(line.encode("utf-8")); h.update(b"\n")
    return h.hexdigest()

def sentence_words(line):
    return TOKEN_RE.findall(line)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--max-per-language", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out", default="science_readonly_audit.json")
    args=ap.parse_args()
    torch.set_num_threads(4)
    runner=ExpertRunner(args.checkpoint,"cpu")
    if runner.domain != DOMAIN or str(runner.ck.get("source_run_id")) != SOURCE_RUN:
        raise SystemExit("FAIL: unexpected expert domain/source provenance")
    vocab=runner.vocab
    vocab_to_id=runner.vocab_to_id
    stop={w for i,w in enumerate(vocab)
          if i < len(runner.vocab_counts) and int(runner.vocab_counts[i]) >= 300}
    counts=collections.Counter()
    train_hash_lines=[]
    accepted=0
    train_token_total=0
    train_known_total=0
    eval_rows={"Hindi":[], "English":[]}
    eval_seen={"Hindi":0,"English":0}
    # Recreate punah.py's first 400,000 accepted 6..60-token sentences.
    with open(args.corpus,encoding="utf-8",errors="ignore") as f:
        for line in f:
            ws=sentence_words(line)
            if not 6 <= len(ws) <= 60:
                continue
            if accepted < MAX_SENTENCES:
                accepted += 1
                counts.update(ws)
                train_token_total += len(ws)
                train_known_total += sum(w in vocab_to_id for w in ws)
                if len(train_hash_lines)<1000:
                    train_hash_lines.append(" ".join(ws))
                continue
            ids=[vocab_to_id[w] for w in ws if w in vocab_to_id]
            if len(ids) < SEQ_LEN+1:
                continue
            has_dev=any("\u0900" <= ch <= "\u097F" for w in ws for ch in w)
            has_latin=any(("a" <= ch.lower() <= "z") for w in ws for ch in w)
            lang="Hindi" if has_dev else ("English" if has_latin else None)
            if lang is None or len(eval_rows[lang]) >= args.max_per_language:
                continue
            # Pick a full 16-token context and a genuine following target.
            max_i=len(ids)-1
            i=SEQ_LEN + (eval_seen[lang] % max(1,max_i-SEQ_LEN+1))
            i=min(i,max_i)
            prefix_ids=ids[max(0,i-SEQ_LEN):i]
            if len(prefix_ids)!=SEQ_LEN:
                continue
            prefix_words=[vocab[t] for t in prefix_ids]
            target=ids[i]
            full_pattern=tuple(w if w in stop else "*" for w in ws)
            eval_rows[lang].append({"prefix":prefix_words,"target":target,"sentence_len":len(ws),"oov":len(ws)-len(ids),"position":i,"full_pattern":full_pattern})
            eval_seen[lang]+=1
            if all(len(v)>=args.max_per_language for v in eval_rows.values()):
                break
    if accepted != MAX_SENTENCES:
        raise SystemExit(f"FAIL: reconstructed training prefix has {accepted} valid sentences, expected {MAX_SENTENCES}")

    rebuilt=[w for w,_ in counts.most_common(len(vocab))]
    vocab_exact=(rebuilt==vocab)
    vocab_same_positions=sum(1 for i,w in enumerate(vocab) if i<len(rebuilt) and rebuilt[i]==w)
    roundtrip_tokens=0
    roundtrip_ok=0
    for w,c in counts.items():
        if w in vocab_to_id:
            idx=vocab_to_id[w]
            roundtrip_tokens+=c
            if vocab[idx]==w: roundtrip_ok+=c

    pattern_counts=collections.Counter()
    pattern_prefix_exact=0
    pattern_full_known=0
    slot_full_available=0
    slot_prefix_available=0
    full_pattern_exact=0
    full_pattern_slot_available=0
    candidate_recall=collections.Counter()
    top1_correct=collections.Counter()
    top1_covered=collections.Counter()
    candidate_perplexity_sum=collections.Counter()
    candidate_perplexity_n=collections.Counter()
    evaluated=collections.Counter()

    # Compare the deployed prefix-pattern path with a strict graph-only ablation.
    # Only rows with an exact prefix pattern need a second model forward pass.
    ablation_metrics = {
        mode: {
            "evaluated": collections.Counter(),
            "covered": collections.Counter(),
            "correct": collections.Counter(),
            "nll_sum": collections.Counter(),
            "nll_n": collections.Counter()
        }
        for mode in ("pattern_on", "pattern_off")
    }
    ablation_compare = {
        "candidate_lists_identical": collections.Counter(),
        "top1_agreement": collections.Counter()
    }
    for lang in ("Hindi","English"):
        rows=eval_rows[lang]
        for start in range(0,len(rows),args.batch_size):
            batch=rows[start:start+args.batch_size]
            tensors=[]
            for row in batch:
                prefix=row["prefix"]
                full_pattern=tuple(w if w in stop else "*" for w in prefix)
                # Training pattern is made from the complete source sentence;
                # the inference input here is only its 16-token prefix.
                # Compare the inference prefix mapping with stored full patterns.
                pid, exact=pattern_id_for(prefix,vocab,runner.vocab_counts,runner.patterns,runner.pattern_to_id)
                if exact: pattern_prefix_exact+=1
                full_pid=runner.pattern_to_id.get(row["full_pattern"])
                if full_pid is not None:
                    full_pattern_exact+=1
                    if (runner.patterns[full_pid],row["position"]) in runner.slot_candidates:
                        full_pattern_slot_available+=1
                if exact and (runner.patterns[pid],len(prefix)) in runner.slot_candidates:
                    slot_prefix_available+=1
                pattern_counts["prefix_exact" if exact else "prefix_fallback"]+=1
                prefix_ids=[vocab_to_id[w] for w in prefix]
                x,nb,pid_t,mask=runner._build_inputs(prefix_ids,pid)
                tensors.append((x,nb,pid_t,mask))
            x=torch.cat([z[0] for z in tensors],dim=0)
            nb=torch.cat([z[1] for z in tensors],dim=0)
            pid_t=torch.cat([z[2] for z in tensors],dim=0)
            mask=torch.cat([z[3] for z in tensors],dim=0)
            hidden=runner.model(x,nb,pid_t,mask)[:,-1,:]
            for j,row in enumerate(batch):
                prefix=row["prefix"]; target=row["target"]
                pid,exact=pattern_id_for(prefix,vocab,runner.vocab_counts,runner.patterns,runner.pattern_to_id)
                last_id=vocab_to_id[prefix[-1]]
                cand=runner._candidate_ids(last_id,pid,len(prefix),exact)
                evaluated[lang]+=1
                if target in cand:
                    candidate_recall[lang]+=1
                    top1_covered[lang]+=1
                    ct=torch.tensor(cand,dtype=torch.long)
                    scores=hidden[j] @ runner.model.tok(ct).T
                    best=int(scores.argmax().item())
                    if cand[best]==target: top1_correct[lang]+=1
                    loc=cand.index(target)
                    logp=torch.log_softmax(scores,dim=0)[loc]
                    candidate_perplexity_sum[lang]+=float((-logp).item())
                    candidate_perplexity_n[lang]+=1

                # Strict ablation: force pattern ID 0 and suppress pattern-slot
                # candidates. When the deployed prefix already falls back to 0,
                # reuse its hidden state; no extra model pass is necessary.
                cand_off=runner._candidate_ids(last_id,0,len(prefix),False)
                if pid == 0:
                    hidden_off=hidden[j]
                else:
                    x0,nb0,pid0,mask0=runner._build_inputs(
                        [vocab_to_id[w] for w in prefix],0)
                    hidden_off=runner.model(x0,nb0,pid0,mask0)[:,-1,:][0]
                off_ids=torch.tensor(cand_off,dtype=torch.long)
                scores_off=hidden_off @ runner.model.tok(off_ids).T
                best_on_id=cand[int(scores.argmax().item())]
                best_off_id=cand_off[int(scores_off.argmax().item())]
                if cand == cand_off:
                    ablation_compare["candidate_lists_identical"][lang]+=1
                if best_on_id == best_off_id:
                    ablation_compare["top1_agreement"][lang]+=1

                for mode,candidates,mode_scores in (
                    ("pattern_on",cand,scores),
                    ("pattern_off",cand_off,scores_off)
                ):
                    stats=ablation_metrics[mode]
                    stats["evaluated"][lang]+=1
                    if target in candidates:
                        stats["covered"][lang]+=1
                        loc=candidates.index(target)
                        nll=float((-torch.log_softmax(mode_scores,dim=0)[loc]).item())
                        stats["nll_sum"][lang]+=nll
                        stats["nll_n"][lang]+=1
                        if candidates[int(mode_scores.argmax().item())] == target:
                            stats["correct"][lang]+=1
    report={
      "status":"PASS",
      "mode":"READ_ONLY_NO_TRAINING_NO_CHECKPOINT_WRITES_NO_QUANTIZATION",
      "source_run_id":SOURCE_RUN,
      "checkpoint":str(args.checkpoint),
      "checkpoint_sha256":runner.ck.get("provenance",{}).get("source_checkpoint_sha256"),
      "corpus_path":str(args.corpus),
      "corpus_reconstruction":{
        "accepted_training_prefix_sentences":accepted,
        "expected_training_limit":MAX_SENTENCES,
        "first_1000_normalized_sentences_sha256":sha256_text(train_hash_lines),
        "warning":"Remote FineWeb/repository data may have changed since source run; byte-for-byte original corpus provenance is not established. Holdout is after the reconstructed 400,000-sentence training limit."
      },
      "tokenizer":{
        "tokenizer_regex":TOKEN_RE.pattern,
        "checkpoint_vocab_size":len(vocab),
        "rebuilt_vocab_size":len(rebuilt),
        "rebuilt_vocab_exact_order_match":vocab_exact,
        "same_position_vocab_rows":vocab_same_positions,
        "training_prefix_token_count":train_token_total,
        "training_prefix_in_vocab_tokens":train_known_total,
        "training_prefix_oov_rate":(1-train_known_total/max(train_token_total,1)),
        "known_token_reverse_roundtrip_count":roundtrip_tokens,
        "known_token_reverse_roundtrip_exact":roundtrip_ok,
        "reverse_roundtrip_pass":roundtrip_tokens==roundtrip_ok
      },
      "pattern_audit":{
        "stored_pattern_count":len(runner.patterns),
        "evaluation_prefix_exact_pattern_matches":pattern_prefix_exact,
        "evaluation_full_sentence_exact_pattern_matches":full_pattern_exact,
        "full_sentence_slot_candidates_at_target_position":full_pattern_slot_available,
        "evaluation_prefix_pattern_fallbacks":sum(pattern_counts.values())-pattern_prefix_exact,
        "prefix_slot_candidate_hits":slot_prefix_available,
        "note":"Training patterns are built from complete sentences; inference patterns are built from the available prefix. Exact-prefix match rate is reported, not assumed."
      },
      "prediction":{
        "languages":{},
        "metrics_definition":{
          "candidate_recall_at_configured_limit":"fraction of held-out targets present in the actual inference candidate list; target is never injected",
          "candidate_set_top1_accuracy_all":"correct candidate argmax divided by all held-out examples; missing target counts incorrect",
          "candidate_set_perplexity":"exp(mean negative log probability) only where target is present in candidate set; not full-vocabulary perplexity"
        }
      }
    }
    report["pattern_ablation"] = {
      "method":"same held-out prefixes and frozen checkpoint; pattern_off forces pattern ID 0 and disables pattern-slot candidates",
      "weights_or_checkpoint_modified":False,
      "modes":{},
      "comparison":{}
    }
    for mode in ("pattern_on","pattern_off"):
        report["pattern_ablation"]["modes"][mode]={"languages":{}}
        for lang in ("Hindi","English"):
            st=ablation_metrics[mode]
            n=st["evaluated"][lang]
            covered=st["covered"][lang]
            report["pattern_ablation"]["modes"][mode]["languages"][lang]={
              "examples":n,
              "candidate_recall":covered/max(n,1),
              "candidate_recall_count":covered,
              "candidate_set_top1_accuracy_all":st["correct"][lang]/max(n,1),
              "candidate_set_top1_correct":st["correct"][lang],
              "candidate_set_top1_accuracy_when_target_covered":st["correct"][lang]/max(covered,1),
              "candidate_set_perplexity_when_target_covered":float(__import__("math").exp(st["nll_sum"][lang]/max(st["nll_n"][lang],1))),
              "candidate_limit":runner.candidate_limit
            }
    for lang in ("Hindi","English"):
        n=evaluated[lang]; covered=candidate_recall[lang]
        report["prediction"]["languages"][lang]={
          "examples":n,
          "candidate_recall":covered/max(n,1),
          "candidate_recall_count":covered,
          "candidate_set_top1_accuracy_all":top1_correct[lang]/max(n,1),
          "candidate_set_top1_correct":top1_correct[lang],
          "candidate_set_top1_accuracy_when_target_covered":top1_correct[lang]/max(covered,1),
          "candidate_set_perplexity_when_target_covered":float(__import__("math").exp(candidate_perplexity_sum[lang]/max(candidate_perplexity_n[lang],1))),
          "candidate_limit":runner.candidate_limit
        }
    report["pattern_ablation"]["comparison"][lang]={
      "candidate_lists_identical":ablation_compare["candidate_lists_identical"][lang],
      "top1_predictions_identical":ablation_compare["top1_agreement"][lang],
      "examples":evaluated[lang]
    }
    Path(args.out).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
