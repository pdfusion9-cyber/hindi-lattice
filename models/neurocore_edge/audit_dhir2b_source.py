"""Read-only Dhīr-2B Hub metadata audit. No upstream Python is executed."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
from tokenizers import Tokenizer

REPO_ID=os.environ.get("DHIR_MODEL_ID","zenteiq-ai/Dhir-2B")
REPORT_PATH=Path(os.environ.get("DHIR_REPORT_PATH","dhir2b_source_audit.json"))

def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def read_json(repo_id,revision,filename):
    local=Path(hf_hub_download(repo_id,filename=filename,revision=revision))
    return json.loads(local.read_text(encoding="utf-8")),local

def main():
    api=HfApi()
    info=api.model_info(REPO_ID,files_metadata=True)
    revision=info.sha
    if not revision or len(revision)!=40:
        raise RuntimeError("Hub did not return an immutable 40-character commit SHA")

    config,config_path=read_json(REPO_ID,revision,"config.json")
    tokenizer_json,tokenizer_path=read_json(REPO_ID,revision,"tokenizer.json")
    tokenizer=Tokenizer.from_file(str(tokenizer_path))
    vocabulary=tokenizer.get_vocab(with_added_tokens=True)
    token_ids=[int(value) for value in vocabulary.values()]
    vocab_size=len(vocabulary)

    facts={
      key:config.get(key) for key in (
        "model_type","architectures","vocab_size","hidden_size","intermediate_size",
        "num_hidden_layers","num_attention_heads","num_key_value_heads","head_dim",
        "max_position_embeddings","rope_theta","rms_norm_eps","hidden_act",
        "tie_word_embeddings","auto_map"
      ) if key in config
    }
    checks={
      "hidden_size_2048":facts.get("hidden_size")==2048,
      "layers_28":facts.get("num_hidden_layers")==28,
      "query_heads_16":facts.get("num_attention_heads")==16,
      "kv_heads_8":facts.get("num_key_value_heads")==8,
      "context_at_least_16384":isinstance(facts.get("max_position_embeddings"),int)
          and facts["max_position_embeddings"]>=16384,
      "rope_theta_1e6":facts.get("rope_theta") is not None
          and abs(float(facts["rope_theta"])-1_000_000.0)<1e-3,
      "silu_activation":facts.get("hidden_act")=="silu",
      "tied_embeddings":facts.get("tie_word_embeddings") is True,
      "intermediate_size_present":isinstance(facts.get("intermediate_size"),int)
          and facts["intermediate_size"]>0,
      "tokenizer_len_matches_config":vocab_size==facts.get("vocab_size"),
      "tokenizer_ids_contiguous":len(token_ids)==vocab_size and set(token_ids)==set(range(vocab_size)),
    }

    siblings=[]
    python_sources=[]
    for sibling in info.siblings:
      name=sibling.rfilename
      siblings.append({"name":name,"size":getattr(sibling,"size",None),
          "lfs_oid":getattr(getattr(sibling,"lfs",None),"oid",None)})
      if name.endswith(".py"):
        source_path=Path(hf_hub_download(REPO_ID,filename=name,revision=revision))
        content=source_path.read_text(encoding="utf-8",errors="replace")
        python_sources.append({"name":name,"size_bytes":source_path.stat().st_size,
          "sha256":sha256(source_path),"source_preview":content[:4000],
          "executed":False,"review_status":"REQUIRES_MANUAL_REVIEW"})

    optional_hashes={}
    for name in ("tokenizer_config.json","special_tokens_map.json","chat_template.jinja"):
      if any(item["name"]==name for item in siblings):
        local=Path(hf_hub_download(REPO_ID,filename=name,revision=revision))
        optional_hashes[name]=sha256(local)

    index_name=next((name for name in ("model.safetensors.index.json","pytorch_model.bin.index.json")
                     if any(item["name"]==name for item in siblings)),None)
    index_checks={"index_present":index_name is not None}
    weight_map={}
    if index_name:
      index,_=read_json(REPO_ID,revision,index_name)
      weight_map=index.get("weight_map",{})
      keys=list(weight_map)
      index_checks["embedding_tensor_present"]=any(
        name.endswith(("embed_tokens.weight","wte.weight")) for name in keys)
      for norm in ("q_norm","k_norm"):
        matched=[any(f"layers.{i}.self_attn.{norm}." in key for key in keys) for i in range(28)]
        index_checks[f"{norm}_present_all_28_layers"]=all(matched)

    report={
      "status":"PASS" if all(checks.values()) else "FAIL",
      "repo_id":REPO_ID,"revision":revision,
      "license":getattr(info.card_data,"license",None) if info.card_data else None,
      "config":facts,"tokenizer_length":vocab_size,
      "tokenizer_vocab_sha256":hashlib.sha256(
        json.dumps(vocabulary,sort_keys=True,ensure_ascii=False).encode("utf-8")).hexdigest(),
      "metadata_file_sha256":{"config.json":sha256(config_path),"tokenizer.json":sha256(tokenizer_path),
                              **optional_hashes},
      "files":siblings,"architecture_checks":checks,
      "weight_index_checks":index_checks,
      "remote_python_sources":python_sources,
      "custom_code_policy":"UPSTREAM_PYTHON_NOT_EXECUTED; inspect source before allowing any custom-code path",
      "weight_shards_downloaded":False,
      "note":"Config/tokenizer/index metadata only; multi-GB weight shards are not downloaded."
    }
    REPORT_PATH.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if not all(checks.values()): raise SystemExit("Dhīr config/tokenizer checks failed")

if __name__=="__main__": main()
