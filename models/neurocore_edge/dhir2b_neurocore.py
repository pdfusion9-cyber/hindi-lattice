"""Dhīr-2B native backbone plus 20 NeuroCore residual adapter experts."""
from __future__ import annotations
import hashlib, json, os, re, shutil
from pathlib import Path
from typing import Any, Optional
import torch
from torch import nn

ARCHITECTURE_VERSION = "0.4.0-dhir2b-adapters"
DEFAULT_MODEL_ID = "zenteiq-ai/Dhir-2B"
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
REVIEWED_UPSTREAM_SHA256 = {
    "configuration_dhir.py": "9ddceca88e4d7780ccab5d194d93bb1bf557238e4f71b5b7abe4d2b540ba5cee",
    "modeling_dhir.py": "9836ca6d3996428bbcd7dc768e885268707dbb7d3dc04ca21df494db893bfc45",
}

def verify_upstream_code(model_id, revision, destination=None):
    from huggingface_hub import hf_hub_download
    observed={}
    for filename,expected in REVIEWED_UPSTREAM_SHA256.items():
        source=Path(hf_hub_download(model_id,filename=filename,revision=revision))
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != expected:
            raise RuntimeError("upstream Dhīr source hash is not the reviewed version: "+filename)
        observed[filename]=digest
        if destination is not None:
            destination.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,destination/filename)
    return observed

def cfg_value(config: Any, key: str, default=None):
    return config.get(key, default) if isinstance(config, dict) else getattr(config, key, default)

def config_facts(config: Any) -> dict:
    keys=("model_type","architectures","vocab_size","hidden_size","intermediate_size",
          "num_hidden_layers","num_attention_heads","num_key_value_heads","head_dim",
          "max_position_embeddings","rope_theta","rms_norm_eps","hidden_act",
          "tie_word_embeddings")
    return {key:cfg_value(config,key) for key in keys if cfg_value(config,key) is not None}

def validate_dhir_config(config: Any, tokenizer: Any) -> dict:
    facts=config_facts(config)
    required=("vocab_size","hidden_size","intermediate_size","num_hidden_layers",
              "num_attention_heads","num_key_value_heads","max_position_embeddings")
    missing=[key for key in required if key not in facts]
    if missing: raise ValueError(f"Dhīr config missing fields: {missing}")
    expected={"hidden_size":2048,"num_hidden_layers":28,
              "num_attention_heads":16,"num_key_value_heads":8}
    for key,value in expected.items():
        if int(facts[key]) != value:
            raise ValueError(f"incompatible Dhīr config: {key}={facts[key]}, expected {value}")
    if int(facts["max_position_embeddings"]) < 16384:
        raise ValueError("checkpoint context is shorter than 16384")
    if "rope_theta" in facts and abs(float(facts["rope_theta"])-1_000_000.0)>1e-3:
        raise ValueError("checkpoint RoPE theta mismatch")
    if facts.get("tie_word_embeddings") is not True:
        raise ValueError("checkpoint must use tied embeddings")
    if facts.get("hidden_act") not in (None,"silu"):
        raise ValueError("checkpoint activation is not SiLU/SwiGLU-compatible")
    size=len(tokenizer)
    if size != int(facts["vocab_size"]):
        raise ValueError(f"tokenizer/config vocabulary mismatch: {size} vs {facts['vocab_size']}")
    vocab=tokenizer.get_vocab()
    token_ids = [int(token_id) for token_id in vocab.values()]
    if (not token_ids or len(token_ids) != size
            or set(token_ids) != set(range(size))):
        raise ValueError("tokenizer IDs must cover the full contiguous model vocabulary")
    facts["tokenizer_length"]=size
    return facts

def find_decoder_layers(backbone: nn.Module):
    for path in ("model.layers","transformer.h","gpt_neox.layers","decoder.layers","layers"):
        obj=backbone
        try:
            for part in path.split("."): obj=getattr(obj,part)
        except AttributeError: continue
        if isinstance(obj,(nn.ModuleList,list,tuple)) and len(obj) and all(isinstance(x,nn.Module) for x in obj):
            return obj
    raise ValueError("decoder block list not found in supported Transformers layouts")

class BottleneckExpert(nn.Module):
    def __init__(self,width:int,rank:int):
        super().__init__()
        if width<1 or rank<1: raise ValueError("width and rank must be positive")
        self.down=nn.Linear(width,rank,bias=False)
        self.up=nn.Linear(rank,width,bias=False)
        nn.init.kaiming_uniform_(self.down.weight,a=5**0.5)
        nn.init.zeros_(self.up.weight)
    def forward(self,x):
        return self.up(torch.nn.functional.silu(self.down(x)))

class NeuroCoreExpertBank(nn.Module):
    def __init__(self,width:int,num_experts=20,top_k=2,rank=8):
        super().__init__()
        if not 1<=top_k<=num_experts: raise ValueError("invalid top_k")
        self.width=width
        self.num_experts=num_experts
        self.top_k=top_k
        self.router=nn.Linear(width,num_experts,bias=False)
        nn.init.zeros_(self.router.weight)
        self.experts=nn.ModuleList([BottleneckExpert(width,rank) for _ in range(num_experts)])
    def forward(self,x,expert_id:Optional[int]=None):
        if expert_id is not None:
            if isinstance(expert_id,bool) or not isinstance(expert_id,int) or not 0<=expert_id<self.num_experts:
                raise ValueError("expert_id outside configured range")
            return x+self.experts[expert_id](x)
        weights,indices=torch.topk(self.router(x.float()),self.top_k,dim=-1)
        weights=torch.softmax(weights,dim=-1).to(x.dtype)
        flat=x.reshape(-1,self.width)
        idx=indices.reshape(-1,self.top_k)
        w=weights.reshape(-1,self.top_k)
        delta=torch.zeros_like(flat)
        for expert_id,expert in enumerate(self.experts):
            positions,slots=torch.where(idx==expert_id)
            if positions.numel():
                values=expert(flat.index_select(0,positions))
                delta.index_add_(0,positions,values*w[positions,slots].unsqueeze(-1))
        return x+delta.reshape_as(x)

def replace_hidden(output,replacement):
    if isinstance(output,tuple): return (replacement,*output[1:])
    if isinstance(output,list): return [replacement,*output[1:]]
    if isinstance(output,torch.Tensor): return replacement
    raise TypeError("decoder layer output must be Tensor, tuple or list")

class Dhir2BNeuroCore(nn.Module):
    def __init__(self,backbone:nn.Module,model_id:str,revision:str,
                 num_experts=20,top_k=2,rank=8,source_facts=None):
        super().__init__()
        if not COMMIT_RE.fullmatch(revision): raise ValueError("revision must be immutable 40-char commit SHA")
        self.backbone=backbone
        self.base_model_id=model_id
        self.base_revision=revision
        self.facts=source_facts or config_facts(backbone.config)
        self.hidden_size=int(self.facts["hidden_size"])
        self.vocab_size=int(self.facts["vocab_size"])
        layers=find_decoder_layers(backbone)
        if len(layers)!=28 or int(self.facts["num_hidden_layers"])!=28:
            raise ValueError("Dhīr adapter path requires 28 independent decoder blocks")
        self.num_experts=num_experts
        self.rank=rank
        self.adapter_layers=nn.ModuleList([
            NeuroCoreExpertBank(self.hidden_size,num_experts,top_k,rank) for _ in range(28)
        ])
        self._forced_expert_id=None
        self._handles=[]
        for i,layer in enumerate(layers):
            self._handles.append(layer.register_forward_hook(self._hook(i)))
        self.freeze_backbone()
    def _hook(self,index):
        def apply(_module,_inputs,output):
            hidden=output[0] if isinstance(output,(tuple,list)) else output
            return replace_hidden(output,self.adapter_layers[index](hidden,self._forced_expert_id))
        return apply
    def freeze_backbone(self):
        self.backbone.eval()
        for p in self.backbone.parameters(): p.requires_grad_(False)
    def prepare_expert_training(self,expert_id:int):
        if isinstance(expert_id,bool) or not isinstance(expert_id,int) or not 0<=expert_id<self.num_experts:
            raise ValueError("invalid expert_id")
        self.freeze_backbone()
        selected=[]
        for layer in self.adapter_layers:
            layer.router.requires_grad_(False)
            for idx,expert in enumerate(layer.experts):
                for p in expert.parameters():
                    p.requires_grad_(idx==expert_id)
                    if idx==expert_id: selected.append(p)
        return selected
    def forward(self,*args,expert_id=None,**kwargs):
        if expert_id is not None and (isinstance(expert_id,bool) or not isinstance(expert_id,int) or not 0<=expert_id<self.num_experts):
            raise ValueError("invalid expert_id")
        previous=self._forced_expert_id
        self._forced_expert_id=expert_id
        try: return self.backbone(*args,**kwargs)
        finally: self._forced_expert_id=previous
    def train(self,mode=True):
        super().train(mode)
        self.backbone.eval()
        self.adapter_layers.train(mode)
        return self
    def load_expert_state_dict(self,expert_id:int,incoming:dict):
        if isinstance(expert_id,bool) or not isinstance(expert_id,int) or not 0<=expert_id<self.num_experts:
            raise ValueError("invalid expert_id")
        expected={f"layer_{i}.{key}":value for i,layer in enumerate(self.adapter_layers)
                  for key,value in layer.experts[expert_id].state_dict().items()}
        if set(incoming)!=set(expected):
            raise ValueError("expert checkpoint tensor key set mismatch")
        for key,target in expected.items():
            value=incoming[key]
            if not isinstance(value,torch.Tensor) or value.shape!=target.shape:
                raise ValueError("expert checkpoint tensor shape mismatch: "+key)
            if not bool(torch.isfinite(value).all()):
                raise ValueError("expert checkpoint contains non-finite tensor: "+key)
        for i,layer in enumerate(self.adapter_layers):
            prefix=f"layer_{i}."
            state={key[len(prefix):]:incoming[key] for key in incoming if key.startswith(prefix)}
            layer.experts[expert_id].load_state_dict(state,strict=True)
    def save_checkpoint(self,directory:str,tokenizer):
        root=Path(directory); root.mkdir(parents=True,exist_ok=True)
        self.backbone.save_pretrained(str(root/"backbone"),safe_serialization=True,max_shard_size="2GB")
        upstream_hashes=verify_upstream_code(self.base_model_id,self.base_revision,root/"backbone")
        tokenizer.save_pretrained(str(root/"tokenizer"))
        torch.save(self.adapter_layers.state_dict(),root/"adapters.pt")
        manifest={"schema_version":1,"architecture_version":ARCHITECTURE_VERSION,
          "status":"IMPORTED_BASE_UNTRAINED_EXPERTS","base_model_id":self.base_model_id,
          "base_revision":self.base_revision,"config":self.facts,"vocab_size":self.vocab_size,
          "num_hidden_layers":28,"num_experts":self.num_experts,"adapter_rank":self.rank,
          "experts_per_token":self.adapter_layers[0].top_k,
          "upstream_code_sha256":upstream_hashes,
          "base_trainable_during_expert_training":False,
          "router_status":"UNTRAINED_REQUIRES_CALIBRATION"}
        (root/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
        return manifest
    @classmethod
    def load_checkpoint(cls,directory:str):
        root=Path(directory)
        manifest=json.loads((root/"manifest.json").read_text())
        if manifest.get("architecture_version")!=ARCHITECTURE_VERSION or manifest.get("schema_version")!=1:
            raise ValueError("unsupported Dhīr NeuroCore checkpoint")
        try:
            from transformers import AutoModelForCausalLM,AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("install requirements-dhir2b.txt") from exc
        local_backbone=root/"backbone"
        expected_code=manifest.get("upstream_code_sha256",{})
        if expected_code != REVIEWED_UPSTREAM_SHA256:
            raise ValueError("checkpoint does not contain the reviewed Dhīr source hashes")
        for filename,digest in expected_code.items():
            source=local_backbone/filename
            if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest()!=digest:
                raise ValueError("local Dhīr source file hash mismatch: "+filename)
        backbone=AutoModelForCausalLM.from_pretrained(
            str(local_backbone),local_files_only=True,trust_remote_code=True
        )
        tokenizer=AutoTokenizer.from_pretrained(
            str(root/"tokenizer"),local_files_only=True,trust_remote_code=False
        )
        facts=validate_dhir_config(backbone.config,tokenizer)
        for key,value in manifest["config"].items():
            if facts.get(key)!=value:
                raise ValueError("checkpoint config/tokenizer provenance mismatch: "+key)
        model=cls(backbone,manifest["base_model_id"],manifest["base_revision"],
                  num_experts=int(manifest["num_experts"]),
                  top_k=int(manifest["experts_per_token"]),
                  rank=int(manifest["adapter_rank"]),source_facts=facts)
        state=torch.load(root/"adapters.pt",map_location="cpu",weights_only=True)
        model.adapter_layers.load_state_dict(state,strict=True)
        return model,tokenizer,manifest
    @classmethod
    def import_pretrained(cls,model_id=DEFAULT_MODEL_ID,revision=None,rank=8,dtype=torch.bfloat16):
        if not revision or not COMMIT_RE.fullmatch(revision):
            raise ValueError("pin model revision to a 40-character commit SHA")
        try:
            from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer
        except ImportError as exc: raise RuntimeError("install requirements-dhir2b.txt") from exc
        code_hashes=verify_upstream_code(model_id,revision)
        config=AutoConfig.from_pretrained(model_id,revision=revision,trust_remote_code=True)
        tokenizer=AutoTokenizer.from_pretrained(
            model_id,revision=revision,trust_remote_code=False
        )
        facts=validate_dhir_config(config,tokenizer)
        backbone=AutoModelForCausalLM.from_pretrained(
            model_id,revision=revision,trust_remote_code=True,
            torch_dtype=dtype,low_cpu_mem_usage=True
        )
        return cls(backbone,model_id,revision,rank=rank,source_facts=facts),tokenizer
