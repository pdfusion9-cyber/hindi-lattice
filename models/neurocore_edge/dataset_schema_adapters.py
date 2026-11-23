#!/usr/bin/env python3
"""Loss-aware text adapters. Multimodal and unknown schemas fail closed."""
from __future__ import annotations
import json
from typing import Any, Iterable
ADAPTER_VERSION = "1.0.0"
class SchemaAdapterError(ValueError): pass
SIDECAR_ONLY = {
 "HuggingFaceM4/DocumentVQA":"document image payload requires a vision encoder and page loader",
 "ai4bharat/indicvoices_r":"audio, speaker rights and a TTS/vocoder sidecar are required",
 "mythicinfinity/libritts":"audio, speaker rights and a TTS/vocoder sidecar are required",
 "ajimeno/PubTabNet":"page image and table-cell geometry require a table/vision sidecar",
 "allenai/olmOCR-mix-0225":"page images and layout coordinates require a vision/OCR sidecar"}
def _text(v:Any)->str:
 return v.strip() if isinstance(v,str) else str(v) if isinstance(v,(int,float,bool)) else ""
def _first(r:dict,ks:Iterable[str])->str:
 for k in ks:
  v=_text(r.get(k))
  if v:return v
 return ""
def _json(v:Any)->str:
 try:return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(",",":"))
 except (TypeError,ValueError) as e:raise SchemaAdapterError(f"nested field is not JSON serializable: {e}") from e
def _pair(task:str,prompt:str,answer:str)->str:
 if not prompt or not answer:raise SchemaAdapterError(f"{task} row needs both prompt/source and target/answer")
 return f"### TASK: {task}\n### INPUT\n{prompt}\n### OUTPUT\n{answer}"
def _chat(r:dict)->str:
 raw=r.get("conversations",r.get("messages",r.get("conversation")))
 if isinstance(raw,str):dialogue=raw.strip()
 elif isinstance(raw,list):
  parts=[]
  for i,m in enumerate(raw):
   if not isinstance(m,dict):raise SchemaAdapterError(f"conversation message {i} is not an object")
   role=_first(m,("from","role","speaker","author")).lower();body=_first(m,("value","content","text","message"))
   if not role or not body:raise SchemaAdapterError(f"conversation message {i} lacks role or content")
   role={"human":"user","gpt":"assistant","bot":"assistant"}.get(role,role)
   parts.append(f"### {role.upper()}\n{body}")
  dialogue="\n".join(parts)
 else:dialogue=""
 if not dialogue:raise SchemaAdapterError("chat row has no non-empty conversations/messages")
 tools=r.get("tools")
 if tools not in (None,"",[],{}):
  if not isinstance(tools,(str,list,dict)):raise SchemaAdapterError("tools must be JSON/text")
  dialogue="### TOOL SCHEMAS\n"+(tools if isinstance(tools,str) else _json(tools))+"\n"+dialogue
 return dialogue
def _gov(r:dict)->str:
 return _pair("document_summarization",_first(r,("report","document","article","text","content")),_first(r,("summary","highlights","target")))
def _gov_qs(r:dict)->list[str]:
 sections=r.get("document_sections",r.get("sections"))
 if not isinstance(sections,list):raise SchemaAdapterError("gov_report_qs requires document_sections list")
 out_sections=[]
 for i,s in enumerate(sections):
  if isinstance(s,str):title,body=f"Section {i+1}",s.strip()
  elif isinstance(s,dict):title=_first(s,("heading","title","section_title","name")) or f"Section {i+1}";body=_first(s,("text","content","body","paragraph"))
  else:raise SchemaAdapterError(f"document section {i} has invalid type")
  if not body:raise SchemaAdapterError(f"document section {i} has no body")
  out_sections.append(f"## {title}\n{body}")
 pairs=r.get("question_summary_pairs",r.get("qa_pairs",r.get("pairs")))
 if not out_sections or not isinstance(pairs,list) or not pairs:raise SchemaAdapterError("gov_report_qs requires question_summary_pairs")
 doc="\n\n".join(out_sections);out=[]
 for i,p in enumerate(pairs):
  if not isinstance(p,dict):raise SchemaAdapterError(f"question-summary pair {i} is not an object")
  out.append(_pair("long_document_planning_and_summary",doc+"\n\n"+_first(p,("question","query")),_first(p,("summary","answer","response"))))
 return out
def _translation(r:dict)->str:
 src=_first(r,("source","src","sentence","translation_source"));tgt=_first(r,("target","tgt","translation","sentence_target"))
 sl=_first(r,("source_lang","src_lang","language1","lang1"));tl=_first(r,("target_lang","tgt_lang","language2","lang2"))
 return _pair("translation",(f"[{sl}] " if sl else "")+src,(f"[{tl}] " if tl else "")+tgt)
def _legal(r:dict)->str:
 q=_first(r,("question","query","instruction","issue"));ctx=_first(r,("context","facts","case_text","judgment","judgement","opinion","document","text"));a=_first(r,("answer","response","holding","reasoning","rationale","output"))
 return _pair("legal_research_not_legal_advice",(ctx+"\n\n" if ctx else "")+q,a)
def adapt_row(dataset_id:str,row:dict,expert_id:int)->list[str]:
 if not isinstance(row,dict):raise SchemaAdapterError("dataset row must be an object")
 if dataset_id in SIDECAR_ONLY:raise SchemaAdapterError(f"SIDECAR_REQUIRED: {dataset_id}: {SIDECAR_ONLY[dataset_id]}")
 if dataset_id=="grammarly/coedit":
  ins=_first(row,("instruction","edit_type","task")) or "Revise while preserving meaning."
  return [_pair("document_editing",ins+"\n\n"+_first(row,("src","source","original","input","text")),_first(row,("tgt","target","edited","correction","output")))]
 if dataset_id in {"teknium/OpenHermes-2.5","hiyouga/glaive-function-calling-v2-sharegpt"}:return [_chat(row)]
 if dataset_id in {"ccdv/govreport-summarization","FiscalNote/billsum","pszemraj/govreport-summarization-8192"}:return [_gov(row)]
 if dataset_id=="launch/gov_report_qs":return _gov_qs(row)
 if dataset_id in {"facebook/flores","openlanguagedata/flores_plus"}:return [_translation(row)]
 if dataset_id in {"HuggingFaceFW/fineweb-edu","ai4bharat/sangraha","ai4bharat/IndicCorpV2"}:
  t=_first(row,("text","content","document","article","sentence","text_normalized"))
  if not t:raise SchemaAdapterError("plain-text row has no recognized text field")
  return [f"### TASK: language_modeling\n### TEXT\n{t}"]
 if dataset_id in {"awinml/legalbench-rag","kaushik-harsh-99/Indian-legal-data-v3","opennyaiorg/InJudgements_dataset","sarthak-wiz01/legalbench"}:return [_legal(row)]
 raise SchemaAdapterError(f"no reviewed adapter registered for {dataset_id!r} (expert {expert_id})")
def adapter_family(d:str)->str:
 if d in SIDECAR_ONLY:return "sidecar_required"
 if d=="grammarly/coedit":return "editing_pair_v1"
 if d in {"teknium/OpenHermes-2.5","hiyouga/glaive-function-calling-v2-sharegpt"}:return "ordered_chat_tool_v1"
 if d=="launch/gov_report_qs":return "hierarchical_document_qa_v1"
 if d in {"facebook/flores","openlanguagedata/flores_plus"}:return "parallel_translation_v1"
 if d in {"ccdv/govreport-summarization","FiscalNote/billsum","pszemraj/govreport-summarization-8192"}:return "document_summary_pair_v1"
 if d in {"HuggingFaceFW/fineweb-edu","ai4bharat/sangraha","ai4bharat/IndicCorpV2"}:return "plain_text_v1"
 if d in {"awinml/legalbench-rag","kaushik-harsh-99/Indian-legal-data-v3","opennyaiorg/InJudgements_dataset","sarthak-wiz01/legalbench"}:return "legal_context_qa_v1"
 return "unregistered"
