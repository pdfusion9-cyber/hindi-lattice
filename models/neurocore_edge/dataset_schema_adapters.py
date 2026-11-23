#!/usr/bin/env python3
"""Loss-aware text adapters. Multimodal and unknown schemas fail closed."""
from __future__ import annotations
import json
from typing import Any, Iterable

ADAPTER_VERSION = "1.0.0"

class SchemaAdapterError(ValueError):
    pass

SIDECAR_ONLY = {
    "HuggingFaceM4/DocumentVQA": "document image payload requires a vision encoder and page loader",
    "ai4bharat/indicvoices_r": "audio, speaker rights and a TTS/vocoder sidecar are required",
    "mythicinfinity/libritts": "audio, speaker rights and a TTS/vocoder sidecar are required",
    "ajimeno/PubTabNet": "page image and table-cell geometry require a table/vision sidecar",
    "allenai/olmOCR-mix-0225": "page images and layout coordinates require a vision/OCR sidecar",
}

def _text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    return ""

def _first(row: dict, keys: Iterable[str]) -> str:
    for key in keys:
        value = _text(row.get(key))
        if value:
            return value
    return ""

def _json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise SchemaAdapterError(f"nested field is not JSON serializable: {exc}") from exc

def _pair(task: str, prompt: str, answer: str) -> str:
    if not prompt or not answer:
        raise SchemaAdapterError(f"{task} row needs both prompt/source and target/answer")
    return f"### TASK: {task}\n### INPUT\n{prompt}\n### OUTPUT\n{answer}"

def _chat(row: dict) -> str:
    raw = row.get("conversations", row.get("messages", row.get("conversation")))
    if isinstance(raw, str):
        dialogue = raw.strip()
    elif isinstance(raw, list):
        parts = []
        for i, message in enumerate(raw):
            if not isinstance(message, dict):
                raise SchemaAdapterError(f"conversation message {i} is not an object")
            role = _first(message, ("from", "role", "speaker", "author")).lower()
            body = _first(message, ("value", "content", "text", "message"))
            if not role or not body:
                raise SchemaAdapterError(f"conversation message {i} lacks role or content")
            role = {"human": "user", "gpt": "assistant", "bot": "assistant"}.get(role, role)
            parts.append(f"### {role.upper()}\n{body}")
        dialogue = "\n".join(parts)
    else:
        dialogue = ""
    if not dialogue:
        raise SchemaAdapterError("chat row has no non-empty conversations/messages")
    tools = row.get("tools")
    if tools not in (None, "", [], {}):
        if not isinstance(tools, (str, list, dict)):
            raise SchemaAdapterError("tools must be JSON/text")
        tool_text = tools if isinstance(tools, str) else _json(tools)
        dialogue = "### TOOL SCHEMAS\n" + tool_text + "\n" + dialogue
    return dialogue

def _gov_report(row: dict) -> str:
    return _pair("document_summarization",
                 _first(row, ("report", "document", "article", "text", "content")),
                 _first(row, ("summary", "highlights", "target")))

def _gov_report_qs(row: dict) -> list[str]:
    sections = row.get("document_sections", row.get("sections"))
    if not isinstance(sections, list):
        raise SchemaAdapterError("gov_report_qs requires a document_sections list")
    rendered_sections = []
    for i, section in enumerate(sections):
        if isinstance(section, str):
            title, body = f"Section {i+1}", section.strip()
        elif isinstance(section, dict):
            title = _first(section, ("heading", "title", "section_title", "name")) or f"Section {i+1}"
            body = _first(section, ("text", "content", "body", "paragraph"))
        else:
            raise SchemaAdapterError(f"document section {i} has invalid type")
        if not body:
            raise SchemaAdapterError(f"document section {i} has no body")
        rendered_sections.append(f"## {title}\n{body}")
    pairs = row.get("question_summary_pairs", row.get("qa_pairs", row.get("pairs")))
    if not rendered_sections or not isinstance(pairs, list) or not pairs:
        raise SchemaAdapterError("gov_report_qs requires question_summary_pairs")
    document = "\n\n".join(rendered_sections)
    out = []
    for i, pair in enumerate(pairs):
        if not isinstance(pair, dict):
            raise SchemaAdapterError(f"question-summary pair {i} is not an object")
        question = _first(pair, ("question", "query"))
        answer = _first(pair, ("summary", "answer", "response"))
        out.append(_pair("long_document_planning_and_summary", document + "\n\n" + question, answer))
    return out

def _translation(row: dict) -> str:
    source = _first(row, ("source", "src", "sentence", "translation_source"))
    target = _first(row, ("target", "tgt", "translation", "sentence_target"))
    src_lang = _first(row, ("source_lang", "src_lang", "language1", "lang1"))
    tgt_lang = _first(row, ("target_lang", "tgt_lang", "language2", "lang2"))
    if src_lang:
        source = f"[{src_lang}] {source}"
    if tgt_lang:
        target = f"[{tgt_lang}] {target}"
    return _pair("translation", source, target)

def _legal(row: dict) -> str:
    question = _first(row, ("question", "query", "instruction", "issue"))
    context = _first(row, ("context", "facts", "case_text", "judgment", "judgement", "opinion", "document", "text"))
    answer = _first(row, ("answer", "response", "holding", "reasoning", "rationale", "output"))
    prompt = (context + "\n\n" if context else "") + question
    return _pair("legal_research_not_legal_advice", prompt, answer)

def adapt_row(dataset_id: str, row: dict, expert_id: int) -> list[str]:
    if not isinstance(row, dict):
        raise SchemaAdapterError("dataset row must be an object")
    if dataset_id in SIDECAR_ONLY:
        raise SchemaAdapterError(f"SIDECAR_REQUIRED: {dataset_id}: {SIDECAR_ONLY[dataset_id]}")
    if dataset_id == "grammarly/coedit":
        instruction = _first(row, ("instruction", "edit_type", "task")) or "Revise while preserving meaning."
        return [_pair("document_editing", instruction + "\n\n" + _first(row, ("src", "source", "original", "input", "text")),
                      _first(row, ("tgt", "target", "edited", "correction", "output")))]
    if dataset_id in {"teknium/OpenHermes-2.5", "hiyouga/glaive-function-calling-v2-sharegpt"}:
        return [_chat(row)]
    if dataset_id in {"ccdv/govreport-summarization", "FiscalNote/billsum", "pszemraj/govreport-summarization-8192"}:
        return [_gov_report(row)]
    if dataset_id == "launch/gov_report_qs":
        return _gov_report_qs(row)
    if dataset_id in {"facebook/flores", "openlanguagedata/flores_plus"}:
        return [_translation(row)]
    if dataset_id in {"HuggingFaceFW/fineweb-edu", "ai4bharat/sangraha", "ai4bharat/IndicCorpV2"}:
        text = _first(row, ("text", "content", "document", "article", "sentence", "text_normalized"))
        if not text:
            raise SchemaAdapterError("plain-text row has no recognized text field")
        return [f"### TASK: language_modeling\n### TEXT\n{text}"]
    if dataset_id in {"awinml/legalbench-rag", "kaushik-harsh-99/Indian-legal-data-v3",
                      "opennyaiorg/InJudgements_dataset", "sarthak-wiz01/legalbench"}:
        return [_legal(row)]
    raise SchemaAdapterError(f"no reviewed adapter registered for {dataset_id!r} (expert {expert_id})")

def adapter_family(dataset_id: str) -> str:
    if dataset_id in SIDECAR_ONLY:
        return "sidecar_required"
    if dataset_id == "grammarly/coedit":
        return "editing_pair_v1"
    if dataset_id in {"teknium/OpenHermes-2.5", "hiyouga/glaive-function-calling-v2-sharegpt"}:
        return "ordered_chat_tool_v1"
    if dataset_id == "launch/gov_report_qs":
        return "hierarchical_document_qa_v1"
    if dataset_id in {"facebook/flores", "openlanguagedata/flores_plus"}:
        return "parallel_translation_v1"
    if dataset_id in {"ccdv/govreport-summarization", "FiscalNote/billsum", "pszemraj/govreport-summarization-8192"}:
        return "document_summary_pair_v1"
    if dataset_id in {"HuggingFaceFW/fineweb-edu", "ai4bharat/sangraha", "ai4bharat/IndicCorpV2"}:
        return "plain_text_v1"
    if dataset_id in {"awinml/legalbench-rag", "kaushik-harsh-99/Indian-legal-data-v3",
                      "opennyaiorg/InJudgements_dataset", "sarthak-wiz01/legalbench"}:
        return "legal_context_qa_v1"
    return "unregistered"
