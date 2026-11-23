import pytest
from dataset_schema_adapters import ADAPTER_VERSION, SchemaAdapterError, adapt_row, adapter_family

def test_coedit():
    x = adapt_row("grammarly/coedit", {"src": "This are wrong", "tgt": "This is wrong"}, 0)[0]
    assert "This are wrong" in x and "This is wrong" in x

def test_chat_roles_tools():
    x = adapt_row("teknium/OpenHermes-2.5", {
        "conversations": [
            {"from": "human", "value": "Read invoice"},
            {"from": "gpt", "value": "I will read."},
            {"from": "tool", "value": "total 42"},
        ],
        "tools": [{"name": "read_pdf"}],
    }, 1)[0]
    assert x.index("### TOOL SCHEMAS") < x.index("### USER") < x.index("### ASSISTANT") < x.index("### TOOL\n")
    assert '"name":"read_pdf"' in x

def test_glaive_string():
    row = {
        "conversations": 'USER: search\nASSISTANT: <tool_call>{"name":"search"}</tool_call>',
        "tools": '[{"name":"search"}]',
    }
    x = adapt_row("hiyouga/glaive-function-calling-v2-sharegpt", row, 19)[0]
    assert "<tool_call>" in x and '"name":"search"' in x

def test_summary():
    x = adapt_row("ccdv/govreport-summarization", {"report": "Long report", "summary": "Outcome"}, 2)[0]
    assert "Long report" in x and "Outcome" in x

def test_hierarchical_qs():
    x = adapt_row("launch/gov_report_qs", {
        "document_sections": [
            {"heading": "Background", "text": "Facts"},
            {"heading": "Decision", "content": "Outcome"},
        ],
        "question_summary_pairs": [
            {"question": "What happened?", "summary": "Facts reviewed."},
            {"query": "Decision?", "answer": "Decision issued."},
        ],
    }, 17)
    assert len(x) == 2
    assert "## Background" in x[0] and "## Decision" in x[0]
    assert "Decision issued." in x[1]

def test_translation():
    x = adapt_row("facebook/flores", {
        "sentence": "नमस्ते", "translation": "Hello",
        "source_lang": "hin", "target_lang": "eng",
    }, 12)[0]
    assert "[hin] नमस्ते" in x and "[eng] Hello" in x
    with pytest.raises(SchemaAdapterError):
        adapt_row("facebook/flores", {"sentence": "नमस्ते"}, 12)

@pytest.mark.parametrize("dataset", [
    "HuggingFaceM4/DocumentVQA", "ai4bharat/indicvoices_r",
    "mythicinfinity/libritts", "ajimeno/PubTabNet", "allenai/olmOCR-mix-0225",
])
def test_sidecars(dataset):
    with pytest.raises(SchemaAdapterError, match="SIDECAR_REQUIRED"):
        adapt_row(dataset, {"text": "partial", "image": b"bytes"}, 3)

def test_unknown_fails_closed():
    with pytest.raises(SchemaAdapterError, match="no reviewed adapter"):
        adapt_row("unknown/source", {"foo": "bar"}, 0)

def test_catalog():
    assert ADAPTER_VERSION == "1.0.0"
    assert adapter_family("hiyouga/glaive-function-calling-v2-sharegpt") == "ordered_chat_tool_v1"

def test_gov_report_qs_actual_huggingface_dictionary_of_lists_schema():
    # Mirrors the documented paragraph/document config feature shape.
    x = adapt_row("launch/gov_report_qs", {
        "document_sections": {
            "title": ["Background", "Decision"],
            "paragraphs": ["Facts and context.", "The agency issued a decision."],
            "depth": [1, 2],
        },
        "question_summary_pairs": {
            "question": ["What happened?", "What was decided?"],
            "summary": ["The report describes events.", "The agency decided the matter."],
            "parent_pair_index": [-1, 0],
        },
    }, 17)
    assert len(x) == 2
    assert "## Background (depth 1)" in x[0]
    assert "## Decision (depth 2)" in x[0]
    assert "What happened?" in x[0]
    assert "Parent question-summary index: 0" in x[1]
    assert "\\n" not in x[0]
    assert "\n## Decision (depth 2)" in x[0]
    assert "The agency decided the matter." in x[1]


def test_legalbench_task_schema_input_output():
    x = adapt_row("sarthak-wiz01/legalbench", {
        "input": "Is the forum-selection clause enforceable?",
        "output": "The clause is enforceable on these facts.",
        "text": "Contract facts and the relevant clause.",
        "answer": "The clause is enforceable on these facts.",
        "slice": "personal_jurisdiction",
        "index": 3,
    }, 13)[0]
    assert "Is the forum-selection clause enforceable?" in x
    assert "Contract facts and the relevant clause." in x
    assert "The clause is enforceable on these facts." in x
