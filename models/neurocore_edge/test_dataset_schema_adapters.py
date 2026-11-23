import pytest
from dataset_schema_adapters import ADAPTER_VERSION, SchemaAdapterError, adapt_row, adapter_family

def test_coedit_pair_keeps_source_target():
    x=adapt_row("grammarly/coedit", {"src":"This are wrong","tgt":"This is wrong"}, 0)[0]
    assert "This are wrong" in x and "This is wrong" in x

def test_openhermes_preserves_roles_order_and_tools():
    x=adapt_row("teknium/OpenHermes-2.5", {"conversations":[
        {"from":"human","value":"Read invoice"},{"from":"gpt","value":"I will read it."},
        {"from":"tool","value":"total 42"}], "tools":[{"name":"read_pdf"}]}, 1)[0]
    assert x.index("### TOOL SCHEMAS") < x.index("### USER") < x.index("### ASSISTANT") < x.index("### TOOL\n")
    assert '"name":"read_pdf"' in x and "total 42" in x

def test_glaive_preserves_serialized_tool_calls():
    x=adapt_row("hiyouga/glaive-function-calling-v2-sharegpt", {
        "conversations":"USER: search\\nASSISTANT: <tool_call>{\\"name\\":\\"search\\"}</tool_call>",
        "tools":"[{\\"name\\":\\"search\\"}]"}, 19)[0]
    assert "<tool_call>" in x and '"name":"search"' in x

def test_govreport_pair():
    x=adapt_row("ccdv/govreport-summarization", {"report":"Long report","summary":"Outcome"}, 2)[0]
    assert "Long report" in x and "Outcome" in x

def test_gov_report_qs_keeps_hierarchy_and_each_question():
    x=adapt_row("launch/gov_report_qs", {
        "document_sections":[{"heading":"Background","text":"Facts"},{"heading":"Decision","content":"Outcome"}],
        "question_summary_pairs":[{"question":"What happened?","summary":"Facts reviewed."},
                                  {"query":"Decision?","answer":"Decision issued."}]}, 17)
    assert len(x)==2 and "## Background" in x[0] and "## Decision" in x[0]
    assert "What happened?" in x[0] and "Decision issued." in x[1]

def test_translation_requires_both_sides_and_keeps_language_tags():
    x=adapt_row("facebook/flores", {"sentence":"नमस्ते","translation":"Hello",
        "source_lang":"hin","target_lang":"eng"}, 12)[0]
    assert "[hin] नमस्ते" in x and "[eng] Hello" in x
    with pytest.raises(SchemaAdapterError):
        adapt_row("facebook/flores", {"sentence":"नमस्ते"}, 12)

@pytest.mark.parametrize("dataset", ["HuggingFaceM4/DocumentVQA","ai4bharat/indicvoices_r",
 "mythicinfinity/libritts","ajimeno/PubTabNet","allenai/olmOCR-mix-0225"])
def test_multimodal_sources_require_sidecar(dataset):
    with pytest.raises(SchemaAdapterError, match="SIDECAR_REQUIRED"):
        adapt_row(dataset, {"text":"partial text","image":b"image"}, 3)

def test_unknown_dataset_fails_closed():
    with pytest.raises(SchemaAdapterError, match="no reviewed adapter"):
        adapt_row("new/dataset", {"unknown":"x"}, 0)

def test_adapter_catalog_is_stable():
    assert ADAPTER_VERSION == "1.0.0"
    assert adapter_family("hiyouga/glaive-function-calling-v2-sharegpt") == "ordered_chat_tool_v1"
