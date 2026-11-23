import json
from hindi_word_lm_smoke import run


def test_hindi_word_level_autoregressive_smoke(tmp_path):
    report_path = tmp_path / "hindi_word_lm_report.json"
    result = run(report_path, steps=50, seed=11)

    assert result["status"] == "HINDI_WORD_LM_SMOKE_PASS"
    assert result["tokenizer"]["type"] == "whitespace-delimited orthographic word"
    assert result["tokenizer"]["vocabulary_built_from_training_only"] is True
    assert result["heldout_used_for_gradient_updates"] is False
    assert result["heldout_unknown_words"] == []
    assert result["final_train"]["loss"] < result["initial_train"]["loss"]
    assert result["final_heldout"]["loss"] < result["initial_heldout"]["loss"]
    assert result["final_heldout"]["next_word_accuracy"] >= 0.30
    assert result["grammar_semantic_probes"]["preferred_count"] >= 5
    assert result["grammar_semantic_probes"]["probe_count"] == 6

    samples = result["generation"]["samples"]
    assert "उगाते" in samples["किसान"]
    assert "लौटते" in samples["बच्चे"]
    assert "पढ़ाते" in samples["शिक्षक"]
    for sample in samples.values():
        assert sample.split()
        assert all(token not in ("<unk>", "<eos>") for token in sample.split())

    assert result["checkpoint"]["generated_continuations_equal_after_reload"] is True
    assert result["checkpoint"]["optimizer_state_restored"] is True
    assert result["provenance"]["external_data_used"] is False
    assert result["provenance"]["repository_corpus_used"] is False
    assert result["production_training_gate"] == "CLOSED"

    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["status"] == result["status"]
