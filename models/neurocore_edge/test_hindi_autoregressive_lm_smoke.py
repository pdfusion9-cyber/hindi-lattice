import json
from hindi_autoregressive_lm_smoke import run


def test_autoregressive_hindi_unicode_char_lm_smoke(tmp_path):
    path = tmp_path / "report.json"
    result = run(path, steps=120, seed=23)
    assert result["status"] == "HINDI_AUTOREGRESSIVE_LM_SMOKE_PASS"
    assert result["task"].startswith("autoregressive")
    assert result["tokenizer"]["type"] == "Unicode code-point"
    assert result["tokenizer"]["unicode_round_trip"] is True
    assert result["tokenizer"]["vocabulary_built_from_training_only"] is True
    assert result["heldout_used_for_gradient_updates"] is False
    assert result["heldout_examples"] == 5
    assert result["heldout_scored_known_codepoints"] > 0
    assert result["final_train_loss"] < result["initial_train_loss"]
    assert result["final_heldout_loss"] < result["initial_heldout_loss"]
    generation = result["generation"]
    assert generation["prompt_is_absent_from_training_sentences"] is True
    assert generation["unicode_valid"] is True
    assert generation["replacement_character_present"] is False
    assert generation["sample"].startswith(generation["prompt"])
    assert len(generation["sample"]) > len(generation["prompt"])
    assert generation["sample"].encode("utf-8").decode("utf-8") == generation["sample"]
    assert result["checkpoint"]["reload_max_abs_logit_difference"] == 0.0
    assert result["checkpoint"]["optimizer_state_restored"] is True
    assert result["provenance"]["external_data_used"] is False
    assert result["provenance"]["repository_corpus_used"] is False
    assert result["production_training_gate"] == "CLOSED"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["status"] == result["status"]
