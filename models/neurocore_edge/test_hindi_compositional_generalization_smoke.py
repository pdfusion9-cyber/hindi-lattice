import json
from hindi_compositional_generalization_smoke import run


def test_unseen_subject_time_compositional_generalization(tmp_path):
    report_path = tmp_path / "compositional_report.json"
    result = run(report_path, steps=700, seed=11)

    assert result["status"] == "HINDI_COMPOSITIONAL_GENERALIZATION_PASS"
    assert result["tokenizer"]["vocabulary_built_from_training_split_only"] is True
    assert result["tokenizer"]["all_heldout_lexical_items_seen_in_training"] is True
    split = result["split"]
    assert split["train_unique_sentences"] == 32
    assert split["heldout_unique_sentences"] == 8
    assert split["heldout_contexts_seen_in_training"] is False
    assert split["heldout_used_for_gradient_updates"] is False
    assert split["heldout_oov_tokens"] == []
    assert result["final_train"]["loss"] < result["initial_train"]["loss"]
    assert result["final_heldout"]["loss"] < result["initial_heldout"]["loss"]
    assert result["final_heldout"]["next_word_accuracy"] >= 0.75
    generation = result["generation"]
    assert generation["exact_sentence_matches"] == 8
    assert generation["heldout_sentence_count"] == 8
    assert all(item["context_seen_in_training"] is False for item in generation["samples"])
    assert all(item["exact_match"] for item in generation["samples"])
    assert result["checkpoint"]["generated_text_equal_after_reload"] is True
    assert result["checkpoint"]["optimizer_state_restored"] is True
    assert result["provenance"]["external_data_used"] is False
    assert result["provenance"]["repository_corpus_used"] is False
    assert result["production_training_gate"] == "CLOSED"

    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["status"] == result["status"]
