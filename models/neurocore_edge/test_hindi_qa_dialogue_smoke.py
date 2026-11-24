import json
from hindi_qa_dialogue_smoke import run_experiment


def test_supervised_hindi_qa_dialogue_smoke(tmp_path):
    report_path = tmp_path / "hindi_qa_dialogue_report.json"
    result = run_experiment(report_path, steps=1200, seed=11)

    assert result["status"] == "HINDI_QA_DIALOGUE_SMOKE_PASS"
    assert result["task"] == "supervised intent classification with fixed response mapping"
    assert result["tokenizer"]["unicode_round_trip"] is True
    assert result["heldout_used_for_gradient_updates"] is False
    assert result["final_train_loss"] < result["initial_train_loss"]
    assert result["final_train"]["accuracy"] >= 0.95
    assert result["final_heldout"]["accuracy"] >= 0.70
    assert result["checkpoint"]["optimizer_state_restored"] is True
    assert result["checkpoint"]["predictions_equal_after_reload"] is True
    assert result["provenance"]["external_data_used"] is False
    assert result["provenance"]["repository_corpus_used"] is False
    assert result["production_training_gate"] == "CLOSED"
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["status"] == result["status"]
