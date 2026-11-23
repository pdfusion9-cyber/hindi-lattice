import json
from hindi_qa_dialogue_smoke import run_experiment

def test_supervised_hindi_qa_dialogue_smoke(tmp_path):
    path=tmp_path/"report.json"
    r=run_experiment(path,steps=500,seed=11)
    assert r["status"]=="HINDI_QA_DIALOGUE_SMOKE_PASS"
    assert r["tokenizer"]["unicode_round_trip"] is True
    assert r["heldout_used_for_gradient_updates"] is False
    assert r["final_train"]["mean_teacher_forced_loss"] < r["initial_train"]["mean_teacher_forced_loss"]
    assert r["final_train"]["exact_match_count"] > 0
    assert r["checkpoint"]["optimizer_state_restored"] is True
    assert r["checkpoint"]["generation_equal_after_reload"] is True
    assert r["provenance"]["external_data_used"] is False
    assert r["provenance"]["repository_corpus_used"] is False
    assert r["production_training_gate"]=="CLOSED"
    saved=json.loads(path.read_text(encoding="utf-8"))
    assert saved["status"]==r["status"]
    assert 0.0 <= saved["final_heldout"]["exact_match_accuracy"] <= 1.0
