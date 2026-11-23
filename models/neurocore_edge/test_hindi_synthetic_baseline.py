import json
from hindi_synthetic_baseline import run

def test_hindi_byte_tokenizer_training_and_checkpoint(tmp_path):
    path=tmp_path/"report.json"
    result=run(path,steps=160,seed=7)
    assert result["status"]=="HINDI_SYNTHETIC_BASELINE_PASS"
    assert result["tokenizer"]["unicode_round_trip"] is True
    assert result["heldout_used_for_gradient_updates"] is False
    assert result["final_train_loss"] < result["initial_train_loss"]
    assert result["checkpoint"]["reload_max_abs_logit_difference"] == 0.0
    assert result["checkpoint"]["optimizer_state_restored"] is True
    assert result["provenance"]["external_data_used"] is False
    assert result["production_training_gate"] == "CLOSED"
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == result["status"]
