import json
from hindi_autoregressive_lm_smoke import run


def test_autoregressive_hindi_byte_lm_smoke(tmp_path):
    path = tmp_path / "report.json"
    result = run(path, steps=240, seed=23)
    assert result["status"] == "HINDI_AUTOREGRESSIVE_LM_SMOKE_PASS"
    assert result["task"].startswith("autoregressive")
    assert result["tokenizer"]["unicode_round_trip"] is True
    assert result["heldout_used_for_gradient_updates"] is False
    assert result["final_train_loss"] < result["initial_train_loss"]
    assert result["generation"]["sample"].startswith("नमस्ते।")
    assert len(result["generation"]["sample"]) > len("नमस्ते।")
    assert result["checkpoint"]["reload_max_abs_logit_difference"] == 0.0
    assert result["checkpoint"]["optimizer_state_restored"] is True
    assert result["provenance"]["external_data_used"] is False
    assert result["provenance"]["repository_corpus_used"] is False
    assert result["production_training_gate"] == "CLOSED"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["status"] == result["status"]
