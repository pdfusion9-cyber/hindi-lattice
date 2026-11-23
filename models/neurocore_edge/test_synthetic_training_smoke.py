import json

from synthetic_training_smoke import run_experiment


def test_synthetic_training_checkpoint_and_resume(tmp_path):
    report_path = tmp_path / "synthetic_training_report.json"
    report = run_experiment(report_path, steps=40, seed=7)

    assert report["status"] == "SYNTHETIC_SMOKE_PASS"
    assert report["provenance"]["external_data_used"] is False
    assert report["provenance"]["user_data_used"] is False
    assert report["production_training_gate"] == "CLOSED"
    assert report["production_dataset_approvals_changed"] is False
    assert report["loss_reduction_fraction"] >= 0.5
    assert report["checkpoint"]["reload_max_abs_logit_difference"] <= 1e-7
    assert report["checkpoint"]["resume_max_abs_parameter_difference"] <= 1e-7
    assert report["checkpoint"]["resume_next_loss_difference"] <= 1e-7

    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["status"] == "SYNTHETIC_SMOKE_PASS"
    assert saved["checkpoint"]["sha256"]
