import json

import pytest
import torch

from native_staged_training import (
    merge_expert_shards,
    run_staged_training,
    tiny_native_config,
)


def test_native_shared_backbone_20_expert_shards_train_and_merge(tmp_path):
    report_path = tmp_path / "native_staged_report.json"
    checkpoint_path = tmp_path / "native_staged_checkpoint.pt"
    report = run_staged_training(
        report_path,
        checkpoint_path,
        base_steps=1,
        expert_steps=1,
        seed=19,
    )

    assert report["status"] == "NATIVE_20_EXPERT_STAGED_SYNTHETIC_PASS"
    assert report["model_under_test"] == "native_neurocore_edge_scaled_reference"
    assert report["training_scale"] == "tiny_20_expert_reference"
    assert report["open_source_model_integrated"] is False
    assert report["external_weights_used"] is False
    assert report["shared_backbone"]["status"] == "TRAINED"
    assert report["expert_shards"]["status"] == "TRAINED_AND_MERGED"
    assert report["expert_shards"]["count"] == 20
    assert report["expert_shards"]["base_fingerprint_match"] is True
    assert report["expert_shards"]["config_fingerprint_match"] is True
    assert report["parameter_count_preserved"] is True
    assert report["production_training_gate"]["status"] == "CLOSED"
    assert report["production_2_87b_training_started"] is False
    assert checkpoint_path.exists()
    assert json.loads(report_path.read_text())["status"] == report["status"]


def test_merge_refuses_missing_duplicate_or_wrong_base_shards():
    cfg = tiny_native_config()
    model_state = {
        key: value.detach().cpu().clone()
        for key, value in __import__(
            "modeling_neurocore_edge"
        ).NeuroCoreEdgeForCausalLM(cfg).state_dict().items()
    }
    from native_staged_training import config_fingerprint, state_fingerprint

    base_hash = state_fingerprint(model_state)
    cfg_hash = config_fingerprint(cfg)
    with pytest.raises(ValueError, match="exactly one shard"):
        merge_expert_shards(
            model_state, [], cfg,
            expected_base_hash=base_hash,
            expected_cfg_hash=cfg_hash,
        )
    duplicate = [
        {
            "expert_id": 0,
            "base_fingerprint": base_hash,
            "config_fingerprint": cfg_hash,
            "expert_state": {},
        },
        {
            "expert_id": 0,
            "base_fingerprint": base_hash,
            "config_fingerprint": cfg_hash,
            "expert_state": {},
        },
    ]
    with pytest.raises(ValueError, match="exactly one shard"):
        merge_expert_shards(
            model_state, duplicate, cfg,
            expected_base_hash=base_hash,
            expected_cfg_hash=cfg_hash,
        )
