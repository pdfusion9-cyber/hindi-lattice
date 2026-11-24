from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "models" / "neurocore_edge"))

from neurocore_native_grouped_answer_word_prediction import (  # noqa: E402
    ALT,
    EOS,
    SPECIAL,
    audit_groups,
    atomic_torch_save,
    collate,
    encode_group,
    load_groups,
    make_vocab,
    split_by_group,
    validate_resume_checkpoint,
)


def test_loader_groups_all_answers_and_deduplicates_normalized_objects(tmp_path):
    corpus = tmp_path / "facts.txt"
    corpus.write_text(
        "दक्षिण अफ्रीका|राजधानी|प्रिटोरिया\n"
        "दक्षिण अफ्रीका|राजधानी|केप टाउन\n"
        "दक्षिण अफ्रीका|राजधानी|प्रिटोरिया\n"
        "भारत|राजधानी|नई दिल्ली\n",
        encoding="utf-8",
    )
    _, rows, groups, counts, markup = load_groups(corpus)
    assert len(rows) == 4
    assert len(groups) == 2
    multi = next(group for group in groups if group["subject"] == "दक्षिण अफ्रीका")
    assert multi["objects"] == ["केप टाउन", "प्रिटोरिया"]
    assert counts["राजधानी"] == 4
    assert audit_groups(rows, groups, markup)["multi_answer_groups"] == 1


def test_group_split_never_leaks_subject_relation_pairs():
    groups = [
        {"subject": f"विषय {i}", "relation": "संबंध", "objects": [f"उत्तर {i}"]}
        for i in range(200)
    ]
    train, valid = split_by_group(groups, validation_percent=20)
    train_keys = {(g["subject"], g["relation"]) for g in train}
    valid_keys = {(g["subject"], g["relation"]) for g in valid}
    assert train and valid
    assert train_keys.isdisjoint(valid_keys)


def test_target_encodes_all_answers_with_alt_and_eos():
    group = {
        "subject": "दक्षिण अफ्रीका",
        "relation": "राजधानी",
        "objects": ["केप टाउन", "प्रिटोरिया"],
    }
    word_to_id, _ = make_vocab({
        "दक्षिण": 1, "अफ्रीका": 1, "राजधानी": 1,
        "केप": 1, "टाउन": 1, "प्रिटोरिया": 1,
    })
    prompt, target = encode_group(group, word_to_id)
    assert target.count(ALT) == 1
    assert target[-1] == EOS
    assert len(target) == 5


def test_collate_masks_prompt_and_padding_from_loss():
    word_to_id, _ = make_vocab({"विषय": 1, "संबंध": 1, "उत्तर": 1})
    examples = [
        encode_group({"subject": "विषय", "relation": "संबंध", "objects": ["उत्तर"]}, word_to_id),
        encode_group({"subject": "विषय", "relation": "संबंध", "objects": ["उत्तर", "विषय"]}, word_to_id),
    ]
    x, y, loss_mask, word_mask = collate(examples)
    assert x.shape == y.shape == loss_mask.shape == word_mask.shape
    assert loss_mask.dtype == torch.bool
    assert word_mask.dtype == torch.bool
    assert int(loss_mask[0].sum()) == len(examples[0][1])
    assert int(loss_mask[1].sum()) == len(examples[1][1])
    assert torch.all(word_mask <= loss_mask)
    assert torch.all(y[loss_mask] >= 0)
    assert torch.all(y[~loss_mask] >= -100)
    assert len(SPECIAL) == 6


def _resume_payload():
    return {
        "model_state_dict": {},
        "optimizer_state_dict": {},
        "step": 40,
        "python_rng_state": (3, (1, 2, 3), None),
        "torch_rng_state": torch.get_rng_state(),
        "validation_curve": [],
        "initial_validation": {"loss": 1.0},
        "elapsed_seconds": 12.5,
        "model_id": "neurocore_edge_native_grouped_answer_word_v1",
        "config": {"hidden_size": 64},
        "training_fingerprint": {"seed": 211},
        "corpus_git_blob_sha1": "abc123",
        "corpus_sha256": "def456",
        "tokenizer_sha256": "789abc",
        "production_training_gate": "CLOSED",
        "internal_experiment_gate": "MICRO_AGENT_APPROVED_BOUNDED_INTERNAL_ONLY",
        "public_artifacts_created": False,
    }


def _validate_payload(payload, **overrides):
    args = {
        "model_config": {"hidden_size": 64},
        "training_fingerprint": {"seed": 211},
        "corpus_blob": "abc123",
        "corpus_sha256": "def456",
        "tokenizer_sha256": "789abc",
        "target_steps": 100,
    }
    args.update(overrides)
    return validate_resume_checkpoint(payload, **args)


def test_atomic_checkpoint_roundtrip_and_resume_validation(tmp_path):
    checkpoint_path = tmp_path / "resume.pt"
    payload = _resume_payload()
    atomic_torch_save(payload, checkpoint_path)
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert _validate_payload(loaded) == 40
    assert not checkpoint_path.with_name(checkpoint_path.name + ".tmp").exists()


@pytest.mark.parametrize(
    "overrides",
    [
        {"corpus_blob": "different"},
        {"corpus_sha256": "different"},
        {"tokenizer_sha256": "different"},
        {"model_config": {"hidden_size": 128}},
        {"training_fingerprint": {"seed": 999}},
    ],
)
def test_resume_rejects_changed_fingerprints(overrides):
    with pytest.raises(ValueError):
        _validate_payload(_resume_payload(), **overrides)


def test_resume_rejects_checkpoint_marked_public_or_wrong_gate():
    payload = _resume_payload()
    payload["public_artifacts_created"] = True
    with pytest.raises(ValueError, match="marked as public"):
        _validate_payload(payload)

    payload = _resume_payload()
    payload["production_training_gate"] = "OPEN"
    with pytest.raises(ValueError, match="production gate"):
        _validate_payload(payload)


def test_resume_rejects_completed_or_invalid_step():
    payload = _resume_payload()
    with pytest.raises(ValueError, match="below the requested total"):
        _validate_payload(payload, target_steps=40)

    payload = _resume_payload()
    payload["step"] = True
    with pytest.raises(ValueError, match="step is invalid"):
        _validate_payload(payload)
