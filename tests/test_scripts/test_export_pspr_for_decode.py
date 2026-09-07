import hashlib
import json

import torch

import pytest

from scripts.export_pspr_for_decode import (
    join_state,
    selector_config,
    split_state,
    validate_policy_provenance,
)


def test_transition_enabled_selector_roundtrips_reference_parameter_names():
    state = {
        "model.layer.weight": torch.randn(2, 3),
        "candidate_selector.delta_mlp.3.weight": torch.randn(3, 4),
        "candidate_selector.gamma": torch.randn(1),
        "candidate_selector.predecessor_codebook": torch.randn(7, 5),
        "candidate_selector.successor_codebook": torch.randn(7, 5),
        "candidate_selector.trans_proj.weight": torch.randn(5, 3),
    }

    backbone, selector = split_state(state)

    assert set(selector) == {
        "dh_mlp.3.weight",
        "gamma",
        "trans_pred",
        "trans_succ",
        "trans_proj.weight",
    }
    assert selector["gamma"].shape == ()
    rejoined = join_state(backbone, selector)
    assert set(rejoined) == set(state)
    for key in state:
        torch.testing.assert_close(rejoined[key], state[key])
        assert rejoined[key].shape == state[key].shape


def test_selector_config_exports_transition_rank_only_when_enabled():
    base = {
        "hidden_size": 8,
        "vocab_size": 17,
        "dflash_config": {"selector_top_k": 4},
    }
    assert "trans_rank" not in selector_config(base)

    enabled = {
        **base,
        "dflash_config": {**base["dflash_config"], "selector_trans_rank": 6},
    }
    assert selector_config(enabled)["trans_rank"] == 6


def _write_config(tmp_path, payload):
    path = tmp_path / "draft.json"
    path.write_text(json.dumps(payload))
    return path


def _keep_repair_policy():
    return {
        "selector_decision_mode": "keep_repair",
        "selector_keep_repair_margin": 0.0,
        "selector_gate_rho": 1.0,
        "selector_gate_tau": 0.0,
        "selector_gate_theta": 0.0,
        "selector_gate_skip_first": False,
    }


def test_policy_provenance_rejects_train_serve_mismatch_and_missing_by_default(tmp_path):
    policy = _keep_repair_policy()
    path = _write_config(tmp_path, policy)
    with pytest.raises(ValueError, match="train-serve policy mismatch"):
        validate_policy_provenance(
            {"dflash2_selector_objective": "multiclass"},
            policy,
            selector_top_k=16,
            draft_config_path=str(path),
            allow_missing=False,
        )
    with pytest.raises(ValueError, match="lacks policy provenance"):
        validate_policy_provenance(
            {},
            policy,
            selector_top_k=16,
            draft_config_path=str(path),
            allow_missing=False,
        )

    legacy = validate_policy_provenance(
        {},
        policy,
        selector_top_k=16,
        draft_config_path=str(path),
        allow_missing=True,
    )
    assert legacy["verified"] is False


def test_policy_provenance_records_verified_training_objective(tmp_path):
    policy = _keep_repair_policy()
    path = _write_config(tmp_path, policy)
    verified = validate_policy_provenance(
        {
            "dflash2_selector_objective": "keep_repair",
            "dflash2_selector_weight_mode": "expected_accept",
            "dflash2_selector_top_k": 16,
            "dflash2_selector_own_denominator": False,
            "draft_config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
        policy,
        selector_top_k=16,
        draft_config_path=str(path),
        allow_missing=False,
    )
    assert verified["verified"] is True
    assert verified["selector_objective"] == "keep_repair"
    assert verified["selector_weight_mode"] == "expected_accept"
    assert verified["selector_top_k"] == 16
    assert verified["selector_own_denominator"] is False

    profitable = validate_policy_provenance(
        {
            "dflash2_selector_objective": "profitable_repair",
            "dflash2_selector_weight_mode": "uniform_frontier_boost",
            "dflash2_selector_top_k": 16,
            "dflash2_selector_own_denominator": False,
            "draft_config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
        policy,
        selector_top_k=16,
        draft_config_path=str(path),
        allow_missing=False,
    )
    assert profitable["verified"] is True
    assert profitable["selector_objective"] == "profitable_repair"


def test_policy_provenance_requires_smoothed_survival_floor_and_rejects_local_mean(tmp_path):
    policy = _keep_repair_policy()
    path = _write_config(tmp_path, policy)
    base = {
        "dflash2_selector_objective": "keep_repair",
        "dflash2_selector_weight_mode": "smoothed_accept",
        "dflash2_selector_top_k": 16,
        "dflash2_selector_own_denominator": False,
        "draft_config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    with pytest.raises(ValueError, match="dflash2_selector_survival_floor"):
        validate_policy_provenance(
            base,
            policy,
            selector_top_k=16,
            draft_config_path=str(path),
            allow_missing=False,
        )

    with pytest.raises(ValueError, match="partition-dependent"):
        validate_policy_provenance(
            {
                **base,
                "dflash2_selector_own_denominator": True,
                "dflash2_selector_survival_floor": 0.5,
            },
            policy,
            selector_top_k=16,
            draft_config_path=str(path),
            allow_missing=False,
        )

    verified = validate_policy_provenance(
        {**base, "dflash2_selector_survival_floor": 0.5},
        policy,
        selector_top_k=16,
        draft_config_path=str(path),
        allow_missing=False,
    )
    assert verified["selector_survival_floor"] == 0.5
