"""Synthetic guard/arithmetic fixtures, never model-quality evidence."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from scripts.calibrate_pspr_gate import METHOD, fit_threshold, sha, validate_for_evaluation, weight_hashes
from scripts.calibrate_pspr_gate import model_identity, validate_capture_identity
from scripts.prepare_pspr_holdout import prompt_key
from scripts.analyze_pspr_holdout import calibration_origin, compare


def test_threshold_uses_only_negative_scores_not_repair_labels():
    slots = np.array([0,1,0,1,0,1])
    margins = np.array([3.,4.,2.,5.,1.,6.], dtype=np.float32)
    kind = np.array([0,1,0,2,0,3])
    a = fit_threshold(kind, margins, slots, 1/3)
    b = fit_threshold(np.array([0,2,0,3,0,1]), margins, slots, 1/3)
    assert a['rho'] == b['rho']
    assert a['counts']['destroyed'] <= a['damage_budget'] == 1
    assert a['log_margin_threshold'] > 2.


def test_zero_budget_ties_and_equal_margin_rejected():
    result = fit_threshold(np.array([0,0,0,1]), np.array([3.,3.,1.,3.]),
                           np.array([0,1,2,3]), 0)
    assert result['counts']['destroyed'] == result['counts']['repaired'] == 0


@pytest.mark.parametrize('rate', [-1.,1.,float('nan'),float('inf')])
def test_bad_rate_rejected(rate):
    with pytest.raises(ValueError, match='destroy rate'):
        fit_threshold(np.array([0,1]), np.array([0.,1.]), np.array([0,1]), rate)


def test_no_negative_or_bad_layout_rejected():
    with pytest.raises(ValueError, match='no correct'):
        fit_threshold(np.array([1]), np.array([1.]), np.array([0]), .005)
    with pytest.raises(AssertionError):
        fit_threshold(np.array([1,0]), np.array([1.,1.]), np.array([0,1]), .005)


def artifact_fixture(tmp_path):
    export = tmp_path/'export'
    bb = export/'backbone'
    target = tmp_path/'target'
    bb.mkdir(parents=True); target.mkdir()
    (export/'selector.pt').write_bytes(b'synthetic selector identity only')
    (bb/'model.safetensors').write_bytes(b'synthetic backbone')
    (target/'model.safetensors').write_bytes(b'synthetic target')
    (target/'config.json').write_text('{}')
    decoder = tmp_path/'decoder.py'
    decoder.write_text('# synthetic source identity only\n')
    prompts = tmp_path/'eval.jsonl'
    prompts.write_text(json.dumps({'turns':['held-out prompt']})+'\n')
    original = {'selector_gate_rho':3., 'selector_gate_tau':0., 'selector_gate_theta':0.}
    artifact = tmp_path/'calibration.json'
    d = dict(schema_version=1, method=METHOD, score_temperature=1.,
             reference_decoder_sha256=sha(decoder),
             calibration=dict(prompts_sha256='cal-only', normalized_prompt_sha256=[prompt_key('calibration prompt')]),
             models={'CE':dict(selector_sha256=sha(export/'selector.pt'), backbone_sha256=weight_hashes(bb),
                               target_sha256=weight_hashes(target), exported_decode_policy=original,
                               fit=dict(rho=2., log_margin_threshold=math.log(2), max_destroy_rate=.005))})
    artifact.write_text(json.dumps(d))
    return artifact, export, target, prompts, decoder, d


def test_valid_calibration_changes_only_rho(tmp_path):
    path, export, target, prompts, decoder, d = artifact_fixture(tmp_path)
    policy, provenance = validate_for_evaluation(path,'CE',export,target,prompts,decoder)
    assert policy == {**d['models']['CE']['exported_decode_policy'], 'selector_gate_rho':2.}
    assert provenance['original_policy']['selector_gate_rho'] == 3.
    assert d['models']['CE']['exported_decode_policy']['selector_gate_rho'] == 3.


def test_normalized_calibration_overlap_rejected(tmp_path):
    path, export, target, prompts, decoder, d = artifact_fixture(tmp_path)
    prompts.write_text(json.dumps({'turns':['  CALIBRATION   prompt ']})+'\n')
    with pytest.raises(ValueError, match='overlaps'):
        validate_for_evaluation(path,'CE',export,target,prompts,decoder)


@pytest.mark.parametrize('field', ['selector', 'backbone', 'target', 'target_config', 'decoder'])
def test_identity_drift_rejected(tmp_path, field):
    path, export, target, prompts, decoder, d = artifact_fixture(tmp_path)
    paths = dict(selector=export/'selector.pt', backbone=export/'backbone/model.safetensors',
                 target=target/'model.safetensors', target_config=target/'config.json', decoder=decoder)
    paths[field].write_bytes(b'changed')
    with pytest.raises(ValueError, match='mismatch'):
        validate_for_evaluation(path,'CE',export,target,prompts,decoder)


@pytest.mark.parametrize('rho', [0.,-1.,float('nan'),float('inf')])
def test_invalid_rho_rejected(tmp_path,rho):
    path, export, target, prompts, decoder, d = artifact_fixture(tmp_path)
    d['models']['CE']['fit']['rho']=rho
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError, match='rho'):
        validate_for_evaluation(path,'CE',export,target,prompts,decoder)


def test_analysis_requires_explicit_pinned_calibration(tmp_path):
    path, export, target, prompts, decoder, d = artifact_fixture(tmp_path)
    policy, provenance = validate_for_evaluation(path,'CE',export,target,prompts,decoder)
    result = dict(decode_policy=policy, calibration_provenance=provenance,
                  selector_sha256=sha(export/'selector.pt'), reference_decoder_sha256=sha(decoder),
                  exported_decode_policy=d['models']['CE']['exported_decode_policy'])
    assert calibration_origin(result)['selector_gate_rho'] == 3.
    result['decode_policy']['selector_gate_theta']=.1
    with pytest.raises(AssertionError,match='only rho'):
        calibration_origin(result)


def test_analysis_rejects_modified_calibration_artifact(tmp_path):
    path, export, target, prompts, decoder, d = artifact_fixture(tmp_path)
    policy, provenance = validate_for_evaluation(path,'CE',export,target,prompts,decoder)
    result = dict(decode_policy=policy, calibration_provenance=provenance,
                  selector_sha256=sha(export/'selector.pt'), reference_decoder_sha256=sha(decoder),
                  exported_decode_policy=d['models']['CE']['exported_decode_policy'])
    path.write_text(path.read_text()+' ')
    with pytest.raises(AssertionError, match='artifact drift'):
        calibration_origin(result)


def test_calibrated_comparison_requires_opt_in_and_default_stays_strict(tmp_path):
    path, export, target, prompts, decoder, d = artifact_fixture(tmp_path)
    policy, provenance = validate_for_evaluation(path,'CE',export,target,prompts,decoder)
    original = d['models']['CE']['exported_decode_policy']
    rows = [dict(prompt_index=i, id=str(i), source='toy', prompt_sha256=str(i),
                 acceptance_lengths=[2], accepted_sum=2, num_blocks=1, mean_acceptance=2.) for i in range(2)]
    native = dict(block_size=16, proposal_slots=15, results=rows, skipped=[],
                  arguments=dict(max_new_tokens=256, max_prompt_tokens=2816, target_model='toy'),
                  prompts_sha256=sha(prompts), reference_decoder_sha256=sha(decoder),
                  selector_sha256=sha(export/'selector.pt'), decode_policy=original)
    calibrated = dict(**native)
    calibrated.update(decode_policy=policy, calibration_provenance=provenance, exported_decode_policy=original)
    a, b = tmp_path/'native.json', tmp_path/'calibrated.json'
    a.write_text(json.dumps(native)); b.write_text(json.dumps(calibrated))
    specs = [f'native={a}', f'calibrated={b}']
    with pytest.raises(AssertionError, match='decode_policy'):
        compare(specs, 'native', expected_prompts=2, bootstrap=10)
    result = compare(specs, 'native', expected_prompts=2, bootstrap=10, allow_calibrated_rho=True)
    assert result['results'][1]['paired_comparisons']['native']['delta'] == 0.
    default = compare([f'native={a}'], 'native', expected_prompts=2, bootstrap=10)
    assert not {'seed', 'allow_calibrated_rho', 'reachable_diagnostics'} & default.keys()


def test_fixed_runner_uses_native_decoder_and_no_training():
    from scripts.run_slotdeep_calibration import jobs_for
    assert len(jobs_for('smoke')) == len(jobs_for('diagnostics')) == len(jobs_for('six')) == 2
    jobs = jobs_for('confirmation')
    assert len(jobs) == 4 and sum(j['calibrated'] for j in jobs) == 2
    for job in jobs:
        cmd = job['command']
        assert cmd[:3] == ['python', '-u', 'scripts/evaluate_pspr_holdout.py']
        assert '--gate-stats' in cmd
        assert ('--calibration' in cmd) == job['calibrated']
        assert not any('train' in part for part in cmd)
    for job in jobs_for('six'):
        cmd = job['command']
        assert cmd[:3] == ['python', '-u', 'scripts/decode_lattice.py']
        assert cmd[cmd.index('--max-samples')+1] == '20'
        assert cmd[cmd.index('--shuffle-seed')+1] == '2026'
        assert '--allow-policy-mismatch' in cmd


def test_capture_and_diagnostic_identity_cannot_substitute_target(tmp_path):
    config = tmp_path/'config.json'
    config.write_text(json.dumps({'dflash_config':{'target_layer_ids':[1,9,17,25,33]}}))
    p = dict(arguments=dict(target_model=str(tmp_path/'target-a'), draft_config=str(config)),
             layer_ids=[1,9,17,25,33])
    validate_capture_identity(p, tmp_path/'target-a', config)
    with pytest.raises(ValueError, match='capture target'):
        validate_capture_identity(p, tmp_path/'target-b', config)
    with pytest.raises(ValueError, match='draft config'):
        validate_capture_identity(p, tmp_path/'target-a', tmp_path/'other.json')
    p['layer_ids']=[1,2,3]
    with pytest.raises(ValueError, match='layer IDs'):
        validate_capture_identity(p, tmp_path/'target-a', config)
    with pytest.raises(ValueError, match='diagnostic target'):
        model_identity(tmp_path/'absent-export', p, tmp_path/'target-b')
