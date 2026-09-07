"""Synthetic six-domain arithmetic and provenance, never ML results."""
import json

import pytest

from scripts import analyze_slotdeep_calibration as module


def setup_six(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'OUT', tmp_path)
    monkeypatch.setattr(module, 'TAPS', tmp_path/'taps')
    monkeypatch.setattr(module, 'export_path', lambda arm:tmp_path/arm)
    native = {arm:tmp_path/f'native_{arm}.json' for arm in ('CE','KD')}
    monkeypatch.setattr(module, 'NATIVE_SIX', native)
    policy = dict(selector_gate_rho=3., selector_gate_tau=0., selector_gate_theta=0.)
    artifact = tmp_path/'calibration.json'
    models = {}
    for arm in ('CE','KD'):
        (tmp_path/arm).mkdir()
        (tmp_path/arm/'selector.pt').write_bytes(b'synthetic identity only')
        models[arm] = dict(selector_sha256=module.sha(tmp_path/arm/'selector.pt'),
                           exported_decode_policy=policy, fit=dict(rho=2.), target_model=str(tmp_path/'target'))
    cal = dict(schema_version=1, method=module.METHOD, models=models, reference_decoder_sha256='toy-decoder')
    artifact.write_text(json.dumps(cal))
    for offset, (arm, calibrated) in enumerate([('CE',False),('CE',True),('KD',False),('KD',True)]):
        args = dict.fromkeys(module.SEMANTICS, False)
        args.update(max_samples=20, max_new_tokens=256, shuffle_seed=2026, eval_reserved=True,
                    gate_rho=2. if calibrated else 3., gate_tau=0., gate_theta=0.,
                    score_temperature=1., target_model=str(tmp_path/'target'), beam=1, path_branch=1,
                    lattice_head=str(tmp_path/arm/'selector.pt'))
        rows = [dict(dataset=d, prompt_index=i, prompt_sha256=f'{d}:{i}', mode='latgate',
                     acceptance_lengths=[2+offset], accepted_sum=2+offset, num_blocks=1, mean_acceptance=2+offset)
                for d in module.DOMAINS for i in range(20)]
        stats = dict(fix_recovered=2, fix_kept_wrong=2, fix_wrong_override=1, fixable_n=5,
                     base_right_kept=8, base_right_destroyed=1, base_right_n=9,
                     unfixable_n=3, base_wrong_n=8, fix_gate_blocked_true_best=1)
        data = dict(block_size=16, proposal_slots=15, results=rows, arguments=args, gate_stats={'latgate':stats})
        path = tmp_path/f'six_{arm}_calibrated.json' if calibrated else native[arm]
        path.write_text(json.dumps(data))
        if calibrated:
            provenance = dict(calibration_provenance=dict(artifact_sha256=module.sha(artifact), arm=arm,
                original_policy=policy, fitted_rho=2., independent_prompt_check=True),
                selector_sha256=models[arm]['selector_sha256'],
                source_sha256={str(module.TAPS/'scripts/decode_lattice.py'):'toy-decoder'})
            (tmp_path/f'six_{arm}_calibrated_PROVENANCE.json').write_text(json.dumps(provenance))
            (tmp_path/f'six_{arm}_calibrated_EXIT.json').write_text(json.dumps({'success':True}))
    return artifact


def test_six_all_predeclared_contrasts_and_symmetric_sensitivity(tmp_path, monkeypatch):
    artifact = setup_six(tmp_path,monkeypatch)
    result = module.six_comparison(artifact,bootstrap=20)
    assert result['population'] == 120 and result['seed'] == 20260906
    rows = {r['label']:r for r in result['results']}
    for a,b,delta in [('KD-calibrated','CE-calibrated',2.),('CE-calibrated','CE-native',1.),('KD-calibrated','KD-native',1.)]:
        assert rows[a]['paired_comparisons'][b] == dict(delta=delta,paired_95ci=[delta,delta])
    assert result['known_overlap_sensitivity']['population'] == 119
    assert result['known_overlap_sensitivity']['results']['CE-native']['macro'] == 2.


def test_six_rejects_nonrho_policy_drift(tmp_path,monkeypatch):
    artifact = setup_six(tmp_path,monkeypatch)
    path=tmp_path/'six_CE_calibrated.json'
    data=json.loads(path.read_text()); data['arguments']['gate_theta']=.1
    path.write_text(json.dumps(data))
    with pytest.raises(AssertionError,match='semantics differ'):
        module.six_comparison(artifact,bootstrap=20)


def test_six_rejects_different_prompts(tmp_path,monkeypatch):
    artifact = setup_six(tmp_path,monkeypatch)
    path=tmp_path/'six_CE_calibrated.json'
    data=json.loads(path.read_text()); data['results'][0]['prompt_sha256']='other'
    path.write_text(json.dumps(data))
    with pytest.raises(AssertionError,match='prompt pairing'):
        module.six_comparison(artifact,bootstrap=20)


def test_six_rejects_artifact_drift(tmp_path,monkeypatch):
    artifact = setup_six(tmp_path,monkeypatch)
    artifact.write_text(artifact.read_text()+' ')
    with pytest.raises(AssertionError):
        module.six_comparison(artifact,bootstrap=20)
