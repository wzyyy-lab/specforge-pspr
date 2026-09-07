"""Synthetic arithmetic/provenance regression, never experiment evidence."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('pb_analysis', Path(__file__).resolve().parents[2] / 'scripts/analyze_pspr_holdout.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(path, extra=0):
    rows = []
    for i, a in enumerate(([2], [4, 4, 4], [2], [4, 4, 4])):
        a = [x + extra for x in a]
        rows.append(dict(prompt_index=i, id=str(i), source=str(i//2), prompt_sha256=str(i),
                         acceptance_lengths=a, accepted_sum=sum(a), num_blocks=len(a), mean_acceptance=sum(a)/len(a)))
    data = dict(block_size=16, proposal_slots=15, results=rows, skipped=[],
                arguments=dict(max_new_tokens=256, max_prompt_tokens=2816, target_model='synthetic-test-only'),
                prompts_sha256='synthetic-test-only', reference_decoder_sha256='synthetic-test-only',
                decode_policy={'selector_gate_rho': 3.})
    path.write_text(json.dumps(data))
    return data


def test_holdout_pools_blocks_and_pairs_prompts(tmp_path):
    a, b = tmp_path/'a.json', tmp_path/'b.json'
    fixture(a)
    fixture(b, 1)
    result = module.compare([f'T0={a}', f'T1={b}'], 'T0', control='T0', bootstrap=20, expected_prompts=4)
    assert result['population'] == 4
    assert result['results'][0]['pooled_acceptance'] == 3.5
    assert result['results'][1]['pooled_acceptance'] == 4.5
    assert result['results'][1]['paired_comparisons']['T0'] == dict(delta=1., paired_95ci=[1., 1.])


@pytest.mark.parametrize('field', ['prompt_sha256', 'source', 'id'])
def test_holdout_rejects_wrong_pairing(tmp_path, field):
    a, b = tmp_path/'a.json', tmp_path/'b.json'
    fixture(a)
    data = fixture(b)
    data['results'][0][field] = 'mismatch'
    b.write_text(json.dumps(data))
    with pytest.raises(AssertionError, match='identity/order'):
        module.compare([f'T0={a}', f'T1={b}'], 'T0', bootstrap=20, expected_prompts=4)


def test_holdout_rejects_wrong_denominator(tmp_path):
    a = tmp_path/'a.json'
    data = fixture(a)
    data['results'][0]['num_blocks'] = 9
    a.write_text(json.dumps(data))
    with pytest.raises(AssertionError):
        module.compare([f'T0={a}'], 'T0', bootstrap=20, expected_prompts=4)


def test_turn_report_is_not_a_warm_start_or_untouched_test_claim():
    title, note = module.report_description('slotdeep-turn')
    assert 'previously consumed' in title
    assert 'random-initialization whole-selector' in note
    assert 'not a full epoch' in note
    assert 'not untouched confirmation' in note


def test_loss_report_default_wording_and_unknown_kind():
    title, note = module.report_description('loss-pilot')
    assert title == 'SlotDeep training-only pilots: independent PB validation'
    assert 'weights-only warm-start pilots' in note
    with pytest.raises(ValueError, match='unknown run kind'):
        module.report_description('not-a-run')
