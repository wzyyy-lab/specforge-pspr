"""Synthetic arithmetic checks only; these fixtures are not experiment evidence."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


_path = Path(__file__).resolve().parents[2] / 'scripts/analyze_pspr_memory_s1.py'
_spec = importlib.util.spec_from_file_location('pspr_s1_analysis', _path)
analysis = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analysis)


def test_safe_repair_requires_undamaged_prefix():
    # First block's repair is unreachable after its harmful first override.
    result = analysis.risk_counts(np.array([0, 1, 0, 1, 0, 0]),
                                  np.array([2., 3., 0., 3., 2., 0.]),
                                  np.array([0, 1, 0, 1, 0, 1]), 1.)
    assert result['repaired'] == 2 and result['destroyed'] == 2
    assert result['safe_first_repairs'] == 1
    assert result['damaged_before_first_error_blocks'] == 1
    assert result['damaged_fully_base_correct_blocks'] == 1


def test_tied_negatives_never_exceed_damage_budget():
    kind = np.array([0, 0, 0, 1])
    margin = np.array([3., 3., 1., 4.])
    threshold = analysis.risk_budget_threshold(margin, kind, 1)
    assert threshold == 3.
    assert ((margin > threshold) & (kind == 0)).sum() == 0
    with pytest.raises(AssertionError, match='risk budget'):
        analysis.risk_budget_threshold(margin, kind, -1)


def test_strict_gate_does_not_replace_equal_margin():
    result = analysis.risk_counts(np.array([0, 1]), np.array([1., 1.]), np.array([0, 1]), 1.)
    assert result['destroyed'] == result['repaired'] == 0


@pytest.mark.parametrize('kind,slot', [([0, 1], [0, 2]), ([1, 0], [0, 1]), ([0, 0], [1, 2])])
def test_invalid_base_blocks_are_rejected(kind, slot):
    with pytest.raises(AssertionError):
        analysis.risk_counts(np.array(kind), np.zeros(2), np.array(slot), 1.)


def test_paired_recipe_control_uses_domain_pooled_ratios():
    base = {d: np.array([[2., 1.], [12., 3.]]) for d in analysis.DOMAINS}
    improved = {d: np.array([[3., 1.], [15., 3.]]) for d in analysis.DOMAINS}
    draws = {d: np.array([[0, 1], [0, 0], [1, 1]]) for d in analysis.DOMAINS}
    result = analysis.paired_control_comparison({'T0': base, 'T1': improved}, 'T0', draws)
    assert set(result) == {'T1'}
    assert result['T1']['reference'] == 'T0'
    assert result['T1']['delta_macro'] == 1.
    assert result['T1']['paired_95ci'] == [1., 1.]
    assert analysis.paired_control_comparison({}, None, draws) == {}
    with pytest.raises(AssertionError, match='paired reference'):
        analysis.paired_control_comparison({'T0': base}, 'unknown', draws)


def test_loss_pilot_scope_cannot_claim_random_initialization():
    description = analysis.loss_pilot_description()
    assert 'weights-only warm start' in description['experiment']
    assert 'fresh optimizer and scheduler' in description['experiment']
    assert 'predeclared step1000' in description['checkpoint_policy']
    assert 'not whole-head Stage1 training from random initialization' in description['scope_note']


def test_turn_recipe_scope_is_fixed_budget_not_full_epoch():
    description = analysis.turn_recipe_description()
    assert 'random initialization' in description['experiment']
    assert 'step1000 is interim' in description['checkpoint_policy']
    assert 'NOT a full epoch' in description['checkpoint_policy']
    assert '199220 turn instances' in description['scope_note']
    assert 'not a prefix-only ablation' in description['scope_note']
    assert 'official DFlash backbone is frozen' in description['scope_note']


def test_teacher_source_identity_gate_and_legacy_warning():
    a, b = 'a' * 64, 'b' * 64
    assert analysis.teacher_source_status({'old': a, 'new': a}, True)['status'] == 'matched'
    assert analysis.teacher_source_status({'old': a, 'new': b})['status'] == 'mixed'
    assert analysis.teacher_source_status({'old': None, 'new': b})['status'] == 'unavailable'
    for sources in ({'old': a, 'new': b}, {'old': None}, {}, {'old': 'bad'}):
        with pytest.raises(AssertionError, match='source hashes'):
            analysis.teacher_source_status(sources, require_match=True)
