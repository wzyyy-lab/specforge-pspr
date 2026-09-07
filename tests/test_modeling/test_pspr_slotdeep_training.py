"""Synthetic loss/gradient/assembly checks, not experiment quality evidence."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from specforge.algorithms.common.dflash_family_model import (
    OnlineDFlashModel, SelectorAuxTerms, selector_training_auxiliaries,
    candidate_distillation_kl, gather_selector_teacher_hidden,
    redistribute_reachable_weights,
)
from specforge.algorithms.dflash.providers import (
    SLOTDEEP_LEGACY_TRAINING_EXTENSION, SLOTDEEP_TRAINING_EXTENSION_KEY, resume_contract,
)
from specforge.config import Config
from specforge.modeling.draft.pspr_slotdeep import SlotDeepCorrector


def auxiliary(scores, target, *, covered=None, valid=None, ranking=None, anchor=False):
    if covered is None:
        covered = torch.ones_like(target, dtype=torch.bool)
    if valid is None:
        valid = torch.ones_like(target, dtype=torch.float)
    if ranking is None:
        ranking = valid
    pick = SlotDeepCorrector.select_margin_gate(scores, rho=3, tau=0, theta=0)
    return selector_training_auxiliaries(scores, target, covered, valid, ranking, pick,
                                         excludes_anchor=not anchor, rho=3,
                                         safe_margin=.1, repair_margin=.1)


def make_model(**options):
    torch.manual_seed(313)
    head = SlotDeepCorrector(hidden_size=32, vocab_size=97, top_k=4, d=16, n_layers=1,
                            n_heads=4, state_dim=16, delta_hidden=48, max_slots=8,
                            slot_layers=2, dropout=0., anchor_fusion='concat',
                            candidate_rank_dim=8, candidate_query_hidden=24)
    embedding = nn.Embedding(97, 32).requires_grad_(False)
    head.bind_target_embedding(embedding)
    nn.init.normal_(head.delta_w2.weight, std=.01)
    nn.init.normal_(head.rank_out.weight, std=.01)
    draft = nn.Module()
    draft.candidate_selector = head
    draft.config = SimpleNamespace(hidden_size=32, num_hidden_layers=1,
                                   dflash_config={'selector_top_k': 4})
    draft.target_layer_ids = [0]
    draft.block_size = 4
    draft.selector_decision_mode = 'margin_gate'
    draft.selector_gate_rho, draft.selector_gate_tau, draft.selector_gate_theta = 3., 0., 0.
    draft.selector_compute_dtype = 'float32'
    draft.transform_unary_logits = lambda x: x.float()
    draft.enforce_selector_compute_dtype = lambda: head.float()
    lm_head = nn.Linear(32, 97, bias=False)
    lm_head.weight = embedding.weight
    opts = dict(selector_objective='multiclass', selector_weight_mode='uniform_frontier_boost',
                selector_frontier_boost=3., selector_err_loss_alpha=1.)
    opts.update(options)
    return OnlineDFlashModel(draft_model=draft, target_lm_head=lm_head,
                             target_embed_tokens=embedding, block_size=4, mask_token_id=5,
                             attention_backend='eager', **opts)


def test_reachable_mixture_conserves_covered_mass_per_block_and_retains_suffix():
    original = torch.tensor([[1., 4., 1., 1.], [1., 4., 1., 0.], [0., 0., 0., 0.]], requires_grad=True)
    reachable = torch.tensor([[True, True, False, False]] * 3)
    covered = torch.tensor([[True, True, True, True], [True, False, True, False], [False]*4])
    mixed = redistribute_reachable_weights(original, reachable, covered, .25)
    assert not mixed.requires_grad
    torch.testing.assert_close((mixed * covered).sum(-1), (original * covered).sum(-1))
    assert (mixed[0, :2] > original[0, :2]).all()
    assert (mixed[0, 2:] > 0).all() and (mixed[0, 2:] < original[0, 2:]).all()
    assert torch.count_nonzero(mixed[original == 0]) == 0
    assert torch.isfinite(mixed).all()
    torch.testing.assert_close(redistribute_reachable_weights(original, reachable, covered, 1.),
                               original, rtol=0, atol=0)
    # Row-local normalization preserves chunk/microbatch/rank partitions.
    split = torch.cat([redistribute_reachable_weights(original[i:i+1], reachable[i:i+1],
                       covered[i:i+1], .25) for i in range(3)])
    torch.testing.assert_close(split, mixed, rtol=0, atol=0)


@pytest.mark.parametrize('floor', [0., -.1, float('inf'), float('nan'), 1.01])
def test_reachable_mixture_invalid_floor_fails_closed(floor):
    with pytest.raises(ValueError, match='floor'):
        redistribute_reachable_weights(torch.ones(1, 3), torch.ones(1, 3).bool(),
                                       torch.ones(1, 3).bool(), floor)


def test_reachable_mixture_contract_no_new_parameters():
    old = make_model()
    new = make_model(selector_weight_mode='reachable_frontier_boost', selector_survival_floor=.25,
                     selector_alt_loss_alpha=.5, selector_safe_loss_alpha=1., selector_repair_loss_alpha=.1)
    assert old.state_dict().keys() == new.state_dict().keys()
    for name, value in old.state_dict().items():
        torch.testing.assert_close(value, new.state_dict()[name], rtol=0, atol=0)
    contract = resume_contract(None, new.draft_model, new)
    assert contract['dflash2_selector_weight_mode'] == 'reachable_frontier_boost'
    assert contract['dflash2_selector_survival_floor'] == .25
    with pytest.raises(ValueError, match='active SlotDeep multiclass'):
        make_model(selector_weight_mode='reachable_frontier_boost', selector_objective='candidate_distill')


def test_alt_ce_has_zero_direct_base_gradient():
    scores = torch.tensor([[[2., 0., 1., -.5]]], requires_grad=True)
    t = auxiliary(scores, torch.tensor([[1]]))
    t.alt_ce_num.backward()
    assert scores.grad[..., 0].item() == 0
    assert scores.grad[..., 1].item() < 0
    assert (scores.grad[..., 2:] > 0).all()


def test_safe_and_repair_gradients_have_correct_signs():
    unsafe = torch.tensor([[[0., 2., -1., -2.]]], requires_grad=True)
    t = auxiliary(unsafe, torch.tensor([[0]]))
    assert t.safe_den == t.safe_violation_num == 1 and t.repair_den == 0
    t.safe_num.backward()
    torch.testing.assert_close(unsafe.grad, torch.tensor([[[-1., 1., 0., 0.]]]))
    missed = torch.tensor([[[2., 0., 1., -2.]]], requires_grad=True)
    t = auxiliary(missed, torch.tensor([[1]]))
    assert t.repair_den == t.repair_violation_num == 1 and t.safe_den == 0
    t.repair_num.backward()
    torch.testing.assert_close(missed.grad, torch.tensor([[[1., -1., 0., 0.]]]))


def test_safe_hinge_does_not_keep_pushing_already_safe_scores():
    scores = torch.tensor([[[3., 0., -1., -2.]]], requires_grad=True)
    t = auxiliary(scores, torch.tensor([[0]]))
    assert t.safe_num == 0 and t.safe_den == 1
    t.safe_num.backward()
    assert torch.count_nonzero(scores.grad) == 0


def test_miss_has_no_fabricated_keep_and_stops_boundary_credit():
    scores = torch.tensor([[[0., 2., -1., -2.], [2., 0., 1., -2.]]], requires_grad=True)
    t = auxiliary(scores, torch.tensor([[0, 1]]), covered=torch.tensor([[False, True]]))
    assert t.safe_num == t.repair_num == t.safe_den == t.repair_den == 0
    assert t.alt_ce_num > 0 and t.alt_weight_den == 1  # uniform ranking continues after miss
    t.alt_ce_num.backward()
    assert torch.count_nonzero(scores.grad[:, 0]) == 0
    assert torch.count_nonzero(scores.grad[:, 1]) > 0


def test_padding_and_hole_cannot_restart_boundary_credit():
    scores = torch.tensor([[[3., 0., -1., -2.], [3., 0., -1., -2.], [0., 2., -1., -2.]]])
    target = torch.zeros(1, 3, dtype=torch.long)
    t = auxiliary(scores, target, valid=torch.tensor([[1., 0., 1.]]))
    assert t.safe_den == 1 and t.safe_num == 0
    empty = auxiliary(scores, target, valid=torch.zeros(1, 3))
    assert all(torch.isfinite(v) and v == 0 for v in empty)


def test_anchor_exclusion_matches_proposal_only():
    scores = torch.tensor([[[9., 0., -1., -2.], [3., 0., -1., -2.], [2., 0., 1., -2.]]])
    target, valid = torch.tensor([[0, 0, 1]]), torch.tensor([[0., 1., 1.]])
    retained = auxiliary(scores, target, valid=valid, anchor=True)
    sliced = auxiliary(scores[:, 1:], target[:, 1:], valid=valid[:, 1:])
    for a, b in zip(retained, sliced):
        torch.testing.assert_close(a, b)


def test_alternative_permutation_preserves_losses():
    scores = torch.tensor([[[0., 2., -1., -2.], [2., 0., 1., -2.]]])
    target, permutation = torch.tensor([[1, 1]]), torch.tensor([0, 3, 1, 2])
    inverse = permutation.argsort()
    a = auxiliary(scores, target)
    b = auxiliary(scores[..., permutation], inverse[target])
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y)


def test_uneven_partition_has_identical_additive_gradients():
    torch.manual_seed(2)
    initial = torch.randn(5, 3, 4)
    target = torch.randint(0, 4, (5, 3))
    covered = torch.rand(5, 3) > .2
    valid = (torch.rand(5, 3) > .2).float()
    valid[0] = 0
    def run(slices):
        scores = initial.clone().requires_grad_()
        totals = None
        for part in slices:
            t = auxiliary(scores[part], target[part], covered=covered[part], valid=valid[part])
            totals = t if totals is None else SelectorAuxTerms(*(x+y for x, y in zip(totals, t)))
        loss = (.5*totals.alt_ce_num + .1*totals.safe_num + .1*totals.repair_num) / valid.sum()
        loss.backward()
        return totals, scores.grad
    whole, grad = run([slice(None)])
    split, split_grad = run([slice(0, 1), slice(1, 3), slice(3, 5)])
    for a, b in zip(whole, split):
        torch.testing.assert_close(a, b)
    torch.testing.assert_close(grad, split_grad)


@pytest.mark.parametrize('bad', [-.1, float('inf'), float('nan')])
def test_nonfinite_or_negative_weights_rejected(bad):
    with pytest.raises(ValueError, match='finite and nonnegative'):
        make_model(selector_alt_loss_alpha=bad)


def test_invalid_policy_and_objective_fail_closed():
    with pytest.raises(ValueError, match='auxiliary losses require'):
        make_model(selector_alt_loss_alpha=.5, selector_weight_mode='expected_accept')
    with pytest.raises(ValueError, match='auxiliary losses require'):
        make_model(selector_alt_loss_alpha=.5, selector_loss_alpha=0.)


def test_typed_config_rejects_ignored_extensions():
    with pytest.raises(ValueError, match="require training.strategy"):
        Config.model_validate(dict(model=dict(target_model_path='/target'),
                                   data=dict(hidden_states_path='/features'),
                                   training=dict(strategy='domino', dflash2_selector_alt_loss_alpha=.5)))


def test_extension_checkpoint_contract_records_even_disabled_arm():
    base, active = make_model(), make_model(selector_alt_loss_alpha=.5, selector_preserve_fp32=True)
    a = resume_contract(None, base.draft_model, base)
    b = resume_contract(None, active.draft_model, active)
    assert a[SLOTDEEP_TRAINING_EXTENSION_KEY] == SLOTDEEP_LEGACY_TRAINING_EXTENSION
    assert b[SLOTDEEP_TRAINING_EXTENSION_KEY]['alt_loss_alpha'] == .5
    assert b[SLOTDEEP_TRAINING_EXTENSION_KEY]['preserve_fp32'] is True
    assert a['pspr_slotdeep_reference_config'] == b['pspr_slotdeep_reference_config']
    assert a[SLOTDEEP_TRAINING_EXTENSION_KEY] != b[SLOTDEEP_TRAINING_EXTENSION_KEY]


@pytest.mark.parametrize('preserve', [False, True])
def test_wrapper_assembly_preserves_values_only_when_requested(preserve):
    from specforge.algorithms import model_providers
    from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead
    model = make_model(selector_preserve_fp32=preserve)
    draft = model.draft_model
    with torch.no_grad():
        draft.candidate_selector.h_in.weight.fill_(1.000123)
    exact = draft.candidate_selector.h_in.weight.detach().clone()
    cfg = Config.model_validate(dict(model=dict(target_model_path='/target'),
                                     data=dict(hidden_states_path='/features'),
                                     training=dict(strategy='dflash', attention_backend='eager')))
    parts = SimpleNamespace(embed_tokens=model.embed_tokens, lm_head=model.lm_head)
    with patch.object(model_providers, '_device', return_value=torch.device('cpu')), \
         patch.object(model_providers, '_torch_dtype', return_value=torch.bfloat16), \
         patch.object(model_providers, '_validate_dflash_block_size'), \
         patch.object(model_providers, '_resolve_mask_token_id', return_value=5), \
         patch.object(TargetEmbeddingsAndHead, 'from_pretrained', return_value=parts):
        model_providers._build_dflash_family_model(cfg, draft, None, lambda common: model)
    expected = exact if preserve else exact.bfloat16().float()
    assert torch.equal(draft.candidate_selector.h_in.weight, expected)
    assert draft.candidate_selector.h_in.weight.dtype == torch.float32


@pytest.mark.parametrize('weight_mode', ['uniform_frontier_boost', 'reachable_frontier_boost'])
def test_real_forward_chunking_and_all_head_gradients(weight_mode):
    options = dict(selector_alt_loss_alpha=.5, selector_safe_loss_alpha=.1,
                   selector_repair_loss_alpha=.1, selector_preserve_fp32=True,
                   selector_weight_mode=weight_mode, selector_survival_floor=.25)
    snapshots = []
    for chunks in (0, 1):
        model = make_model(objective_chunk_blocks=chunks, **options)
        torch.manual_seed(77)
        hidden = torch.randn(2, 2, 4, 32)
        ids = model.lm_head(hidden).topk(4).indices
        labels = ids[..., 0].clone()
        labels[..., 2] = ids[..., 2, 1]
        labels[..., 3] = ids[..., 3, 2]
        input_ids = labels.reshape(2, 8)
        anchors = torch.tensor([[0, 4], [0, 4]])
        model._forward_draft_blocks = lambda **kw: (anchors, torch.ones_like(anchors).bool(),
                                                     hidden.reshape(2, 8, 32))
        loss, _, metrics = model(input_ids, torch.zeros(2, 8, 32), torch.ones(2, 8))
        assert torch.isfinite(loss)
        assert 'selector_aux_loss_per_base_weight' in metrics['ratio_metrics']
        loss.backward()
        grad = {}
        for name, param in model.draft_model.candidate_selector.named_parameters():
            assert param.grad is not None and torch.isfinite(param.grad).all(), name
            grad[name] = param.grad.clone()
        snapshots.append((loss.detach(), grad, metrics['ratio_metrics']))
    torch.testing.assert_close(snapshots[0][0], snapshots[1][0], rtol=2e-6, atol=2e-6)
    for name, grad in snapshots[0][1].items():
        torch.testing.assert_close(grad, snapshots[1][1][name], rtol=2e-5, atol=2e-6)
    for name, values in snapshots[0][2].items():
        for a, b in zip(values, snapshots[1][2][name]):
            torch.testing.assert_close(a, b, rtol=2e-6, atol=2e-6)


def test_candidate_distill_additivity_masking_and_frozen_teacher():
    torch.manual_seed(17)
    initial = torch.randn(5, 3, 4)
    teacher = torch.randn_like(initial, requires_grad=True)
    mask = torch.tensor([[1., 1., 0.], [0., 0., 0.], [1., 1., 1.],
                         [1., 0., 1.], [1., 1., 1.]])
    results = []
    for slices in ([slice(None)], [slice(0, 1), slice(1, 3), slice(3, 5)]):
        scores = initial.clone().requires_grad_()
        terms = [candidate_distillation_kl(scores[p], teacher[p], mask[p], 1.5) for p in slices]
        numerator = sum(t[0] for t in terms)
        denominator = sum(t[1] for t in terms)
        (numerator / denominator).backward()
        assert teacher.grad is None
        assert not torch.count_nonzero(scores.grad[mask == 0])
        results.append((numerator.detach(), denominator, scores.grad))
    for a, b in zip(*results):
        torch.testing.assert_close(a, b)


def test_candidate_distill_zero_at_teacher_and_logit_shift_invariance():
    student = torch.tensor([[[1., -2., 3., .5]]], requires_grad=True)
    teacher = student.detach() + 123.
    numerator, denominator = candidate_distillation_kl(student, teacher, torch.ones(1, 1))
    assert denominator == 1
    assert abs(numerator.item()) < 2e-7
    numerator.backward()
    assert student.grad.abs().max() < 2e-7


def test_candidate_distill_supplies_nonuniform_ranking_without_a_hard_class():
    # Production KL has no hard-class argument: even on an uncovered slot the
    # same teacher candidates provide relative supervision, not a fake KEEP label.
    student = torch.zeros(1, 1, 4, requires_grad=True)
    teacher = torch.tensor([[[0., -1., 4., 1.]]])
    numerator, _ = candidate_distillation_kl(student, teacher, torch.ones(1, 1))
    numerator.backward()
    assert student.grad[0, 0, 2] < 0
    assert student.grad[0, 0, 0] > 0
    assert len(torch.unique(student.grad)) == 4


def test_teacher_hidden_alignment_is_previous_position_not_same_position():
    teacher = torch.arange(24.).reshape(2, 6, 2).requires_grad_()
    positions = torch.tensor([[[0, 1, 2], [3, 4, 5]], [[0, 1, 2], [3, 4, 5]]])
    aligned = gather_selector_teacher_hidden(teacher, positions)
    for b in range(2):
        for a in range(2):
            for k in range(3):
                torch.testing.assert_close(aligned[b, a, k], teacher[b, max(int(positions[b, a, k])-1, 0)])
    assert not aligned.requires_grad


def test_distill_forward_chunking_no_new_parameters_or_inference_input():
    snapshots = []
    base = make_model()
    for chunks in (0, 1):
        model = make_model(selector_objective='candidate_distill', objective_chunk_blocks=chunks,
                           selector_preserve_fp32=True)
        assert list(model.state_dict()) == list(base.state_dict())
        assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in base.parameters())
        torch.manual_seed(77)
        hidden = torch.randn(2, 2, 4, 32)
        labels = model.lm_head(hidden).topk(4).indices[..., 1].clone()
        input_ids = labels.reshape(2, 8)
        anchors = torch.tensor([[0, 4], [0, 4]])
        model._forward_draft_blocks = lambda **kw: (anchors, torch.ones_like(anchors).bool(), hidden.reshape(2, 8, 32))
        teacher = torch.randn(2, 8, 32, requires_grad=True)
        seen = []
        original = model.draft_model.candidate_selector.score_candidates
        def checked_score(**kwargs):
            assert 'selector_teacher_hidden' not in kwargs and 'target_last_hidden_states' not in kwargs
            value = original(**kwargs)
            seen.append(value[0].detach().clone())
            return value
        model.draft_model.candidate_selector.score_candidates = checked_score
        with pytest.raises(ValueError, match='online target_last_hidden_states'):
            model(input_ids, torch.zeros(2, 8, 32), torch.ones(2, 8))
        loss, _, metrics = model(input_ids, torch.zeros(2, 8, 32), torch.ones(2, 8),
                                 target_last_hidden_states=teacher)
        assert torch.isfinite(loss)
        assert metrics['ratio_metrics']['selector_distill_weight_per_valid'][0] == 12
        assert metrics['ratio_metrics']['selector_distill_weight_per_valid'][1] == 12
        before = [x.clone() for x in seen]
        seen.clear()
        other, _, _ = model(input_ids, torch.zeros(2, 8, 32), torch.ones(2, 8),
                             target_last_hidden_states=-teacher)
        assert other != loss
        assert len(before) == len(seen)
        for a, b in zip(before, seen):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        loss.backward()
        assert teacher.grad is None
        grad = {}
        for name, param in model.draft_model.candidate_selector.named_parameters():
            assert param.grad is not None and torch.isfinite(param.grad).all(), name
            grad[name] = param.grad.clone()
        snapshots.append((loss.detach(), grad))
    torch.testing.assert_close(snapshots[0][0], snapshots[1][0], atol=2e-6, rtol=2e-6)
    for name, grad in snapshots[0][1].items():
        torch.testing.assert_close(grad, snapshots[1][1][name], atol=2e-6, rtol=2e-5)


def test_candidate_distill_resume_contract_and_fail_closed_options():
    base = make_model()
    model = make_model(selector_objective='candidate_distill')
    a = resume_contract(None, base.draft_model, base)
    b = resume_contract(None, model.draft_model, model)
    key = 'pspr_slotdeep_candidate_distill_v1'
    assert key not in a
    assert b[key]['alpha'] == .5 and b[key]['temperature'] == 1.
    assert b['dflash2_selector_objective'] == 'candidate_distill'
    with pytest.raises(ValueError, match='active SlotDeep CE/KL'):
        make_model(selector_objective='candidate_distill', selector_distill_alpha=0)
    with pytest.raises(ValueError, match='positive'):
        make_model(selector_objective='candidate_distill', selector_distill_temperature=0)


def test_candidate_teacher_contract_is_online_opt_in_and_does_not_mutate_registry():
    from pathlib import Path
    from specforge.config import load_config
    from specforge.application import resolve_run, bind_run
    from specforge.algorithms.builtin import builtin_algorithm_registry
    root = Path(__file__).resolve().parents[2]
    cfg = load_config(str(root/'examples/configs/online/disaggregated/managed-local/qwen3-4b-pspr-slotdeep.yaml'))
    registry = builtin_algorithm_registry()
    original = registry.resolve('dflash')
    assert resolve_run(cfg, registry).algorithm is original
    cfg.training.dflash2_selector_objective = 'candidate_distill'
    run = resolve_run(cfg, registry)
    assert run.algorithm is not original
    assert original.providers.server_streaming_for('text').layout.last_hidden_feature is None
    contract = run.algorithm.spec.feature_contract('streaming', 'text')
    assert 'target_last_hidden_states' in contract.required_tensors
    assert contract.default_target_representation == 'hidden_state'
    stream = run.algorithm.providers.server_streaming_for('text')
    assert stream.layout.last_hidden_feature == 'target_last_hidden_states'
    assert stream.target_representation == 'hidden_state'
    assert bind_run(cfg, run.algorithm).algorithm == run.algorithm
    collated = stream.build_collator()([dict(input_ids=torch.ones(1, 2, dtype=torch.long),
        loss_mask=torch.ones(1, 2), hidden_states=torch.zeros(1, 2, 160),
        target_last_hidden_states=torch.zeros(1, 2, 32))])
    assert collated['target_last_hidden_states'].shape == (1, 2, 32)
