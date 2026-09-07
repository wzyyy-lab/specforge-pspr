"""Stage2 config/gradient guards; synthetic tensors are not quality evidence."""
import json
from pathlib import Path

import pytest
import torch
from transformers import Qwen3Config

from specforge.modeling.draft.pspr_slotdeep import PSPRSlotDeepDraftModel
from specforge.optimizer import BF16Optimizer
from tests.test_modeling.test_pspr_slotdeep_training import make_model

ROOT = Path(__file__).resolve().parents[2]


def test_joint_config_changes_only_freeze_and_trains_both_halves():
    original = json.loads((ROOT/'configs/qwen3-4b-pspr-slotdeep.json').read_text())
    joint = json.loads((ROOT/'configs/qwen3-4b-pspr-slotdeep-joint.json').read_text())
    assert original['dflash_config']['freeze_backbone'] is True
    assert joint['dflash_config']['freeze_backbone'] is False
    with torch.device('meta'):
        model = PSPRSlotDeepDraftModel(Qwen3Config(**joint))
    assert model.apply_backbone_freeze() == 0
    params = dict(model.named_parameters())
    assert len(params) == 162 and all(p.requires_grad for p in params.values())
    assert sum(p.numel() for n,p in params.items() if n.startswith('candidate_selector.')) == 47152913
    assert sum(p.numel() for n,p in params.items() if not n.startswith('candidate_selector.')) == 537427200
    joint['dflash_config']['freeze_backbone'] = True
    assert joint == original


@pytest.mark.parametrize('stopped', [False, True])
def test_selector_loss_reaches_backbone_hidden_unless_explicitly_stopped(stopped):
    model = make_model(selector_stop_gradient=stopped, selector_alt_loss_alpha=.5,
                       selector_safe_loss_alpha=1., selector_repair_loss_alpha=.1)
    hidden = torch.randn(2,4,32,requires_grad=True)
    logits = model.lm_head(hidden)
    labels = logits.detach().topk(4).indices[...,1]
    terms = model._selector_chunk_terms(model.draft_model.candidate_selector, logits,
        hidden, labels, torch.zeros_like(labels), torch.ones_like(labels).float(),
        torch.ones_like(labels).float())
    loss = terms.ce_num + terms.err_ce_num + .5*terms.alt_ce_num + terms.safe_num + .1*terms.repair_num
    loss.backward()
    if stopped:
        assert hidden.grad is None
    else:
        assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
        assert hidden.grad.abs().max() > 0
    assert model.lm_head.weight.grad is None


def test_joint_lr_prefix_rules_do_not_throttle_selector():
    inner = torch.nn.Module()
    inner.candidate_selector = torch.nn.Linear(3,4)
    inner.layers = torch.nn.Linear(2,2)
    model = torch.nn.Module()
    model.draft_model = inner
    rules = tuple(sorted({'candidate_selector.':1., '':.3}.items(), key=lambda kv:-len(kv[0])))
    optimizer = BF16Optimizer(model,lr=1e-4,total_steps=20,warmup_ratio=.06,lr_scale_rules=rules)
    assert optimizer.scheduler.base_lrs == [1e-4,3e-5]
    assert [sum(p.numel() for p in g['params']) for g in optimizer.optimizer.param_groups] == [16,6]

