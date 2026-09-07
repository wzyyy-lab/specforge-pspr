"""Tiny deterministic contract tests; synthetic tensors are NOT quality evidence."""
import copy
import pytest
import torch
from torch import nn
from transformers import Qwen3Config
from specforge.modeling.draft.pspr_cloze import ClozeCorrector
from specforge.modeling.draft.pspr_memory import (
    MemoryCorrector, PSPRMemoryDraftModel, VerifiedMemoryCache, gather_verified_memory)


KW = dict(hidden_size=32, vocab_size=97, d=16, n_layers=1, n_heads=4,
          state_dim=16, delta_hidden=32, max_slots=8, dropout=0.0, top_k=4)


def pair():
    torch.manual_seed(42)
    old = ClozeCorrector(**KW).eval()
    nn.init.normal_(old.delta_w2.weight, std=.02)  # mature, nonzero old scorer
    new = MemoryCorrector(**KW, memory_length=8, memory_dropout=0.0).eval()
    result = new.load_state_dict(old.state_dict(), strict=False)
    assert all(k.startswith('memory_refiner.') for k in result.missing_keys)
    emb = nn.Embedding(97, 32)
    old.bind_target_embedding(emb)
    new.bind_target_embedding(emb)
    return old, new


def inputs():
    torch.manual_seed(51)
    return dict(hidden_states=torch.randn(2, 3, 4, 32),
                candidate_ids=torch.randint(0, 97, (2, 3, 4, 4)),
                predecessor_ids=torch.randint(0, 97, (2, 3, 4)),
                unary_logits=torch.randn(2, 3, 4, 4).log_softmax(-1),
                lattice_scalars=torch.randn(2, 3, 4, 3))


def test_strict_prefix_no_future_and_cache_alignment():
    x = torch.arange(2*20*64).reshape(2, 20, 64).float()
    a = torch.tensor([[0, 3, 17], [1, 5, 20]])
    memory, valid = gather_verified_memory(x, a, 8, 32)
    for b in range(2):
        for n in range(3):
            anchor = int(a[b,n])
            torch.testing.assert_close(memory[b,n][valid[b,n]], x[b,max(0,anchor-8):anchor,-32:])
    assert not valid[0,0].any()
    a = torch.tensor([[7], [7]])
    changed = x.clone()
    changed[:,7:] += 12345
    assert torch.equal(gather_verified_memory(x,a,8,32)[0], gather_verified_memory(changed,a,8,32)[0])
    cache = VerifiedMemoryCache(8,32)
    cache.append(x[:,:5],0)
    cache.append(x[:,5:9],5)
    cm, cv = cache.read()
    gm, gv = gather_verified_memory(x,torch.tensor([[9],[9]]),8,32)
    assert torch.equal(cm,gm[:,0]) and torch.equal(cv,gv[:,0])
    with pytest.raises(ValueError):
        cache.append(x[:,:2],8)
    cache.append(x[:,:2],0)
    assert cache.end == 2


def test_warm_start_exact_and_missing_memory_fails():
    old, new = pair()
    data = inputs()
    memory, valid = torch.randn(2,3,8,32), torch.ones(2,3,8,dtype=torch.bool)
    expected = old.score_candidates(**data, return_err=True, return_full=True)
    actual = new.score_candidates(**data, target_memory=memory, memory_mask=valid,
                                   return_err=True, return_full=True)
    for a,b in zip(expected,actual):
        assert torch.equal(a,b)
    with pytest.raises(RuntimeError):
        new.score_candidates(**data)


def test_context_live_after_zero_output_opens_and_padding_masked():
    _, new = pair()
    nn.init.normal_(new.memory_refiner.out.weight, std=.03)
    data = inputs()
    memory = torch.randn(2,3,8,32)
    valid = torch.ones(2,3,8,dtype=torch.bool)
    valid[...,:3] = False
    a = new.score_candidates(**data, target_memory=memory, memory_mask=valid)
    changed = memory.clone()
    changed[...,:3,:] += 1000
    b = new.score_candidates(**data, target_memory=changed, memory_mask=valid)
    assert torch.equal(a,b)
    changed[...,3:,:] = torch.randn_like(changed[...,3:,:])
    c = new.score_candidates(**data, target_memory=changed, memory_mask=valid)
    assert not torch.allclose(a,c)
    c.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in new.memory_refiner.parameters())


def test_null_prefix_and_controls_are_finite():
    _, new = pair()
    data = inputs()
    nn.init.normal_(new.memory_refiner.out.weight, std=.03)
    for source in ('verified','blind','draft_future'):
        new.memory_source = source
        for causal in (False,True):
            new.memory_future_causal = causal
            scores = new.score_candidates(**data, target_memory=torch.zeros(2,3,8,32),
                                           memory_mask=torch.zeros(2,3,8,dtype=torch.bool))
            assert torch.isfinite(scores).all()


def test_draft_init_freezing_and_zero_restore():
    cfg = Qwen3Config(architectures=['PSPRMemoryDraftModel'], block_size=4,
        hidden_size=32, intermediate_size=64, num_attention_heads=4, num_key_value_heads=2,
        num_hidden_layers=1, head_dim=8, max_position_embeddings=64, vocab_size=97,
        num_target_layers=4)
    cfg._attn_implementation='eager'
    cfg.dflash_config=dict(mask_token_id=5,target_layer_ids=[0],selector_top_k=4,
        selector_dim=16,selector_layers=1,selector_heads=4,selector_delta_hidden=32,
        selector_state_dim=16,selector_max_slots=8,selector_dropout=.1,
        selector_compute_dtype='float32',freeze_backbone=True,memory_length=8)
    model=PSPRMemoryDraftModel(cfg)
    model.apply_backbone_freeze()
    model.train()
    assert not model.candidate_selector.encoder.training
    assert model.candidate_selector.memory_refiner.training
    assert torch.count_nonzero(model.candidate_selector.memory_refiner.out.weight)==0
    assert all(n.startswith('candidate_selector.memory_refiner.')
               for n,p in model.named_parameters() if p.requires_grad)


def test_chunk_objective_with_explicit_memory():
    from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
    from types import SimpleNamespace
    _, selector = pair()
    draft = nn.Module()
    draft.candidate_selector = selector
    draft.config = SimpleNamespace(hidden_size=32)
    draft.transform_unary_logits = lambda x: x.float()
    draft.selector_decision_mode = 'margin_gate'
    draft.selector_gate_theta = 0.0
    draft.selector_gate_rho = 3.0
    target_embed = nn.Embedding(97,32)
    selector.bind_target_embedding(target_embed)
    target_head = nn.Linear(32,97,bias=False)
    model = OnlineDFlashModel(draft_model=draft, target_lm_head=target_head,
        target_embed_tokens=target_embed, block_size=5, mask_token_id=5,
        attention_backend='eager', selector_weight_mode='uniform_frontier_boost',
        selector_frontier_boost=3.0, selector_objective='multiclass')
    h = torch.randn(2,3,5,32,requires_grad=True)
    ids = torch.randint(0,97,(2,3,5))
    pred = torch.cat([ids[...,:1],ids[...,:-1]],-1)
    w = torch.ones(2,3,5)
    w[...,0] = 0
    terms = model._dflash_objective_chunk_terms(h,ids,w,pred,None,
        torch.randn(2,3,8,32),torch.ones(2,3,8,dtype=torch.bool))
    assert all(torch.isfinite(x).all() for x in terms)
    (terms.ce_loss_num+terms.selector_ce_num).backward()
    assert selector.memory_refiner.out.weight.grad is not None
