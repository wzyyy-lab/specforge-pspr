"""Active SlotDeep recipe: real objective gradients, policy parity and artifact contracts.

Small synthetic tensors are unit-test fixtures, never performance evidence.
"""
import importlib.util
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from transformers import Qwen3Config

from specforge.modeling.draft.pspr_slotdeep import SlotDeepCorrector, PSPRSlotDeepDraftModel

ROOT = Path(__file__).resolve().parents[2]


def build(**overrides):
    torch.manual_seed(71)
    args = dict(hidden_size=32, vocab_size=97, top_k=4, d=16, n_layers=2,
                n_heads=4, state_dim=16, delta_hidden=48, max_slots=8,
                slot_layers=2, dropout=0.0, anchor_fusion="concat",
                candidate_rank_dim=8, candidate_query_hidden=24)
    args.update(overrides)
    head = SlotDeepCorrector(**args)
    head.bind_target_embedding(torch.nn.Embedding(97, 32))
    return head


def inputs():
    torch.manual_seed(19)
    return dict(hidden_states=torch.randn(2, 3, 32),
                candidate_ids=torch.randint(0, 97, (2, 3, 4)),
                predecessor_ids=torch.randint(0, 97, (2, 3)),
                unary_logits=torch.log_softmax(torch.randn(2, 3, 4), -1).sort(-1, descending=True).values,
                lattice_scalars=torch.rand(2, 3, 3))


def states(head, x):
    z = head.cloze_states(x["hidden_states"], x["candidate_ids"], x["predecessor_ids"][:, 0],
                          x["unary_logits"], x["lattice_scalars"])
    state = head.causal_states(F.embedding(x["predecessor_ids"], head._embedding()))
    return z, state


def test_zero_init_and_full_head_learning():
    head, x = build(), inputs()
    scores = head.score_candidates(**x)
    assert torch.equal(scores, x["unary_logits"])
    index = head.select_margin_gate(scores, rho=3, tau=0, theta=0)
    assert not torch.count_nonzero(index)
    labels = torch.tensor([[0, 2, 1], [3, 0, 1]])
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0)
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        scores, err = head.score_candidates(**x, return_err=True)
        loss = F.cross_entropy(scores.reshape(-1, 4), labels.reshape(-1))
        loss += F.binary_cross_entropy_with_logits(err, labels.ne(0).float())
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
        opt.step()
    for prefix in ("slot_blocks.", "anchor_fuse.", "rank_query_in.", "rank_query_out.",
                   "rank_candidate_in.", "rank_out.", "gru.", "delta_w2."):
        assert any(p.grad.abs().max() > 0 for n, p in head.named_parameters() if n.startswith(prefix)), prefix


def test_slot_locality_and_prefix_stream():
    head, x = build(), inputs()
    z, state = states(head, x)
    changed = dict(x, candidate_ids=(x["candidate_ids"] + 1) % 97)
    assert torch.equal(states(head, changed)[0], z)  # no concrete candidate enters z
    h = x["hidden_states"].clone()
    h[:, 1] += torch.randn_like(h[:, 1])
    dz = states(head, dict(x, hidden_states=h))[0] - z
    assert not torch.count_nonzero(dz[:, [0, 2]])
    assert torch.count_nonzero(dz[:, 1])
    pred = x["predecessor_ids"].clone()
    pred[:, 1] = (pred[:, 1] + 1) % 97
    _, other = states(head, dict(x, predecessor_ids=pred))
    assert torch.equal(other[:, 0], state[:, 0])
    assert not torch.equal(other[:, 1:], state[:, 1:])


def test_parallel_teacher_prefix_equals_incremental_scoring():
    head, x = build(), inputs()
    torch.nn.init.normal_(head.rank_out.weight, std=.1)
    torch.nn.init.normal_(head.delta_w2.weight, std=.02)
    z, _ = states(head, x)
    batched = head.score_candidates(**x)
    recurrent, rows = None, []
    emb = head._embedding()
    for i in range(3):
        out, recurrent = head.gru(F.embedding(x["predecessor_ids"][:, i], emb)[:, None], recurrent)
        rows.append(head.score(z[:, i], x["hidden_states"][:, i],
                               F.embedding(x["candidate_ids"][:, i], emb),
                               x["unary_logits"][:, i], out[:, 0]))
    torch.testing.assert_close(torch.stack(rows, 1), batched, atol=2e-6, rtol=2e-6)


def test_alternative_permutation_equivariance():
    head, x = build(), inputs()
    torch.nn.init.normal_(head.rank_out.weight, std=.1)
    z, state = states(head, x)
    ce = F.embedding(x["candidate_ids"], head._embedding())
    perm = torch.tensor([0, 3, 1, 2])
    a = head.score(z, x["hidden_states"], ce, x["unary_logits"], state)
    b = head.score(z, x["hidden_states"], ce[..., perm, :], x["unary_logits"][..., perm], state)
    torch.testing.assert_close(b, a[..., perm], atol=1e-6, rtol=1e-6)


def test_config_roundtrip_and_fail_closed_controls():
    head = build()
    copy = SlotDeepCorrector.from_reference_config(head.reference_config())
    copy.load_state_dict(head.state_dict(), strict=True)
    assert copy.reference_config() == head.reference_config()
    with pytest.raises(ValueError, match="no attention"):
        build(bidirectional=False)
    head.bidirectional = False  # decoder's --cloze-causal changes the runtime attribute
    with pytest.raises(ValueError, match="no attention"):
        states(head, inputs())
    head.bidirectional = True
    with pytest.raises(ValueError, match="top-K only"):
        head.full_vocab_logits(None, None)
    c = head.reference_config()
    del c["anchor_fusion"]
    with pytest.raises(KeyError):
        SlotDeepCorrector.from_reference_config(c)


def test_real_config_freezes_backbone_trains_entire_head_and_obeys_budget():
    cfg = json.loads((ROOT / "configs/qwen3-4b-pspr-slotdeep.json").read_text())
    with torch.device("meta"):
        model = PSPRSlotDeepDraftModel(Qwen3Config(**cfg))
    model.apply_backbone_freeze()
    trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all(n.startswith("candidate_selector.") for n, _ in trainable)
    assert 42_100_000 < sum(size for _, size in trainable) < 50_830_000
    assert all(p.requires_grad for p in model.candidate_selector.parameters())
    assert model.candidate_selector.anchor_fusion == "concat"
    assert model.candidate_selector.candidate_rank_dim == 256


def test_active_weight_decay_rules():
    import yaml
    cfg = yaml.safe_load((ROOT / "examples/configs/online/disaggregated/managed-local/qwen3-4b-pspr-slotdeep.yaml").read_text())
    rules = cfg["training"]["weight_decay_rules"]
    head = build()
    for n, p in head.named_parameters():
        name = "candidate_selector." + n
        hits = [v for k, v in rules.items() if name.startswith(k) or f".{k}" in name]
        assert len(hits) <= 1, (n, hits)
        embedding_like = n in {"pos_emb", "role_emb.weight"}
        if p.ndim >= 2 and not embedding_like:
            assert hits == [.002], (n, hits)
        else:
            assert not hits, (n, hits)


def test_export_builder_and_real_decoder_loader(tmp_path):
    spec = importlib.util.spec_from_file_location("slotdeep_export_test", ROOT / "scripts/export_pspr_cloze_for_decode.py")
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    head = build()
    artifact = dict(model_type="SlotDeepCorrector", config=head.reference_config(), state_dict=head.state_dict())
    built = exporter.build(artifact["config"])
    built.load_state_dict(artifact["state_dict"], strict=True)
    path = tmp_path / "selector.pt"
    torch.save(artifact, path)
    spec = importlib.util.spec_from_file_location("slotdeep_decode_test", ROOT.parent / "TAPS-SP/scripts/decode_lattice.py")
    decoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(decoder)
    restored, payload = decoder.load_selector_checkpoint(path)
    assert type(restored) is SlotDeepCorrector
    assert payload["config"] == artifact["config"]
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in head.state_dict().items())
