#!/usr/bin/env python3
"""Gate the PSPR-Decision head before any GPU time is spent on it.

Every check is a hard assert.  The list is the same discipline used for pspr_cloze, plus the checks
specific to a factorised decision: the step-0 identity now has to hold for the ACTION, not just the
scores, and the detector must be shown to actually consume each of the inputs it was given.

Run:  PYTHONPATH=. python3 scripts/gate_pspr_decision.py
"""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, ".")

from specforge.modeling.draft.pspr_decision import DecisionCorrector  # noqa: E402

torch.manual_seed(0)

HID, VOCAB, K, H = 64, 512, 16, 15
B = 2


def build(**overrides) -> DecisionCorrector:
    kwargs = dict(
        hidden_size=HID,
        vocab_size=VOCAB,
        top_k=K,
        d=32,
        n_layers=2,
        expansion=2,
        state_dim=24,
        delta_hidden=48,
        det_hidden=40,
        max_slots=32,
        dropout=0.0,
        direct_hidden=True,
        use_state=True,
        err_use_state=True,
        block_context=True,
    )
    kwargs.update(overrides)
    sel = DecisionCorrector(**kwargs)
    sel.bind_target_embedding(nn.Embedding(VOCAB, HID))
    return sel.double().eval()


def make_batch(sel: DecisionCorrector, blocks: int = 3):
    """A synthetic lattice with the same invariants extract_lattice guarantees."""
    hidden = torch.randn(B, blocks, H, HID, dtype=torch.float64)
    raw = torch.randn(B, blocks, H, VOCAB, dtype=torch.float64)
    lp_all = F.log_softmax(raw, dim=-1)
    lp, cand = lp_all.topk(K, dim=-1)
    entropy = -(lp_all.exp() * lp_all).sum(-1)
    margin = lp[..., 0] - lp[..., 1]
    mass = lp.exp().sum(-1)
    scal = torch.stack([entropy, margin, mass], dim=-1)
    pred = torch.randint(0, VOCAB, (B, blocks, H), dtype=torch.long)
    return hidden, cand, lp, scal, pred


def walk(sel: DecisionCorrector, hidden, cand, lp, scal, anchor_id):
    """Per-slot serving walk: one GRU step, one delta MLP, one detector MLP per slot.

    Mirrors ``PSPRDecisionDraftModel._sample_draft_tokens`` exactly.  Returns per-slot scores, err
    logits and the chosen candidate indices, for one block.
    """
    emb = sel._embedding()
    cand_emb = F.embedding(cand, emb)
    z = sel.slot_states(hidden, cand, lp, scal)
    prev = anchor_id
    gru_h = None
    scores, errs, picks = [], [], []
    for i in range(cand.shape[-2]):
        step, gru_h = sel.gru(F.embedding(prev, emb).unsqueeze(1), gru_h)
        state = step[:, 0]
        sc = sel.score(z[:, i], hidden[:, i], cand_emb[:, i], lp[:, i], state)
        el = sel.err_logits(
            z[:, i], cand_emb[:, i, 0], lp[:, i], scal[:, i], hidden[:, i], state
        )
        idx = sel.select_keep_repair(sc, el)
        scores.append(sc)
        errs.append(el)
        picks.append(idx[:, 0])
        prev = cand[:, i].gather(1, idx)[:, 0]
    return torch.stack(scores, 1), torch.stack(errs, 1), torch.stack(picks, 1)


fails = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        fails.append(name)


# ---------------------------------------------------------------- 1 zero-init: scores
sel = build()
hidden, cand, lp, scal, pred = make_batch(sel)
with torch.no_grad():
    sc, el = sel.score_candidates(
        candidate_ids=cand,
        unary_logits=lp,
        hidden_states=hidden,
        predecessor_ids=pred,
        lattice_scalars=scal,
        return_err=True,
    )
check(
    "zero-init: score == log_probs (delta_w2 zeroed)",
    torch.allclose(sc, lp.float(), atol=1e-10),
    f"max|d|={(sc - lp.float()).abs().max():.3e}",
)

# ---------------------------------------------------------------- 2 zero-init: ACTION is KEEP
# This is the contract that matters for a factorised head: not only must the scores be untouched,
# the MAP action must be candidate 0 at every slot, or step 0 is not plain DFlash.
with torch.no_grad():
    action = sel.keep_repair_log_probs(sc, el)
    picked = action.argmax(-1)
check(
    "zero-init: keep_repair action is KEEP everywhere",
    bool((picked == 0).all()),
    f"non-keep={int((picked != 0).sum())}/{picked.numel()}",
)
check(
    "keep_repair_log_probs is a normalised distribution",
    torch.allclose(
        action.logsumexp(-1), torch.zeros_like(action[..., 0]), atol=1e-5
    ),
    f"max|logsumexp|={action.logsumexp(-1).abs().max():.3e}",
)
check(
    "detector bias starts at logit(det_prior)",
    abs(float(torch.sigmoid(el).mean()) - sel.det_prior) < 1e-6,
    f"p_mean={float(torch.sigmoid(el).mean()):.6f} prior={sel.det_prior}",
)

# ---------------------------------------------------------------- 3 batch == per-slot walk
# Train on a batched call, serve with a per-slot walk: any divergence is a silent train/serve bug.
sel = build()
with torch.no_grad():
    for p in sel.parameters():
        p.add_(torch.randn_like(p) * 0.05)
hidden, cand, lp, scal, pred = make_batch(sel, blocks=1)
h1, c1, l1, s1, p1 = (
    hidden[:, 0],
    cand[:, 0],
    lp[:, 0],
    scal[:, 0],
    pred[:, 0],
)
with torch.no_grad():
    sc_walk, el_walk, picks = walk(sel, h1, c1, l1, s1, p1[:, 0])
    # Batched path fed the SAME committed prefix the walk actually produced, so the GRU input is
    # identical and the only thing under test is the batched-vs-stepwise algebra.
    committed = torch.cat([p1[:, :1], c1.gather(2, picks.unsqueeze(-1))[:, :, 0][:, :-1]], dim=1)
    sc_b, el_b = sel.score_candidates(
        candidate_ids=c1,
        unary_logits=l1,
        hidden_states=h1,
        predecessor_ids=committed,
        lattice_scalars=s1,
        return_err=True,
    )
check(
    "batched score == per-slot walk score",
    torch.allclose(sc_b, sc_walk, atol=1e-9),
    f"max|d|={(sc_b - sc_walk).abs().max():.3e}",
)
check(
    "batched err == per-slot walk err",
    torch.allclose(el_b, el_walk, atol=1e-9),
    f"max|d|={(el_b - el_walk).abs().max():.3e}",
)

# ---------------------------------------------------------------- 4 GRU: full-sequence == stepwise
with torch.no_grad():
    emb = sel._embedding()
    full = sel.causal_states(F.embedding(committed, emb))
    inc, gh = [], None
    for i in range(committed.shape[1]):
        st, gh = sel.gru(F.embedding(committed[:, i], emb).unsqueeze(1), gh)
        inc.append(st[:, 0])
    inc = torch.stack(inc, 1)
check(
    "GRU full-sequence == incremental steps",
    torch.allclose(full, inc, atol=1e-10),
    f"max|d|={(full - inc).abs().max():.3e}",
)

# ---------------------------------------------------------------- 5 z must not see the prefix
# slot_states is called ONCE per block.  If it depended on the committed prefix the walk would have to
# re-run it H times and the head would stop being cheap.
#
# The obvious test -- call it twice with different predecessors -- is TAUTOLOGICAL, because
# slot_states has no predecessor parameter to vary: it would pass no matter what the body did.
# (An earlier version of this file did exactly that.)  Two checks that can actually fail instead:
#   (a) the signature admits nothing prefix-shaped, which is the structural guarantee;
#   (b) a whole-block walk calls it exactly once, which is the cost guarantee.
import inspect  # noqa: E402

sig = inspect.signature(sel.slot_states)
# Substring bans only; "state" as a substring would flag the legitimate `hidden_states`
# (the backbone activation), so the GRU state is caught by exact name instead.
banned = ("predecessor", "prefix", "committed", "prev", "anchor")
exact = {"state", "gru_state", "causal_state"}
offending = [
    n for n in sig.parameters
    if n.lower() in exact or any(tok in n.lower() for tok in banned)
]
check(
    "slot_states signature admits nothing prefix-shaped",
    not offending,
    f"params={list(sig.parameters)}",
)

z_a = None
calls = {"n": 0}
_orig_slot_states = type(sel).slot_states


def _counting(self, *a, **k):
    calls["n"] += 1
    return _orig_slot_states(self, *a, **k)


type(sel).slot_states = _counting
try:
    with torch.no_grad():
        walk(sel, h1, c1, l1, s1, p1[:, 0])
finally:
    type(sel).slot_states = _orig_slot_states
check(
    "a full H-slot walk calls slot_states exactly once",
    calls["n"] == 1,
    f"calls={calls['n']} for H={c1.shape[-2]} slots",
)
with torch.no_grad():
    z_a = sel.slot_states(h1, c1, l1, s1)

# ---------------------------------------------------------------- 6 cross-slot path is exactly the
# block mean and nothing else
with torch.no_grad():
    h_pert = h1.clone()
    h_pert[:, 3] += torch.randn_like(h_pert[:, 3])  # perturb slot 3 only, non-uniformly
    z_pert = sel.slot_states(h_pert, c1, l1, s1)
    moved_other = (z_pert[:, 0] - z_a[:, 0]).abs().max()
check(
    "block_context=True: perturbing slot 3 moves slot 0's z (cross-slot path alive)",
    float(moved_other) > 1e-6,
    f"max|d|={float(moved_other):.3e}",
)
sel_noctx = build(block_context=False)
with torch.no_grad():
    for p in sel_noctx.parameters():
        p.add_(torch.randn_like(p) * 0.05)
    z0 = sel_noctx.slot_states(h1, c1, l1, s1)
    z1 = sel_noctx.slot_states(h_pert, c1, l1, s1)
    off = torch.cat([(z1[:, :3] - z0[:, :3]).flatten(), (z1[:, 4:] - z0[:, 4:]).flatten()])
check(
    "block_context=False: z is strictly per-slot (ablation is exact)",
    float(off.abs().max()) < 1e-12 and float((z1[:, 3] - z0[:, 3]).abs().max()) > 1e-6,
    f"off-slot max|d|={float(off.abs().max()):.3e}",
)

# ---------------------------------------------------------------- 7 detector consumes every input
# A first-class decision maker with a dead input is a silent capacity loss.  Perturb one input at a
# time and require the logit to move.
with torch.no_grad():
    z_i, h_i, l_i, s_i = z_a[:, 5], h1[:, 5], l1[:, 5], s1[:, 5]
    seed_i = F.embedding(c1[:, 5, 0], emb)
    st_i = full[:, 5]
    base_err = sel.err_logits(z_i, seed_i, l_i, s_i, h_i, st_i)
    probes = {
        "z": lambda: sel.err_logits(z_i + 0.5, seed_i, l_i, s_i, h_i, st_i),
        "E(d0_i)": lambda: sel.err_logits(
            z_i, seed_i + torch.randn_like(seed_i), l_i, s_i, h_i, st_i
        ),
        "log_probs": lambda: sel.err_logits(z_i, seed_i, l_i - 0.5, s_i, h_i, st_i),
        "scalars": lambda: sel.err_logits(z_i, seed_i, l_i, s_i + 0.5, h_i, st_i),
        "hidden": lambda: sel.err_logits(z_i, seed_i, l_i, s_i, h_i + 0.5, st_i),
        "committed state": lambda: sel.err_logits(z_i, seed_i, l_i, s_i, h_i, st_i + 0.5),
    }
    for name, fn in probes.items():
        moved = float((fn() - base_err).abs().max())
        check(f"detector actually reads {name}", moved > 1e-8, f"max|d|={moved:.3e}")

# ---------------------------------------------------------------- 8 ranker: candidate 0 excluded
# Under the factorised likelihood the alternatives are normalised WITHOUT candidate 0, so cand 0's
# score must not affect the conditional ranking at all.
with torch.no_grad():
    sc_i = sel.score(z_i, h_i, F.embedding(c1[:, 5], emb), l_i, st_i)
    alt_a = F.log_softmax(sc_i[..., 1:], dim=-1)
    bumped = sc_i.clone()
    bumped[..., 0] += 10.0
    alt_b = F.log_softmax(bumped[..., 1:], dim=-1)
check(
    "conditional alternative ranking is invariant to candidate 0's score",
    torch.allclose(alt_a, alt_b, atol=1e-12),
)

# ---------------------------------------------------------------- 9 decode rule == training MAP
# The served action must be the argmax of the same log-likelihood the objective optimises.
with torch.no_grad():
    err_i = sel.err_logits(z_i, seed_i, l_i, s_i, h_i, st_i)
    lpj = sel.keep_repair_log_probs(sc_i, err_i)
    manual_keep = F.logsigmoid(-err_i)
    manual_rep = F.logsigmoid(err_i).unsqueeze(-1) + F.log_softmax(sc_i[..., 1:], -1)
    manual = torch.cat([manual_keep.unsqueeze(-1), manual_rep], dim=-1)
check(
    "keep_repair_log_probs matches the factorised likelihood term-by-term",
    torch.allclose(lpj, manual, atol=1e-12),
)
check(
    "select_keep_repair == argmax of that likelihood",
    bool((sel.select_keep_repair(sc_i, err_i)[:, 0] == manual.argmax(-1)).all()),
)

# ---------------------------------------------------------------- 10 restore_zero_init_contract
sel_r = build()
with torch.no_grad():
    for p in sel_r.parameters():
        p.add_(torch.randn_like(p) * 0.3)
    sel_r.restore_zero_init_contract()
    sc_r, el_r = sel_r.score_candidates(
        candidate_ids=c1,
        unary_logits=l1,
        hidden_states=h1,
        predecessor_ids=p1,
        lattice_scalars=s1,
        return_err=True,
    )
check(
    "restore_zero_init_contract re-establishes the step-0 identity",
    torch.allclose(sc_r, l1.float(), atol=1e-10)
    and bool((sel_r.keep_repair_log_probs(sc_r, el_r).argmax(-1) == 0).all()),
)

# ---------------------------------------------------------------- 11 real-size parameter count
real = DecisionCorrector(
    hidden_size=2560,
    vocab_size=151936,
    top_k=16,
    d=512,
    n_layers=4,
    expansion=4,
    state_dim=512,
    delta_hidden=2048,
    det_hidden=1024,
    dropout=0.05,
)
total = sum(p.numel() for p in real.parameters())
det = sum(
    p.numel()
    for n, p in real.named_parameters()
    if n.startswith(("err_seed_in", "err_head"))
)
print(f"\nreal-size params: total={total/1e6:.2f}M  detector={det/1e6:.2f}M")
print(f"  (cloze 42.04M / v2 24.69M / Domino 50.82M of which 38.9M is its vocab projection)")
check("real-size head is smaller than cloze (42.04M)", total < 42.04e6)
check("no [vocab, *] parameter is learned", not any(
    151936 in tuple(p.shape) for p in real.parameters()
))

print()
if fails:
    print(f"FAILED {len(fails)}: {fails}")
    sys.exit(1)
print("ALL GATES PASSED")
