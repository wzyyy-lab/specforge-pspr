"""Gate: the ``full_vocab`` selector objective is wired end to end and is not silently a no-op.

Checks, on a small synthetic problem so it runs on CPU in seconds:

1. ``OnlineDFlashModel`` accepts ``selector_objective='full_vocab'`` only for a selector that
   advertises ``supports_full_vocab``, and rejects a train/serve policy mismatch.
2. ``_selector_chunk_terms`` returns a finite CE whose denominator counts EVERY valid slot, not just
   the ones whose target happens to sit in the top-K.  This is the whole point of the objective: the
   K-way path leaves ~21% of slots with zero gradient.
3. The CE actually reaches the corrector's parameters (gradient is non-zero), and it is genuinely
   full-vocabulary -- a slot whose target is outside the lattice still contributes loss, which the
   multiclass objective provably cannot do.
4. ``delta == 0`` at init, so the module starts exactly equal to plain DFlash and any measured gain
   is attributable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel  # noqa: E402
from specforge.modeling.draft.pspr_v2 import LatticeCorrector  # noqa: E402

VOCAB, HID, K, H, B = 512, 32, 8, 6, 4
CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""), flush=True)


class _StubDraft(nn.Module):
    """Minimal stand-in for PSPRv2DraftModel: the objective only reads these attributes."""

    def __init__(self, selector, decision_mode):
        super().__init__()
        self.candidate_selector = selector
        self.selector_decision_mode = decision_mode
        self.block_size = H + 1
        self.selector_gate_theta = 0.0
        self.selector_gate_rho = 1.0
        self.selector_gate_tau = 0.0
        self.selector_gate_skip_first = False
        self.selector_keep_repair_margin = 0.0


def build(objective, decision_mode, **kw):
    torch.manual_seed(0)
    selector = LatticeCorrector(
        hidden_size=HID, vocab_size=VOCAB, top_k=K, d=16, n_layers=1, n_heads=2,
        state_dim=16, delta_hidden=32, max_slots=32, dropout=0.0, cand_slots=0, **kw,
    )
    embed = nn.Embedding(VOCAB, HID)
    selector.bind_target_embedding(embed)
    return OnlineDFlashModel(
        draft_model=_StubDraft(selector, decision_mode),
        target_lm_head=nn.Linear(HID, VOCAB, bias=False),
        target_embed_tokens=embed,
        mask_token_id=0,
        block_size=H + 1,
        selector_objective=objective,
        selector_weight_mode="uniform",
        selector_loss_alpha=1.0,
        selector_err_loss_alpha=1.0,
    ), selector


# 1. configuration guards -------------------------------------------------------------------------
try:
    build("full_vocab", "margin_gate")
    check("train/serve mismatch rejected", False, "no error raised")
except ValueError as exc:
    check("train/serve mismatch rejected", "mismatch" in str(exc))

try:
    from specforge.modeling.draft.pspr import LatticePathSelector

    torch.manual_seed(0)
    v1 = LatticePathSelector(hidden_size=HID, vocab_size=VOCAB, top_k=K, d=16, n_layers=1,
                             n_heads=2, state_dim=16, delta_hidden=32, dropout=0.0)
    v1.bind_target_embedding(nn.Embedding(VOCAB, HID))
    OnlineDFlashModel(
        draft_model=_StubDraft(v1, "full_vocab"),
        target_lm_head=nn.Linear(HID, VOCAB, bias=False),
        target_embed_tokens=nn.Embedding(VOCAB, HID),
        mask_token_id=0, block_size=H + 1, selector_objective="full_vocab",
        selector_weight_mode="uniform", selector_loss_alpha=1.0,
    )
    check("K-way selector rejected for full_vocab", False, "no error raised")
except ValueError as exc:
    check("K-way selector rejected for full_vocab", "supports_full_vocab" in str(exc))

# 2/3. the objective runs, covers every slot, and reaches the parameters --------------------------
model, selector = build("full_vocab", "full_vocab")
torch.manual_seed(1)
# The host slices the anchor off before any lattice extraction, so feed H+1 slots.
logits = torch.randn(B, H + 1, VOCAB)
hidden = torch.randn(B, H + 1, HID)
target_ids = torch.randint(0, VOCAB, (B, H + 1))
predecessor_ids = torch.randint(0, VOCAB, (B, H + 1))
weight_mask = torch.ones(B, H + 1)

# Force a known number of off-lattice targets: make slot 1's target a token the lattice cannot hold.
lattice_top = logits[:, 1:, :].topk(K, dim=-1).indices
target_ids[:, 1] = logits[:, 1, :].argmax(-1)          # in-lattice (rank 0)
off = torch.full((B,), -1)
for b in range(B):
    banned = set(lattice_top[b, 1].tolist())
    off[b] = next(t for t in range(VOCAB) if t not in banned)
target_ids[:, 2] = off                                  # provably outside the top-K

terms = model._selector_chunk_terms(
    selector, logits, hidden, target_ids, predecessor_ids, weight_mask.clone(), weight_mask.clone()
)
den = terms.weight_den.item()
cov = terms.covered_num.item()
check("CE finite", torch.isfinite(terms.ce_num).item(), f"ce_num={terms.ce_num.item():.4f}")
check(
    "every valid slot supervised",
    abs(den - B * H) < 1e-6,
    f"weight_den={den:.1f} expected {B*H} (K-way support would be covered_num={cov:.1f})",
)
check(
    "off-lattice slots are the gap the objective closes",
    cov < den,
    f"{den - cov:.0f} of {den:.0f} slots have no K-way label",
)

terms.ce_num.backward()
grads = {n: p.grad for n, p in selector.named_parameters() if p.grad is not None}
gnorm = sum(float(g.pow(2).sum()) for g in grads.values()) ** 0.5
out_layer = f"delta_mlp.{len(selector.delta_mlp) - 1}.weight"
check("gradient reaches corrector output", grads.get(out_layer) is not None
      and float(grads[out_layer].abs().sum()) > 0, f"{out_layer} |grad_total|={gnorm:.3e}")
check("gradient reaches encoder", any(n.startswith("encoder.") for n in grads))
check("gradient reaches GRU", any(n.startswith("gru.") for n in grads))

# 4. init is an exact no-op ------------------------------------------------------------------------
torch.manual_seed(2)
sel2 = LatticeCorrector(hidden_size=HID, vocab_size=VOCAB, top_k=K, d=16, n_layers=1, n_heads=2,
                        state_dim=16, delta_hidden=32, dropout=0.0, cand_slots=0)
emb2 = nn.Embedding(VOCAB, HID)
sel2.bind_target_embedding(emb2)
sel2.eval()
with torch.no_grad():
    lg = torch.randn(B, H, VOCAB)
    ul, cid, sc = sel2.extract_lattice(lg)
    scores = sel2.score_candidates(
        candidate_ids=cid, unary_logits=ul, hidden_states=torch.randn(B, H, HID),
        predecessor_ids=torch.randint(0, VOCAB, (B, H)), lattice_scalars=sc,
    )
check("init is exact DFlash no-op", torch.allclose(scores, ul, atol=0),
      f"max|score-lp|={(scores-ul).abs().max().item():.2e}")

# prefix-rank variant constructs and differs from the plain GRU state
torch.manual_seed(3)
sel3 = LatticeCorrector(hidden_size=HID, vocab_size=VOCAB, top_k=K, d=16, n_layers=1, n_heads=2,
                        state_dim=16, delta_hidden=32, dropout=0.0, cand_slots=0, prefix_rank=True)
sel3.bind_target_embedding(nn.Embedding(VOCAB, HID))
sel3.eval()
with torch.no_grad():
    cid3 = torch.randint(0, VOCAB, (B, H, K))
    pred3 = cid3[:, :, 0].roll(1, dims=1)
    ranks = sel3.prefix_ranks(cid3, pred3)
check("prefix_ranks anchor bucket", bool((ranks[:, 0] == K + 1).all()), f"{ranks[0].tolist()}")
check("prefix_ranks resolves in-lattice ranks", bool((ranks[:, 1:] <= K).all()))

print()
bad = [n for n, ok, _ in CHECKS if not ok]
print(f"{len(CHECKS) - len(bad)}/{len(CHECKS)} PASS")
sys.exit(1 if bad else 0)
