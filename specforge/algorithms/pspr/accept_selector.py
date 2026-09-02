"""Selector objective and calibrated-gate evaluation, ported verbatim from the dh2048 recipe.

Provenance: every function below is a line-for-line port of
``TAPS-SP/scripts/train_accept_selector.py``, the script that produced
``TAPS-SP/outputs/dh2048/best.pt`` (``cal_gain=+0.4401``, ``tau=(0.0, 4.5, False, 0.35)``).

The only substitution is the model entry point: the reference calls the reference
``LatticeSelector.forward``; here we call SpecForge's ``LatticePathSelector.score_candidates``, which
the reference-equivalence gate verified is bit-identical (max|delta| = 0 on scores, encode, causal
states, delta term and err logits under the dh2048 weights).

Reference docstrings are kept verbatim because they carry the measurements that justify the recipe.
"""

from __future__ import annotations

from collections import defaultdict

import torch

from specforge.data.lattice_trace_data import collate

TAUS = [float("inf"), 0.45, 0.25, 0.12, 0.04, 0.0]
THETAS = [0.35, 0.0]
# A gate is (tau, rho):  override the draft's top-1 with the best alternative iff
#     p(alt) > tau + rho * p(top-1).
# rho == 1 recovers the ADDITIVE gate used so far (p_alt - p_top1 > tau); tau == 0 gives a
# MULTIPLICATIVE gate (p_alt / p_top1 > rho).  The distinction matters because the two error modes
# are NOT symmetric: at the frontier, replacing an already-WRONG top-1 with another wrong token is
# entirely free (the block truncates at that slot either way, and the token actually committed is
# the target's own g_t), whereas replacing a RIGHT top-1 truncates the block.  So the only real cost
# is destruction, whose probability scales with p(top-1) -- an additive gate is exactly the wrong
# shape there: it blocks hardest where p(top-1) is smallest, i.e. where overriding is nearly free.
RHOS = [1.30, 1.70, 2.20, 3.00, 4.50, 6.50, 9.00, 14.0, 22.0, 35.0, 60.0]
_G = (
    [(t, 1.0) for t in TAUS]
    + [(0.0, r) for r in RHOS]
    + [(0.04, 1.30), (0.04, 1.70), (0.12, 1.30), (0.12, 1.70)]
)
# skip0: never override slot 0.  Official Domino does exactly this (pure_draft_prefix_len=1) and the
# frontier probe measured the same thing here -- the module LOSES ~1pp of accuracy at slot 0, which
# is the most expensive slot to break because it is reached in 100% of blocks.
GATES = [(t, r, s0) for (t, r) in _G for s0 in (False, True)]

__all__ = [
    "TAUS",
    "THETAS",
    "RHOS",
    "GATES",
    "slot_probs",
    "frontier_mask",
    "expected_accept",
    "dpace_weights",
    "eval_gate",
    "report_gate",
]


def slot_probs(model, b, with_err=False):
    """p[B,H] = prob assigned to the ground-truth candidate (0 where g not in candidate set)."""
    out = model(
        candidate_ids=b["top"],
        unary_logits=b["lp"],
        hidden_states=b["dh"],
        predecessor_ids=b["prefix_ids"],
        lattice_scalars=b["scal"],
        return_err=with_err,
    )
    sc, el = out if with_err else (out, None)
    logp = torch.log_softmax(sc, dim=-1)
    ti = b["ti"]
    ok = ti >= 0
    p = logp.gather(-1, ti.clamp(min=0).unsqueeze(-1)).squeeze(-1).exp()
    p = torch.where(ok, p, torch.zeros_like(p))
    return (sc, p, el) if with_err else (sc, p)


def frontier_mask(valid, top_ok):
    """Slots up to AND INCLUDING the first top-1 error: the only decisions that occur at decode,
    and the only slots whose trace label is still valid (g_block past the draft's first mismatch was
    produced with a wrong target context)."""
    right = torch.cumsum(((~valid) | (~top_ok)).long(), dim=1) == 0
    incl = torch.zeros_like(right)
    L = right.sum(dim=1)
    H = valid.shape[1]
    ar = torch.arange(valid.shape[0], device=valid.device)
    ok = L < H
    incl[ar[ok], L[ok].clamp(max=H - 1)] = True
    return (right | incl) & valid


def expected_accept(p):
    """sum_t prod_{j<=t} p_j -> [B]; differentiable surrogate for accept length (minus the +1)."""
    return torch.cumprod(p, dim=1).sum(dim=1)


def dpace_weights(p):
    """Per-slot CE weights = d E[accept] / d log p_k = sum_{t>=k} prod_{j<=t} p_j.   [B,H]

    D-PACE weighting (arXiv 2605.18810) applied to a SELECTION head rather than a drafter.  It is the
    correct middle ground between the two extremes that both failed here:
      * maximizing E[accept] directly -> gradient flows through the product, the head turns
        risk-averse and collapses onto top-1 (recovery 23.4% -> 2.3%, measured).
      * a fixed binary frontier mask  -> ignores that the acceptance-limiting position MOVES as the
        head improves (measured: only +0.04 accept length over uniform CE).
    Detached weights keep plain CE's "just be correct" signal while allocating capacity to the slots
    that currently cap acceptance: w_0 = E[accept] (largest) and w_k -> 0 once reaching k is unlikely,
    and as p improves cumprod grows so weight shifts deeper automatically.
    """
    C = torch.cumprod(p, dim=1)  # [B,H]: prob of reaching slot t
    return C.flip(1).cumsum(1).flip(1)


def _lead(ok):
    """number of leading True per row -> [B] (vectorized run length)."""
    return (torch.cumsum((~ok).long(), dim=1) == 0).sum(dim=1)


def _frontier(valid, ref_ok, mod_ok):
    """Vectorized destruction/recovery around the frontier of ref_ok.

    right_mask = slots reached with ref (top-1) still correct; destroyed = mod broke one of them.
    The FIRST slot where ref is wrong (if valid) is the recovery opportunity.
    """
    right = torch.cumsum(((~valid) | (~ref_ok)).long(), dim=1) == 0  # [B,H]
    L = right.sum(dim=1)  # [B]
    H = valid.shape[1]
    Lc = L.clamp(max=H - 1)
    is_wrong = (L < H) & valid.gather(1, Lc[:, None]).squeeze(1)
    rec = is_wrong & mod_ok.gather(1, Lc[:, None]).squeeze(1)
    return (
        right.sum().item(),
        (right & ~mod_ok).sum().item(),
        is_wrong.sum().item(),
        rec.sum().item(),
    )


@torch.no_grad()
def eval_gate(model, records, embed, device, gates=GATES, thetas=THETAS, bs=256):
    """Teacher-forced accept length + destruction/recovery, per dataset, over the 2-D gate
    (tau = softmax margin, theta = P(top-1 wrong) from the dedicated detector)."""
    model.eval()
    gt = torch.tensor([g[0] for g in gates], device=device)
    gr = torch.tensor([g[1] for g in gates], device=device)
    g0 = torch.tensor([g[2] for g in gates], device=device)
    th = torch.tensor(thetas, device=device)
    keys = [(t, r, s0, o) for (t, r, s0) in gates for o in thetas]
    per = {k: defaultdict(lambda: defaultdict(float)) for k in keys}
    base = defaultdict(lambda: defaultdict(float))
    for i in range(0, len(records), bs):
        chunk = records[i : i + bs]
        b = collate(chunk, embed, device)
        sc, _, el = slot_probs(model, b, with_err=True)
        prob = torch.softmax(sc, dim=-1)
        p_top1 = prob[..., 0]
        best_o, best_i = prob[..., 1:].max(dim=-1)
        best_i = best_i + 1
        # allow[a] = gate a fires:  p_alt > tau_a + rho_a * p_top1
        p_wrong = torch.sigmoid(el)  # [B,H]
        ti, g = b["ti"], b["g"]
        valid = g >= 0
        top_ok = (ti == 0) & valid
        run_top1 = _lead(top_ok) + 1
        allow = best_o.unsqueeze(0) > (
            gt.view(-1, 1, 1) + gr.view(-1, 1, 1) * p_top1.unsqueeze(0)
        )
        slot0 = torch.zeros_like(allow[0], dtype=torch.bool)
        slot0[:, 0] = True
        allow = allow & ~(g0.view(-1, 1, 1) & slot0.unsqueeze(0))
        pw = p_wrong.unsqueeze(0) > th.view(-1, 1, 1)  # [O,B,H]
        ds = [r["dataset"] for r in chunk]
        uds = sorted(set(ds))
        dsi = torch.tensor([uds.index(d) for d in ds], device=device)
        for k, d in enumerate(uds):
            m = dsi == k
            base[d]["n"] += m.sum().item()
            base[d]["run"] += run_top1[m].sum().item()
        for a, (tau, rho, s0) in enumerate(gates):
            for c, theta in enumerate(thetas):
                ov = allow[a] & pw[c]
                pick = torch.where(ov, best_i, torch.zeros_like(best_i))
                mod_ok = (pick == ti) & (ti >= 0)
                for k, d in enumerate(uds):
                    m = dsi == k
                    mo = mod_ok[m]
                    s = per[(tau, rho, s0, theta)][d]
                    s["run"] += (_lead(mo) + 1).sum().item()
                    rn, de, wn, rc = _frontier(valid[m], top_ok[m], mo)
                    s["right_n"] += rn
                    s["destroyed"] += de
                    s["wrong_n"] += wn
                    s["recovered"] += rc
    return base, per


def report_gate(tag, base, per, gates=GATES, thetas=THETAS, verbose=True, top=8, log=print):
    """MACRO-average over datasets so one big domain cannot bias the gate."""
    dss = sorted(base.keys())
    bt = {d: base[d]["run"] / max(base[d]["n"], 1) for d in dss}
    macro_bt = sum(bt.values()) / len(bt)
    rows = []
    for tau, rho, s0 in gates:
        for theta in thetas:
            s_ = per[(tau, rho, s0, theta)]
            accs, de, rn, rc, wn, dd = [], 0.0, 0.0, 0.0, 0.0, []
            for d in dss:
                s = s_[d]
                a = s["run"] / max(base[d]["n"], 1)
                accs.append(a)
                dd.append(f"{d[:4]}{a-bt[d]:+.2f}")
                de += s["destroyed"]
                rn += s["right_n"]
                rc += s["recovered"]
                wn += s["wrong_n"]
            rows.append(
                (
                    sum(accs) / len(accs),
                    tau,
                    rho,
                    s0,
                    theta,
                    de / max(rn, 1),
                    rc / max(wn, 1),
                    " ".join(dd),
                )
            )
    rows.sort(key=lambda r: -r[0])
    if verbose:
        log(
            f"[{tag}] DFlash top-1 macro={macro_bt:.3f} | "
            + " ".join(f"{d}={bt[d]:.2f}" for d in dss)
        )
        log(
            f"  {'tau':>6} {'rho':>5} {'skip0':>5} {'theta':>6} {'macroAcc':>9} {'delta':>8}"
            f" {'destroy%':>9} {'recover%':>9}   per-domain delta"
        )
        for a, tau, rho, s0, theta, dr, rr, dd in rows[:top]:
            log(
                f"  {tau:6.2f} {rho:5.2f} {str(s0):>5} {theta:6.2f} {a:9.3f} {a-macro_bt:+8.3f}"
                f" {dr:9.2%} {rr:9.2%}   {dd}"
            )
    best = rows[0]
    if verbose:
        log(
            f"  -> best tau={best[1]} rho={best[2]} skip0={best[3]} theta={best[4]} "
            f"macroAcc={best[0]:.3f} (delta={best[0]-macro_bt:+.3f})"
        )
    return (best[1], best[2], best[3], best[4]), best[0] - macro_bt
