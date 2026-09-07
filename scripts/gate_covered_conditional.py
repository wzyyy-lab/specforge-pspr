#!/usr/bin/env python3
"""Gate for the `covered_conditional` selector objective (structured top-16-or-NONE likelihood).

The previous round's lesson is the reason this file exists.  PSPR-Decision was trained for 5700
steps on a `profitable_repair` likelihood whose KEEP branch was
``log(1-p)`` with ``1-p = P(y=c_0) + P(y not in C)`` -- KEEP absorbed the miss mass and was
overestimated by 0.10-0.16.  That defect was *recorded* and then not tested, and the whole run was
spent measuring its consequences.  So every claim this objective makes is checked here before any
GPU time is spent:

  1. the loss is a proper normalized likelihood over 17 outcomes;
  2. a miss contributes ONLY the coverage term (it has no "which candidate" label);
  3. a covered slot's conditional term is exactly multiclass CE (so the ranker is not weakened);
  4. KEEP stays inside the K-way softmax (the property `latrepair` violated, costing 0.69 accept);
  5. the coverage head is neither frozen nor starved of `err_logits` by the three control paths
     that key off `_FACTORIZED_SELECTOR_OBJECTIVES`;
  6. the serving rule `argmax(r)` is *identical* to `margin_gate` at rho=1, so "is MAP optimal?"
     is measurable by moving one scalar rather than by swapping decode rules;
  7. `selector_err_loss_alpha > 0` is rejected (it would be a second, conflicting label on the
     same logits).
"""

from __future__ import annotations

import json
import pathlib
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from specforge.algorithms.common import dflash_family_model as H  # noqa: E402
from specforge.modeling.draft.pspr import LatticePathSelector  # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def reference_nll(scores, none_logit, target_index, covered):
    """Independent re-derivation of the flat (K+1)-way NLL, one slot at a time, in log space.

    z = [s_0 .. s_{K-1}, s_none];  L = -log softmax(z)[observed].

    Deliberately not vectorised and deliberately dtype-preserving.  The first version of this gate
    reported a 1.37e-7 mismatch and I mis-attributed it twice (log-vs-prob space; softmax reduction
    order) before measuring the real cause: the reference wrote ``scores.float()`` on a float64 input
    and 1.37e-7 is float32 epsilon.  The host was never wrong; the reference was.
    """
    out = torch.empty_like(none_logit)
    K = scores.shape[-1]
    flat_scores = scores.reshape(-1, K)
    flat_none = none_logit.reshape(-1)
    flat_idx = target_index.reshape(-1)
    flat_cov = covered.reshape(-1)
    for i in range(none_logit.numel()):
        z = torch.cat([flat_scores[i], flat_none[i].reshape(1)])
        observed = int(flat_idx[i]) if flat_cov[i] else K
        out.reshape(-1)[i] = -F.log_softmax(z, dim=-1)[observed]
    return out


def main() -> None:
    torch.manual_seed(0)
    B, T, K = 2, 7, 16
    scores = torch.randn(B, T, K, dtype=torch.float64) * 2.0
    none_logit = torch.randn(B, T, dtype=torch.float64) * 1.5
    covered = torch.rand(B, T) > 0.25
    target_index = torch.randint(0, K, (B, T))
    target_index = torch.where(covered, target_index, torch.zeros_like(target_index))

    # ---- host arithmetic -------------------------------------------------------------------------
    extended = torch.cat([scores, none_logit.unsqueeze(-1)], dim=-1)
    flat_target = torch.where(covered, target_index, torch.full_like(target_index, K))
    host = F.cross_entropy(
        extended.reshape(-1, K + 1), flat_target.reshape(-1), reduction="none"
    ).reshape(B, T)
    ref = reference_nll(scores, none_logit, target_index, covered)
    err = (host - ref).abs().max().item()
    check("loss equals the flat (K+1)-way NLL exactly", err < 1e-12, f"max|delta|={err:.2e}")

    miss = ~covered
    check(
        "on a miss the observed class is NONE, not a fabricated candidate",
        torch.allclose(host[miss], -F.log_softmax(extended, dim=-1)[..., K][miss]),
        f"{int(miss.sum())} miss slots",
    )

    # ---- THE central claim: a miss is a negative example for s_0 ---------------------------------
    # This is the entire reason for choosing flat over the q/r factorisation.  Under q/r the ranker
    # sees nothing at all on a miss; here it must see a correctly-signed push DOWN on s_0.
    sc = scores.float().clone().requires_grad_(True)
    nl = none_logit.float().clone().requires_grad_(True)
    ext = torch.cat([sc, nl.unsqueeze(-1)], dim=-1)
    loss = F.cross_entropy(
        ext.reshape(-1, K + 1), flat_target.reshape(-1), reduction="none"
    ).reshape(B, T).sum()
    loss.backward()
    g = sc.grad
    probs = torch.softmax(extended, dim=-1)
    check(
        "miss slots push s_0 DOWN (dL/ds_0 = P(c_0) > 0)",
        bool((g[..., 0][miss] > 0).all()),
        f"min grad on miss = {g[..., 0][miss].min():.4f}",
    )
    check(
        "and that gradient equals P(c_0) analytically",
        torch.allclose(g[..., 0][miss].double(), probs[..., 0][miss], atol=1e-6),
    )
    check(
        "covered target=c_0 slots still push s_0 UP (dL/ds_0 = P(c_0) - 1 < 0)",
        bool((g[..., 0][covered & target_index.eq(0)] < 0).all())
        if bool((covered & target_index.eq(0)).any()) else True,
    )
    # s_0 must fall RELATIVE to the alternatives on a miss, otherwise the objective moves nothing an
    # argmax decision can see.  P(c_0) > mean P(c_alt) is what makes this true, and c_0 being the
    # draft's own argmax is why it holds in practice; assert it on the realistic case where s_0 leads.
    lead = scores.clone()
    lead[..., 0] = scores.max(dim=-1).values + 1.0
    ext_lead = torch.cat([lead, none_logit.unsqueeze(-1)], dim=-1)
    p_lead = torch.softmax(ext_lead, dim=-1)
    check(
        "when c_0 leads (the real case), s_0 is pushed down harder than any alternative",
        bool((p_lead[..., 0:1] > p_lead[..., 1:K]).all()),
    )

    # ---- comparability with the multiclass baseline it must beat ----------------------------------
    mc = F.cross_entropy(
        scores.reshape(-1, K), target_index.reshape(-1), reduction="none"
    ).reshape(B, T)
    # On a covered slot the flat NLL is the multiclass NLL plus log(1 + exp(s_none)/Z_K), i.e. it
    # differs only by the mass reserved for NONE -- the RANKING signal is unchanged.
    z_k = torch.logsumexp(scores, dim=-1)
    offset = torch.log1p(torch.exp(none_logit - z_k))
    check(
        "on covered slots flat NLL = multiclass NLL + log(1 + e^{s_none}/Z_K)",
        torch.allclose(host[covered], (mc + offset)[covered], atol=1e-12),
    )

    # ---- 4: KEEP is a class of the same softmax, not a separate branch --------------------------
    # If KEEP had been factored out (the latrepair rule), candidate 0's score could not influence
    # the relative ordering of the alternatives.  Under the conditional softmax it must.
    bumped = scores.clone()
    bumped[..., 0] += 5.0
    r0 = torch.softmax(scores, dim=-1)
    r1 = torch.softmax(bumped, dim=-1)
    keep_share_rose = bool((r1[..., 0] > r0[..., 0]).all())
    alts_fell = bool((r1[..., 1:].sum(-1) < r0[..., 1:].sum(-1)).all())
    factorized = LatticePathSelector.keep_repair_log_probs(
        scores.float(), none_logit.float()
    )
    factorized_bumped = LatticePathSelector.keep_repair_log_probs(
        bumped.float(), none_logit.float()
    )
    # The factorised rule renormalises over scores[1:], so raising scores[0] leaves the repair
    # branch untouched -- this is precisely the information it throws away.
    factorized_blind = torch.allclose(
        factorized[..., 1:], factorized_bumped[..., 1:], atol=1e-6
    )
    check(
        "candidate 0 participates in the conditional softmax",
        keep_share_rose and alts_fell,
    )
    check(
        "and the factorised rule is provably blind to it (the 0.69-accept defect)",
        factorized_blind,
    )

    # ---- 6: argmax(r) == margin_gate at rho=1 --------------------------------------------------
    native = scores.float().argmax(dim=-1)
    gated = LatticePathSelector.select_margin_gate(
        scores.float(), rho=1.0, tau=0.0, theta=0.0
    ).squeeze(-1)
    check(
        "argmax(r) is exactly margin_gate(rho=1, tau=0)",
        torch.equal(native, gated),
        f"{int((native != gated).sum())} disagreements",
    )
    conservative = LatticePathSelector.select_margin_gate(
        scores.float(), rho=3.0, tau=0.0, theta=0.0
    ).squeeze(-1)
    check(
        "rho=3 is a strictly more conservative version of the SAME rule",
        bool(((conservative == 0) | (conservative == native)).all()),
    )

    # ---- 5/7: host control-flow contracts ------------------------------------------------------
    check(
        "covered_conditional is a registered objective",
        "covered_conditional" in H._VALID_SELECTOR_OBJECTIVES,
    )
    check(
        "it is NOT in the factorised set (it does not decode keep/repair)",
        "covered_conditional" not in H._FACTORIZED_SELECTOR_OBJECTIVES,
    )
    check(
        "but it IS in the main-likelihood-err set (so the head is requested and stays trainable)",
        "covered_conditional" in H._MAIN_LIKELIHOOD_ERR_OBJECTIVES,
    )
    check(
        "the factorised set is a subset of the main-likelihood-err set",
        H._FACTORIZED_SELECTOR_OBJECTIVES <= H._MAIN_LIKELIHOOD_ERR_OBJECTIVES,
    )

    src = pathlib.Path(H.__file__).read_text()
    # These three sites decide (a) whether err_logits are computed at all, (b) whether the head is
    # frozen, (c) whether the head is required to exist.  Each one keyed off the factorised set
    # before this change; leaving any of them behind silently trains a frozen coverage head.
    for marker, why in [
        ("or self.selector_objective in _MAIN_LIKELIHOOD_ERR_OBJECTIVES", "wants_err"),
        (
            "and self.selector_objective not in _MAIN_LIKELIHOOD_ERR_OBJECTIVES",
            "freeze guard",
        ),
        (
            "if self.selector_objective in _MAIN_LIKELIHOOD_ERR_OBJECTIVES and not (",
            "head-required guard",
        ),
    ]:
        check(f"control path uses the new set: {why}", marker in src)

    check(
        "standalone err BCE is rejected for covered_conditional",
        'if selector_objective == "covered_conditional" and selector_err_loss_alpha > 0'
        in src,
    )
    check(
        "margin_gate accepts covered_conditional (train/serve contract)",
        '"margin_gate": {"multiclass", "profitable_action", "covered_conditional"}' in src,
    )
    check(
        "misses stay supervised (selector_supervised = ones)",
        'if self.selector_objective == "covered_conditional":\n'
        "            # A miss is a first-class outcome" in src,
    )
    check(
        "the NONE class has its own telemetry",
        '"selector_none_nll"' in src and '"selector_none_accuracy"' in src,
    )

    # ---- gradient reachability, and the q/r contrast that decided the design --------------------
    # Under q/r the assertion here was "the ranker gets NO gradient on miss slots".  Under the flat
    # form the assertion is the exact opposite, and that reversal IS the design decision: q/r leaves
    # the ranker blind on 14-19% of slots that each carry a hard negative for s_0.
    sc2 = scores.float().clone().requires_grad_(True)
    nl2 = none_logit.float().clone().requires_grad_(True)
    ext2 = torch.cat([sc2, nl2.unsqueeze(-1)], dim=-1)
    F.cross_entropy(
        ext2.reshape(-1, K + 1), flat_target.reshape(-1), reduction="none"
    ).sum().backward()
    check("ranker receives gradient", sc2.grad is not None and sc2.grad.abs().sum() > 0)
    check("NONE logit receives gradient", nl2.grad is not None and nl2.grad.abs().sum() > 0)
    check(
        "ranker DOES get gradient on miss slots (the whole point; q/r gives zero here)",
        bool((sc2.grad[miss].abs().sum(dim=-1) > 0).all()),
    )
    check(
        "NONE logit gets gradient on EVERY slot",
        bool((nl2.grad.abs() > 0).all()),
    )
    # Contrast arm: the q/r loss on the same tensors, showing the miss gradient really is absent.
    sc_qr = scores.float().clone().requires_grad_(True)
    q_qr = none_logit.float().clone().requires_grad_(True)
    (
        F.binary_cross_entropy_with_logits(q_qr, covered.float(), reduction="none")
        + torch.where(
            covered,
            F.cross_entropy(
                sc_qr.reshape(-1, K), target_index.reshape(-1), reduction="none"
            ).reshape(B, T),
            torch.zeros(B, T),
        )
    ).sum().backward()
    check(
        "control: under q/r the ranker gradient on miss slots is exactly zero",
        bool(sc_qr.grad[miss].abs().sum() == 0),
    )
    # I asserted these were identical on covered slots.  They are NOT, and the discrepancy is the
    # most important thing this gate found.  With alpha = Z_K/(Z_K + e^{s_none}) = P(covered):
    #     dL_flat/ds_j = alpha * P_K(c_j) - delta_{j,target}
    #     dL_qr  /ds_j =         P_K(c_j) - delta_{j,target}
    # so on a covered slot the flat form pushes every candidate UP relative to q/r (they must
    # out-compete NONE), by an amount proportional to P_K(c_j) -- largest for c_0.  That is the exact
    # opposite of the miss-slot effect this objective was chosen for.
    #
    # Consequence, and it is a real limitation: at convergence the flat model fits the true 17-way
    # distribution, P(c_j) = P(covered) * P(c_j | covered).  P(covered) is a per-slot scalar, so it
    # cancels inside a slot's argmax, and the serving decision argmax_{j<K} s_j is IDENTICAL to what
    # multiclass converges to.  Flat 17-way is therefore a REPARAMETERISATION, not a different
    # ranking objective; any benefit must come from finite-capacity/optimisation effects, not from
    # the asymptotic solution.  Asserting the two gradient fields differ is the honest check.
    diff = (sc_qr.grad[covered] - sc2.grad[covered]).abs().max().item()
    alpha = (
        torch.logsumexp(scores, dim=-1)
        - torch.logaddexp(torch.logsumexp(scores, dim=-1), none_logit)
    ).exp()
    predicted = (
        ((alpha.unsqueeze(-1) - 1.0) * torch.softmax(scores, dim=-1))[covered].abs().max().item()
    )
    check(
        "covered-slot gradients differ by exactly (alpha-1)*P_K(c_j), alpha=P(covered)",
        diff > 1e-6 and abs(diff - predicted) < 1e-5,
        f"measured {diff:.6f} vs predicted {predicted:.6f} -- flat is a reparameterisation, "
        "so it cannot change the converged ranking",
    )

    # ---- launch-boundary checks: a new objective name must be registered in FOUR places ---------
    # The 25 mathematical checks above all passed while the run would still have died twice: once at
    # startup on the pydantic Literal, and once after 9 hours of training when the exporter refused
    # the checkpoint.  Correct maths is not a launchable change; these are the gates that would
    # actually have caught it.
    import importlib

    schema_src = pathlib.Path(
        importlib.import_module("specforge.config.schema").__file__
    ).read_text()
    check(
        "registered in the pydantic Literal (else the run dies at config load)",
        '"covered_conditional",\n    ] = "multiclass"' in schema_src,
    )
    for exporter in (
        "export_pspr_cloze_for_decode.py",
        "export_pspr_decision_for_decode.py",
        "export_pspr_for_decode.py",
    ):
        text = (pathlib.Path(__file__).parent / exporter).read_text()
        check(
            f"registered in {exporter} (else 9h of training exports nothing)",
            '"margin_gate": {"multiclass", "profitable_action", "covered_conditional"}' in text,
        )

    # The real config must load through the real loader, not just be valid YAML.
    from specforge.config.schema import load_config

    cfg = load_config(
        str(
            pathlib.Path(__file__).parents[1]
            / "examples/configs/online/disaggregated/managed-local/qwen3-4b-pspr-cloze-cc.yaml"
        )
    )
    t = cfg.training
    check(
        "the real training yaml loads and carries the intended settings",
        t.dflash2_selector_objective == "covered_conditional"
        and t.dflash2_selector_weight_mode == "uniform"
        and float(t.dflash2_selector_err_loss_alpha) == 0.0,
        f"objective={t.dflash2_selector_objective!r} weight_mode={t.dflash2_selector_weight_mode!r} "
        f"err_alpha={t.dflash2_selector_err_loss_alpha}",
    )
    draft_cfg = json.loads(
        (pathlib.Path(__file__).parents[1] / "configs/qwen3-4b-pspr-cloze-cc.json").read_text()
    )["dflash_config"]
    baseline_cfg = json.loads(
        (pathlib.Path(__file__).parents[1] / "configs/qwen3-4b-pspr-cloze.json").read_text()
    )["dflash_config"]
    differing = {
        k for k in set(draft_cfg) | set(baseline_cfg)
        if draft_cfg.get(k) != baseline_cfg.get(k)
    }
    check(
        "draft config differs from the 6.0498 baseline ONLY in selector_gate_rho",
        differing == {"selector_gate_rho"},
        f"differing keys = {sorted(differing)}",
    )
    check(
        "and its native serving rho is 1.0 (the trained MAP), not a hand-tuned 3.0",
        float(draft_cfg["selector_gate_rho"]) == 1.0,
    )

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        raise SystemExit(1)
    print("all gates passed")


if __name__ == "__main__":
    main()
