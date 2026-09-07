#!/usr/bin/env python3
"""Gate for using the expected-accept survival weights with the `multiclass` selector objective.

Why this arm exists, stated so the run cannot be re-justified after the fact:

The champion (cloze stage-1, macro 6.0498) trains a per-slot 0-1 classification loss with
`uniform_frontier_boost`, but decoding maximises a SEQUENCE quantity, the accepted prefix length.
Those differ in exactly one way that matters under finite capacity: an error in slot 0 destroys the
whole 15-slot block while an error in slot 14 costs one token, yet the per-slot loss prices them the
same.  `expected_accept` is the exact gradient of E[accepted prefix]; `smoothed_accept` is its
floor-mixed surrogate.  Both were already implemented and both were locked to `keep_repair` by a
whitelist that this change relaxes.

Unlike `covered_conditional` -- which this project's own gradient audit showed to be a mere
REPARAMETERISATION, since P(covered) is a per-slot scalar that cancels inside the serving argmax --
a slot reweighting changes the finite-capacity optimum.  That is the whole reason it gets the GPU
budget instead.

What is deliberately NOT claimed: that this explains why a conservative serving rho beats the
trained MAP.  Four candidate mechanisms for that have already been falsified in PSPR_DECISION.md
(frontier_boost bias, per-slot miss rate, generic overconfidence, corpus-vs-greedy shift).  The new
`selector_base_right` telemetry is here to MEASURE the remaining prior-mismatch hypothesis, not to
assume it.

Checks:
  1. exact-gradient identity: a CE weighted by the detached floor-0 weights has the same gradient
     as -E[A] computed by autograd, to float64 tolerance;
  2. the identity holds only at floor 0, so `smoothed_accept` must not be labelled exact;
  3. floor=1 degenerates to the pure positional ramp (H, H-1, ..., 1) and is logit-independent;
  4. a miss zeroes its own and all later credit at floor 0, but not at floor > 0;
  5. invalid slots never splice two spans (1...0...1 stays broken);
  6. `smoothed_accept` + multiclass is accepted at the champion's rho=3, `expected_accept` is not;
  7. `covered_conditional` + accept weights is rejected (its q is a joint, not the emittable
     conditional);
  8. the base_right telemetry counts a top-k miss as NOT base_right despite the placeholder index 0;
  9. the telemetry is suppressed when the err objective owns the err_* fields.
"""

from __future__ import annotations

import pathlib
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from specforge.algorithms.common import dflash_family_model as H  # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def expected_accept_scalar(q: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """E[A] = sum_t prod_{j<=t} q_j, computed straightforwardly and only over the valid prefix."""
    total = q.new_zeros(())
    running = q.new_ones(())
    for t in range(q.shape[-1]):
        if not bool(valid[t]):
            break
        running = running * q[t]
        total = total + running
    return total


def main() -> None:
    torch.manual_seed(0)
    B, S, Hs, K = 2, 3, 15, 16

    # ---- 1 & 2: exact-gradient identity, on ONE slot row so the reference stays trivially right.
    logits = torch.randn(Hs, K, dtype=torch.float64, requires_grad=True)
    target = torch.randint(0, K, (Hs,))
    valid_row = torch.ones(Hs, dtype=torch.bool)

    ce = F.cross_entropy(logits, target, reduction="none")
    q = torch.exp(-ce)
    (-expected_accept_scalar(q, valid_row)).backward()
    autograd_grad = logits.grad.clone()

    logits2 = logits.detach().clone().requires_grad_(True)
    ce2 = F.cross_entropy(logits2, target, reduction="none")
    w = H.accept_survival_weights(
        valid=valid_row.unsqueeze(0),
        correct_action_probability=torch.exp(-ce2).detach().unsqueeze(0),
        excludes_anchor=True,
        survival_floor=0.0,
    ).squeeze(0)
    (w * ce2).sum().backward()

    err = (autograd_grad - logits2.grad).abs().max().item()
    check(
        "expected_accept weights reproduce the autograd gradient of -E[A] exactly",
        err < 1e-12,
        f"max|d(-E[A]) - d(w*CE)| = {err:.3e}",
    )

    logits3 = logits.detach().clone().requires_grad_(True)
    ce3 = F.cross_entropy(logits3, target, reduction="none")
    w_smooth = H.accept_survival_weights(
        valid=valid_row.unsqueeze(0),
        correct_action_probability=torch.exp(-ce3).detach().unsqueeze(0),
        excludes_anchor=True,
        survival_floor=0.5,
    ).squeeze(0)
    (w_smooth * ce3).sum().backward()
    smooth_err = (autograd_grad - logits3.grad).abs().max().item()
    check(
        "floor=0.5 is NOT the exact gradient (so calling it a surrogate is the honest label)",
        smooth_err > 1e-6,
        f"max deviation = {smooth_err:.3e}",
    )

    # ---- 3: floor=1 is a pure positional ramp, independent of the model.
    ramp_a = H.accept_survival_weights(
        valid=torch.ones(1, Hs, dtype=torch.bool),
        correct_action_probability=torch.rand(1, Hs, dtype=torch.float64),
        excludes_anchor=True,
        survival_floor=1.0,
    )
    ramp_b = H.accept_survival_weights(
        valid=torch.ones(1, Hs, dtype=torch.bool),
        correct_action_probability=torch.rand(1, Hs, dtype=torch.float64),
        excludes_anchor=True,
        survival_floor=1.0,
    )
    want = torch.arange(Hs, 0, -1, dtype=torch.float64).unsqueeze(0)
    check(
        "floor=1 gives the logit-independent positional ramp H..1",
        torch.equal(ramp_a, ramp_b) and torch.allclose(ramp_a, want),
        f"first three = {ramp_a[0, :3].tolist()}, last = {ramp_a[0, -1].item()}",
    )
    check(
        "the ramp really does price slot 0 an order of magnitude above the last slot",
        abs(ramp_a[0, 0].item() / ramp_a[0, -1].item() - Hs) < 1e-12,
        f"ratio = {ramp_a[0, 0].item() / ramp_a[0, -1].item():.1f}x",
    )

    # ---- 4: a miss (q=0) truncates credit at floor 0 but not above it.
    qmiss = torch.full((1, Hs), 0.9, dtype=torch.float64)
    qmiss[0, 4] = 0.0
    exact = H.accept_survival_weights(
        valid=torch.ones(1, Hs, dtype=torch.bool),
        correct_action_probability=qmiss,
        excludes_anchor=True,
        survival_floor=0.0,
    )
    smoothed = H.accept_survival_weights(
        valid=torch.ones(1, Hs, dtype=torch.bool),
        correct_action_probability=qmiss,
        excludes_anchor=True,
        survival_floor=0.5,
    )
    check(
        "floor=0: an uncoverable slot zeroes its own and every later slot's credit",
        bool((exact[0, 4:] == 0).all()) and bool(exact[0, 3] > 0),
        f"w[3]={exact[0, 3].item():.4f}, w[4:]={exact[0, 4:].abs().sum().item():.1e}",
    )
    check(
        "floor>0: the same slots keep a curriculum signal (the starvation fix)",
        bool((smoothed[0, 4:] > 0).all()),
        f"w[4]={smoothed[0, 4].item():.4f}, w[14]={smoothed[0, 14].item():.4f}",
    )

    # ---- 5: 1...0...1 must not splice two spans together.
    spliced = torch.ones(1, Hs, dtype=torch.bool)
    spliced[0, 5] = False
    w_split = H.accept_survival_weights(
        valid=spliced,
        correct_action_probability=torch.full((1, Hs), 0.9, dtype=torch.float64),
        excludes_anchor=True,
        survival_floor=1.0,
    )
    check(
        "an invalid slot terminates the block; the tail after it gets zero credit",
        bool((w_split[0, 5:] == 0).all()) and abs(w_split[0, 0].item() - 5.0) < 1e-12,
        f"w[0]={w_split[0, 0].item()} (=5 valid slots), w[5:]={w_split[0, 5:].abs().sum().item():.1e}",
    )

    # ---- 6 & 7: the configuration whitelist.
    class FakeDraft:
        def __init__(self, **kw):
            self.selector_decision_mode = "margin_gate"
            self.selector_gate_rho = 1.0
            self.selector_gate_tau = 0.0
            self.selector_gate_theta = 0.0
            self.selector_gate_skip_first = False
            self.selector_keep_repair_margin = 0.0
            for k, v in kw.items():
                setattr(self, k, v)

    src = pathlib.Path(H.__file__).read_text()
    accepts_multiclass = (
        'if selector_objective not in {"keep_repair", "multiclass"}' in src
    )
    check(
        "the whitelist admits multiclass alongside keep_repair",
        accepts_multiclass,
        "expected_accept/smoothed_accept no longer hard-require keep_repair",
    )
    check(
        "covered_conditional is still refused (its q is the JOINT, not the emittable conditional)",
        accepts_multiclass and "covered_conditional is deliberately NOT allowed" in src,
    )
    check(
        "only expected_accept demands argmax serving; smoothed_accept may run at the champion rho=3",
        'elif selector_weight_mode == "expected_accept":' in src
        and "use 'smoothed_accept' for a conservative " in src,
    )

    # ---- 8: base_right must not count the placeholder index of a top-k miss.
    candidate_ids = torch.tensor([[[7, 3, 9], [7, 3, 9], [7, 3, 9]]])
    labels = torch.tensor([[7, 3, 42]])  # base_right, repairable, miss
    matches = candidate_ids.eq(labels.unsqueeze(-1))
    is_cand = matches.any(dim=-1)
    idx = matches.long().argmax(dim=-1)
    naive = idx.eq(0)
    guarded = idx.eq(0) & is_cand
    check(
        "a miss yields placeholder index 0, so the naive count would overstate base_right",
        bool(naive[0, 2]) and not bool(guarded[0, 2]),
        f"naive={naive.tolist()}, guarded={guarded.tolist()}",
    )
    check(
        "the guarded count is what the host uses",
        "(target_candidate_index.eq(0) & target_is_candidate).float() * weight_mask" in src,
    )

    # ---- 9: the telemetry must survive the err objective.  The first version of this change
    # borrowed the three err_* fields (which are zero for multiclass) and was therefore silently
    # dead in the only configuration that matters: the champion trains err_loss_alpha=1.0, and the
    # err branch reassigns all three fields further down the same function.  Dedicated fields now.
    check(
        "base_right does NOT borrow err_* (which the err objective reassigns at alpha=1.0)",
        "base_right_num: torch.Tensor" in src
        and "selector_base_right_num: torch.Tensor" in src
        and "err_ce_num = (base_right" not in src,
    )
    check(
        "it is computed for every objective, not gated on multiclass",
        'if self.selector_objective == "multiclass" and not self._selector_err_objective_enabled:'
        not in src,
    )
    check(
        "registration is still guarded by has_selector_objective (else plain DFlash logs 0/0)",
        "if has_selector_objective:\n            #   selector_base_right" in src,
    )
    # `checkpointed_chunk_reduce` returns a positional tuple that the caller unpacks by hand, so a
    # field added to DFlashObjectiveTerms without a matching name in the unpack target silently
    # shifts every later metric.  Compare the two arities structurally instead of grepping strings.
    import ast as _ast

    tree = _ast.parse(src)
    declared = [
        n.target.id
        for cls in tree.body
        if isinstance(cls, _ast.ClassDef) and cls.name == "DFlashObjectiveTerms"
        for n in cls.body
        if isinstance(n, _ast.AnnAssign)
    ]
    unpacked = None
    for node in _ast.walk(tree):
        if (
            isinstance(node, _ast.Assign)
            and isinstance(node.value, _ast.Call)
            and getattr(node.value.func, "id", "") == "checkpointed_chunk_reduce"
            and isinstance(node.targets[0], _ast.Tuple)
            and len(node.targets[0].elts) > 10
        ):
            unpacked = [e.id for e in node.targets[0].elts if isinstance(e, _ast.Name)]
            break
    # One pre-existing benign alias: the field is `accuracy_den`, the local is `accuracy_denom`.
    # Anything else means a field was added on one side only, which would shift every later metric.
    known_aliases = {("accuracy_den", "accuracy_denom")}
    divergences = [
        (a, b) for a, b in zip(declared, unpacked or []) if a != b and (a, b) not in known_aliases
    ]
    check(
        "DFlashObjectiveTerms arity matches the positional unpack of checkpointed_chunk_reduce",
        unpacked is not None and len(declared) == len(unpacked) and not divergences,
        f"declared={len(declared)}, unpacked={len(unpacked) if unpacked else None}, "
        f"unexpected divergences={divergences}",
    )
    check(
        "the three new diagnostic fields are the tail of that tuple",
        declared[-3:]
        == [
            "selector_base_right_num",
            "selector_slot_weight_num",
            "selector_weight_den_all",
        ],
        f"tail={declared[-3:]}",
    )

    # ---- real config must actually load ---------------------------------------------------------
    cfg_path = pathlib.Path(__file__).resolve().parents[1] / (
        "examples/configs/online/disaggregated/managed-local/qwen3-4b-pspr-cloze-sa.yaml"
    )
    if cfg_path.exists():
        from specforge.config.schema import load_config  # noqa: E402

        try:
            cfg = load_config(str(cfg_path))
            t = cfg.training
            check(
                "the launch YAML loads and requests smoothed_accept on multiclass",
                t.dflash2_selector_objective == "multiclass"
                and t.dflash2_selector_weight_mode == "smoothed_accept",
                f"objective={t.dflash2_selector_objective}, "
                f"weight_mode={t.dflash2_selector_weight_mode}, "
                f"floor={getattr(t, 'dflash2_selector_survival_floor', None)}",
            )
        except Exception as exc:  # noqa: BLE001
            check("the launch YAML loads", False, f"{type(exc).__name__}: {exc}")
    else:
        check("the launch YAML exists", False, f"missing {cfg_path}")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
