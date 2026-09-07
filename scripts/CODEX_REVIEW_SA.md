# Review request: switching the PSPR-cloze selector to expected-accept slot weights

You are reviewing a change that is about to consume ~9 GPU-hours. Be adversarial. The author of this
change has a documented failure mode of over-generalising from single experiments and of declaring
directions exhausted while holding unfixed bugs; assume that is happening again unless the evidence
below rules it out.

## The state of the project

Frozen official DFlash-b16 backbone, PSPR selector head trained on top (stage-1). The metric is
macro accept length over six domains. Ceiling `oracle@16 = 10.4422`.

| arm | macro accept |
|---|---|
| PSPR-cloze stage-1, served at gate rho=3 | **6.0498** (champion) |
| PSPR-Decision, best margin 0.1 | 5.8372 |
| stage-1 baseline used for the S2 study | 5.7217 |

The exact decision-chain accounting on the champion (held-out, `DETAIL_cloze7115_rho3.log`),
`decision_n = 26353`:

- `base_right` 19872 = 75.4%
- `fixable` 5789 = 22.0% (target is in top-16 but is not candidate 0)
- `unfixable` 692 = 2.6%
- of the 5789 fixable: `alternative_rank_0` = 3870 = **66.86%** (the target is the best NON-base
  candidate, i.e. reachable if base did not compete), `selector_rank_0` = 2550 = **44.05%**,
  gate-blocked 425 = 7.34%, `recovered` = 2125 = **36.71%** (2550 − 425 = 2125, exact).

So the headroom is the 22.8pp between 66.86% and 44.05%, and it is **entirely** "the base score
ranks ahead of the target". The gate itself only costs 7.34pp.

## Things already falsified (do not re-propose them)

1. **Bidirectional cross-slot attention is worthless.** Paired per-prompt test, causal (single
   direction, cross-slot retained) vs full: `+0.0070 ± 0.0182, t=0.38, p=0.70`, 95% CI
   [−0.029, +0.043].
2. **`frontier_boost` is not a one-sided bias.** It boosts the selector's OWN first error slot and
   depends dynamically on `selected_index`; it is self-balancing negative feedback.
3. **Per-slot miss rate does not explain the rho=3 win.** Zero correlation with the per-domain
   rho1−rho3 gap (mbpp miss 0.0850 needs −0.2486; alpaca miss 0.1624 needs only −0.1204).
4. **Generic overconfidence does not explain rho=3.** rho=1 at tau=theta=0 IS argmax, and argmax is
   invariant to any global temperature. A 5-point temperature sweep returned bit-identical numbers
   (−0.0936 at every T), confirming the invariance empirically. Therefore the defect is
   candidate-0 specific, not a softmax-sharpness problem, and temperature / label smoothing cannot
   fix it.
5. **Corpus-vs-target-greedy conditioning shift does not explain rho=3 either — it predicts the
   wrong sign.** Training pairs are (corpus prefix, corpus next token) at
   `dflash_family_model.py:1547` with `selector_target_greedy_labels: false`; decoding is on the
   target's greedy rollout. But the draft is distilled from the target, so draft-top1 agrees with
   target-greedy MORE than with the corpus ⟹ training class-0 rate < decode base_right ⟹ class 0
   would be UNDER-scored ⟹ rho<1 should win. rho=3 wins. Falsified.
6. **`covered_conditional` (flat 17-way "top-16 or NONE") is only a reparameterisation.** Its own
   gate (`scripts/gate_covered_conditional.py`, 25/25) derives, with
   `alpha = Z_K/(Z_K+e^{s_none}) = P(covered)`:
   `dL_flat/ds_j = alpha*P_K(c_j) - delta`, `dL_qr/ds_j = P_K(c_j) - delta`.
   At convergence flat fits `P(c_j) = P(covered)*P(c_j|covered)`; `P(covered)` is a per-slot scalar
   that cancels inside the serving argmax, so the deployed decision is IDENTICAL to what multiclass
   converges to. It is therefore **not** given the GPU budget. (You previously recommended this
   objective. This is the reason it is being declined; say so if you think the derivation is wrong.)

**The rho=3 mechanism remains UNKNOWN and this change does not claim to explain it.**

## The change under review

Decoding maximises the accepted prefix length, a sequence quantity. The champion trains a per-slot
0-1 CE (`uniform_frontier_boost`) that prices a slot-0 error identically to a slot-14 error, even
though slot 0 destroys all 15 slots and slot 14 costs one token.

`expected_accept` / `smoothed_accept` weight modes were **already implemented** in the host but a
whitelist at `dflash_family_model.py:388` locked them to `selector_objective == "keep_repair"`. The
change relaxes that to admit `multiclass`, because for multiclass
`selector_probability = exp(-CE)` is exactly `P(argmax over the K deployable candidates hits the
target)` — precisely the survival factor `q_i` the weights need. The weight code touches only
`weight_mask`, `target_is_candidate`, `selector_probability`, `selector_excludes_anchor`.

Files:
- `specforge/algorithms/common/dflash_family_model.py`
  - new module-level `accept_survival_weights(...)`, extracted from the loss body so the gate tests
    the real code and not a copy;
  - `:388`-area whitelist relaxed to `{keep_repair, multiclass}`; `covered_conditional` explicitly
    refused because its `selector_probability` is the JOINT over K+1 classes while NONE is not
    emittable, so it would understate `q_i`;
  - the argmax-serving requirement is scoped to `expected_accept` only (which claims exactness);
    `smoothed_accept` is documented as a surrogate and may be served at the champion's rho=3, which
    is what keeps this a one-variable comparison;
  - new telemetry `selector_base_right`, `selector_mean_slot_weight`, guarded off when the err
    objective owns the `err_*` fields (the champion runs `err_loss_alpha: 1.0`, i.e. exactly the
    colliding configuration).
- `scripts/gate_smoothed_accept.py` — 14 checks, all passing. The central one verifies against
  autograd that the floor-0 weights reproduce `d(-E[A])/d logits` to **5.42e-20**, using
  `dq_i/dCE_i = -q_i` ⟹ `dE[A]/dCE_i = -sum_{t>=i} prod_{j<=t} q_j`.
- `examples/configs/.../qwen3-4b-pspr-cloze-sa.yaml` — diff vs the champion yaml is exactly:
  `weight_mode: uniform_frontier_boost -> smoothed_accept`, `frontier_boost: 3.0 -> 0.0` (inert in
  this mode), `survival_floor: 0.5`, and the four output paths re-pointed so the champion checkpoint
  cannot be overwritten. `draft_model_config` deliberately unchanged.

## What I want challenged

- **C1.** Is `smoothed_accept` genuinely not a reparameterisation? The claim is that a slot
  reweighting changes the finite-capacity optimum whereas `P(covered)` cancelling in an argmax does
  not. If both are equally inert, this run is worthless — say so.
- **C2.** With floor=0.5 the survival factor is `0.5 + 0.5*q`, so the "value" of slot i is at least
  `2^-k`-ish decayed rather than truly `prod q`. Is the resulting ramp dominated by the positional
  term (floor=1 gives exactly 15,14,...,1) to the point that the model-dependent part is noise? If
  so, the honest first arm is floor=1.0, a pure positional ramp with zero extra machinery.
- **C3.** Is it legitimate to train with survival weights derived from the argmax probability while
  serving at rho=3? I scoped the fail-closed check to `expected_accept` and argued the surrogate does
  not need the identity. Attack that.
- **C4.** `uniform_frontier_boost` already applies `policy_reachable` masking plus a frontier bonus,
  so the champion is not purely uniform. Quantify whether `smoothed_accept` is actually different
  from it in the regime that matters, or whether I am re-running a variant of the champion.
- **C5.** Given the accounting above says the entire 22.8pp headroom is "base out-ranks target",
  is slot reweighting even aimed at the right target? Name the single change you would fund instead
  with 9 GPU-hours. Candidates I have NOT implemented: target top-16 soft-distribution distillation
  (dense ranking signal at 100% of slots vs the 22% that hard argmax labels supply); a cascade where
  the detector sees the ranker scores; true leave-one-out cross-attention; anchor-position sampling
  matched to decode's `start += accept+1`.
- **C6.** The telemetry `selector_base_right` is meant to test a prior-mismatch hypothesis by
  comparing the training class-0 rate against the decoder's 75.4%. But it is suppressed exactly when
  the err objective is on, which is the champion configuration. Is the measurement therefore
  unobtainable in the run that matters, and what is the correct way to get it?

Output format: `VERDICT: SOUND | FLAWED | FATAL` per C1-C6 with file:line evidence, then
`WHAT TO CHANGE BEFORE LAUNCH` as an ordered list, then
`WHAT TO WATCH IN THE FIRST 500 STEPS` with metric names and the numeric values that would mean
abort. If you conclude the run should not happen at all, say that first and plainly.
