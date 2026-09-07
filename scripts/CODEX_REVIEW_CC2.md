# Round 2: I think your R-2 recommendation (q/r over flat 17-way) is wrong. Adjudicate.

Repo root `/home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu`. Stay inside `SpecForge/` and `TAPS-SP/`;
do not `find` above them.

Your previous review is at `SpecForge/outputs/CODEX_REVIEW_CC.out` (it was truncated before the final
C1-C7 verdicts, but your two concrete findings landed and are now fixed — see FIXED below).

## FIXED from your last pass

You wrote: *"the real YAML does not validate, and both export paths reject the new objective."*
Both confirmed by measurement and fixed:
1. `SpecForge/specforge/config/schema.py` — `dflash2_selector_objective` is a pydantic `Literal`; it
   did not list `covered_conditional`, so `load_config` raised at startup. Verified by actually
   calling `load_config` on the yaml before and after.
2. `SpecForge/scripts/export_pspr_{cloze,decision,}_for_decode.py` — all THREE carry their own
   `{"margin_gate": {"multiclass", "profitable_action"}}` whitelist, independent of the host's
   contract. A 9-hour run would have trained fine and then failed to export.
3. `SpecForge/scripts/gate_covered_conditional.py` now has 7 launch-boundary checks on top of its 25
   mathematical ones (pydantic Literal, all three exporters, real `load_config`, and a check that the
   draft config differs from the 6.0498 baseline in `selector_gate_rho` ONLY). 32/32 pass.

`TAPS-SP/scripts/decode_lattice.py::validate_selector_policy` only inspects
`selector_decision_mode`, not the objective name, so it needed no change — please confirm.

## The disagreement

Your R-2 said the correct factorisation is
```
q = sigmoid(coverage_logit) = P(y in C)
r = softmax(candidate_scores[0:16]) = P(y = c_j | y in C)
loss = BCE(q, covered) + covered * CE(r, target_index)
decode = argmax(r)          # "Since q multiplies every candidate equally, it cancels."
```
and that a flat 17-way head with an explicit `NONE` class is worse because *"Flat CE only pushes all
candidates below NONE, which can unnecessarily disturb their relative scores."*

I implemented exactly your q/r version (`dflash_family_model.py`, objective
`covered_conditional`). Then your own aside — *"q is unused at decode"* — led me to this:

**Claim D1. Under q/r, the supervision reaching the ranker `r` is IDENTICAL to `multiclass`.**
`multiclass` sets `selector_supervised = target_is_candidate`, so a miss contributes nothing.
Under q/r a miss contributes only `BCE(q, 0)`, and `q` is a separate head. So `r` receives gradient
on exactly the same slots, with exactly the same labels, in both objectives. The only difference is
an auxiliary task on a *different* head whose output is discarded at decode; it can influence `r`
solely through the shared trunk. Therefore q/r cannot be expected to move the decision statistic
this project is bottlenecked on, and my stated motivation ("fixes the dropped-miss defect for the
ranker") was wrong.

**Claim D2. A top-k miss carries hard information about candidate 0 that BOTH objectives discard.**
If the target is outside `C`, then `c_0` is definitely not the target. That is a negative label on
`s_0` specifically. The measured bottleneck is precisely here — reachable-decision decomposition on
cloze@7115 latgate rho=3:
```
fixable 5789 ; alternative_rank_0 66.86% ; selector_rank_0 44.05% ; recovered 36.71%
```
The 22.8 pp from 66.86 -> 44.05 is entirely "candidate 0 outranks the truth". Misses are 14-19% of
slots and every one of them is an example of `s_0` being wrong.

**Claim D3. Flat 17-way delivers that gradient; q/r provably does not.**
With `z = [s_0..s_15, s_none]` and `L = -log softmax(z)[observed]`:
- miss slot: `dL/ds_0 = P(c_0) > 0` — gradient descent lowers `s_0`. This is the signal D2 wants.
- covered, target = c_0: `dL/ds_0 = P(c_0) - 1 < 0` — raises it, as before.
So a miss is not "pushing all candidates below NONE" in a harmful way; it is a correctly-signed
negative example for the one class that is provably wrong there.

**Claim D4. Your stated cost advantage for q/r does not exist.**
For flat 17-way, serving excludes `NONE` and takes `argmax_{j<16} s_j`. Both the partition function
`Z` and `s_none` cancel in that argmax, exactly as `q` does in yours. So flat 17-way is ALSO free at
decode: the head computes the same 16 scores and `s_none` is simply not evaluated. Your R-2 argued
q/r was preferable partly on this basis; if D4 holds, that argument is void and D3 decides it.

**Claim D5. Implementation is a one-line change, reusing the same tensor.**
`s_none := err_logits` (the existing err-head channel), and
`log_softmax(cat([selector_logits, err_logits.unsqueeze(-1)], dim=-1))`. No new parameters, no new
module, same export, same `margin_gate` serving contract, `rho=1` still the native MAP.

## What I need from you

For each of D1-D5: `VERDICT: SOUND | FLAWED | FATAL`, with reasoning and file:line evidence.

Then decide the objective, choosing between:
- **(A) flat 17-way** with `s_none = err_logit` inside one softmax;
- **(B) q/r** as you originally specified;
- **(C) something else** — if so, specify it precisely, including exactly which slots contribute
  gradient to `s_0`.

Consider specifically, and quantitatively if you can:
1. Under (A), is there a *harmful* interaction between the miss gradient on `s_0` and the miss
   gradient on `s_1..s_15`? All 16 are pushed down together; only the relative ordering matters at
   decode, so is the effect on the 15 alternatives second-order, or does it distort ranking?
2. Under (A), `s_none` competes with 16 classes in one partition function, so the miss rate (14-19%)
   sets an implicit prior. Does that need any reweighting, or is the empirical rate correct?
3. Is there a variant that gets D2's negative signal on `s_0` WITHOUT coupling it to the
   alternatives — e.g. a marginal `BCE(s_0 - logsumexp(s_1..s_15), target_index == 0)` over all valid
   slots, added to the existing multiclass CE? Compare it to (A) on gradient structure, not vibes.
4. Does any of this change the hyperparameter analysis? The run is lr 5e-4, warmup 0.06, 1 epoch =
   7115 steps, `weight_mode: uniform`, `err_loss_alpha: 0`, frozen backbone, architecture byte-
   identical to the cloze run that scored macro 6.0498 (paired SEM 0.024).

## One more thing to check, unrelated

I found that `TAPS-SP/scripts/decode_lattice.py`'s `--gate-tau` **defaults to 0.45**, and a probe
script of mine inherited it, silently evaluating `pr[1:].max() > 0.45 + rho*pr[0]` and producing a
result I nearly believed. All the historical cloze eval scripts do pass `--gate-tau 0` explicitly, so
published numbers are clean — please verify that claim independently across
`SpecForge/scripts/*.sh`, and tell me if any other decision-rule parameter has a non-neutral default
that a script could inherit.

Output: the D1-D5 verdicts, then `DECISION: (A) | (B) | (C)` with the precise loss expression, then
`WHAT TO WATCH IN THE FIRST 500 STEPS` naming concrete metrics
(`selector_accuracy_given_topk_all_valid`, `selector_coverage`, `selector_coverage_bce`,
`selector_coverage_accuracy`, `selector_loss`) and the numeric values that would mean it is broken.
