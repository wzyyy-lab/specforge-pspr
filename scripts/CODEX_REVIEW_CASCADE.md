# Review request: PSPR-cascade, the keep/repair gate reads the ranker's corrected scores

This implements the change you recommended under C5 of `CODEX_REVIEW_SA.out`. Be adversarial about
the implementation, not just the idea — you proposed it, so you are not a neutral judge of whether it
is worth funding; judge whether what was built is what you described and whether it can fail
silently.

## Corrections already applied from your last review

All accepted, with no defence attempted:

1. **C3.** `exp(-CE)` is not "the probability argmax hits the target"; argmax correctness is 0/1. The
   `smoothed_accept` whitelist justification was wrong and the autograd gate was circular (it defined
   `q`, defined `E[A]` from that same `q`, then differentiated). That arm is shelved, not launched.
2. **C4.** `uniform_frontier_boost` (`dflash_family_model.py:1139`) applies **no** reachability mask;
   I had misread `teacher_forced_frontier` (`:1178`). The champion's measured mean slot weight is
   1.174, matching `[1,…,4,…,1]/15`, so the floor-1 ramp would have been a 6.8× scale confound.
3. **C6.** The rho sign error was still in the source comment; fixed, with the derivation spelled
   out (`p_alt > tau + rho*p0` ⟹ larger rho favours candidate 0 ⟹ an under-scored class 0 calls for
   rho>1).
4. **Temperature claim retracted.** You were right: `PSPR_README.md:186` runs
   `regenerate_train_data.py --temperature 0` for this exact output path and
   `PSPR_STAGE2_PLAN.md:35` audits 98.76% greedy agreement. I had inferred 0.8 from the script's
   generic docstring example. The residual train/decode gap is therefore an occupancy effect, and the
   populations are now labelled as unmatched wherever they appear.

## What was built

The champion's gate is structurally blind to the quantity its decision is about:
`ClozeCorrector.err_logits` receives `z`, the seed embedding, `confidence(log_probs, scalars)` built
from the **base** unary logits, optionally `hidden_states` and the GRU state — never the corrected
candidate scores that serving compares. Meanwhile serving uses a hand-set scalar rule
`max_alt_prob > rho * p0`, rho tuned to 3.

The ceiling argument, which is the reason this is the funded arm: every rule of that comparison form
is capped at `selector_rank_0` = 44.05% regardless of rho, because the cap belongs to the comparison
and not to the threshold. A separate gate is not bound by it, so its ceiling is the ranker's own
`alternative_rank_0` = 66.86%. That difference is exactly the 22.81pp in the accounting.

Files:

- `specforge/modeling/draft/pspr_cascade.py` — `CascadeCorrector(ClozeCorrector)` and
  `PSPRCascadeDraftModel(PSPRClozeDraftModel)`. `pspr_cloze.py` is untouched (hard constraint).
  - `score_features` (7 channels): `s0-max_alt`, `logsumexp(alt)-s0`, `p0`, entropy,
    `alt_top1-alt_top2`, `alt_top2-alt_top3`, `s0`. Order statistics only, so permutation-invariant
    over alternatives; all but the last channel shift-invariant.
  - Read-out zero-initialised **and** re-zeroed in an overridden `restore_zero_init_contract`, so
    step 0 is bit-identical to the champion's gate.
  - Features detached by default (`selector_err_score_detach: true`), so the gate can read the ranker
    and provably cannot distort it.
  - `err_only_module_names` extended, so the existing `selector_train_scope: err_only` freeze path
    trains the new head with no host change.
  - `score_candidates` is duplicated from the parent (there is no seam: the parent computes `scores`
    then immediately consumes `z`/`state` for the gate). The gate asserts bit-identical `scores`
    against the parent to catch drift.
- `configs/qwen3-4b-pspr-cascade.json` — architecture swapped; `selector_decision_mode: keep_repair`;
  `selector_gate_rho/tau/theta` **removed** (unread under keep_repair, so leaving them would imply a
  threshold that no longer exists); `selector_train_scope: err_only`; `selector_err_use_state: false`
  (deliberately: enabling it would add a second, also zero-initialised, capability and a gain could
  not be attributed).
- `examples/configs/.../qwen3-4b-pspr-cascade.yaml` — warm start from
  `outputs/qwen3-4b-pspr-cloze/qwen3-4b-pspr-cloze-step7115` (the 6.0498 checkpoint),
  `selector_objective: profitable_repair`, `err_loss_alpha: 0.0`, `max_steps: 500`,
  `save_interval: 100`, weight mode unchanged, output paths re-pointed.
- `scripts/gate_pspr_cascade.py` — 15 checks, all passing, including `scores` max|d| = 0.000e+00
  versus the parent, gate logit max|d| = 0.000e+00 at zero-init, zero-init surviving a
  post_init-style re-init, 13 trainable / 53 frozen tensors under `err_only`, and zero gate gradient
  reaching `delta_*`.
- `scripts/export_pspr_cascade_for_decode.py` and a `CascadeCorrector` branch in
  `TAPS-SP/scripts/decode_lattice.py:load_selector_checkpoint`, both keyed on
  `model_type: "CascadeCorrector"` with strict loading.

No new objective was written: `profitable_repair` already labels REPAIR iff the target is one of
candidates 1..K-1 with KEEP for both base-right and top-k misses
(`dflash_family_model.py:1035`), which is the labelling you specified.

## What I want challenged

- **D1.** Is the ceiling argument correct? Specifically: with the ranker frozen, is
  `alternative_rank_0` = 66.86% actually the cascade's ceiling, or does serving through
  `keep_repair_log_probs` reintroduce a base-versus-alternative comparison that re-imposes the 44.05%
  cap? `pspr.py:485` and `decode_lattice.py`'s `latrepair` branch are the relevant code.
- **D2.** The 7 features. Is anything load-bearing missing, or is any channel redundant to the point
  of hurting a 500-step warm-start? You listed "entropy/full score vector"; I used order statistics
  instead of the full vector to keep permutation invariance. Attack that choice.
- **D3.** Detaching. It makes the one-variable claim structural, but it also means the ranker can
  never adapt to the existence of a better gate. Is detach the right default for a 500-step
  calibration arm, and is it the wrong default for anything after it?
- **D4.** The previous `latrepair` evaluation of a cloze checkpoint scored 6.371 on gsm8k n=20 against
  7.057 for `latgate` rho=3 — i.e. the cascade serving path has historically been *worse*. That was a
  gate trained without score features and, in the PSPR-Decision case, with a KEEP branch that
  absorbed the miss mass. Is that sufficient to explain the 0.69 gap, or is there a defect in the
  `latrepair` serving path itself that this arm will inherit?
- **D5.** Silent-failure surface. Where can this configuration appear to work while decoding as
  something else? I am specifically worried about: the warm start loading the champion's selector
  under a different architecture name; `err_only` interacting with the frozen backbone; and the
  duplicated `score_candidates`.
- **D6.** Is 500 steps with `err_only` enough for a 7→512→1 head to learn anything, given
  `selector_loss_alpha: 1.0` and the fact that the binary BCE is now the whole selector loss on
  base-right slots? Give the metric and threshold that would show it is learning rather than drifting.

Output `VERDICT: SOUND | FLAWED | FATAL` per D1-D6 with file:line evidence, then
`WHAT TO CHANGE BEFORE LAUNCH` as an ordered list, then `WHAT TO WATCH IN THE FIRST 500 STEPS` with
metric names and abort values. If the run should not happen, say so first and plainly.
