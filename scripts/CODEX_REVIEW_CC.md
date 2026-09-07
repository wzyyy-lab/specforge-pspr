# Review request: `covered_conditional` selector objective, before any GPU time

You are reviewing a change that is about to consume ~9 GPU-hours. The previous round wasted a full
run because a modelling defect was *recorded* and then not tested. Be adversarial. If the design is
wrong, say so and say why with file:line evidence.

Repo root: `/home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu`
Do NOT run `find` outside the two project directories below; everything you need is in:
- `SpecForge/` (training host + heads)
- `TAPS-SP/` (decode/eval harness + logs)

## Files to read

Implementation (all edits are mine, this round):
- `SpecForge/specforge/algorithms/common/dflash_family_model.py`
  - `_VALID_SELECTOR_OBJECTIVES`, `_FACTORIZED_SELECTOR_OBJECTIVES`, and the NEW
    `_MAIN_LIKELIHOOD_ERR_OBJECTIVES` near the top
  - the `covered_conditional` branch in the objective dispatch (search `covered_conditional`)
  - the coverage telemetry block (search `selector_coverage_bce`)
  - the three control paths I retargeted from the factorised set to the new set:
    `wants_err`, the err-head freeze guard, the head-required guard
  - `selector_supervised` for this objective
- `SpecForge/scripts/gate_covered_conditional.py` (25 checks, all passing)
- `SpecForge/configs/qwen3-4b-pspr-cloze-cc.json` (architecture identical to cloze; only
  `selector_gate_rho` 3.0 -> 1.0)
- `SpecForge/examples/configs/online/disaggregated/managed-local/qwen3-4b-pspr-cloze-cc.yaml`
- Reference, unchanged: `.../qwen3-4b-pspr-cloze.yaml`, `SpecForge/configs/qwen3-4b-pspr-cloze.json`

Evidence this round (read these before judging the design):
- `TAPS-SP/PSPR_DECISION.md` steps 17-23
- `SpecForge/scripts/paired_compare.py`

## The design

Structured "top-16 or NONE" likelihood, replacing 16-way multiclass CE:

```
q_i  = sigmoid(coverage_logit_i)              = P(y in C_i)
r_ij = softmax(scores_i)_j  over ALL 16       = P(y = c_ij | y in C_i),  candidate 0 = KEEP
P(NONE) = 1 - q_i ,  P(c_ij) = q_i * r_ij
loss = BCE(q, target_is_candidate) + target_is_candidate * CE(r, target_candidate_index)
serve: argmax(r)      # q multiplies all 16 equally -> cancels -> never computed at serving
```

The coverage logit reuses the existing `err_head` output channel; no module, no shape, no serving
code changes. `selector_err_loss_alpha` is forced to 0 (a standalone err BCE would be a second,
contradictory label on the same logits: `~base_top1_ok` vs `target_is_candidate`).

Config deltas vs the cloze run that scored macro 6.0498:
1. `dflash2_selector_objective: multiclass` -> `covered_conditional`
2. `dflash2_selector_weight_mode: uniform_frontier_boost` + `frontier_boost: 3.0` -> `uniform`
3. `dflash2_selector_err_loss_alpha: 1.0` -> `0.0`
4. draft config `selector_gate_rho: 3.0` -> `1.0`
Everything else byte-identical (same 6-layer bidirectional encoder, d=512, state_dim=512,
delta_hidden=2048, lr 5e-4, warmup 0.06, 1 epoch = 7115 steps, chunk_blocks 64, same
weight_decay_rules, frozen backbone).

## Measurements this design is built on (all mine, this session)

Decision decomposition, cloze@7115 latgate rho=3, `TAPS-SP/outputs/DETAIL_cloze7115_rho3.log`:
```
decision_n 26353 ; base-right 19872 (kept 19802 = 99.65%) ; base-wrong fixable 5789 ; unfixable 692
alternative_rank_0 3870 = 66.86%   (truth first among alternatives, candidate 0 excluded)
selector_rank_0    2550 = 44.05%   (truth first once candidate 0 is included)
gate blocked        425 =  7.34%
fix_recovered      2125 = 36.71%   (2550 - 425 = 2125, exact)
```

Paired per-prompt comparisons (new tool; paired macro SEM 0.024 vs unpaired 0.24):
| comparison | macro diff | paired SEM | t | p |
|---|---|---|---|---|
| causal vs full (bidirectionality removed) | +0.0070 | 0.0182 | 0.38 | 0.70 |
| anchor_self vs full (all cross-slot removed) | -0.0465 | 0.0244 | -1.90 | 0.057 |
| zero_context vs full (positive control) | -0.5550 | 0.0495 | -11.20 | 4e-29 |
| **rho=1 vs rho=3** | **-0.1110** | 0.0436 | -2.54 | 0.011 |

Factorised decode on trained cloze weights (gsm8k n=20), i.e. discarding `scores[0]`:
| rule | base-right kept | recovered | destroy | accept |
|---|---|---|---|---|
| latgate rho=3 (native) | 99.65% | 36.71% | 70 | 7.057 |
| latrepair m=0 | 96.98% | 35.63% | 111 | 6.371 |
| latrepair m=3 | 99.05% | 18.69% | 35 | 6.112 |

PSPR-Decision margin sweep at step5600, six domains n=20: macro 5.7400 / 5.8372 / 5.8275 / 5.8152 /
5.8152 at margin 0 / 0.1 / 0.2 / 0.3 / 0.4. Best 5.8372 vs cloze 6.0498.

## Claims I want attacked

**C1.** `frontier_boost=3.0` is `+log(3) = +1.10` nats of standing override bias, and the optimal
serving `rho=3.0` is *also* 1.10 nats, so the gate exists only to undo a bias the loss injected.
Therefore removing the boost should make `rho=1` (the trained MAP) optimal, and that is this run's
primary read-out. Is the nats accounting right? `rho` enters as
`pr[1:].max() > tau + rho*pr[0]` (`TAPS-SP/scripts/decode_lattice.py` latgate branch) and the boost
enters as a loss *weight* multiplier, not a logit offset — do these two actually live in the same
units, or am I equating a per-sample loss weight with a decision-threshold offset illegitimately?
This is the single load-bearing claim of the whole design; if it is wrong the run is pointless.

**C2.** Keeping KEEP inside the 16-way conditional softmax is required, not stylistic, because the
latrepair table above shows discarding `scores[0]` costs 0.69 accept and doubles destroys. But that
experiment used the cloze `err_head`, which was trained with `err_target = ~base_top1_ok` and whose
KEEP branch has the very miss-mass defect this design fixes. Does that confound invalidate C2? If it
does, the design should arguably be a *properly calibrated* factorisation instead.

**C3.** MAP is optimal for maximising `E[accept] = sum_i P(all decisions <= i correct)` because the
continuation value multiplies every candidate equally at a given slot. Hence any benefit from a
conservative `rho` is *pure miscalibration*, not a genuine asymmetric-cost effect. Counter-argument I
can see: destroying a correct base token truncates the block whereas failing to repair a wrong one
was already truncating it, so the two errors have different consequences. Resolve this properly.
If asymmetry is real, the objective needs a cost-weighted term and this run is under-specified.

**C4.** Reusing `err_head` as the coverage head is free. Risks I see: (a) its input features were
chosen for "is top-1 wrong", which is a different question from "is the target present anywhere in
top-16"; (b) `pspr_cloze.py`'s `err_logits` signature and inputs — check whether it can even express
coverage; (c) the `err_*` telemetry channel reuse could collide with something I missed.

**C5.** The three control paths I retargeted are the complete set. If any site still keys off
`_FACTORIZED_SELECTOR_OBJECTIVES` where it should key off `_MAIN_LIKELIHOOD_ERR_OBJECTIVES`, the
coverage head is silently frozen or `err_logits` is missing. Grep exhaustively and tell me if I
missed one. Also: does anything downstream (export, decode loader, `validate_selector_policy` in
`TAPS-SP/scripts/decode_lattice.py`) need to know about the new objective name?

**C6.** Hyperparameters. `lr 5e-4` and `warmup 0.06` are carried over unchanged on the argument that
the architecture is identical and the new BCE term only touches the 1.85M err_head, leaving the
ranker's gradient scale unchanged. Check that. Also: the loss now has a term on 100% of slots instead
of 81-86%, so `selector_weight_den` grows — does that change the effective LR on the shared trunk,
and should `dflash2_selector_loss_alpha` compensate?

**C7.** Is this the right *next* experiment at all? The alternative is stage-2 (unfreeze backbone,
LR 3e-5), which `TAPS-SP/EXPERIMENT_LOG.md:2345` shows delivered macro 5.7217 -> 6.1565 (+0.4348),
six domains monotonically up, zero forgetting, and was stopped at step 2400 of 7115. Frozen-backbone
oracle@16 is 10.4422 and current best is 6.0498. Argue the ordering.

## Things I already got wrong this session, for calibration

- Claimed "3.7x ranker sample starvation" explained Decision's deficit. Wrong: misses give no
  positive ranking labels under either objective; you told me this last round and you were right.
- Claimed the cross-slot ablation proved bidirectionality worthless. The number (-0.0445) was a
  quarter of one standard error. Only after switching to a paired estimator did the question become
  answerable — and the answer turned out to be that bidirectionality really is worthless
  (CI excludes >0.043) while cross-slot is weakly positive.
- Claimed base error rate was 3.5%. It is 24.6%.
- Claimed a 1.37e-7 gate discrepancy was a log-vs-prob issue, then a reduction-order issue. It was a
  float32 downcast in my own reference implementation.

Output format: for each of C1-C7 give `VERDICT: SOUND | FLAWED | FATAL` plus the reasoning and
file:line evidence. Then a section `WHAT TO CHANGE BEFORE LAUNCH` with a concrete ordered list, and
`WHAT TO WATCH IN THE FIRST 500 STEPS` with specific metric names and the numeric values that would
mean the run is broken.
