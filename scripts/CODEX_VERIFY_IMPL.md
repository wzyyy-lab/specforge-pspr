# Task: adversarially review a NEW speculative-decoding selector head before 1.2 GPU-hours are spent on it.

I wrote this head today. Find what is WRONG with it. Concrete bugs and design errors, with file:line.
Do not summarise what it does back to me. Do not be encouraging.

Repo roots:
- SpecForge: /home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/SpecForge
- TAPS-SP:   /home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/TAPS-SP

## Context you need

A FROZEN DFlash-b16 draft backbone drafts a block of 16 tokens (1 anchor + 15 proposal slots) in a
single forward pass with [MASK] at every proposal slot. Per slot i it emits a hidden state h_i; a
top-16 candidate lattice is read off the frozen tied `lm_head(h_i)`. A small trainable "selector"
head may override the backbone's top-1 at each slot. Accept length = length of the longest correct
prefix the target model verifies. The backbone stays bit-frozen; only the head trains.

New files (mine, today):
- `SpecForge/specforge/modeling/draft/pspr_decision.py`  <- the head
- `SpecForge/configs/qwen3-4b-pspr-decision.json`
- `SpecForge/examples/configs/online/disaggregated/managed-local/qwen3-4b-pspr-decision.yaml`
- `SpecForge/scripts/gate_pspr_decision.py`, `scripts/gate_pspr_decision_decay.py`
- `SpecForge/scripts/export_pspr_decision_for_decode.py`

Predecessors for reference (DO NOT propose edits to these, they carry shipped checkpoints):
`pspr.py`, `pspr_v2.py`, `pspr_cloze.py`.

Host objective code: `SpecForge/specforge/algorithms/common/dflash_family_model.py`
(the `profitable_repair` branch is around line 840; label/weight logic around 1100-1240).
Decoder: `TAPS-SP/scripts/decode_lattice.py` (mode `latrepair`).
Design rationale and all measurements: `TAPS-SP/PSPR_DECISION.md`.

## The design in one paragraph

The predecessor used a 6-layer bidirectional Transformer over H*(H+1) cells. Inference-time
ablations showed removing the future costs 0.0012 macro accept length, removing ALL cross-slot
information costs 0.0445, and zeroing the encoder output costs 0.5086 -- so ~91% of that path was a
per-slot transform. This head replaces the attention with a per-slot residual MLP tower plus one
mean-over-block broadcast, and factorises the decision:

    p_i = sigmoid(detector(feat_i))          P(a usable repair exists at slot i)
    r_i = softmax(ranker(feat_i)) in R^15    P(target = cand_j | a usable repair exists)
    P(target = cand_0) = 1 - p_i             absorbs "base is right" AND "target not in top-16"
    P(target = cand_j) = p_i * r_i[j-1]
    decision = argmax_j P(target = cand_j)   -- no tau/rho/theta

Trained with `dflash2_selector_objective: profitable_repair`, whose detector label is
`repair_is_covered` (target is one of candidates 1..15) and whose alternative CE is conditional on
that event.

## What I want you to attack

### A. Correctness of the factorisation
A1. Is `1 - p_i` the correct probability for candidate 0 given the label `repair_is_covered`? The
    event "base top-1 is right" and the event "target is outside top-16" are merged into it. Does
    that break the decode rule anywhere?
A2. `score()` returns `log_probs + <E(cand_j), delta>` for ALL 16 candidates, but only `[..., 1:]`
    is ever log-softmaxed under this objective. Is candidate 0's score therefore a dead output that
    still receives gradient through `delta`? If yes, does that create a conflicting pull on `delta`?
A3. `keep_repair_log_probs` (in `pspr.py`, reused here) is
    `[logsigmoid(-err), logsigmoid(err) + log_softmax(scores[1:])]`. Verify the step-0 zero-init
    claim: with `delta_w2` zeroed and the detector's last layer zeroed (bias = logit(0.22)), is the
    argmax provably 0 at every slot, for ANY lattice? What if two candidates tie in log-prob, or
    K=2, or the log-probs are all equal?

### B. Calibration -- I believe this is the real risk
B1. The training recipe sets `dflash2_selector_frontier_boost: 3.0`, which multiplies the weight of
    each block's first base error by 4. `repair_is_covered` is ~0.84 at those slots and ~0.22
    overall. I estimate the detector's effective prior moves 22% -> ~32%, biasing it toward
    override. Check my arithmetic against the actual weighting code
    (`uniform_frontier_boost` branch, `dflash_family_model.py`) and tell me whether the bias is
    (a) a roughly constant logit shift, correctable by the scalar `--repair-margin`, or
    (b) slot-dependent, in which case a scalar cannot fix it.
B2. At step 400 of 7115, real decoding gave `base-right kept = 93.24%` against the predecessor's
    99.65%, and NET accepted slots per block = -0.026 (i.e. worse than not overriding at all).
    Is that consistent with "detector not yet calibrated, 5.6% into training", or is there a
    structural reason it will not converge? Look for anything that would make the detector
    permanently miscalibrated.
B3. Is `det_prior=0.22` as the output-bias initialisation actually consistent with the label
    `repair_is_covered` under the boosted weighting? Should it be the weighted rate instead?

### C. Train/serve consistency
C1. During training the causal GRU consumes `predecessor_ids` built from CORPUS ground-truth tokens;
    at decode it consumes the tokens the selector actually committed. The detector is now a
    first-class decision maker and reads that state (`err_use_state: true`). Does that make this
    head MORE sensitive to the teacher-forcing gap than its predecessor, which used the state only
    for scoring? Quantify the risk if you can.
C2. `slot_states()` must not depend on the committed prefix, or serving would need H passes.
    Verify it truly does not. Also verify the batched training path and the per-slot decode walk
    (`_sample_draft_tokens` in pspr_decision.py, and `lattice_block_inputs` / the `latrepair` loop
    in decode_lattice.py) compute the same thing.
C3. The block-mean term `hidden.mean(dim=-2)` is taken over the H slots. In training,
    `hidden_states` is `[B, num_blocks, H, hidden]`; at decode it is `[1, H, hidden]`. Confirm the
    reduction axis is the same object in both, and that no cross-BLOCK leakage occurs.
C4. `ctx_ln` is a LayerNorm applied AFTER the mean. LayerNorm subtracts the mean over the feature
    dimension, so any uniform shift of the block summary is annihilated. Is that a meaningful loss
    of information here, or irrelevant?

### D. The recipe
D1. `dflash2_selector_err_loss_alpha: 0.0` -- confirm this is REQUIRED (not merely allowed) for a
    factorised objective, i.e. that a nonzero value would apply a second, contradictory label to the
    same logit. Find the enforcement.
D2. `objective_chunk_blocks: 128` (predecessor used 64 on smaller cards). Measured trainer memory is
    22 GiB of 141. Is there any correctness (not just memory) consequence of the chunk size?
D3. weight decay: `training.weight_decay: 0.0` plus 16 explicit suffix rules. The matcher is
    `name.startswith(prefix) or f".{prefix}" in name`, sorted by descending prefix length
    (`specforge/optimizer.py:94`). Find any rule that matches something I did not intend, and any
    2-D weight that gets no decay. `scripts/gate_pspr_decision_decay.py` claims 22 decayed / 44 not.
D4. lr 5e-4 and warmup_ratio 0.06 were inherited from the predecessor, whose head was a 6-layer
    Transformer. This head is a 4-layer residual MLP tower with a different gradient path. Is
    inheriting them defensible, and is there a specific reason to expect a different optimum?

### E. Anything I have not thought of
Read `pspr_decision.py` end to end and list any bug, dead code path, silent dtype problem,
shape-broadcast hazard, or place where an ablation switch does not do exactly what its name says.

## Output format
Sections A-E. For each numbered item: VERDICT (correct / bug / risk / unclear) + file:line + a
one-or-two-sentence reason. Then a final section "THE THREE THINGS MOST LIKELY TO MAKE THIS FAIL",
ranked, max 150 words. If you think the whole approach is misguided, say so and say why.
