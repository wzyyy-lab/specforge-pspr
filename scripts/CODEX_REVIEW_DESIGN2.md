# Review this design BEFORE I implement it. Attack it. I have already wasted one 1.2-GPU-hour run.

You reviewed my previous head (`pspr_decision.py`) and were right about several things. That run has
now finished and LOST. I need you to find what is wrong with the replacement design while it is still
cheap to change.

Repos: SpecForge `/home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/SpecForge`,
TAPS-SP `/home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/TAPS-SP`.
Full measurement log: `TAPS-SP/PSPR_DECISION.md`. Prior head: `specforge/modeling/draft/pspr_decision.py`.
Objective host: `specforge/algorithms/common/dflash_family_model.py`. Decoder: `TAPS-SP/scripts/decode_lattice.py`.

## What is fixed and not up for discussion
A FROZEN DFlash-b16 draft backbone drafts 16 positions (1 anchor + H=15 proposal slots) in ONE
forward with [MASK] at every proposal slot, emitting h_i per slot. A top-16 lattice is read off the
frozen tied `lm_head(h_i)`. A trainable head may override the backbone's top-1 per slot. The target
verifies the block; accept length = longest correct prefix. Stage 1 keeps the backbone frozen.

## Measured facts you should treat as ground truth (all n=20, six domains, same hardware)

Accept length:
    oneshot (no head)      5.3043
    Domino                 5.8868
    cloze@4000             5.9532
    cloze@7115             6.0498
    Decision@4000          5.6975      <- my factorised head, LOST by 0.2557 at equal steps
    oracle@16              10.4422     <- ceiling for any top-16 selector on this lattice

Per-block latency, measured today (`scripts/bench_head_latency.py`, `scripts/bench_block_breakdown.py`):
    target decode, 1 tok       26.46 ms
    target verify, 16 tok      29.93 ms     (only +13% over 1 token: memory-bandwidth bound)
    draft backbone, 16 tok      3.45 ms
    lm_head readout             0.21 ms
    block without any head     33.59 ms
    Domino head                 3.10 ms
    PSPR-v2 head                6.46 ms
    PSPR-cloze head             8.24 ms
    PSPR-Decision head         11.62 ms

Therefore, throughput = accept / block_ms:
    no speculation      37.8 tok/s
    oneshot            157.9
    Domino             160.4      <- the only head that pays for itself, and only by 1.6%
    cloze@7115         144.6
    Decision@4000      126.0
Break-even accept vs oneshot: 5.79 at a 3.1 ms head, 6.61 at 8.24 ms, 7.14 at 11.62 ms.

Repair diagnostics at MATCHED conservatism (this is the key comparison):
    Decision@1600, margin 0.5 : base-right kept 99.93%, repair rate 29.14%
    cloze@7115,    rho 3      : base-right kept 99.65%, repair rate 36.71%
i.e. at equal destroy, my factorised head repaired 7.6pp FEWER first errors. Its ROC is worse, so
this was not a threshold/working-point problem.

Ablations of the cloze head (same trained weights, inference-time, so parameters and FLOPs are
identical): removing the FUTURE costs 0.0012 macro; removing ALL cross-slot info costs 0.0445;
zeroing the encoder output costs 0.5086. n=20 cannot resolve 0.04, so only the 0.5086 is solid.

## My diagnosis of why Decision lost. Tell me if I am wrong.

D-1. **Ranker sample starvation.** `multiclass` supervises the ranker at every slot whose target is
     inside top-16 (`selector_coverage` measured 0.81-0.86). `profitable_repair`'s alternative CE is
     conditional on `repair_is_covered`, which is ~0.22 of slots. So the ranker saw ~3.7x less data.
     I only reasoned about the 77.4% class-0 prior polluting the ranking and never counted this.
D-2. **Per-slot capacity cut.** cloze's encoder is ~31.5M; my per-slot tower is 8.4M. Under
     `anchor_self` the cloze encoder is still 6 layers of (self-attention-to-self + FFN), which is a
     strong per-slot transform. I replaced it with 4 residual MLP blocks and assumed equivalence.
D-3. **Latency was never measured.** 11.62 ms is not arithmetic: my FLOPs are an order of magnitude
     BELOW Domino (no `[*, vocab]` matrix at all; Domino pays `256x151936` per slot). It is
     kernel-launch overhead from H serialised small ops: per slot one GRU step, one
     `3584->2048->2560` delta MLP, and one `4116->1024->1024->1` detector MLP. cloze looked cheaper
     partly because at `theta=0` it never calls its detector at all.

## The replacement design. Four changes.

### R-1. Latency: move everything that does not depend on the committed prefix out of the loop
The only genuinely serial quantity is the GRU state S_i over committed tokens. Today S_i is
concatenated into the delta MLP and the detector MLP, which forces BOTH big MLPs to run H times.
Proposal: the state enters only through cheap low-rank terms, so the big matrices run ONCE, batched:

    batched, once per block:
        z_i            = tower(h_i, conf_i, lp_i, E(d0_i), pos_i, ctx)        [H, d]
        delta_static_i = W2 · silu(W1_h·LN(h_i) + W1_z·LN(z_i))               [H, hidden]
        base_score_ij  = lp_ij + <E(c_ij), delta_static_i>                    [H, K]
        v_ij           = V·E(c_ij)                                            [H, K, r], r ~ 32
        err_static_i   = errhead(h_i, z_i, conf_i, lp_i, E(d0_i))             [H]
    per slot, serial:
        S_i            = GRU_step(E(committed_{i-1}), S_{i-1})
        u_i            = U·S_i                                                [r]
        score_ij       = base_score_ij + <u_i, v_ij>
        err_i          = err_static_i + w·S_i
        pick           = decision_rule(score_i, err_i)

Per slot this is one GRU step, one 512->32 matvec, K dot products of length r, one dot product for
err. Target: <= 4 ms, i.e. within ~1.3x of Domino.
Questions: (a) Is the bilinear `<U·S_i, V·E(c_j)>` expressive enough to replace the state's current
full participation in the delta MLP, or does this throw away most of what the causal stream was
contributing? (b) Would you instead keep S_i in the delta MLP and shrink/quantise elsewhere?
(c) Is there a fundamentally cheaper way to make a top-16 decision that I am missing?

### R-2. Objective: 17-way softmax over {c_0..c_15, NONE}
Trained at EVERY valid slot. Label = index of the target in the lattice, or NONE when the target is
outside top-16. Decode = `argmax over {c_0..c_15}` (NONE is never selectable).
Claims I want checked:
 (a) It fixes your A1: KEEP's score no longer absorbs the out-of-lattice mass, because NONE holds it.
 (b) It fixes D-1: the ranker now gets gradient at 100% of valid slots, MORE than multiclass's 81-86%,
     because misses become a real supervised class instead of being dropped.
 (c) The 77.4% class-0 prior is a TRUE prior, not a defect, so leaving KEEP inside the same softmax
     is correct as long as the model is not also forced to explain the miss mass with it.
 (d) No tau/rho/theta: the decode rule is the constrained MAP action.
Is (c) right? My previous head's whole premise was that KEEP must be factored OUT of the softmax, and
that head lost. Am I now over-correcting? Note this needs a NEW objective in
`dflash_family_model.py`; tell me what it must do about `selector_supervised`, the weight modes, and
the accuracy telemetry so that the reported numbers stay comparable to multiclass runs.

### R-3. Capacity: restore the per-slot budget cloze had (~30M in the tower)
Widen/deepen the per-slot residual tower to roughly match cloze's encoder parameter count. Cost is
batched matmul, not launches, so it should be nearly free in wall-clock. Do you agree, and what would
you spend the budget on -- depth, width, or expansion?

### R-4. Make the bidirectional/cloze idea actually testable
My cloze head was NOT a cloze: with `direct_hidden=True` the query row still received h_i at full
width, and `argmax(lm_head(h_i))` IS the masked token, so the answer was never hidden. A genuine
leave-one-out feature would be:

    z_loo_i = attention at position i over {h_j, E(d0_j) : j != i}, with h_i EXCLUDED

implemented as ONE attention pass with the diagonal masked (`mask[i,i] = -inf`), giving all H
positions their leave-one-out representation in a single forward -- 1/H the cost of the H-row cloze.
STRICTLY ONE LAYER, because with two layers position i sees position j's output, which already
contains h_i, and the exclusion leaks. For more capacity I would run several independent
single-layer heads in parallel rather than stack.

The intended value is NOT new information: `{h_j}` is closed under the frozen lm_head, so nothing
here exceeds it. It is that the DISAGREEMENT between `z_loo_i` (what the rest of the block implies
slot i should be) and `h_i` (what the backbone actually put there) is exactly the evidence a
keep/repair decision needs, and it is a feature the model is unlikely to construct by itself.

Questions: (a) Is the one-layer leakage argument correct, and is one layer enough to be useful?
(b) Is "disagreement between a leave-one-out prediction and the actual h_i" a sound signal here, or
is it circular given the closure argument? (c) Since `anchor_self` measured only -0.0445 (inside
n=20 noise) for ALL cross-slot information, is R-4 justified at all, or am I chasing a quantity the
data says is ~0? (d) If it is worth testing, how would you make the test conclusive at n=20?

## Also tell me
S-1. Given break-even accept is ~5.8-6.0 at a 3-4 ms head, and cloze already reaches 6.05 at 8.24 ms:
     is the highest-value move actually to make the EXISTING cloze head cheaper rather than to build a
     better one? Be concrete.
S-2. Stage 2 unfreezes the backbone and trains jointly. `EXPERIMENT_LOG.md` records that joint
     training deletes the very candidates the selector relies on (top-K recall drops). Find that entry
     and tell me what it actually says, and what it implies for the order of operations.
S-3. Anything in R-1..R-4 that will not work for a reason I have not mentioned.

## Output
Sections R-1, R-2, R-3, R-4, S-1, S-2, S-3. For each: VERDICT (sound / flawed / fatal) + file:line
where relevant + reasoning. Then "WHAT I WOULD BUILD INSTEAD", max 250 words, concrete enough to
implement. If the honest answer is that no top-16 selector head can beat Domino on throughput on this
backbone, say that plainly and say what the evidence for it is.
