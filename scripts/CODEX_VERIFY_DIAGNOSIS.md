# Task: independently verify 5 code-level claims. Answer TRUE / FALSE / PARTIAL for each, with file:line evidence.

Do NOT trust my reasoning. Read the actual code. If a claim is wrong, say exactly how.

Repo roots:
- SpecForge: /home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/SpecForge
- TAPS-SP:   /home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/TAPS-SP

The system under test: a "candidate selector" head (`ClozeCorrector` in
`SpecForge/specforge/modeling/draft/pspr_cloze.py`) bolted onto a FROZEN DFlash-b16 draft backbone
(`SpecForge/specforge/modeling/draft/dflash.py`). Block size 16: one anchor + 15 proposal slots.
The backbone emits per-slot hidden states h_i; a top-16 candidate lattice is read off
`lm_head(h_i)`. The head may override the backbone's top-1 at each slot.
Training config: `SpecForge/examples/configs/online/disaggregated/managed-local/qwen3-4b-pspr-cloze.yaml`
Model config: `SpecForge/configs/qwen3-4b-pspr-cloze.json`
Decoder: `TAPS-SP/scripts/decode_lattice.py`, deployed mode `latgate` with gate_tau=0, gate_rho=3, gate_theta=0.

## CLAIM 1 — the "cloze" head cannot be a real masked-LM, by construction
Sub-claims:
 1a. The DFlash backbone attention is NON-causal within a block (full attention over all 16
     positions), so h_i already aggregates the whole block.
 1b. The draft's top-1 token at slot j is a deterministic function of h_j alone, via the frozen
     tied `lm_head`.
 1c. Therefore the entire cloze input sequence (seed tokens d0_j, their log-probs, margins) is a
     deterministic function of {h_1..h_H} — it carries zero information beyond {h_j}.
 1d. In `ClozeCorrector`, the QUERY row for slot i still receives h_i directly (config has
     `direct_hidden: true`), so masking the *token id* at position i does not hide the answer:
     `argmax(lm_head(h_i))` recovers exactly the masked token. Hence it is NOT a genuine cloze.
Report which of 1a-1d hold.

## CLAIM 2 — the deployed bottleneck is the KEEP-vs-REPAIR decision, not "which alternative"
Read `TAPS-SP/scripts/decode_lattice.py` around lines 615-710 (the `stats` block) and 1060-1105
(the printing). Verify that:
 2a. `fixable_n` counts reachable slots (accepted prefix + frontier slot) where the backbone top-1
     is wrong AND the true token IS inside the top-16 lattice.
 2b. `alternative_rank_0 / fixable_n` = fraction where the selector ranks the true token FIRST
     among candidates 1..15 (i.e. excluding candidate 0 = KEEP).
 2c. `selector_rank_0 / fixable_n` = fraction where the true token ranks first among ALL 16
     candidates (KEEP competing in the same softmax).
 2d. `fix_recovered / fixable_n` = fraction actually repaired after the margin gate
     `best_alt_prob > tau + rho * P(keep)` is applied.
 2e. Confirm from the logs that these three are 66.86%, 45.07%, 36.71%:
       SpecForge/outputs/DETAIL_cloze7115_rho1_altstats.json (rho=1, has alternative_rank)
       SpecForge/outputs/DETAIL_cloze7115_rho1.log  (recovered 45.07%)
       SpecForge/outputs/DETAIL_cloze7115_rho3.log  (recovered 36.71%)
     The altstats run has no .log; recompute what you can and state what you could not verify.
 2f. State whether the following mechanism is a correct reading of the code: candidate 0 (KEEP)
     competes inside the SAME softmax as the 15 alternatives, and its score is anchored on
     `lp[0]`, the largest top-k log-prob. Look at `lattice_score_slot` and
     `ClozeCorrector.score_candidates`. Is KEEP structurally advantaged?

## CLAIM 3 — the trained err_head is dead code at serving time
 3a. In `decode_lattice.py` mode `latgate`, with `gate_theta=0`, `slot_err_logit` is NEVER called
     (short-circuit `if ok and gate_theta > 0`).
 3b. Even with gate_theta>0, err_head can only VETO an override the margin rule already admitted;
     it can never admit an override the margin rule rejected. So any theta sweep can only lower
     the override rate.
 3c. `selector_err_use_state` is absent from `configs/qwen3-4b-pspr-cloze.json`, so it defaults to
     False and the err_head does NOT consume the committed-prefix GRU state during training.
 3d. `selector_err_weight_mode` is absent from the training yaml, so it defaults to
     `"base_frontier"`. Explain precisely which slots that trains on. Specifically: does it train
     ONLY on the backbone's own uncorrected contiguous-correct prefix plus its first error, i.e.
     roughly one positive (err=1) sample per block and ~5 negatives? And does that mean the
     detector is never trained on the positions PSPR actually reaches AFTER repairing an error?

## CLAIM 4 — label handling for top-16 misses
 4a. For `selector_objective: multiclass` (what this run uses), a slot whose target is outside the
     top-16 is DROPPED from the selector CE (weight 0) — it is NOT relabelled as KEEP/class 0.
     Check `selector_supervised = target_is_candidate.float()`.
 4b. There EXISTS another objective, `profitable_action`, which instead maps a miss to class 0
     (KEEP). It is not the one used here.
 4c. The err_head BCE label is `err_target = (~base_top1_ok)`, which is TRUE on a top-16 miss.
     So the detector IS trained to fire on slots where no repair is possible. Under
     `base_frontier` weighting, estimate what fraction of the err=1 positives are such
     unrepairable misses. Frontier top-K recall measured at decode is 0.8437
     (SpecForge/outputs/DETAIL_cloze7115_rho3.log), i.e. ~15.6% of frontier slots are misses.
     Is ~15.6% the right estimate for the contamination of the positive class, or is the correct
     number different because of how `frontier_mask` selects slots? Show the reasoning.

## CLAIM 5 — train/serve mismatch of the GRU state
 5a. During training, the causal GRU consumes `predecessor_ids` built from the CORPUS ground-truth
     tokens (teacher forcing). Find where `predecessor_ids` is constructed
     (`dflash_family_model.py`, around line 1440) and confirm.
 5b. During decoding, the GRU consumes the tokens the selector actually COMMITTED (which may be
     overrides, and may be wrong). See the `latgate` loop in `decode_lattice.py` (~line 481-505).
 5c. So the head is trained on a prefix distribution it never sees at serving time. Confirm or
     refute.

## Output format
For each of 1a..5c: TRUE / FALSE / PARTIAL + file:line + one-sentence justification.
Then a section "WHAT I THINK IS THE REAL BOTTLENECK" of at most 200 words, based only on what you read.
Be adversarial. I would rather be told I am wrong now than after a 9-hour training run.
