#!/usr/bin/env bash
# Discriminating experiment: is the rho>1 advantage generic overconfidence, or candidate-0 specific?
#
# NOT a sampling temperature.  Decoding is fully greedy throughout (argmax target, argmax draft, topk
# candidate set), exactly as DFlash and Domino are evaluated.  T divides only the SELECTOR's own K
# candidate scores before the gate's softmax, and T=1.0 is a bit-exact no-op.
#
# Background.  MAP over the K conditional scores is provably optimal for E[accept] under a calibrated
# model: the block's accepted run stops at the first wrong slot regardless of WHICH way the decision
# was wrong (base right and overridden, or base wrong and kept -- both truncate at slot i), and the
# continuation value multiplies every candidate at a slot equally.  So rho=3 beating rho=1 by
# -0.1110 +- 0.0436 (paired, p=0.011) on the old card is a calibration failure, not an asymmetric-cost
# effect.  Two mechanisms I proposed for it were both refuted by measurement (frontier_boost as a
# one-way bias; miss-rate correlation across domains), so this probe attacks the structure instead.
#
# Two structurally different calibration failures predict different outcomes:
#   * GENERIC OVERCONFIDENCE -- the whole softmax is too peaked.  A single temperature T>1 flattens it
#     uniformly and should reproduce rho=3's behaviour.  Fix is cheap and objective-agnostic
#     (temperature / label smoothing / a calibration term).
#   * CANDIDATE-0 SPECIFIC bias -- only class 0's mass is wrong, which is what the dropped-miss and
#     KEEP-inflation arguments predict.  Then no single T can reproduce rho=3, because T rescales
#     candidate 0 and the alternatives together while rho moves them apart.  Fix must be structural,
#     which is exactly what `covered_conditional` is.
#
# EVERY gate parameter is passed explicitly.  The first version of this script omitted --gate-tau and
# silently inherited decode_lattice's default of 0.45, making the rule
# `pr[1:].max() > 0.45 + rho*pr[0]` -- a different judgement from the tau=0 runs that every historical
# number was produced with, so that comparison (which appeared to show rho=1 winning by +0.118) was
# meaningless.  Never rely on a default for a decision-rule parameter.
#
# Arms, all on the SAME trained weights (cloze@7115) and the same prompts, with --json-output so the
# comparison is paired:
#   rho=1, T in {1.0, 1.5, 2.0, 3.0, 5.0}   and   rho=3, T=1.0 (the reference operating point)
set -u
SF=/home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/SpecForge
TS=/home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/TAPS-SP
DFLASH=$TS/models/Qwen3-4B-DFlash-b16
HEAD=$SF/outputs/qwen3-4b-pspr-cloze_step7115_decode/selector.pt
SAMPLES=${SAMPLES:-20}
DATASETS=${DATASETS:-gsm8k,math500,humaneval,mbpp,alpaca,mt-bench}

run () {
  local rho=$1 temp=$2 gpu=$3
  local tag="TEMP_rho${rho}_T${temp}"
  CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=".:$SF" \
    python3 -u "$TS/scripts/decode_lattice.py" \
    --target-model "$TS/models/Qwen3-4B" \
    --draft-model "$DFLASH" \
    --lattice-head "$HEAD" \
    --modes latgate --gate-tau 0 --gate-rho "$rho" --gate-theta 0 \
    --score-temperature "$temp" --gate-stats \
    --allow-policy-mismatch \
    --json-output "$SF/outputs/${tag}.json" \
    --eval-reserved --max-samples "$SAMPLES" --max-new-tokens 256 \
    --datasets "$DATASETS" > "$SF/outputs/${tag}.log" 2>&1
  echo "done $tag"
}

cd "$TS"
run 3 1.0 0 &
run 1 1.0 1 &
run 1 1.5 2 &
run 1 2.0 3 &
run 1 3.0 4 &
run 1 5.0 5 &
wait
echo "== all arms complete"
for T in 1.0 1.5 2.0 3.0 5.0; do
  printf "rho=1 T=%-4s vs rho=3 T=1.0: " "$T"
  python3 "$SF/scripts/paired_compare.py" \
    --a "$SF/outputs/TEMP_rho3_T1.0.json" --b "$SF/outputs/TEMP_rho1_T$T.json" \
    --mode latgate --label-a rho3 --label-b "rho1_T$T" 2>/dev/null | grep "^macro difference"
done
