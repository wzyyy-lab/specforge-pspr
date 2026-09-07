#!/usr/bin/env bash
# Six domains across six GPUs for the CLOZE head at its native policy (latgate tau=0/rho=3/theta=0).
# Exists so a Decision-vs-cloze comparison can be made at the SAME step count on the SAME hardware:
# the archived cloze numbers were measured on the 96GB cards and are not comparable (pure-backbone
# oneshot moved 6.007 -> 5.827 with no head in the loop).
set -u
SF=/home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/SpecForge
TS=/home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/TAPS-SP
STEP=${1:?usage: par_eval_cloze.sh <step>}
SAMPLES=${SAMPLES:-20}
HEAD=$SF/outputs/qwen3-4b-pspr-cloze_step${STEP}_decode/selector.pt
[ -f "$HEAD" ] || { echo "MISSING $HEAD"; exit 2; }

i=1
for D in gsm8k math500 humaneval mbpp alpaca mt-bench; do
  LOG=$SF/outputs/NEWHW_cloze${STEP}_$D.log
  ( cd "$TS" && CUDA_VISIBLE_DEVICES=$i HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=. \
      python3 -u scripts/decode_lattice.py \
      --target-model "$TS/models/Qwen3-4B" --draft-model "$TS/models/Qwen3-4B-DFlash-b16" \
      --lattice-head "$HEAD" --modes latgate \
      --gate-tau 0 --gate-rho 3 --gate-theta 0 --gate-stats \
      --eval-reserved --max-samples "$SAMPLES" --max-new-tokens 256 \
      --datasets "$D" > "$LOG" 2>&1 ) &
  i=$((i + 1))
done
wait
echo "== cloze step$STEP done"
for D in gsm8k math500 humaneval mbpp alpaca mt-bench; do
  tr '\r' '\n' < "$SF/outputs/NEWHW_cloze${STEP}_$D.log" | grep -E "^\[$D\]" | tail -1
done
