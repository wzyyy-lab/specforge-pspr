#!/usr/bin/env bash
# Six-domain validation of the frontier-triggered gate, on two different heads.
#
# `--front-theta 0.5 --front-rho 1.0` was selected on gsm8k alone, so the other five domains are
# held out for this run.  Both heads are decoded with `--modes latgate,latfront` so the shipped
# margin gate is re-measured on the same blocks rather than quoted from an earlier log.
#
# Running it on the stage-1 head as well as the v2 head is the transfer check: a gate change that
# only helps the checkpoint it was tuned on is a tuning artefact, not a decode rule.
set -u
TS=/home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/TAPS-SP
SF=/home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/SpecForge
D=$TS/models/Qwen3-4B-DFlash-b16
DS=gsm8k,math500,humaneval,mbpp,alpaca,mt-bench

cd "$TS"

run () {
  local tag=$1 gpu=$2 head=$3
  shift 3
  CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=. \
    nohup python3 -u scripts/decode_lattice.py \
    --target-model "$TS/models/Qwen3-4B" \
    --draft-model "$D" \
    --lattice-head "$head" \
    --modes latgate,latfront \
    --gate-tau 0 --gate-rho 3 --gate-theta 0 --gate-stats \
    --front-theta 0.5 --front-rho 1.0 \
    --eval-reserved --max-samples 20 --max-new-tokens 256 \
    --datasets "$DS" "$@" > "$SF/outputs/FRONT6_$tag.log" 2>&1 &
  echo "launched $tag on gpu $gpu"
}

run v2step4000 0 "$SF/outputs/v2_step4000_decode/selector.pt"
run stage1     1 "$SF/outputs/pspr_dh2048_exact/best.pt" --allow-policy-mismatch
wait
