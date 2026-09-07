#!/usr/bin/env bash
# Sweep the frontier-triggered gate against the shipped margin gate, paired on the same blocks.
#
# Every run decodes BOTH `latgate` and `latfront` in the same process over the same prompts, so the
# comparison is paired and the shipped gate is re-measured in each cell rather than quoted.
#
# Cell 0 is the identity check: `--front-theta 1.0` can never fire (a sigmoid is < 1), so latfront
# must equal latgate to the digit.  If it does not, the branch is wired wrong and every other cell
# in this sweep is meaningless.
set -u
SF=/home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/SpecForge
TS=/home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/TAPS-SP
DFLASH=$TS/models/Qwen3-4B-DFlash-b16
HEAD=$SF/outputs/v2_step4000_decode/selector.pt
DATASETS=${DATASETS:-gsm8k}
SAMPLES=${SAMPLES:-20}
OUT=$SF/outputs/frontgate
mkdir -p "$OUT"

run () {
  local tag=$1 gpu=$2 theta=$3 rho=$4
  cd "$TS"
  CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=. \
    nohup python3 -u scripts/decode_lattice.py \
    --target-model "$TS/models/Qwen3-4B" \
    --draft-model "$DFLASH" \
    --lattice-head "$HEAD" \
    --modes latgate,latfront \
    --gate-tau 0 --gate-rho 3 --gate-theta 0 --gate-stats \
    --front-theta "$theta" --front-rho "$rho" \
    --eval-reserved --max-samples "$SAMPLES" --max-new-tokens 256 \
    --datasets "$DATASETS" > "$OUT/$tag.log" 2>&1 &
  echo "launched $tag on gpu $gpu (theta=$theta rho=$rho)"
}

run t40_r1 0 0.40 1.0
run t45_r1 1 0.45 1.0
run t50_r05 2 0.50 0.5
run t50_r15 3 0.50 1.5
run t50_r20 4 0.50 2.0
run t55_r1 5 0.55 1.0
run t60_r1 6 0.60 1.0
run t60_r15 7 0.60 1.5
wait
