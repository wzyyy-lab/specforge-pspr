#!/usr/bin/env bash
# Six-domain head-to-head: PSPR-v2 (PerfectBlend online) vs the stage-1 selector, same backbone.
#
# Both arms decode with the SAME backbone bytes -- the official DFlash-b16 release -- because this
# run froze the backbone and `export_pspr_v2_for_decode.py --verify-frozen` proved all 58 tensors are
# bit-identical to it.  So the only thing that differs between the arms is the head, and the only
# thing that differs between v2 and stage-1 is how much data trained it.
#
# The gate is pinned to stage-1's own shipped policy (tau=0, rho=3, skip0=False, theta=0) for both
# arms.  That gives v2 zero tuning degrees of freedom: it is not allowed to pick a friendlier gate on
# the very 40 prompts it is scored on.  A calibration-selected gate is reported separately.
set -uo pipefail

SF=/home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/SpecForge
TS=/home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/TAPS-SP
DFLASH=$TS/models/Qwen3-4B-DFlash-b16
DATASETS=${DATASETS:-gsm8k,math500,humaneval,mbpp,alpaca,mt-bench}
SAMPLES=${SAMPLES:-40}
GPU=${GPU:-0}

usage () { echo "usage: eval_v2.sh baseline|<step> [gpu]"; exit 2; }
WHAT=${1:?$(usage)}
GPU=${2:-$GPU}

if [ "$WHAT" = "baseline" ]; then
  HEAD=$SF/outputs/pspr_dh2048_exact/best.pt
  TAG=BASELINE_stage1
  EXTRA="--allow-policy-mismatch"      # legacy artifact, no verified training_policy block
else
  RUN=${RUN:-qwen3-4b-pspr-v2-stage1}
  OUT=$SF/outputs/${RUN}_step${WHAT}_decode
  if [ ! -d "$OUT" ]; then
    CKPT=$SF/outputs/${RUN}/${RUN}-step${WHAT}/training_state.pt
    [ -f "$CKPT" ] || { echo "MISSING $CKPT"; exit 2; }
    cd "$SF"
    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python3 scripts/export_pspr_v2_for_decode.py \
      --checkpoint "$CKPT" \
      --draft-config configs/qwen3-4b-pspr-v2.json \
      --backbone-template "$DFLASH" \
      --verify-frozen "$DFLASH" \
      --out "$OUT" || exit 3
  fi
  HEAD=$OUT/selector.pt
  TAG=${RUN#qwen3-4b-pspr-v2-}_step${WHAT}
  EXTRA=""
fi

LOG=$SF/outputs/EVAL_${TAG}.log
echo "== $TAG | head=$HEAD | backbone=$DFLASH | gpu=$GPU -> $LOG"
cd "$TS"
CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=. \
  python3 -u scripts/decode_lattice.py \
  --target-model "$TS/models/Qwen3-4B" \
  --draft-model "$DFLASH" \
  --lattice-head "$HEAD" \
  --modes oneshot,latgate \
  --gate-tau 0 --gate-rho 3 --gate-theta 0 --gate-stats \
  --eval-reserved --max-samples "$SAMPLES" --max-new-tokens 256 \
  --datasets "$DATASETS" $EXTRA 2>&1 | tee "$LOG" | tail -30
