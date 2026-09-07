#!/usr/bin/env bash
# Wait for a cloze checkpoint to finish landing, then run its six-domain head-to-head.
# `training_state.pt` is written last and non-atomically, so poll until its size stops changing.
set -uo pipefail
SF=/home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/SpecForge
STEP=${1:?usage: autoeval_cloze.sh <step> [gpu] [samples]}
GPU=${2:-7}
SAMPLES=${3:-20}
RUN=${RUN:-qwen3-4b-pspr-cloze}
CKPT=$SF/outputs/${RUN}/${RUN}-step${STEP}/training_state.pt
while [ ! -f "$CKPT" ]; do sleep 60; done
prev=0
while true; do
  cur=$(stat -c %s "$CKPT" 2>/dev/null || echo 0)
  if [ "$cur" != "0" ] && [ "$cur" = "$prev" ]; then break; fi
  prev=$cur
  sleep 20
done
cd "$SF"
RUN=$RUN SAMPLES=$SAMPLES bash scripts/eval_cloze.sh "$STEP" "$GPU"
