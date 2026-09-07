#!/usr/bin/env bash
# Wait for a checkpoint to finish landing, then run the six-domain head-to-head for it.
#
# `training_state.pt` is written last and non-atomically, so a directory appearing is not proof the
# file is complete.  Poll until its size stops changing before exporting, otherwise the exporter
# reads a truncated archive and the run dies half an hour later for no reason.
set -uo pipefail
SF=/home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/SpecForge
STEP=${1:?usage: autoeval_v2.sh <step> [gpu] [samples]}
GPU=${2:-7}
SAMPLES=${3:-20}
RUN=${RUN:-qwen3-4b-pspr-v2-stage1}
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
RUN=$RUN SAMPLES=$SAMPLES bash scripts/eval_v2.sh "$STEP" "$GPU"
