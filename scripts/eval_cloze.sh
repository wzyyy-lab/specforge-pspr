#!/usr/bin/env bash
# Six-domain head-to-head: PSPR-cloze (PerfectBlend online) on the frozen DFlash-b16 backbone.
#
# Byte-for-byte the v2 harness apart from the exporter and the draft config, so a cloze number and a
# v2 number come from the same decoder, the same prompts, the same gate and the same backbone bytes.
# The backbone is the OFFICIAL DFlash-b16 release and `--verify-frozen` proves all 58 tensors are
# bit-identical to it, so the only thing that differs between the arms is the head.
#
# The gate is pinned to stage-1's own shipped policy (tau=0, rho=3, skip0=False, theta=0).  cloze
# therefore gets zero tuning degrees of freedom: it is not allowed to pick a friendlier gate on the
# very prompts it is scored on.
set -uo pipefail

SF=/home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/SpecForge
TS=/home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/TAPS-SP
DFLASH=$TS/models/Qwen3-4B-DFlash-b16
DATASETS=${DATASETS:-gsm8k,math500,humaneval,mbpp,alpaca,mt-bench}
SAMPLES=${SAMPLES:-40}
GPU=${GPU:-0}

usage () { echo "usage: eval_cloze.sh <step> [gpu]"; exit 2; }
WHAT=${1:?$(usage)}
GPU=${2:-$GPU}

RUN=${RUN:-qwen3-4b-pspr-cloze}
OUT=$SF/outputs/${RUN}_step${WHAT}_decode
if [ ! -d "$OUT" ]; then
  CKPT=$SF/outputs/${RUN}/${RUN}-step${WHAT}/training_state.pt
  [ -f "$CKPT" ] || { echo "MISSING $CKPT"; exit 2; }
  cd "$SF"
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python3 scripts/export_pspr_cloze_for_decode.py \
    --checkpoint "$CKPT" \
    --draft-config configs/qwen3-4b-pspr-cloze.json \
    --backbone-template "$DFLASH" \
    --verify-frozen "$DFLASH" \
    --out "$OUT" || exit 3
fi
HEAD=$OUT/selector.pt
TAG=cloze_step${WHAT}

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
  --datasets "$DATASETS" 2>&1 | tee "$LOG" | tail -30
