#!/usr/bin/env bash
set -euo pipefail

SF=/home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/SpecForge
TS=/home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/TAPS-SP
RUN="$SF/outputs/qwen3-4b-pspr-joint-online"

STEP="${1:?usage: pilot_eval.sh <step> [datasets] [eval_gpu] [tag]}"
DATASETS="${2:-gsm8k}"
GPU="${3:-0}"
TAG="${4:-S2}"

CKPT="$RUN/qwen3-4b-pspr-joint-online-step${STEP}/training_state.pt"
OUT="$SF/outputs/${TAG}_step${STEP}_decode"

if [ ! -f "$CKPT" ]; then
  echo "MISSING: $CKPT"
  exit 2
fi

if [ -d "$OUT" ]; then
  echo "REFUSING to overwrite existing export: $OUT"
  exit 3
fi

cd "$SF"
PYTHONPATH=. python scripts/export_pspr_for_decode.py \
  --checkpoint "$CKPT" \
  --draft-config configs/qwen3-4b-pspr-joint.json \
  --backbone-template /kl_infra_infer_intern/wangzhuoyu/TAPS-SP/models/Qwen3-4B-DFlash-b16 \
  --output-dir "$OUT"

cd "$TS"
CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=. python scripts/decode_lattice.py \
  --draft-model "$OUT/backbone" \
  --lattice-head "$OUT/selector.pt" \
  --modes oneshot,latgate \
  --gate-tau 0 --gate-rho 3 --gate-theta 0 --gate-stats \
  --eval-reserved --max-samples 40 --max-new-tokens 256 \
  --datasets "$DATASETS"
