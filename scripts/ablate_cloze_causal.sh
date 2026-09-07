#!/usr/bin/env bash
# Cloze bidirectionality ablation: the same trained head, scored with and without the future half of
# each cloze row's attention.
#
# This is an INFERENCE-time ablation, so it answers "is the head reading the future at serving time,
# and does that reading help" -- not "would a causally-trained head have been worse".  The latter
# needs a separate 7115-step run.  What makes this one meaningful is that parameters, FLOPs and the
# lattice are bit-identical between the arms: only the attention mask differs.
set -uo pipefail
SF=/home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/SpecForge
TS=/home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/TAPS-SP
DFLASH=$TS/models/Qwen3-4B-DFlash-b16
STEP=${1:-7115}
GPU=${2:-0}
SAMPLES=${SAMPLES:-20}
DATASETS=${DATASETS:-gsm8k,math500,humaneval,mbpp,alpaca,mt-bench}
HEAD=$SF/outputs/qwen3-4b-pspr-cloze_step${STEP}_decode/selector.pt
[ -f "$HEAD" ] || { echo "MISSING $HEAD -- run eval_cloze.sh $STEP first"; exit 2; }

LOG=$SF/outputs/ABLATE_cloze_causal_step${STEP}.log
echo "== cloze@${STEP} CAUSAL ablation | gpu=$GPU -> $LOG"
cd "$TS"
CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=. \
  python3 -u scripts/decode_lattice.py \
  --target-model "$TS/models/Qwen3-4B" \
  --draft-model "$DFLASH" \
  --lattice-head "$HEAD" \
  --cloze-causal \
  --modes oneshot,latgate \
  --gate-tau 0 --gate-rho 3 --gate-theta 0 --gate-stats \
  --eval-reserved --max-samples "$SAMPLES" --max-new-tokens 256 \
  --datasets "$DATASETS" 2>&1 | tee "$LOG" | tail -20
