#!/usr/bin/env bash
# Six-domain evaluation for PSPR-Decision on the frozen official DFlash-b16 backbone.
#
# Differences from eval_cloze.sh, and why:
#   * mode is `latrepair`, not `latgate`.  This head's decode rule is the MAP action of its own
#     training likelihood (`keep_repair_log_probs` argmax), so there is no tau/rho/theta to pin.
#     Passing the old margin gate would run a policy the head was never trained for.
#   * `--repair-margin` is the ONLY decode knob, and it is a pure scalar log-probability penalty on
#     every override.  It exists because `frontier_boost` re-weights the detector's BCE and therefore
#     shifts its calibration; a scalar offset is the matched correction for a constant shift.
#     REPAIR_MARGIN=0 is the native exported policy.
#
# Baselines on this hardware, all n=20 (see TAPS-SP/PSPR_DECISION.md):
#   oneshot 5.3043 | Domino 5.8868 | cloze@7115 rho=3 6.0498 | oracle@16 10.4422
set -uo pipefail

SF=/home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/SpecForge
TS=/home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/TAPS-SP
DFLASH=$TS/models/Qwen3-4B-DFlash-b16
DATASETS=${DATASETS:-gsm8k,math500,humaneval,mbpp,alpaca,mt-bench}
SAMPLES=${SAMPLES:-20}
REPAIR_MARGIN=${REPAIR_MARGIN:-0}
# A nonzero margin is by definition NOT the exported native policy, and decode_lattice fails closed
# on that -- correctly, because it is the difference between reporting a result and tuning on the set
# you report.  Any run with a margin is therefore explicitly labelled a CALIBRATION run.
EXTRA=""
if [ "$REPAIR_MARGIN" != "0" ]; then EXTRA="--allow-policy-mismatch"; fi
MODES=${MODES:-oneshot,latrepair}
GPU=${GPU:-0}

usage () { echo "usage: eval_decision.sh <step> [gpu]   (env: DATASETS SAMPLES REPAIR_MARGIN MODES TAGSUF)"; exit 2; }
WHAT=${1:?$(usage)}
GPU=${2:-$GPU}

RUN=${RUN:-qwen3-4b-pspr-decision}
OUT=$SF/outputs/${RUN}_step${WHAT}_decode
if [ ! -d "$OUT" ]; then
  CKPT=$SF/outputs/${RUN}/${RUN}-step${WHAT}/training_state.pt
  [ -f "$CKPT" ] || { echo "MISSING $CKPT"; exit 2; }
  cd "$SF"
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python3 scripts/export_pspr_decision_for_decode.py \
    --checkpoint "$CKPT" \
    --draft-config configs/qwen3-4b-pspr-decision.json \
    --backbone-template "$DFLASH" \
    --verify-frozen "$DFLASH" \
    --out "$OUT" || exit 3
fi
HEAD=$OUT/selector.pt
TAG=decision_step${WHAT}${TAGSUF:-}

LOG=$SF/outputs/EVAL_${TAG}.log
echo "== $TAG | margin=$REPAIR_MARGIN | modes=$MODES | gpu=$GPU -> $LOG"
cd "$TS"
CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=. \
  python3 -u scripts/decode_lattice.py \
  --target-model "$TS/models/Qwen3-4B" \
  --draft-model "$DFLASH" \
  --lattice-head "$HEAD" \
  --modes "$MODES" \
  --repair-margin "$REPAIR_MARGIN" --gate-stats $EXTRA \
  --json-output "$SF/outputs/EVAL_${TAG}.json" \
  --eval-reserved --max-samples "$SAMPLES" --max-new-tokens 256 \
  --datasets "$DATASETS" 2>&1 | tee "$LOG" | tail -32
