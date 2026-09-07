#!/usr/bin/env bash
# Decode the ALREADY-TRAINED cloze head with the factorised keep/repair rule instead of the margin
# gate.  No training, no new weights -- only the decode rule changes.
#
# Why this is the highest-value cheap experiment available (see PSPR_DECISION.md step 17):
# the reachable-decision decomposition on the cloze rho=3 trajectory loses 22.8 pp between
# "truth ranked first among alternatives" (66.86%) and "truth ranked first overall" (44.05%).
# That entire loss is candidate 0 outscoring the truth.  `latgate` reads pr[0] in its threshold;
# `latrepair` never reads scores[0] at all.  If the loss is candidate-0 inflation, this rule
# recovers it on a ranker that was trained on every covered slot.
#
# The detector's operating point must be swept: it was trained where the base is wrong ~40-50% of
# the time and is queried where the base is wrong 3.5% of the time, which is a constant logit
# offset of roughly log(.42/.58) - log(.035/.965) ~= 3.0 nats.  So margin ~3 is the a-priori
# estimate, not 0.
#
# Every run is --allow-policy-mismatch: multiclass weights decoded with the factorised rule are by
# definition not the exported native policy.  CALIBRATION only.
set -u
SF=/home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/SpecForge
TS=/home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/TAPS-SP
DFLASH=$TS/models/Qwen3-4B-DFlash-b16
STEP=${STEP:-7115}
HEAD=$SF/outputs/qwen3-4b-pspr-cloze_step${STEP}_decode/selector.pt
DATASETS=${DATASETS:-gsm8k}
SAMPLES=${SAMPLES:-20}
MARGIN=${MARGIN:-0}
GPU=${GPU:-0}
TAG=clozerepair_step${STEP}_m$(echo "$MARGIN" | tr -d '.')${TAGSUF:-}
LOG=$SF/outputs/EVAL_${TAG}.log

[ -f "$HEAD" ] || { echo "MISSING $HEAD"; exit 2; }
echo "== $TAG | margin=$MARGIN | gpu=$GPU -> $LOG"
cd "$TS"
CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=.:$SF \
  python3 -u scripts/decode_lattice.py \
  --target-model "$TS/models/Qwen3-4B" \
  --draft-model "$DFLASH" \
  --lattice-head "$HEAD" \
  --modes latrepair \
  --repair-margin "$MARGIN" --gate-stats --allow-policy-mismatch \
  --json-output "$SF/outputs/EVAL_${TAG}.json" \
  --eval-reserved --max-samples "$SAMPLES" --max-new-tokens 256 \
  --datasets "$DATASETS" 2>&1 | tee "$LOG" | tail -30
