#!/usr/bin/env bash
set -uo pipefail

TS=/home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/TAPS-SP
SF=/home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/SpecForge
OUT="${1:?usage: gate_sweep.sh <exported_dir> <datasets> [gpu] [tag]}"
DATASETS="${2:-gsm8k}"
GPU="${3:-0}"
TAG="${4:-sweep}"

cd "$TS"
for RHO in 3 2 1 0.5; do
  for THETA in 0 0.35; do
    LOG="$SF/outputs/GATE_${TAG}_rho${RHO}_theta${THETA}.log"
    echo "=========== rho=$RHO theta=$THETA -> $LOG ==========="
    CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=. \
      python scripts/decode_lattice.py \
      --draft-model "$OUT/backbone" \
      --lattice-head "$OUT/selector.pt" \
      --modes latgate \
      --gate-tau 0 --gate-rho "$RHO" --gate-theta "$THETA" --gate-stats \
      --eval-reserved --max-samples 40 --max-new-tokens 256 \
      --datasets "$DATASETS" > "$LOG" 2>&1
    echo "exit=$?"
    grep -aE "^\[[a-z0-9-]+\] n=|recall AT|NET accepted|destroy \(at|base wrong|override rate" "$LOG" || tail -5 "$LOG"
  done
done
