#!/usr/bin/env bash
# Stage-1 reproduction gate: export the trained selector and measure it through the SAME reference
# decoder and protocol that produced 5.7048 for dh2048 in this same session.
#
# Usage: bash scripts/run_pspr_reproduction_gate.sh <run_dir> <gpu> [gate_rho]
set -euo pipefail

RUN_DIR="${1:?run dir, e.g. outputs/qwen3-4b-pspr-frozen}"
GPU="${2:?gpu id}"
RHO="${3:-9}"
SF=/kl_infra_infer_intern/wangzhuoyu/SpecForge
TS=/kl_infra_infer_intern/wangzhuoyu/TAPS-SP
EXPORT_DIR="$SF/outputs/decode_$(basename "$RUN_DIR")"

CKPT=$(ls -t "$SF/$RUN_DIR"/*/training_state.pt "$SF/$RUN_DIR"/training_state.pt 2>/dev/null | head -1)
echo "checkpoint: $CKPT"

cd "$SF"
OMP_NUM_THREADS=6 PYTHONPATH=. python scripts/export_pspr_for_decode.py \
    --checkpoint "$CKPT" --output-dir "$EXPORT_DIR"

cd "$TS"
HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 \
CUDA_VISIBLE_DEVICES="$GPU" python scripts/decode_lattice.py \
    --draft-model "$EXPORT_DIR/backbone" --lattice-head "$EXPORT_DIR/selector.pt" \
    --modes oneshot,latgate --gate-tau 0 --gate-rho "$RHO" --gate-skip0 \
    --eval-reserved --max-samples 40 --max-new-tokens 256
