#!/bin/bash
# The decisive weighting experiment, on 960 training sequences (5.2M slots) so the 64-sequence
# overfitting confound is gone.
#   A  uniform loss           -- the objective the deployed selector was trained with
#   B  decode slot weights    -- matches the training slot mix to decode's chained decision mix
#   C  B + base-wrong upweight-- also fixes the 77% of gradient spent where the selector is 99.65% right
# All three are model-selected on decode-weighted alt_rank_0, and all three report the SAME held-out
# benchmark slots, so the comparison isolates the objective's importance weighting.
set -e
ROOT=/home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/SpecForge
cd "$ROOT"
COMMON="--train-features outputs/TF_FEAT_pbbig.npz --eval-features outputs/TF_FEAT_eval.npz \
--chain-ladder outputs/TF_LADDER_eval_chain.json --fit-on train --epochs 4 --ablate full"
SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 PYTHONPATH=/home/wangzhuoyu/sglang-patched:. \
CUDA_VISIBLE_DEVICES=1 nohup python3 -u scripts/probe_score_head.py $COMMON \
  --slot-weight uniform --json-output outputs/PROBE_W_A_uniform.json \
  > outputs/PROBE_W_A_uniform.log 2>&1 &
SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 PYTHONPATH=/home/wangzhuoyu/sglang-patched:. \
CUDA_VISIBLE_DEVICES=2 nohup python3 -u scripts/probe_score_head.py $COMMON \
  --slot-weight decode --json-output outputs/PROBE_W_B_slot.json \
  > outputs/PROBE_W_B_slot.log 2>&1 &
SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 PYTHONPATH=/home/wangzhuoyu/sglang-patched:. \
CUDA_VISIBLE_DEVICES=3 nohup python3 -u scripts/probe_score_head.py $COMMON \
  --slot-weight decode --fixable-weight 4.0 --json-output outputs/PROBE_W_C_slotfix.json \
  > outputs/PROBE_W_C_slotfix.log 2>&1 &
wait
echo all done
