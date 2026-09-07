#!/bin/bash
# The corrected ranker probe: the residual head now sees the SAME inputs the deployed head reads,
# h included.  Without h the earlier negative result was unfalsifiable.
set -e
ROOT=/home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/SpecForge
cd "$ROOT"
COMMON="--train-features outputs/TF_FEATH_pb.npz --eval-features outputs/TF_FEATH_eval.npz \
--chain-ladder outputs/TF_LADDERH_eval.json --fit-on train --epochs 4 --batch-size 1024"
SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 PYTHONPATH=/home/wangzhuoyu/sglang-patched:. \
CUDA_VISIBLE_DEVICES=1 nohup python3 -u scripts/probe_score_head.py $COMMON \
  --ablate full,candidate_only --slot-weight uniform \
  --json-output outputs/PROBE_H_uniform.json > outputs/PROBE_H_uniform.log 2>&1 &
SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 PYTHONPATH=/home/wangzhuoyu/sglang-patched:. \
CUDA_VISIBLE_DEVICES=2 nohup python3 -u scripts/probe_score_head.py $COMMON \
  --ablate full --slot-weight decode --fixable-weight 4.0 \
  --json-output outputs/PROBE_H_weighted.json > outputs/PROBE_H_weighted.log 2>&1 &
wait
echo done
