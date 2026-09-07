#!/bin/bash
set -e
cd /home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/SpecForge
SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
PYTHONPATH=/home/wangzhuoyu/sglang-patched:. \
CUDA_VISIBLE_DEVICES=2 \
python3 -u scripts/probe_score_head.py \
  --train-features outputs/TF_FEAT_pb.npz \
  --eval-features outputs/TF_FEAT_eval.npz \
  --fit-on eval --epochs 6 \
  --json-output outputs/PROBE_HEAD_incorpus.json
