#!/bin/bash
set -e
cd /home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/SpecForge
SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
PYTHONPATH=/home/wangzhuoyu/sglang-patched:. \
CUDA_VISIBLE_DEVICES=1 \
python3 -u scripts/probe_detector.py \
  --train-features outputs/TF_FEAT_pbbig.npz \
  --eval-features outputs/TF_FEAT_eval.npz \
  --chain-ladder outputs/TF_LADDER_eval_chain.json \
  --epochs 5 \
  --json-output outputs/PROBE_DETECTOR.json
