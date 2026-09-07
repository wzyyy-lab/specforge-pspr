#!/bin/bash
# Arm 2 feature dump: the training corpus itself, so the G3 fidelity gate can bite.
set -e
cd /home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/SpecForge
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python3 -u scripts/dump_hidden_states_hf.py \
  --target-model /kl_infra_infer_intern/wangzhuoyu/TAPS-SP/models/Qwen3-4B \
  --draft-config configs/qwen3-4b-pspr-cloze.json \
  --data-path cache/dataset/perfectblend_qwen3-4b_regen.jsonl \
  --output cache/hidden_states/pb_probe \
  --chat-template qwen-nosys \
  --max-length 3072 \
  --num-samples 64 \
  --log-every 10
