#!/bin/bash
# Arm 1 feature dump: target-greedy continuations of the same 120 reserved eval prompts.
set -e
cd /home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/SpecForge
HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
python3 -u scripts/dump_eval_traces.py \
  --output cache/hidden_states/eval_traces \
  --datasets gsm8k,math500,humaneval,mbpp,alpaca,mt-bench \
  --max-samples 20 --shuffle-seed 2026 --max-new-tokens 256
