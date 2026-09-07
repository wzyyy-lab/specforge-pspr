#!/bin/bash
# Feature dump, benchmark corpus (target-greedy continuations of the 120 reserved prompts).
# Same populations the decode log measures, so a head fitted on the pb dump can be scored here
# as a true train/test split along the axis that actually matters: corpus generalisation.
set -e
cd /home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/SpecForge
SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
PYTHONPATH=/home/wangzhuoyu/sglang-patched:. \
CUDA_VISIBLE_DEVICES=2 \
python3 -u scripts/measure_ranker_teacher_forced.py \
  --features cache/hidden_states/eval_traces \
  --checkpoint outputs/qwen3-4b-pspr-cloze/qwen3-4b-pspr-cloze-step7115/training_state.pt \
  --num-samples 120 --num-anchors 512 --chunk-blocks 32 \
  --gate-rho 3.0 --chain \
  --dump-features outputs/TF_FEAT_eval.npz \
  --json-output outputs/TF_LADDER_eval_chain.json
