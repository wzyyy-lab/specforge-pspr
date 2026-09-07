#!/bin/bash
# Arm 1: teacher-forced ladder on the SAME text decode measures (target-greedy continuations of the
# same 120 reserved prompts).  Reports both populations from one forward:
#   POLICY = uniform anchors, i.e. the training-time anchor distribution
#   CHAIN  = exact simulation of decode's block chaining on identical lattices
# POLICY vs CHAIN isolates the anchor-occupancy effect with text, weights and labels held fixed.
# CHAIN vs the decode log validates the whole harness end to end.
set -e
cd /home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/SpecForge
SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
PYTHONPATH=/home/wangzhuoyu/sglang-patched:. \
CUDA_VISIBLE_DEVICES=1 \
python3 -u scripts/measure_ranker_teacher_forced.py \
  --features cache/hidden_states/eval_traces \
  --checkpoint outputs/qwen3-4b-pspr-cloze/qwen3-4b-pspr-cloze-step7115/training_state.pt \
  --num-samples 120 --num-anchors 512 --chunk-blocks 32 \
  --gate-rho 3.0 --chain \
  --json-output outputs/TF_LADDER_eval_chain.json
