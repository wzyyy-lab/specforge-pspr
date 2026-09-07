#!/bin/bash
# Arm 2: teacher-forced ladder on the TRAINING corpus, uniform anchors exactly as training samples
# them.  Validates the harness against PSPR_CLOZE.log telemetry and gives the training-distribution
# reference the loss is actually averaged over.
set -e
cd /home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/SpecForge
SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
PYTHONPATH=/home/wangzhuoyu/sglang-patched:. \
CUDA_VISIBLE_DEVICES=1 \
python3 -u scripts/measure_ranker_teacher_forced.py \
  --features cache/hidden_states/pb_probe \
  --checkpoint outputs/qwen3-4b-pspr-cloze/qwen3-4b-pspr-cloze-step7115/training_state.pt \
  --num-samples 64 --num-anchors 512 --chunk-blocks 32 \
  --gate-rho 3.0 --gate-training-corpus --gate-tol 0.03 \
  --json-output outputs/TF_LADDER_pb_uniform.json
