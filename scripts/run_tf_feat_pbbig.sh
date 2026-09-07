#!/bin/bash
# Scaled feature dump on the training corpus (1024 rows total once merged with pb_probe).
# The 64-row probe overfit inside one epoch, so its negative capacity verdict was uninterpretable.
set -e
cd /home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/SpecForge
SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
PYTHONPATH=/home/wangzhuoyu/sglang-patched:. \
CUDA_VISIBLE_DEVICES=1 \
python3 -u scripts/measure_ranker_teacher_forced.py \
  --features cache/hidden_states/pb_big \
  --checkpoint outputs/qwen3-4b-pspr-cloze/qwen3-4b-pspr-cloze-step7115/training_state.pt \
  --num-samples 960 --num-anchors 512 --chunk-blocks 32 \
  --gate-rho 3.0 --gate-training-corpus --gate-tol 0.03 \
  --dump-features outputs/TF_FEAT_pbbig.npz \
  --json-output outputs/TF_LADDER_pbbig_uniform.json
