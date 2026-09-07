#!/bin/bash
# Re-dump with the backbone hidden h included, so the ranker probe can test the FULL input set the
# deployed head reads.  Two arms: 320 training sequences (fit) and the 120 benchmark traces (report).
set -e
ROOT=/home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/SpecForge
cd "$ROOT"
COMMON="--checkpoint outputs/qwen3-4b-pspr-cloze/qwen3-4b-pspr-cloze-step7115/training_state.pt \
--num-anchors 512 --chunk-blocks 16 --gate-rho 3.0 --dump-hidden"
SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 PYTHONPATH=/home/wangzhuoyu/sglang-patched:. \
CUDA_VISIBLE_DEVICES=1 nohup python3 -u scripts/measure_ranker_teacher_forced.py $COMMON \
  --features cache/hidden_states/pb_big --num-samples 320 \
  --gate-training-corpus --gate-tol 0.03 \
  --dump-features outputs/TF_FEATH_pb.npz --json-output outputs/TF_LADDERH_pb.json \
  > outputs/TF_FEATH_pb.log 2>&1 &
SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 PYTHONPATH=/home/wangzhuoyu/sglang-patched:. \
CUDA_VISIBLE_DEVICES=2 nohup python3 -u scripts/measure_ranker_teacher_forced.py $COMMON \
  --features cache/hidden_states/eval_traces --num-samples 120 --chain \
  --dump-features outputs/TF_FEATH_eval.npz --json-output outputs/TF_LADDERH_eval.json \
  > outputs/TF_FEATH_eval.log 2>&1 &
wait
echo done
