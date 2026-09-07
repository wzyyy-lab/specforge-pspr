#!/bin/bash
# 8 single-GPU SGLang servers for the target, one per H20, for target-greedy regeneration.
set -e
MODEL=/kl_infra_infer_intern/wangzhuoyu/TAPS-SP/models/Qwen3-4B
LOGDIR=/home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/SpecForge/outputs/logs
mkdir -p "$LOGDIR"
for g in 0 1 2 3 4 5 6 7; do
  PORT=$((30000 + g * 10))
  SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=$g nohup python3 -m sglang.launch_server \
    --model "$MODEL" \
    --mem-fraction-static 0.80 \
    --tp 1 \
    --trust-remote-code \
    --cuda-graph-max-bs 128 \
    --host 127.0.0.1 \
    --port $PORT \
    --dtype bfloat16 \
    > "$LOGDIR/sglang_$PORT.log" 2>&1 &
done
echo "launched 8 servers on ports 30000..30070"
