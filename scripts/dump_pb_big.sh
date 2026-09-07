#!/bin/bash
# Scale the training-corpus feature dump.  The 64-sequence probe overfits within one epoch, so a
# negative capacity result there is uninterpretable; 960 more disjoint rows (start>=64, so no
# overlap with the existing pb_probe shard) take the probe from 0.14M to ~2.2M slots.
set -e
cd /home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/SpecForge
mkdir -p cache/hidden_states/pb_big outputs/logs
GPUS="3 4 5 6 7 0"
i=0
for g in $GPUS; do
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=$g nohup python3 -u scripts/dump_hidden_states_hf.py \
    --target-model /kl_infra_infer_intern/wangzhuoyu/TAPS-SP/models/Qwen3-4B \
    --draft-config configs/qwen3-4b-pspr-cloze.json \
    --data-path cache/dataset/perfectblend_qwen3-4b_regen.jsonl \
    --output cache/hidden_states/pb_big \
    --chat-template qwen-nosys \
    --max-length 3072 \
    --num-samples 160 \
    --start $((64 + i)) --stride 6 \
    --log-every 40 \
    > outputs/logs/dump_pb_big_$g.log 2>&1 &
  i=$((i + 1))
done
wait
echo "shards done: $(ls cache/hidden_states/pb_big | wc -l) files"
