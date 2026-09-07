#!/bin/bash
set -e
cd /home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/SpecForge
PYTHONPATH=. python3 -u scripts/analyze_domain_conditional.py \
  --train-features outputs/TF_FEATH_pb.npz \
  --eval-features outputs/TF_FEATH_eval.npz \
  --tokenizer /kl_infra_infer_intern/wangzhuoyu/TAPS-SP/models/Qwen3-4B \
  --json-output outputs/DOMAIN_COND.json
