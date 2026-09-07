#!/bin/bash
# Smoke test for the slotdeep stage-2 (joint) arm: prove the training path runs end to end before
# spending 2.4 h.  `max_steps` is a dotted override so the smoke run and the real run share ONE
# config file; a copied yaml would be a second thing to keep in sync.
set -e
cd /home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/SpecForge
source scripts/env_sglang_patched.sh
RUN=qwen3-4b-pspr-slotdeep-joint-s2-smoke-20260906
CFG=examples/configs/online/disaggregated/managed-local/qwen3-4b-pspr-slotdeep-joint.yaml
PYTHONPATH=/home/wangzhuoyu/sglang-patched-0.5.15:. python3 -u -m specforge.cli train --config "$CFG" \
  training.max_steps=40 \
  training.save_interval=1000000 \
  runtime.producer_concurrency=1 \
  run_id="$RUN" \
  output_dir="./outputs/$RUN" \
  deployment.disaggregated.control_dir="outputs/$RUN/control" \
  deployment.disaggregated.consumer_state_dir="outputs/$RUN/consumer-state"
