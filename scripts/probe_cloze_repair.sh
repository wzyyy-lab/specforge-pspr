#!/usr/bin/env bash
# Two-endpoint probe: cloze weights, factorised decode rule, margin 0 vs margin 3.
set -u
SF=/home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/SpecForge
MARGIN=3 GPU=0 SAMPLES=20 DATASETS=gsm8k bash "$SF/scripts/eval_cloze_repair.sh" > "$SF/outputs/RUN_clozerepair_m3.out" 2>&1 &
MARGIN=0 GPU=7 SAMPLES=20 DATASETS=gsm8k bash "$SF/scripts/eval_cloze_repair.sh" > "$SF/outputs/RUN_clozerepair_m0.out" 2>&1 &
wait
echo "== both done"
