#!/usr/bin/env bash
# PSPR-v2 architecture probe matrix.
#
# Single-variable chain so every delta is attributable:
#   A  v1-like    : h_t reaches the corrector only through the 512-d encoder output, 16-way CE
#   B  +direct h  : h_t wired into the corrector at full 2560 width          (A + one wire)
#   C  +full CE   : dense cross-entropy over E(h+delta) instead of 16-way    (B + one loss)
#   D  -crossslot : C with the encoder's off-diagonal attention masked out   (C - cross-slot)
#
# cand_slots stays 0 in all four so the candidate-identity change is not a confound; it is probed
# separately only if D shows cross-slot is worth anything.
#
# Two GPUs per run, four runs in parallel. Traces are read-only, so the runs share the trace dir.
set -u
cd /home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/SpecForge

EP=${EP:-2}
BS=${BS:-128}
EV=${EV:-400}
OUT=outputs/pspr_v2_probe
mkdir -p "$OUT"

run () {
  local tag=$1 gpus=$2 port=$3
  shift 3
  CUDA_VISIBLE_DEVICES="$gpus" PYTHONPATH=. nohup torchrun \
    --nproc_per_node=2 --master_port="$port" \
    scripts/train_pspr_v2_probe.py \
    --output "$OUT/$tag.pt" --epochs "$EP" --batch-size "$BS" --eval-every "$EV" \
    "$@" > "$OUT/$tag.log" 2>&1 &
  echo "launched $tag on GPU $gpus (port $port) pid $!"
}

run A_v1like  0,1 29511 --no-direct-hidden --cand-slots 0 --loss ce16 --cross-slot full
run B_direct  2,3 29512 --direct-hidden    --cand-slots 0 --loss ce16 --cross-slot full
run C_full    4,5 29513 --direct-hidden    --cand-slots 0 --loss full --cross-slot full
run D_nocross 6,7 29514 --direct-hidden    --cand-slots 0 --loss full --cross-slot none
wait
