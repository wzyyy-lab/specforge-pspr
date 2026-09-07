#!/usr/bin/env bash
# The 2x2 that probe_matrix.sh gated away, and shouldn't have.
#
# probe_matrix.sh held cand_slots at 0 in all eight cells and said the candidate-identity variant
# would be probed "only if D shows cross-slot is worth anything".  D measured +0.023 for cross-slot,
# so it never ran.  That gate is invalid: the deep audit's claim is that cross-slot attention is
# worthless *because* the probability-weighted centroid destroys candidate identity before attention
# ever runs.  Under that hypothesis a near-zero cross-slot delta at cand_slots=0 is exactly the
# predicted observation, so D cannot discriminate between "cross-slot is useless" and "cross-slot is
# starved".  Only the interaction term can.
#
#                     cross_slot=full   cross_slot=none
#   cand_slots=0        P1 (control)      P2
#   cand_slots=4        P3                P4
#
# audit right  =>  (P3 - P4)  >>  (P1 - P2)     -- keep the module, feed it candidate identity
# audit wrong  =>  (P3 - P4)  ~=  (P1 - P2) ~= 0.02 -- the module is dead weight, delete it
#
# Held at the production config otherwise: no direct hidden wire (wave 1 refuted it, +0.373 vs
# +0.418), 16-way covered-slot CE.  Two ranks at batch 128 reproduces wave 1's launch geometry
# exactly, so P1 must land near A's +0.418; if it does not, the comparison is void.
#
# Runs alone on all 8 GPUs.  Do NOT run this next to an online training job: four of these starved
# the mooncake data plane and killed the PerfectBlend run at step 4250 with `get_into failed -707`.
set -u
cd /home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/SpecForge

EP=${EP:-2}
BS=${BS:-128}
EV=${EV:-400}
OUT=outputs/pspr_v2_probe3
mkdir -p "$OUT"

run () {
  local tag=$1 gpus=$2 port=$3
  shift 3
  CUDA_VISIBLE_DEVICES="$gpus" PYTHONPATH=. nohup torchrun \
    --nproc_per_node=2 --master_port="$port" \
    scripts/train_pspr_v2_probe.py \
    --output "$OUT/$tag.pt" --epochs "$EP" --batch-size "$BS" --eval-every "$EV" \
    --no-direct-hidden --loss ce16 \
    "$@" > "$OUT/$tag.log" 2>&1 &
  echo "launched $tag on GPU $gpus (port $port) pid $!"
}

run P1_cand0_full 0,1 29531 --cand-slots 0 --cross-slot full
run P2_cand0_none 2,3 29532 --cand-slots 0 --cross-slot none
run P3_cand4_full 4,5 29533 --cand-slots 4 --cross-slot full
run P4_cand4_none 6,7 29534 --cand-slots 4 --cross-slot none
wait
