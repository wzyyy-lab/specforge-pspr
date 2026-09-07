#!/usr/bin/env bash
# PSPR-v2 probe, wave 2.
#
# Wave 1 ranked the variants at 276k anchors: A (v1-like) +0.418 > B (+direct h) +0.373 >
# C (+full-vocab CE) +0.348 > D (C without cross-slot) +0.325.
#
# That ranking cannot be taken at face value, because the head is known to be data-limited
# (EXPERIMENT_LOG.md:1608 -- halving the trace costs 25% of the gain) and the real run has ~100x more
# data.  A fixed-276k probe systematically penalises exactly the variants that buy capacity or take
# on a harder objective.  So wave 2 measures the SLOPE, not another point:
#
#   G_Ahalf / H_Chalf   A and C at data-frac 0.5.  Combined with wave 1 these give each variant's
#                       gain-per-doubling; whichever slope is steeper wins at PerfectBlend scale, and
#                       the two lines may well cross between here and there.
#   E_rank              A plus the committed token's rank inside the previous slot's candidate list.
#                       In an all-MASK block every slot is conditioned on the same prefix, so
#                       cross-slot attention carries no new evidence (measured 3x at ~+0.02); the
#                       committed path is the only new conditioning there is, and the GRU currently
#                       sees only its token embeddings, not where it sits in the lattice.
#   F_hybrid            A plus a dense full-vocab auxiliary at 0.5.  Keeps the 16-way term that
#                       matches the lattice-restricted serving decision while closing the 20.93%
#                       supervision hole; should inherit the steeper of the two slopes.
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

run E_rank   0,1 29521 --no-direct-hidden --cand-slots 0 --loss ce16   --cross-slot full --prefix-rank
run F_hybrid 2,3 29522 --no-direct-hidden --cand-slots 0 --loss hybrid --cross-slot full --hybrid-lambda 0.5
run G_Ahalf  4,5 29523 --no-direct-hidden --cand-slots 0 --loss ce16   --cross-slot full --data-frac 0.5
run H_Chalf  6,7 29524 --direct-hidden    --cand-slots 0 --loss full   --cross-slot full --data-frac 0.5
wait
