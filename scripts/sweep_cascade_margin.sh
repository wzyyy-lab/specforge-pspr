#!/usr/bin/env bash
# Repair-margin calibration sweep for the cascade arm and its blind control.
#
# Why this is the required next measurement, not a fishing expedition
# ------------------------------------------------------------------
# At margin 0 the trained cascade loses -0.2348 +- 0.0550 macro to the champion `latgate rho=3`
# (t=-4.26), while its gate BCE is 0.21 nats BETTER than the blind control's.  Those two facts are only
# contradictory if per-slot likelihood were the objective.  It is not: accept length truncates at the
# FIRST error, so destroying a correct base token forfeits the rest of the block while a missed repair
# costs only what was already going to be lost.  The MAP rule prices those symmetrically; the
# champion's rho=3 is a hand-set conservative margin that does not.  This is the same effect already
# recorded for rho=1 vs rho=3 (-0.111, t=-2.54, p=0.011).
#
# `--repair-margin m` subtracts m from every REPAIR log-probability, so the factorized rule becomes
# `e + log r_* > m`.  With the analytic anchor that is `s_max_alt - s0 > log(3) + m`, i.e. an effective
# rho of `3 * exp(m)`.  So the margin IS rho, re-parameterised, and comparing the arms at a single
# margin compares one point on each operating curve rather than the curves.
#
# `--allow-policy-mismatch` is required and correct here: the exported policy records margin 0, and the
# decoder is fail-closed about serving a policy it was not exported for.  This is the labelled
# calibration/ablation case the flag exists for.
#
# DISCIPLINE: these are the same 20 prompts per domain that the champion comparison uses, so the best
# margin found here is SELECTED on the comparison set.  It may be reported as a calibration curve, and
# the winning operating point must then be re-verified on a disjoint prompt set before any promotion
# claim.  scripts/paired_compare.py matches on (dataset, prompt_sha256), so a re-verification run with
# a different --shuffle-seed aligns correctly.
set -u

TAPS=/home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/TAPS-SP
OUT=/home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/SpecForge/outputs
DATASETS=gsm8k,math500,humaneval,mbpp,alpaca,mt-bench
COMMON="--target-model models/Qwen3-4B --draft-model models/Qwen3-4B-DFlash-b16 \
--modes latrepair --gate-stats --datasets ${DATASETS} --max-samples 20 --max-new-tokens 256 \
--shuffle-seed 2026 --eval-reserved --allow-policy-mismatch"

gpu=0
for arm in cascade blind; do
  case "${arm}" in
    cascade) head="${OUT}/qwen3-4b-pspr-cascade_step500_decode/selector.pt" ;;
    blind)   head="${OUT}/qwen3-4b-pspr-cascade-blind_step500_decode/selector.pt" ;;
  esac
  for m in 0.5 1.0 1.5 2.0; do
    tag="${arm}_m${m}"
    cd "${TAPS}" || exit 1
    CUDA_VISIBLE_DEVICES=${gpu} SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
      nohup python3 scripts/decode_lattice.py ${COMMON} \
        --lattice-head "${head}" --repair-margin "${m}" \
        --json-output "${OUT}/SWEEP_${tag}.json" \
        > "${OUT}/SWEEP_${tag}.log" 2>&1 &
    echo "launched ${tag} on gpu ${gpu}"
    gpu=$((gpu + 1))
    sleep 3
  done
done
wait
echo "sweep complete"
