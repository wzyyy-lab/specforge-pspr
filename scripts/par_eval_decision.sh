#!/usr/bin/env bash
# Fan the six domains across six GPUs so a full-macro read costs ~5 min instead of ~25.
# Each domain is an independent decode_lattice process; the prompts per domain are identical to the
# serial run (same --eval-reserved / --shuffle-seed / --max-samples), so the per-domain numbers and
# therefore the macro are directly comparable to the serial baselines.
set -u
SF=/home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/SpecForge
STEP=${1:?usage: par_eval_decision.sh <step> [margin]}
MARGIN=${2:-0}
SAMPLES=${SAMPLES:-20}
SUF=${SUF:-}

i=1
for D in gsm8k math500 humaneval mbpp alpaca mt-bench; do
  DATASETS="$D" SAMPLES="$SAMPLES" REPAIR_MARGIN="$MARGIN" MODES=latrepair \
    TAGSUF="${SUF}_$D" bash "$SF/scripts/eval_decision.sh" "$STEP" "$i" >/dev/null 2>&1 &
  i=$((i + 1))
done
wait
echo "== step$STEP margin=$MARGIN done"
for D in gsm8k math500 humaneval mbpp alpaca mt-bench; do
  L="$SF/outputs/EVAL_decision_step${STEP}${SUF}_${D}.log"
  [ -f "$L" ] && tr '\r' '\n' < "$L" | grep -E "^\[$D\]" | tail -1
done
