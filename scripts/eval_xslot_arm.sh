#!/usr/bin/env bash
# Re-run the cross-slot (bidirectional) ablation at n=60 instead of n=20.
#
# Why it must be redone: the ablation that produced the verdict "the bidirectional encoder's
# cross-slot attention is worthless" measured -0.0445 macro (5.9993 vs 6.0438) at n=20.  The
# subsequent latfront comparison showed a +0.0415 result on the old card flipping to -0.344 on the
# new one, i.e. n=20 cannot resolve differences of this magnitude at all.  The verdict was therefore
# never actually established -- it was read off a number inside the noise floor.
#
# Three arms, identical weights (cloze@7115), identical decode policy (latgate rho=3), n=60:
#   full          -- unmodified: every slot attends to every slot
#   anchor_self   -- attention_pattern="anchor_self": cross-slot rows removed, per-slot tower kept
#   zero_context  -- z zeroed: the encoder output is removed entirely (upper bound on its value)
#
# `zero_context` is included because it is the only arm with a large enough effect (-0.51 at n=20) to
# serve as a positive control: if it does not reproduce at n=60, the whole measurement setup is
# suspect and the other two arms mean nothing either.
set -u
SF=/home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/SpecForge
TS=/home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/TAPS-SP
DFLASH=$TS/models/Qwen3-4B-DFlash-b16
HEAD=$SF/outputs/qwen3-4b-pspr-cloze_step7115_decode/selector.pt
SAMPLES=${SAMPLES:-60}
ARM=${ARM:?arm: full|anchor_self|zero_context}
DATASETS=${DATASETS:?datasets}
GPU=${GPU:-0}

EXTRA=""
case "$ARM" in
  full) ;;
  anchor_self) EXTRA="--cloze-anchor-self" ;;
  zero_context) EXTRA="--cloze-zero-context" ;;
  *) echo "bad ARM=$ARM"; exit 2 ;;
esac

TAG=XSLOT_n${SAMPLES}_${ARM}_${DATASETS}
LOG=$SF/outputs/${TAG}.log
echo "== $TAG | gpu=$GPU -> $LOG"
cd "$TS"
CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=.:$SF \
  python3 -u scripts/decode_lattice.py \
  --target-model "$TS/models/Qwen3-4B" \
  --draft-model "$DFLASH" \
  --lattice-head "$HEAD" \
  --modes latgate --gate-rho 3 --gate-stats $EXTRA \
  --json-output "$SF/outputs/${TAG}.json" \
  --eval-reserved --max-samples "$SAMPLES" --max-new-tokens 256 \
  --datasets "$DATASETS" 2>&1 | tee "$LOG" | tail -20
