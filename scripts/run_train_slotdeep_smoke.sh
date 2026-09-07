#!/bin/bash
# Smoke test for the SlotDeep arm; pass "full" to launch the original YAML's full epoch.
# `max_steps` is a dotted override so the smoke run and the real run share ONE config file; a copied
# yaml would be a second thing to keep in sync.
set -euo pipefail
cd /home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/SpecForge
CFG=examples/configs/online/disaggregated/managed-local/qwen3-4b-pspr-slotdeep.yaml
# tmux's server environment predates this shell and lacks no_proxy. Make the
# workload-scoped environment explicit; do not alter the global tmux environment.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 HF_HUB_OFFLINE=1
export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1
export PYTHONPATH=/home/wangzhuoyu/sglang-patched-0.5.15:.
SLOTDEEP_PHASE=${1:-smoke}
if [[ "$SLOTDEEP_PHASE" == stage2-eval ]]; then
  SLOTDEEP_RUN=qwen3-4b-pspr-slotdeep-stage2-20260906
  SLOTDEEP_TAG=SLOTDEEP_STAGE2_STEP9052_20260906
  SLOTDEEP_TAPS=/kl_infra_infer_intern/wangzhuoyu/TAPS-SP
  SLOTDEEP_REPO=/kl_infra_infer_intern/wangzhuoyu/SpecForge
  SLOTDEEP_EXPORT="$SLOTDEEP_REPO/outputs/${SLOTDEEP_RUN}_step9052_decode"
  if [[ "${SLOTDEEP_PLAN_ONLY:-0}" == 1 ]]; then
    bash scripts/run_train_slotdeep_smoke.sh stage2
    exit 0
  fi
  for SLOTDEEP_PATH in "outputs/$SLOTDEEP_RUN" "$SLOTDEEP_EXPORT" \
      outputs/TRAIN_SLOTDEEP_stage2_20260906.log "outputs/EXPORT_$SLOTDEEP_TAG.log" \
      "outputs/EVAL_$SLOTDEEP_TAG.log" "outputs/EVAL_$SLOTDEEP_TAG.json"; do
    if [[ -e "$SLOTDEEP_PATH" ]]; then
      echo "refusing to overwrite $SLOTDEEP_PATH" >&2
      exit 2
    fi
  done
  bash scripts/run_train_slotdeep_smoke.sh stage2 2>&1 | tee outputs/TRAIN_SLOTDEEP_stage2_20260906.log
  export CUDA_VISIBLE_DEVICES=7
  # Joint backbone is EXPECTED to change. Never pass Stage1's --verify-frozen.
  PYTHONPATH=. python -u scripts/export_pspr_cloze_for_decode.py \
    --checkpoint "$SLOTDEEP_REPO/outputs/$SLOTDEEP_RUN/$SLOTDEEP_RUN-step9052/training_state.pt" \
    --draft-config configs/qwen3-4b-pspr-slotdeep-joint.json \
    --backbone-template "$SLOTDEEP_TAPS/models/Qwen3-4B-DFlash-b16" --verify-roundtrip \
    --out "$SLOTDEEP_EXPORT" 2>&1 | tee "outputs/EXPORT_$SLOTDEEP_TAG.log"
  (
    cd "$SLOTDEEP_TAPS"
    PYTHONPATH=. python -u scripts/decode_lattice.py --target-model models/Qwen3-4B \
      --draft-model "$SLOTDEEP_EXPORT/backbone" --lattice-head "$SLOTDEEP_EXPORT/selector.pt" \
      --modes oneshot,latgate,oracle16 --datasets gsm8k,math500,humaneval,mbpp,alpaca,mt-bench \
      --gate-tau 0 --gate-rho 3 --gate-theta 0 --gate-stats --eval-reserved \
      --max-samples 20 --max-new-tokens 256 --shuffle-seed 2026 \
      --json-output "$SLOTDEEP_REPO/outputs/EVAL_$SLOTDEEP_TAG.json"
  ) 2>&1 | tee "outputs/EVAL_$SLOTDEEP_TAG.log"
  exit 0
fi
if [[ "$SLOTDEEP_PHASE" == turn-s1-eval || "$SLOTDEEP_PHASE" == distill-s1-eval ]]; then
  SLOTDEEP_TRAIN_PHASE=${SLOTDEEP_PHASE%-eval}
  SLOTDEEP_DATE=20260905
  SLOTDEEP_KIND=TURN
  if [[ "$SLOTDEEP_TRAIN_PHASE" == distill-s1 ]]; then
    SLOTDEEP_DATE=20260906
    SLOTDEEP_KIND=DISTILL
  fi
  if [[ "${SLOTDEEP_PLAN_ONLY:-0}" == 1 ]]; then
    bash scripts/run_train_slotdeep_smoke.sh "$SLOTDEEP_TRAIN_PHASE"
    exit 0
  fi
  SLOTDEEP_REPO=/kl_infra_infer_intern/wangzhuoyu/SpecForge
  SLOTDEEP_TAPS=/kl_infra_infer_intern/wangzhuoyu/TAPS-SP
  SLOTDEEP_RUN="qwen3-4b-pspr-slotdeep-${SLOTDEEP_TRAIN_PHASE}-${SLOTDEEP_DATE}"
  SLOTDEEP_CHECKPOINT="$SLOTDEEP_REPO/outputs/$SLOTDEEP_RUN/$SLOTDEEP_RUN-step7115/training_state.pt"
  SLOTDEEP_EXPORT="$SLOTDEEP_REPO/outputs/${SLOTDEEP_RUN}_step7115_decode"
  SLOTDEEP_TAG="SLOTDEEP_${SLOTDEEP_KIND}_S1_STEP7115_${SLOTDEEP_DATE}"
  SLOTDEEP_TRAIN_LOG="outputs/TRAIN_SLOTDEEP_${SLOTDEEP_TRAIN_PHASE}_${SLOTDEEP_DATE}.log"
  SLOTDEEP_PB_TAG="PB_VALIDATION_${SLOTDEEP_KIND}_S1_${SLOTDEEP_DATE}"
  SLOTDEEP_FRESH=(
    "outputs/$SLOTDEEP_RUN" "$SLOTDEEP_EXPORT"
    "$SLOTDEEP_TRAIN_LOG"
    "outputs/EXPORT_$SLOTDEEP_TAG.log" "outputs/EVAL_$SLOTDEEP_TAG.log"
    "outputs/EVAL_$SLOTDEEP_TAG.json" "outputs/TF_$SLOTDEEP_TAG.log"
    "outputs/RISK_$SLOTDEEP_TAG.json" "outputs/RISK_$SLOTDEEP_TAG.npz"
    "outputs/$SLOTDEEP_PB_TAG.log" "outputs/$SLOTDEEP_PB_TAG.json"
  )
  for SLOTDEEP_PATH in "${SLOTDEEP_FRESH[@]}"; do
    if [[ -e "$SLOTDEEP_PATH" ]]; then
      echo "refusing to overwrite $SLOTDEEP_PATH" >&2
      exit 2
    fi
  done
  bash scripts/run_train_slotdeep_smoke.sh "$SLOTDEEP_TRAIN_PHASE" 2>&1 | tee "$SLOTDEEP_TRAIN_LOG"
  export CUDA_VISIBLE_DEVICES=7
  PYTHONPATH=. python -u scripts/export_pspr_cloze_for_decode.py \
    --checkpoint "$SLOTDEEP_CHECKPOINT" --draft-config configs/qwen3-4b-pspr-slotdeep.json \
    --backbone-template "$SLOTDEEP_TAPS/models/Qwen3-4B-DFlash-b16" \
    --verify-frozen "$SLOTDEEP_TAPS/models/Qwen3-4B-DFlash-b16" --verify-roundtrip \
    --out "$SLOTDEEP_EXPORT" 2>&1 | tee "outputs/EXPORT_$SLOTDEEP_TAG.log"
  (
    cd "$SLOTDEEP_TAPS"
    PYTHONPATH=. python -u scripts/decode_lattice.py --target-model models/Qwen3-4B \
      --draft-model "$SLOTDEEP_EXPORT/backbone" --lattice-head "$SLOTDEEP_EXPORT/selector.pt" \
      --modes latgate --datasets gsm8k,math500,humaneval,mbpp,alpaca,mt-bench \
      --gate-tau 0 --gate-rho 3 --gate-theta 0 --gate-stats --eval-reserved \
      --max-samples 20 --max-new-tokens 256 --shuffle-seed 2026 \
      --json-output "$SLOTDEEP_REPO/outputs/EVAL_$SLOTDEEP_TAG.json"
  ) 2>&1 | tee "outputs/EVAL_$SLOTDEEP_TAG.log"
  PYTHONPATH=. python -u scripts/measure_ranker_teacher_forced.py \
    --features cache/hidden_states/slotdeep_eval_20260905 \
    --draft-config configs/qwen3-4b-pspr-slotdeep.json --checkpoint "$SLOTDEEP_CHECKPOINT" \
    --target-model "$SLOTDEEP_TAPS/models/Qwen3-4B" --num-samples 120 --num-anchors 512 \
    --chunk-blocks 64 --max-length 3072 --seed 2026 --gate-rho 3 --gate-tau 0 --gate-theta 0 \
    --preserve-selector-fp32 --per-sample-diagnostics --include-base-decisions \
    --dump-decisions "outputs/RISK_$SLOTDEEP_TAG.npz" --json-output "outputs/RISK_$SLOTDEEP_TAG.json" \
    2>&1 | tee "outputs/TF_$SLOTDEEP_TAG.log"
  PYTHONPATH=. python -u scripts/evaluate_pspr_holdout.py \
    --prompts cache/dataset/pspr_holdout_20260905/validation.jsonl --export "$SLOTDEEP_EXPORT" \
    --output "outputs/$SLOTDEEP_PB_TAG.json" --max-new-tokens 256 --max-prompt-tokens 2816 \
    2>&1 | tee "outputs/$SLOTDEEP_PB_TAG.log"
  exit 0
fi
# New loss arms reuse this exact architecture/YAML. Separate run directories are
# experiment artifacts, not new model implementations. Never resume the finished
# step7115 optimizer: these are weights-only, fixed-step continuations.
if [[ "$SLOTDEEP_PHASE" == loss-suite ]]; then
  for SLOTDEEP_ARM in loss-t0 loss-t1 loss-t2; do
    SLOTDEEP_ARM_LOG="outputs/TRAIN_SLOTDEEP_${SLOTDEEP_ARM}_20260905.log"
    if [[ "${SLOTDEEP_PLAN_ONLY:-0}" == 1 ]]; then
      bash scripts/run_train_slotdeep_smoke.sh "$SLOTDEEP_ARM"
      continue
    fi
    if [[ -e "$SLOTDEEP_ARM_LOG" ]]; then
      echo "refusing to overwrite $SLOTDEEP_ARM_LOG" >&2
      exit 2
    fi
    bash scripts/run_train_slotdeep_smoke.sh "$SLOTDEEP_ARM" 2>&1 | tee "$SLOTDEEP_ARM_LOG"
  done
  exit 0
fi
SLOTDEEP_OVERRIDES=(
  training.max_steps=40
  training.save_interval=20
  run_id=qwen3-4b-pspr-slotdeep-smoke-r3-20260905
  output_dir=./outputs/qwen3-4b-pspr-slotdeep-smoke-r3-20260905
  deployment.disaggregated.control_dir=outputs/qwen3-4b-pspr-slotdeep-smoke-r3-20260905/control
  deployment.disaggregated.consumer_state_dir=outputs/qwen3-4b-pspr-slotdeep-smoke-r3-20260905/consumer-state
)
SLOTDEEP_PROVENANCE=outputs/SLOTDEEP_SMOKE_R3_20260905_PROVENANCE
case "$SLOTDEEP_PHASE" in
  smoke) ;;
  full)
    SLOTDEEP_OVERRIDES=()
    SLOTDEEP_PROVENANCE=outputs/SLOTDEEP_S1_20260905_PROVENANCE
    ;;
  stage2-smoke|stage2-smoke-r2|stage2)
    # User-requested Stage2 from the numerically best completed native SlotDeep
    # run (T3). Same architecture/loss/gate; fresh optimizer, official backbone
    # unfrozen. No calibration artifact or KD branch enters this run.
    SLOTDEEP_RUN="qwen3-4b-pspr-slotdeep-${SLOTDEEP_PHASE}-20260906"
    SLOTDEEP_STEPS=9052
    SLOTDEEP_SAVE=500
    if [[ "$SLOTDEEP_PHASE" == stage2-smoke* ]]; then
      SLOTDEEP_STEPS=20
      SLOTDEEP_SAVE=20
    fi
    SLOTDEEP_OVERRIDES=(
      model.draft_model_config=configs/qwen3-4b-pspr-slotdeep-joint.json
      model.draft_checkpoint_path=/kl_infra_infer_intern/wangzhuoyu/SpecForge/outputs/qwen3-4b-pspr-slotdeep-loss-t3-20260905/qwen3-4b-pspr-slotdeep-loss-t3-20260905-step1000/training_state.pt
      data.train_data_path=./cache/dataset/perfectblend_qwen3-4b_regen_turnwise_20260905.jsonl
      training.num_epochs=1
      "training.max_steps=$SLOTDEEP_STEPS"
      "training.save_interval=$SLOTDEEP_SAVE"
      training.learning_rate=1.0e-4
      'training.lr_scale_rules={"candidate_selector.":1.0,"":0.3}'
      training.warmup_ratio=0.06
      training.prompt_seed=42
      runtime.producer_concurrency=1
      training.dflash2_selector_preserve_fp32=true
      training.dflash2_selector_alt_loss_alpha=0.5
      training.dflash2_selector_safe_loss_alpha=1.0
      training.dflash2_selector_repair_loss_alpha=0.1
      training.dflash2_selector_safe_margin=0.1
      training.dflash2_selector_repair_margin=0.1
      "run_id=$SLOTDEEP_RUN"
      "output_dir=./outputs/$SLOTDEEP_RUN"
      "deployment.disaggregated.control_dir=outputs/$SLOTDEEP_RUN/control"
      "deployment.disaggregated.consumer_state_dir=outputs/$SLOTDEEP_RUN/consumer-state"
    )
    SLOTDEEP_PROVENANCE="outputs/SLOTDEEP_${SLOTDEEP_PHASE}_20260906_PROVENANCE"
    # Current host image differs from yesterday's ledger. The workload overlay
    # pins the matching kernel; system packages and old run artifacts remain intact.
    export SLOTDEEP_ENV_SPEC=/home/wangzhuoyu/pspr-sglang-0515-stage2-h20-spec.json
    if [[ "${SLOTDEEP_PLAN_ONLY:-0}" != 1 ]]; then
      python -c 'import hashlib,json; from pathlib import Path; p=Path("cache/dataset/perfectblend_qwen3-4b_regen_turnwise_20260905.jsonl"); m=json.loads(p.with_suffix(".manifest.json").read_text()); assert m["counts"]["kept_turns"]==253480; f=p.open("rb"); assert hashlib.file_digest(f,"sha256").hexdigest()==m["output_sha256"]; f.close(); assert m["counts"]["kept_turns"]//28==9052; print("STAGE2_TURN_DATA_VERIFIED",m["counts"])'
    fi
    ;;
  loss-smoke|loss-t0|loss-t1|loss-t2|loss-t3)
    SLOTDEEP_RUN="qwen3-4b-pspr-slotdeep-${SLOTDEEP_PHASE}-20260905"
    SLOTDEEP_STEPS=1000
    SLOTDEEP_SAVE=500
    SLOTDEEP_ALT=0.0
    SLOTDEEP_SAFE=0.0
    SLOTDEEP_REPAIR=0.0
    case "$SLOTDEEP_PHASE" in
      loss-t1) SLOTDEEP_ALT=0.5 ;;
      loss-t2|loss-smoke) SLOTDEEP_ALT=0.5; SLOTDEEP_SAFE=0.1; SLOTDEEP_REPAIR=0.1 ;;
      # Follow-up single-coefficient contrast against completed T2: the
      # safety hinge receives 10x weight, all other semantics stay fixed.
      # This is another warm-start pilot, NOT random-init full Stage1.
      loss-t3) SLOTDEEP_ALT=0.5; SLOTDEEP_SAFE=1.0; SLOTDEEP_REPAIR=0.1 ;;
    esac
    if [[ "$SLOTDEEP_PHASE" == loss-smoke ]]; then
      SLOTDEEP_STEPS=20
      SLOTDEEP_SAVE=20
    fi
    SLOTDEEP_OVERRIDES=(
      "model.draft_checkpoint_path=/kl_infra_infer_intern/wangzhuoyu/SpecForge/outputs/qwen3-4b-pspr-slotdeep-s1-20260905/qwen3-4b-pspr-slotdeep-s1-20260905-step7115/training_state.pt"
      "training.max_steps=$SLOTDEEP_STEPS"
      "training.save_interval=$SLOTDEEP_SAVE"
      training.learning_rate=2.0e-5
      training.warmup_ratio=0.05
      training.prompt_seed=42
      # One in-flight lease preserves prompt order. With concurrency=2 the
      # producer publishes whichever capture request completes first, so a
      # shared seed alone does not create a paired training-data experiment.
      runtime.producer_concurrency=1
      training.dflash2_selector_preserve_fp32=true
      "training.dflash2_selector_alt_loss_alpha=$SLOTDEEP_ALT"
      "training.dflash2_selector_safe_loss_alpha=$SLOTDEEP_SAFE"
      "training.dflash2_selector_repair_loss_alpha=$SLOTDEEP_REPAIR"
      training.dflash2_selector_safe_margin=0.1
      training.dflash2_selector_repair_margin=0.1
      "run_id=$SLOTDEEP_RUN"
      "output_dir=./outputs/$SLOTDEEP_RUN"
      "deployment.disaggregated.control_dir=outputs/$SLOTDEEP_RUN/control"
      "deployment.disaggregated.consumer_state_dir=outputs/$SLOTDEEP_RUN/consumer-state"
    )
    SLOTDEEP_PROVENANCE="outputs/SLOTDEEP_${SLOTDEEP_PHASE}_20260905_PROVENANCE"
    ;;
  turn-smoke|turn-s1|distill-smoke|distill-s1)
    # Both recipes use the SAME completed turn-data contract. Distill changes
    # only the online supervision, NOT architecture or a mature-head continuation.
    # The original YAML loads the official DFlash directory: selector is random.
    SLOTDEEP_DATE=20260905
    if [[ "$SLOTDEEP_PHASE" == distill-* ]]; then SLOTDEEP_DATE=20260906; fi
    SLOTDEEP_RUN="qwen3-4b-pspr-slotdeep-${SLOTDEEP_PHASE}-${SLOTDEEP_DATE}"
    SLOTDEEP_STEPS=7115
    SLOTDEEP_SAVE=1000
    if [[ "$SLOTDEEP_PHASE" == *-smoke ]]; then
      SLOTDEEP_STEPS=20
      SLOTDEEP_SAVE=20
    fi
    SLOTDEEP_OVERRIDES=(
      data.train_data_path=./cache/dataset/perfectblend_qwen3-4b_regen_turnwise_20260905.jsonl
      "training.max_steps=$SLOTDEEP_STEPS"
      "training.save_interval=$SLOTDEEP_SAVE"
      training.prompt_seed=42
      runtime.producer_concurrency=1
      training.dflash2_selector_preserve_fp32=true
      "run_id=$SLOTDEEP_RUN"
      "output_dir=./outputs/$SLOTDEEP_RUN"
      "deployment.disaggregated.control_dir=outputs/$SLOTDEEP_RUN/control"
      "deployment.disaggregated.consumer_state_dir=outputs/$SLOTDEEP_RUN/consumer-state"
    )
    if [[ "$SLOTDEEP_PHASE" == distill-* ]]; then
      SLOTDEEP_OVERRIDES+=(
        training.dflash2_selector_objective=candidate_distill
        training.dflash2_selector_distill_alpha=0.5
        training.dflash2_selector_distill_temperature=1.0
      )
    fi
    SLOTDEEP_PROVENANCE="outputs/SLOTDEEP_${SLOTDEEP_PHASE}_${SLOTDEEP_DATE}_PROVENANCE"
    if [[ "${SLOTDEEP_PLAN_ONLY:-0}" != 1 ]]; then
      # No launch on a partially prepared JSONL, including failed worker output.
      python -c 'import hashlib,json; from pathlib import Path; p=Path("cache/dataset/perfectblend_qwen3-4b_regen_turnwise_20260905.jsonl"); m=json.loads(p.with_suffix(".manifest.json").read_text()); assert m["counts"]["kept_turns"] > 0; f=p.open("rb"); assert hashlib.file_digest(f,"sha256").hexdigest()==m["output_sha256"]; f.close(); print("TURN_DATA_VERIFIED",m["counts"])'
    fi
    ;;
  *) echo 'usage: bash scripts/run_train_slotdeep_smoke.sh [smoke|full|loss-smoke|loss-t0|loss-t1|loss-t2|loss-t3|loss-suite|turn-smoke|turn-s1|turn-s1-eval|distill-smoke|distill-s1|distill-s1-eval|stage2-smoke|stage2-smoke-r2|stage2|stage2-eval]' >&2; exit 2 ;;
esac
if [[ "${SLOTDEEP_PLAN_ONLY:-0}" == 1 ]]; then
  python -m specforge.cli train --plan -c "$CFG" "${SLOTDEEP_OVERRIDES[@]}"
  exit 0
fi
PROVENANCE_ARGS=()
for SLOTDEEP_OVERRIDE in "${SLOTDEEP_OVERRIDES[@]}"; do
  PROVENANCE_ARGS+=(--override "$SLOTDEEP_OVERRIDE")
done
python scripts/record_slotdeep_launch.py --config "$CFG" \
  --out "$SLOTDEEP_PROVENANCE" "${PROVENANCE_ARGS[@]}"
PYTHONPATH=/home/wangzhuoyu/sglang-patched-0.5.15:. python -u -m specforge.cli train -c "$CFG" "${SLOTDEEP_OVERRIDES[@]}"
