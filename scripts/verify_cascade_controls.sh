#!/usr/bin/env bash
# Two things at once, both of which the current evidence cannot do without.
#
# 1) DETERMINISM CONTROL (gpu 0).  The untrained cascade at margin 0 is algebraically the champion's
#    `latgate rho=3`, and the paired macro difference agrees (+0.0092 +- 0.0427, p=0.83).  But block 0
#    -- the only zero-noise comparison, since its prefix is the prompt itself -- matches on 107/120
#    prompts, not 120/120.  fp32 tie-flipping cannot explain 13/120: a flip needs
#    |s_max - s0 - log 3| < ~1e-6, which over 1800 block-0 slot decisions should happen ~1e-3 times.
#    So EITHER the two code paths differ systematically, OR the decoder is not run-to-run
#    deterministic.  Re-decoding the CHAMPION against its own stored records settles it: if
#    champion-vs-champion also disagrees on ~13/120 block-0 lengths, the residual is nondeterminism and
#    the anchor claim stands; if it matches 120/120, there is a real rule difference to find.
#
# 2) HELD-OUT REPLICATION (gpu 1-3).  `--repair-margin 0.5` was selected as the best operating point on
#    the same 20 prompts/domain used for the comparison, so the headline numbers at m=0.5 are
#    in-sample.  A different `--shuffle-seed` draws a disjoint prompt sample; paired_compare matches on
#    (dataset, prompt_sha256), so the three new runs pair with each other correctly.  The cascade-vs-
#    blind comparison is what needs replicating: it is the arm's actual question.
set -u

TAPS=/home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/TAPS-SP
OUT=/home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/SpecForge/outputs
DATASETS=gsm8k,math500,humaneval,mbpp,alpaca,mt-bench
BASE="--target-model models/Qwen3-4B --draft-model models/Qwen3-4B-DFlash-b16 \
--gate-stats --datasets ${DATASETS} --max-samples 20 --max-new-tokens 256 --eval-reserved"

CHAMP=${OUT}/qwen3-4b-pspr-cloze_step7115_decode/selector.pt
CASC=${OUT}/qwen3-4b-pspr-cascade_step500_decode/selector.pt
BLIND=${OUT}/qwen3-4b-pspr-cascade-blind_step500_decode/selector.pt

cd "${TAPS}" || exit 1

# 1) same seed, same flags, same weights as DETAIL_cloze7115_rho3.json
CUDA_VISIBLE_DEVICES=0 SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 nohup python3 scripts/decode_lattice.py \
  ${BASE} --modes latgate --gate-tau 0 --gate-rho 3 --gate-theta 0 --shuffle-seed 2026 \
  --lattice-head "${CHAMP}" --json-output "${OUT}/REPLAY_champion_rho3.json" \
  > "${OUT}/REPLAY_champion_rho3.log" 2>&1 &
echo "launched determinism replay on gpu 0"
sleep 3

# 2) held-out sample, three matched arms
CUDA_VISIBLE_DEVICES=1 SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 nohup python3 scripts/decode_lattice.py \
  ${BASE} --modes latgate --gate-tau 0 --gate-rho 3 --gate-theta 0 --shuffle-seed 7777 \
  --lattice-head "${CHAMP}" --json-output "${OUT}/HELDOUT_champion_rho3.json" \
  > "${OUT}/HELDOUT_champion_rho3.log" 2>&1 &
echo "launched held-out champion on gpu 1"
sleep 3

CUDA_VISIBLE_DEVICES=2 SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 nohup python3 scripts/decode_lattice.py \
  ${BASE} --modes latrepair --repair-margin 0.5 --allow-policy-mismatch --shuffle-seed 7777 \
  --lattice-head "${CASC}" --json-output "${OUT}/HELDOUT_cascade_m0.5.json" \
  > "${OUT}/HELDOUT_cascade_m0.5.log" 2>&1 &
echo "launched held-out cascade m=0.5 on gpu 2"
sleep 3

CUDA_VISIBLE_DEVICES=3 SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 nohup python3 scripts/decode_lattice.py \
  ${BASE} --modes latrepair --repair-margin 0.5 --allow-policy-mismatch --shuffle-seed 7777 \
  --lattice-head "${BLIND}" --json-output "${OUT}/HELDOUT_blind_m0.5.json" \
  > "${OUT}/HELDOUT_blind_m0.5.log" 2>&1 &
echo "launched held-out blind m=0.5 on gpu 3"

wait
echo "all done"
