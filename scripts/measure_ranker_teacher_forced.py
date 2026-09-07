#!/usr/bin/env python3
"""Teacher-forced ranker ladder: what the selector can rank when the prefix is given for free.

WHY THIS EXISTS
---------------
The decode-side accounting (``TAPS-SP/scripts/decode_lattice.py:818-852``) says that on
policy-reachable slots the cloze selector recovers only 36.71% of fixable first errors, and
decomposes the shortfall as

    alternative_rank_0 / fixable = 66.86%   <- ranks the truth first AMONG the 15 alternatives
    selector_rank_0    / fixable = 44.05%   <- still first once candidate 0 (KEEP) joins
    fix_recovered      / fixable = 36.71%   <- survives the margin gate

So 33.1pp is "picked the wrong alternative", 22.8pp is "the known-wrong base outscored the truth",
7.3pp is the gate.  The open question that decides the next round is whether those losses are an
OPTIMISATION/EXPOSURE artefact (the ranker is better on the distribution it was trained on, and
degrades on the block alignment decode actually produces) or a CAPACITY/OBJECTIVE limit (the ranker
is simply this good, everywhere).  Those two diagnoses have completely different fixes: DAgger /
on-policy prefixes versus a new objective or a bigger per-slot readout.

This script measures the same ladder on the TRAINING-TIME construction: uniformly sampled anchors,
ground-truth (teacher-forced) predecessors, exactly the tensors ``OnlineDFlashModel.forward`` builds.
The rank bookkeeping is transcribed from the decoder, but this is not a matched decode population:
HF-generated feature trajectories and sampled anchors can differ from actual block-wise decoding.
Matched heads on the SAME features can be compared; a TF/decode gap alone does not isolate exposure
bias or block alignment.

FIDELITY: WHAT IS SHARED WITH TRAINING, AND WHAT IS NOT
------------------------------------------------------
Shared, by construction (same functions, not reimplementations):
  * ``OnlineDFlashModel._forward_draft_blocks`` -> anchors, MASK-filled block, DFlash block mask
  * label / predecessor / weight_mask algebra transcribed from ``dflash_family_model.py:1649-1690``
  * ``LatticePathSelector.extract_lattice`` (candidate 0 forced to argmax, not topk[0])
  * ``ClozeCorrector.score_candidates`` (cloze rows + GRU + additive delta)
  * ``LatticePathSelector.select_margin_gate`` (the deployed rho/tau/theta rule)
  * ``frontier_mask`` for the reachability restriction

NOT shared, and why it is acceptable:
  * ``hidden_states`` here come from plain HF ``output_hidden_states`` (``dump_hidden_states_hf.py``),
    while training captured them through patched SGLang.  This is the RIGHT choice for this
    measurement: ``decode_lattice.py`` also runs the target through HF, so the numbers being compared
    against are HF-numerics too.  The capture/HF gap is a property of how the model was trained, and
    is held fixed on both sides of the comparison.
  * anchor sampling is fresh RNG every call (``dflash_family_model.py:629-632``), so the exact anchor
    set never occurred in training.  With ``--num-anchors 512`` uniform-without-replacement over the
    supervised span the statistic is an unbiased estimate of the training-time slot distribution;
    ``--seed`` pins it so two arms are compared on identical anchors.
  * a single sequence per forward, so ``max_valid_anchors`` is per-sequence rather than the
    per-microbatch max used with ``batch_size: 2``.  That only changes how many padded anchor columns
    get masked out, never which sampled anchors are valid.

POPULATIONS (this is the part that is easy to get wrong)
-------------------------------------------------------
``dflash_family_model.py:1368-1382`` states, with numbers, that the training slot population and the
decode reachable population are NOT comparable: base-right is 0.4749 over all valid corpus slots
against 0.7744 over decode's covered reachable slots -- a 28pp pure occupancy effect.  Reporting the
ladder over all valid slots and calling it "the training-distribution repair rate" would therefore
manufacture a fake exposure-bias result.  Three populations are reported:

  ALL      every supervised slot                        -- the population the LOSS is averaged over
  POLICY   frontier_mask(valid, policy_ok)              -- MATCHED to decode: accepted prefix + first
                                                          policy error, i.e. exactly decode's
                                                          ``range(min(accept + 1, len(pick)))``
  BASE     frontier_mask(valid, base_top1_ok)           -- the err-head's own BCE population
                                                          (``dflash_family_model.py:1423``)

POLICY uses the same reachability RULE as decode, not necessarily the same sequences or population.
BASE is useful for a fixed-head comparison because its denominators do not depend on the selector.
ALL is also reported for corpus diagnostics (see GATES).

GATES (fail loudly rather than report a corrupted ladder)
--------------------------------------------------------
G1 selector is actually trained: ``|delta_w2.weight| > 0`` (zero-init means untrained).
G2 selector runs in fp32: the decision-precision contract (``pspr_cloze.py:709``).  In bf16 the
   correction is smaller than the log-prob spacing and the ladder silently collapses to the base
   ranking (``pspr.py:330-334``).
G3 ALL-population ``coverage`` and ``base_right`` reproduce the values recorded in
   ``dflash_family_model.py:1376-1378`` (0.8086 / 0.4749) within ``--gate-tol``.  These were measured
   on capture-side hidden states over the training corpus, so agreement is simultaneous evidence that
   the HF dump is faithful, the anchor sampling matches, and the label/weight algebra is right.
   Only enforced with ``--gate-training-corpus`` (it is meaningless on a benchmark-prompt dump).
G4 accounting closes: ``base_right + unfixable + fixable == decision_n`` and
   ``recovered + kept_wrong + wrong_override == fixable`` in every population.

Usage
-----
    PYTHONPATH=. python3 scripts/measure_ranker_teacher_forced.py \
        --features cache/hidden_states/pb_probe \
        --checkpoint outputs/qwen3-4b-pspr-cloze/qwen3-4b-pspr-cloze-step7115/training_state.pt \
        --target-model /kl_infra_infer_intern/wangzhuoyu/TAPS-SP/models/Qwen3-4B \
        --num-samples 40 --gate-training-corpus --json-output outputs/TF_LADDER_pb.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch

SPECFORGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPECFORGE))

from specforge.algorithms.common.dflash_family_model import (  # noqa: E402
    OnlineDFlashModel,
    frontier_mask,
)
from specforge.data.preprocessing import process_offline_dflash_sample  # noqa: E402
from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead  # noqa: E402
from specforge.runtime.data_plane.offline_reader import list_feature_files  # noqa: E402
from specforge.training.model_loading import warm_start_draft_model  # noqa: E402

POPULATIONS = ("ALL", "POLICY", "BASE", "CHAIN")
TOP_K = 16
COUNTERS = (
    "decision_n",
    "base_right_n",
    "base_right_kept",
    "base_right_destroyed",
    "base_wrong_n",
    "unfixable_n",
    "fixable_n",
    "fix_recovered",
    "fix_kept_wrong",
    "fix_gate_blocked_true_best",
    "fix_wrong_override",
    "covered_n",
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_draft(draft_config: str, checkpoint: str, device, dtype):
    from transformers import Qwen3Config

    from specforge.modeling.draft.pspr_cloze import PSPRClozeDraftModel

    raw = json.load(open(draft_config))
    architecture = raw.get("architectures", ["PSPRClozeDraftModel"])[0]
    draft_class = PSPRClozeDraftModel
    if architecture == "PSPRSlotDeepDraftModel":
        from specforge.modeling.draft.pspr_slotdeep import PSPRSlotDeepDraftModel
        draft_class = PSPRSlotDeepDraftModel
    elif architecture != "PSPRClozeDraftModel":
        raise ValueError(f"this diagnostic does not support architecture {architecture!r}")
    cfg = Qwen3Config(**raw)
    cfg.dflash_config = raw["dflash_config"]
    cfg.block_size = raw["block_size"]
    cfg.num_target_layers = raw["num_target_layers"]
    cfg._attn_implementation = "flex_attention"
    torch.manual_seed(0)
    with torch.device("cpu"):
        draft = draft_class(cfg)
    if architecture == "PSPRSlotDeepDraftModel":
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("pspr_slotdeep_reference_config") != draft.candidate_selector.reference_config():
            raise ValueError("SlotDeep diagnostic checkpoint/config semantic mismatch")
        # Warm-start loaders may allow a freshly initialized optional selector.
        # A trained-head diagnostic must not silently accept that fallback.
        prefix = "candidate_selector."
        selector_state = {k[len(prefix):]: v for k, v in payload["draft_state_dict"].items()
                          if k.startswith(prefix) and not k.endswith("target_embedding")}
        draft.candidate_selector.load_state_dict(selector_state, strict=True)
        del payload, selector_state
    report = warm_start_draft_model(
        draft,
        checkpoint,
        draft_config=cfg,
        strategy="dflash",
    )
    print(f"warm start: {report.loaded_keys} tensors from {Path(checkpoint).name}", flush=True)
    return draft, cfg



def threshold_field(args, gate_margin, base_logprob, device):
    """log rho for every slot, as a tensor broadcastable to ``gate_margin``.

    ``global``      one scalar, exactly the deployed rule.
    ``per_slot``    15 free thresholds, the most expressive and the most overfittable.
    ``affine_slot`` log rho_h = a + b*h            (2 params, monotone in slot index)
    ``affine_base`` log rho = a + b*log p0         (2 params, adapts to base confidence)
    ``affine_both`` log rho = a + b*h + c*log p0   (3 params)
    """
    mode = args.rho_mode
    if mode == "global":
        return gate_margin.new_full((), math.log(args.gate_rho))
    horizon = gate_margin.shape[-1]
    slot = torch.arange(horizon, device=device, dtype=gate_margin.dtype)
    if mode == "per_slot":
        values = [float(x) for x in args.rho_values.split(",")]
        if len(values) != horizon:
            raise SystemExit(f"--rho-values needs {horizon} entries, got {len(values)}")
        return torch.tensor([math.log(v) for v in values], device=device,
                            dtype=gate_margin.dtype)
    coefficients = [float(x) for x in args.rho_values.split(",")]
    if mode == "affine_slot":
        a, b = coefficients
        return a + b * slot
    if mode == "affine_base":
        a, b = coefficients
        return a + b * base_logprob
    if mode == "affine_both":
        a, b, c = coefficients
        return a + b * slot + c * base_logprob
    raise SystemExit(f"unknown --rho-mode {mode}")


def new_stats(device):
    return {
        pop: {k: torch.zeros((), dtype=torch.long, device=device) for k in COUNTERS}
        for pop in POPULATIONS
    }


def new_hists(device):
    return {
        pop: {
            name: torch.zeros(TOP_K, dtype=torch.long, device=device)
            for name in ("draft_rank", "selector_rank", "alternative_rank")
        }
        for pop in POPULATIONS
    }


def accumulate(
    stats,
    hists,
    per_slot,
    pop_mask,
    pop_name,
):
    """Transcription of decode_lattice.py:818-852, vectorised over [B, N, H]."""
    tic = per_slot["target_is_candidate"]
    trank = per_slot["target_rank"]
    srank = per_slot["selector_rank"]
    arank = per_slot["alternative_rank"]
    pick = per_slot["pick_index"]

    base_right = pop_mask & tic & trank.eq(0)
    base_wrong = pop_mask & ~(tic & trank.eq(0))
    unfixable = pop_mask & ~tic
    fixable = pop_mask & tic & trank.ne(0)

    s = stats[pop_name]
    s["decision_n"] += pop_mask.sum()
    s["covered_n"] += (pop_mask & tic).sum()
    s["base_right_n"] += base_right.sum()
    s["base_right_kept"] += (base_right & pick.eq(0)).sum()
    s["base_right_destroyed"] += (base_right & pick.ne(0)).sum()
    s["base_wrong_n"] += base_wrong.sum()
    s["unfixable_n"] += unfixable.sum()
    s["fixable_n"] += fixable.sum()

    recovered = fixable & pick.eq(trank)
    kept_wrong = fixable & pick.eq(0)
    s["fix_recovered"] += recovered.sum()
    s["fix_kept_wrong"] += kept_wrong.sum()
    s["fix_gate_blocked_true_best"] += (kept_wrong & srank.eq(0)).sum()
    s["fix_wrong_override"] += (fixable & pick.ne(0) & pick.ne(trank)).sum()

    h = hists[pop_name]
    for name, values in (
        ("draft_rank", trank),
        ("selector_rank", srank),
        ("alternative_rank", arank),
    ):
        h[name] += torch.bincount(
            values[fixable].reshape(-1).clamp_(0, TOP_K - 1), minlength=TOP_K
        )


def chain_population(anchor_positions, block_keep_mask, valid, policy_ok):
    """Exact simulation of decode's block chaining, at zero extra forward cost.

    ``decode_lattice.py:854`` advances ``start += accept + 1``, so block ``t+1`` anchors at
    ``anchor_t + accept_t + 1`` -- one position past the slot that ended the accepted run.  Easy
    regions are therefore covered by one block per 15 tokens while hard regions get a block every
    two, which length-biases decode's slot population towards hard regions.  Training samples anchors
    uniformly (``dflash_family_model.py:629-632``) and has no such bias.

    The simulation is exact rather than approximate for two reasons:
      * the lattice, ``h`` and ``z`` at anchor ``a`` are a pure function of ``a`` and the ground-truth
        context, with no dependence on how the walk reached ``a``;
      * inside the accepted run every committed token equals the target token, so the on-policy GRU
        prefix decode feeds (``decode_lattice.py:687`` sets ``prev = di``) is bit-identical to the
        teacher-forced ``predecessor_ids`` used here.  The two only diverge past the frontier, which
        is outside the reachable set by construction.

    Returns a bool mask over ``[N, H]`` selecting ``range(min(accept + 1, H))`` for the blocks the
    chain actually visits, plus bookkeeping.
    """
    keep = block_keep_mask[0].bool()
    pos = anchor_positions[0]
    n_blocks, horizon = valid.shape
    row_of = {}
    for row in range(n_blocks):
        if bool(keep[row]):
            row_of[int(pos[row])] = row
    if not row_of:
        return None, {"chain_blocks": 0, "chain_missing": 0, "chain_accepts": []}

    valid_l = valid.tolist()
    ok_l = policy_ok.tolist()
    mask = [[False] * horizon for _ in range(n_blocks)]
    ordered = sorted(row_of)
    cursor = ordered[0]
    last = ordered[-1]
    blocks = missing = 0
    accepts = []
    guard = 0
    while cursor <= last and guard <= 4 * len(ordered):
        guard += 1
        row = row_of.get(cursor)
        if row is None:
            # No anchor was sampled at this position: only possible when anchor coverage is partial.
            missing += 1
            nxt = [p for p in ordered if p > cursor]
            if not nxt:
                break
            cursor = nxt[0]
            continue
        accept = 0
        while accept < horizon and valid_l[row][accept] and ok_l[row][accept]:
            accept += 1
        for h in range(min(accept + 1, horizon)):
            if valid_l[row][h]:
                mask[row][h] = True
        blocks += 1
        accepts.append(accept)
        cursor = cursor + accept + 1
    return mask, {"chain_blocks": blocks, "chain_missing": missing, "chain_accepts": accepts}


def slot_stratify(table, per_slot, pop_mask):
    """Per-slot ladder plus the destroy exposure that any slot-dependent rule has to pay for."""
    tic = per_slot["target_is_candidate"]
    trank = per_slot["target_rank"]
    fixable = pop_mask & tic & trank.ne(0)
    base_right = pop_mask & tic & trank.eq(0)
    horizon = fixable.shape[-1]
    for h in range(horizon):
        f = fixable[..., h]
        br = base_right[..., h]
        n = int(f.sum())
        if not n and not int(br.sum()):
            continue
        row = table.setdefault(h, [0, 0, 0, 0, 0, 0, 0])
        row[0] += n
        row[1] += int((f & per_slot["alternative_rank"][..., h].eq(0)).sum())
        row[2] += int((f & per_slot["selector_rank"][..., h].eq(0)).sum())
        row[3] += int((f & per_slot["pick_index"][..., h].eq(trank[..., h])).sum())
        row[4] += int(pop_mask[..., h].sum())
        row[5] += int(br.sum())
        row[6] += int((br & per_slot["pick_index"][..., h].ne(0)).sum())


@torch.no_grad()
def run_sample(model, draft, selector, sample, args, device, dtype, stats, hists, slot_table,
               chain_info, equivalence, decisions=None, features=None, sample_index=0):
    input_ids = sample["input_ids"].to(device)
    loss_mask = sample["loss_mask"].to(device)
    hidden_states = sample["hidden_states"].to(device=device, dtype=dtype)
    bsz, seq_len = input_ids.shape

    anchor_positions, block_keep_mask, output_hidden = model._forward_draft_blocks(
        input_ids=input_ids,
        hidden_states=hidden_states,
        loss_mask=loss_mask,
    )

    # --- dflash_family_model.py:1649-1690, verbatim algebra -------------------------------------
    block = model.block_size
    label_offsets = torch.arange(0, block, device=device).view(1, 1, -1)
    label_indices = anchor_positions.unsqueeze(-1) + label_offsets
    valid_label_mask = label_indices < seq_len
    safe_label_indices = label_indices.clamp(max=seq_len - 1)
    target_ids = torch.gather(
        input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1), 2, safe_label_indices
    )
    predecessor_ids = torch.cat([target_ids[:, :, :1], target_ids[:, :, :-1]], dim=-1)
    weight_mask = block_keep_mask.unsqueeze(-1).expand(-1, -1, block).float()
    weight_mask = weight_mask * valid_label_mask.float()
    pos_in_block = torch.arange(block, device=device).view(1, 1, -1)
    weight_mask = weight_mask * (pos_in_block > 0).float()
    weight_mask = weight_mask * torch.gather(
        loss_mask.unsqueeze(1).expand(-1, anchor_positions.size(1), -1), 2, safe_label_indices
    )
    hidden_4d = output_hidden.reshape(bsz, anchor_positions.shape[1], block, -1)

    n_blocks = hidden_4d.shape[1]
    step = max(1, args.chunk_blocks)
    parts = []
    for start in range(0, n_blocks, step):
        stop = min(start + step, n_blocks)
        h = hidden_4d[:, start:stop]
        t = target_ids[:, start:stop]
        p = predecessor_ids[:, start:stop]
        w = weight_mask[:, start:stop]

        logits = model.lm_head(h.reshape(h.shape[0], -1, h.shape[-1])).reshape(
            *h.shape[:3], -1
        )
        objective_logits = draft.transform_unary_logits(logits)

        # selector_excludes_anchor is True for ClozeCorrector (pspr_cloze.py:124)
        objective_logits = objective_logits[..., 1:, :]
        h = h[..., 1:, :]
        t = t[..., 1:]
        p = p[..., 1:]
        w = w[..., 1:]

        unary_logits, candidate_ids, lattice_scalars = selector.extract_lattice(objective_logits)
        scores, err_logits = selector.score_candidates(
            candidate_ids=candidate_ids,
            unary_logits=unary_logits,
            hidden_states=h,
            predecessor_ids=p,
            lattice_scalars=lattice_scalars,
            return_err=True,
        )
        reference_pick = selector.select_margin_gate(
            scores,
            err_logits=err_logits,
            rho=args.gate_rho,
            tau=args.gate_tau,
            theta=args.gate_theta,
        ).squeeze(-1)

        target_matches = candidate_ids.eq(t.unsqueeze(-1))
        target_is_candidate = target_matches.any(dim=-1)
        target_rank = target_matches.long().argmax(dim=-1)

        # stable descending argsort, matching decode_lattice.py:697
        order = torch.argsort(scores.float(), dim=-1, descending=True, stable=True)
        inverse = torch.zeros_like(order)
        inverse.scatter_(
            -1, order, torch.arange(order.shape[-1], device=device).expand_as(order)
        )
        selector_rank = inverse.gather(-1, target_rank.unsqueeze(-1)).squeeze(-1)
        # rank among the 15 alternatives == selector rank minus 1 iff KEEP outranks the truth
        base_ahead = inverse[..., 0] < selector_rank
        alternative_rank = selector_rank - base_ahead.long()

        # Sufficient statistic of the deployed gate at tau=0.  The rule
        # `softmax(scores)[1:].max() > rho * softmax(scores)[0]` is monotone in
        # `d = scores[best_alt] - scores[0]` because the softmax normaliser cancels, so
        # `override <=> d > log rho`.  Recording d (plus whether the best alternative IS the truth,
        # and whether the base is right) makes the exact (recovered, wrong_override, destroy) triple
        # available for ANY per-slot threshold from a single forward.
        log_probs_g = torch.log_softmax(scores.float(), dim=-1)
        base_logprob = log_probs_g[..., 0]
        best_alt_logprob, best_alt_offset = log_probs_g[..., 1:].max(dim=-1)
        gate_margin = best_alt_logprob - base_logprob
        best_alt_is_truth = target_is_candidate & (best_alt_offset + 1).eq(target_rank)

        # Variable-threshold gate.  `override <=> gate_margin > log rho_eff` is algebraically the
        # deployed rule at tau=0; `equivalence_mismatch` counts fp32 tie disagreements against the
        # library implementation so a silent divergence cannot masquerade as a gain (gate G6).
        log_rho = threshold_field(args, gate_margin, base_logprob, device)
        use_alt = gate_margin > log_rho
        pick_index = torch.where(
            use_alt, best_alt_offset + 1, torch.zeros_like(best_alt_offset)
        )
        if args.rho_mode == "global":
            equivalence[0] += int((pick_index != reference_pick).sum())
            equivalence[1] += int(pick_index.numel())
        else:
            equivalence[1] += int(pick_index.numel())

        if features is not None:
            # Frozen-trunk feature dump for the score-head headroom probe.  `z` and `S` are the two
            # learned representations the score head consumes; dumping them lets a NEW per-candidate
            # head be fitted offline on top of the existing trunk, which lower-bounds what an
            # architecture change to the head alone could buy -- at minutes rather than hours.
            z = selector.cloze_states(
                h, candidate_ids, p[..., 0], unary_logits, lattice_scalars
            )
            state = selector.causal_states(
                torch.nn.functional.embedding(p, selector._embedding())
            )
            sel_f = (w > 0)
            if args.dump_hidden:
                # The deployed head reads [LN(h); LN(z); LN(S)], so a residual probe that only sees
                # z and S is structurally unable to learn anything that needs h -- which makes a
                # negative probe result uninterpretable.  h is 2560-wide, hence the separate flag.
                features["h"].append(h[sel_f].to(torch.float16).cpu())
            features["z"].append(z[sel_f].to(torch.float16).cpu())
            features["state"].append(state[sel_f].to(torch.float16).cpu())
            features["lp"].append(unary_logits[sel_f].to(torch.float16).cpu())
            features["scores"].append(scores[sel_f].to(torch.float16).cpu())
            features["scalars"].append(lattice_scalars[sel_f].to(torch.float16).cpu())
            features["cand"].append(candidate_ids[sel_f].to(torch.int32).cpu())
            features["target_rank"].append(
                torch.where(target_is_candidate, target_rank,
                            torch.full_like(target_rank, -1))[sel_f].to(torch.int8).cpu()
            )
            slot_idx = torch.arange(sel_f.shape[-1], device=device).expand_as(sel_f)
            features["slot"].append(slot_idx[sel_f].to(torch.int16).cpu())
            # Global block id within the sample.  Needed offline to rebuild the frontier: policy_ok
            # chains along the slots of ONE block, so the slot index alone cannot group them.
            blk_idx = (
                torch.arange(sel_f.shape[1], device=device).view(1, -1, 1).expand_as(sel_f) + start
            )
            features["block"].append(blk_idx[sel_f].to(torch.int32).cpu())
            # Absolute anchor token position.  Without it an offline simulator cannot chain blocks
            # (`cursor += accept + 1`), so it could not evaluate a rule that changes the accept
            # length -- which is every rule worth evaluating.
            anch = anchor_positions[:, start:stop].unsqueeze(-1).expand_as(sel_f)
            features["anchor"].append(anch[sel_f].to(torch.int32).cpu())
            features["pick_index"].append(pick_index[sel_f].to(torch.int8).cpu())
            features["sample"].append(
                torch.full((int(sel_f.sum()),), sample_index, dtype=torch.int32)
            )

        parts.append(
            dict(
                valid=w > 0,
                target_is_candidate=target_is_candidate,
                target_rank=target_rank,
                selector_rank=selector_rank,
                alternative_rank=alternative_rank,
                pick_index=pick_index,
                gate_margin=gate_margin,
                base_logprob=base_logprob,
                best_alt_is_truth=best_alt_is_truth,
            )
        )

    per_slot = {k: torch.cat([p[k] for p in parts], dim=1) for k in parts[0]}
    valid = per_slot.pop("valid")
    base_top1_ok = per_slot["target_is_candidate"] & per_slot["target_rank"].eq(0)
    policy_ok = per_slot["target_is_candidate"] & per_slot["pick_index"].eq(
        per_slot["target_rank"]
    )

    pops = {
        "ALL": valid,
        "POLICY": frontier_mask(valid, policy_ok),
        "BASE": frontier_mask(valid, base_top1_ok),
    }
    chain_mask, info = (None, None)
    if args.chain:
        chain_mask, info = chain_population(
            anchor_positions, block_keep_mask, valid[0], policy_ok[0]
        )
    if chain_mask is not None:
        pops["CHAIN"] = torch.tensor(chain_mask, device=device).unsqueeze(0) & valid
        chain_info["blocks"] += info["chain_blocks"]
        chain_info["missing"] += info["chain_missing"]
        chain_info["accept_sum"] += sum(info["chain_accepts"])
        chain_info["anchors_available"] += int(block_keep_mask.sum())
    for name, mask in pops.items():
        accumulate(stats, hists, per_slot, mask, name)
    slot_stratify(slot_table["POLICY"], per_slot, pops["POLICY"])
    if "CHAIN" in pops:
        slot_stratify(slot_table["CHAIN"], per_slot, pops["CHAIN"])
    if decisions is not None:
        for name in decisions:
            if name not in pops:
                continue
            m = pops[name]
            tic = per_slot["target_is_candidate"]
            trank = per_slot["target_rank"]
            base_right = tic & trank.eq(0)
            fixable = tic & trank.ne(0)
            kind = torch.full_like(trank, 3)
            kind = torch.where(fixable & per_slot["best_alt_is_truth"],
                               torch.ones_like(kind), kind)
            kind = torch.where(fixable & ~per_slot["best_alt_is_truth"],
                               torch.full_like(kind, 2), kind)
            kind = torch.where(base_right, torch.zeros_like(kind), kind)
            slot_idx = torch.arange(m.shape[-1], device=m.device).expand_as(m)
            sel = m
            decisions[name]["sample"].append(
                torch.full((int(sel.sum()),), sample_index, dtype=torch.int32)
            )
            decisions[name]["slot"].append(slot_idx[sel].to(torch.int16).cpu())
            decisions[name]["kind"].append(kind[sel].to(torch.int8).cpu())
            decisions[name]["margin"].append(per_slot["gate_margin"][sel].float().cpu())
            decisions[name]["base_lp"].append(per_slot["base_logprob"][sel].float().cpu())
            decisions[name]["truth_top_alt"].append(
                (per_slot["alternative_rank"][sel].eq(0) & tic[sel]).cpu()
            )


def pct(num, den):
    return float(num) / float(den) * 100.0 if den else float("nan")


def report(stats, hists, slot_table, chain_info, args):
    out = {"arguments": vars(args), "populations": {}, "slot_strata": {},
           "chain": dict(chain_info)}
    for pop in POPULATIONS:
        s = {k: int(v.item()) for k, v in stats[pop].items()}
        if not s["decision_n"]:
            continue
        h = {k: v.tolist() for k, v in hists[pop].items()}
        closes_a = s["base_right_n"] + s["base_wrong_n"] == s["decision_n"]
        closes_b = s["unfixable_n"] + s["fixable_n"] == s["base_wrong_n"]
        closes_c = (
            s["fix_recovered"] + s["fix_kept_wrong"] + s["fix_wrong_override"] == s["fixable_n"]
        )
        fx = max(s["fixable_n"], 1)
        derived = {
            "coverage": s["covered_n"] / max(s["decision_n"], 1),
            "base_right_rate": s["base_right_n"] / max(s["decision_n"], 1),
            "base_right_rate_given_covered": s["base_right_n"] / max(s["covered_n"], 1),
            "destroy_rate": s["base_right_destroyed"] / max(s["base_right_n"], 1),
            "alt_rank_0": h["alternative_rank"][0] / fx,
            "selector_rank_0": h["selector_rank"][0] / fx,
            "draft_rank_1": h["draft_rank"][1] / fx,
            "recovered": s["fix_recovered"] / fx,
            "gate_blocked": s["fix_gate_blocked_true_best"] / fx,
            "wrong_override": s["fix_wrong_override"] / fx,
            "accounting_closes": bool(closes_a and closes_b and closes_c),
        }
        out["populations"][pop] = {"counters": s, "hists": h, "derived": derived}

        print(f"\n=== population {pop} ===")
        print(f"  decision_n {s['decision_n']}   covered {s['covered_n']} "
              f"({pct(s['covered_n'], s['decision_n']):.2f}%)")
        print(f"  base-right kept        {s['base_right_kept']}/{s['base_right_n']} = "
              f"{pct(s['base_right_kept'], s['base_right_n']):.2f}%   "
              f"destroy {s['base_right_destroyed']} "
              f"({pct(s['base_right_destroyed'], s['base_right_n']):.3f}%)")
        print(f"  base-wrong fixable     {s['fixable_n']}   unfixable {s['unfixable_n']}")
        print(f"    draft alt-rank-1     {h['draft_rank'][1]} = "
              f"{pct(h['draft_rank'][1], fx):.2f}%   <- truth is the draft's best alternative")
        print(f"    alternative_rank_0   {h['alternative_rank'][0]} = "
              f"{pct(h['alternative_rank'][0], fx):.2f}%   <- selector ranks truth 1st of 15")
        print(f"    selector_rank_0      {h['selector_rank'][0]} = "
              f"{pct(h['selector_rank'][0], fx):.2f}%   <- still 1st with KEEP in")
        print(f"    recovered            {s['fix_recovered']} = "
              f"{pct(s['fix_recovered'], fx):.2f}%")
        print(f"      gate blocked true best {s['fix_gate_blocked_true_best']} = "
              f"{pct(s['fix_gate_blocked_true_best'], fx):.2f}%")
        print(f"      wrong override         {s['fix_wrong_override']} = "
              f"{pct(s['fix_wrong_override'], fx):.2f}%")
        print(f"  accounting closes: {derived['accounting_closes']}")

    if chain_info["blocks"]:
        mean_accept = chain_info["accept_sum"] / chain_info["blocks"]
        out["chain"]["mean_accept_slots"] = mean_accept
        print(f"\n=== chain simulation ===")
        print(f"  blocks {chain_info['blocks']}  missing-anchor jumps {chain_info['missing']}  "
              f"anchors available {chain_info['anchors_available']}")
        print(f"  mean accepted slots per block {mean_accept:.3f}   "
              f"(+1 bonus token => accept length {mean_accept + 1:.3f})")

    for pop in ("POLICY", "CHAIN"):
        table = slot_table.get(pop) or {}
        if not table:
            continue
        print(f"\n=== {pop} population by within-block slot index ===")
        print(f"  {'slot':>4} {'decis':>7} {'fixable':>8} {'fix%':>6} {'alt_r0':>8} {'sel_r0':>8} "
              f"{'recov':>8} {'baseOK':>7} {'destroy':>8}")
        out["slot_strata"][pop] = {}
        for h in sorted(table):
            n, a, s0, r, dec, br, de = table[h]
            out["slot_strata"][pop][h] = {
                "fixable": n, "alt_rank_0": a, "selector_rank_0": s0, "recovered": r,
                "decision_n": dec, "base_right_n": br, "base_right_destroyed": de,
            }
            print(f"  {h:>4} {dec:>7} {n:>8} {pct(n, dec):>5.1f}% {pct(a, n):>7.2f}% "
                  f"{pct(s0, n):>7.2f}% {pct(r, n):>7.2f}% {pct(br, dec):>6.1f}% "
                  f"{pct(de, br):>7.3f}%")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--draft-config", default=str(SPECFORGE / "configs/qwen3-4b-pspr-cloze.json"))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--target-model",
                    default="/kl_infra_infer_intern/wangzhuoyu/TAPS-SP/models/Qwen3-4B")
    ap.add_argument("--num-samples", type=int, default=40)
    ap.add_argument("--num-anchors", type=int, default=512)
    ap.add_argument("--chunk-blocks", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=3072)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--gate-rho", type=float, default=3.0)
    ap.add_argument("--gate-tau", type=float, default=0.0)
    ap.add_argument("--gate-theta", type=float, default=0.0)
    ap.add_argument("--attention-backend", default="flex_attention")
    ap.add_argument("--gate-training-corpus", action="store_true",
                    help="enforce G3 against the constants recorded in dflash_family_model.py")
    ap.add_argument("--chain", action="store_true",
                    help="also report block chaining on these fixed teacher-prefix features, "
                         "not an exact replay of real decode. Requires every visited anchor, "
                         "so use --num-anchors >= the supervised token count (G5 checks it).")
    ap.add_argument("--gate-tol", type=float, default=0.02)
    ap.add_argument("--json-output", default=None)
    ap.add_argument("--preserve-selector-fp32", action="store_true",
                    help="restore exact checkpoint FP32 selector tensors after the wrapper's bf16 "
                         "cast; required for current SlotDeep diagnostics, opt-in for legacy cloze")
    ap.add_argument("--per-sample-diagnostics", action="store_true",
                    help="save fixed-BASE counters and source hashes per feature sequence for paired "
                         "prompt-level comparisons; this adds no training or decoding actions")
    ap.add_argument("--rho-mode", default="global",
                    choices=("global", "per_slot", "affine_slot", "affine_base", "affine_both"))
    ap.add_argument("--rho-values", default="",
                    help="comma list: 15 rho values for per_slot, else the affine coefficients "
                         "in LOG-rho space")
    ap.add_argument("--dump-hidden", action="store_true",
                    help="also dump the 2560-wide backbone hidden h, which the deployed score head "
                         "reads; without it a frozen-feature probe cannot test h's contribution")
    ap.add_argument("--dump-features", default=None,
                    help="npz of frozen trunk features (z, GRU state, log-probs, current scores, "
                         "lattice scalars, candidate ids, target rank) over the ALL population, for "
                         "fitting a replacement score head offline.")
    ap.add_argument("--dump-decisions", default=None,
                    help="npz of per-decision (sample, slot, kind, gate margin d, base logprob) for "
                         "the POLICY and CHAIN populations. kind: 0=base_right 1=fixable&best-alt-is"
                         "-truth 2=fixable&best-alt-wrong 3=unfixable. Enables exact (recovered, "
                         "wrong_override, destroy) at ANY per-slot threshold with no extra forward.")
    ap.add_argument("--include-base-decisions", action="store_true",
                    help="also dump fixed-BASE decisions for matched-population risk/repair diagnostics")
    args = ap.parse_args()
    if args.include_base_decisions and not args.dump_decisions:
        ap.error("--include-base-decisions requires --dump-decisions")
    if args.gate_tau != 0.0 or args.gate_theta != 0.0:
        ap.error("this log-margin diagnostic supports only --gate-tau 0 --gate-theta 0; "
                 "nonzero values require complete deployed gate semantics")
    if args.per_sample_diagnostics and args.json_output and Path(args.json_output).exists():
        raise FileExistsError(f"refusing to overwrite paired diagnostic: {args.json_output}")
    input_provenance = ({
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "draft_config_sha256": file_sha256(args.draft_config),
        "diagnostic_source_sha256": file_sha256(__file__),
    } if args.per_sample_diagnostics else None)

    device, dtype = torch.device("cuda"), torch.bfloat16
    draft, cfg = build_draft(args.draft_config, args.checkpoint, device, dtype)
    if type(draft).__name__ == "PSPRSlotDeepDraftModel" and not args.preserve_selector_fp32:
        raise ValueError("SlotDeep diagnostics require --preserve-selector-fp32")
    saved_selector = ({k: v.detach().cpu().clone() for k, v in draft.candidate_selector.state_dict().items()
                       if k != "target_embedding"} if args.preserve_selector_fp32 else None)
    parts = TargetEmbeddingsAndHead.from_pretrained(
        args.target_model, device="cuda", dtype=dtype
    )
    draft.bind_target_decoder(parts.embed_tokens)
    draft.apply_backbone_freeze()
    model = (
        OnlineDFlashModel(
            draft_model=draft,
            target_lm_head=parts.lm_head,
            target_embed_tokens=parts.embed_tokens,
            mask_token_id=cfg.dflash_config["mask_token_id"],
            block_size=cfg.block_size,
            attention_backend=args.attention_backend,
            num_anchors=args.num_anchors,
            loss_decay_gamma=None,
            objective_chunk_blocks=args.chunk_blocks,
        )
        .to(device=device, dtype=dtype)
        .eval()
    )
    # model_providers.py:397-399 -- must come AFTER the parent cast.
    draft.enforce_selector_compute_dtype()
    selector = draft.candidate_selector
    if saved_selector is not None:
        selector.load_state_dict(saved_selector, strict=True)
        assert all(torch.equal(selector.state_dict()[k].detach().cpu(), v)
                   for k, v in saved_selector.items()), "selector checkpoint precision was not preserved"
        del saved_selector
        print("selector checkpoint fidelity: FP32 tensors restored exactly after wrapper cast", flush=True)

    gates = {}
    gates["G1 selector is trained (|delta_w2| > 0)"] = bool(
        selector.delta_w2.weight.abs().max().item() > 0
    )
    gates["G2 selector computes in fp32"] = selector.delta_w2.weight.dtype == torch.float32

    files = list_feature_files(args.features)[: args.num_samples]
    if not files:
        raise SystemExit(f"no feature files under {args.features}")
    print(f"features: {len(files)} samples from {args.features}", flush=True)

    stats, hists = new_stats(device), new_hists(device)
    slot_table = {"POLICY": {}, "CHAIN": {}}
    chain_info = {"blocks": 0, "missing": 0, "accept_sum": 0, "anchors_available": 0}
    equivalence = [0, 0]
    feature_keys = ("z", "state", "lp", "scores", "scalars", "cand", "target_rank", "slot",
                    "block", "anchor", "pick_index", "sample")
    if args.dump_hidden:
        feature_keys = ("h",) + feature_keys
    features = {k: [] for k in feature_keys} if args.dump_features else None
    keys = ("sample", "slot", "kind", "margin", "base_lp", "truth_top_alt")
    decisions = (
        {pop: {k: [] for k in keys} for pop in
         (("POLICY", "CHAIN", "BASE") if args.include_base_decisions else ("POLICY", "CHAIN"))}
        if args.dump_decisions else None
    )
    tic = time.time()
    sample_base_records = []
    for i, path in enumerate(files):
        feature_sha = file_sha256(path) if args.per_sample_diagnostics else None
        sample = process_offline_dflash_sample(
            torch.load(path, map_location="cpu"), args.max_length
        )
        torch.manual_seed(args.seed + i)
        before_base = (torch.stack([stats["BASE"][k] for k in COUNTERS]).cpu().tolist()
                       if args.per_sample_diagnostics else None)
        run_sample(model, draft, selector, sample, args, device, dtype, stats, hists, slot_table,
                   chain_info, equivalence, decisions=decisions, features=features,
                   sample_index=i)
        if before_base is not None:
            after_base = torch.stack([stats["BASE"][k] for k in COUNTERS]).cpu().tolist()
            sample_base_records.append({
                "sample_index": i, "feature_file": str(path), "feature_sha256": feature_sha,
                "base_counts": {k: int(a - b) for k, a, b in zip(COUNTERS, after_base, before_base)},
            })
        if (i + 1) % 5 == 0 or i + 1 == len(files):
            done = int(stats["ALL"]["decision_n"].item())
            print(f"  [{i + 1}/{len(files)}] ALL slots={done} "
                  f"({time.time() - tic:.0f}s)", flush=True)

    payload = report(stats, hists, slot_table, chain_info, args)
    if args.per_sample_diagnostics:
        payload["input_provenance"] = input_provenance
        payload["first_base_error_per_sample"] = sample_base_records
    # BASE's wrong slots contain at most ONE first-base-error per sampled block.
    # Its denominator does not change when the selector policy changes (fixed
    # backbone, feature sequences and anchor seed). This is teacher-prefix
    # diagnostic evidence, not end-to-end decoding or the POLICY population.
    base_stats = payload["populations"]["BASE"]["counters"]
    ratio = lambda numerator, denominator: numerator / denominator if denominator else None
    payload["first_base_error_diagnostic"] = {
        "population": "fixed sampled teacher-prefix anchors; one first-base-error per non-perfect block",
        "first_errors": base_stats["base_wrong_n"],
        "first_errors_in_topk": base_stats["fixable_n"],
        "repaired": base_stats["fix_recovered"],
        "repair_all": ratio(base_stats["fix_recovered"], base_stats["base_wrong_n"]),
        "repair_given_topk": ratio(base_stats["fix_recovered"], base_stats["fixable_n"]),
        "topk_coverage": ratio(base_stats["fixable_n"], base_stats["base_wrong_n"]),
        "base_prefix_right": base_stats["base_right_n"],
        "base_prefix_destroyed": base_stats["base_right_destroyed"],
        "teacher_prefix_destroy_rate": ratio(base_stats["base_right_destroyed"], base_stats["base_right_n"]),
        "note": "Not a task-answer metric or a rollout; corpus/feature labels and anchor distribution are fixed.",
    }

    all_derived = payload["populations"]["ALL"]["derived"]
    if args.gate_training_corpus:
        # cloze@7110 training telemetry, PSPR_CLOZE.log: train/selector_coverage = 0.8312.
        # This is a DIAGNOSTIC, not a correctness gate: training captured hidden states through
        # patched SGLang while this dump uses plain HF, and decode also uses HF, so a residual gap
        # here is expected and is the price of matching the decode-side numerics.
        gates["G3 coverage ~= 0.8312 (PSPR_CLOZE.log step7110)"] = (
            abs(all_derived["coverage"] - 0.8312) <= args.gate_tol
        )
    for pop in payload["populations"]:
        gates[f"G4 accounting closes in {pop}"] = payload["populations"][pop]["derived"][
            "accounting_closes"
        ]
    if args.rho_mode == "global":
        gates["G6 d-form gate == select_margin_gate"] = equivalence[0] == 0
        print(f"\n  gate equivalence mismatches: {equivalence[0]}/{equivalence[1]}")
    if chain_info["blocks"]:
        # Every position the chain lands on must have been sampled as an anchor, else the simulated
        # chain silently skips forward and the CHAIN population is not decode's.
        gates["G5 chain never skipped a missing anchor"] = chain_info["missing"] == 0

    print("\n=== gates ===")
    for name, ok in gates.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    payload["gates"] = gates
    ok = all(gates.values())

    if features is not None:
        import numpy as np

        dest = Path(args.dump_features)
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez(dest, files=np.array([str(f) for f in files]),
                 **{k: torch.cat(v).numpy() for k, v in features.items()})
        print(f"wrote {dest}")

    if decisions is not None:
        import numpy as np

        arrays = {}
        for pop, cols in decisions.items():
            if not cols["slot"]:
                continue
            for name, chunks in cols.items():
                arrays[f"{pop}_{name}"] = torch.cat(chunks).numpy()
        dest = Path(args.dump_decisions)
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest, files=np.array([str(f) for f in files]), **arrays)
        print(f"wrote {dest} ({len(arrays)} arrays)")

    if args.json_output:
        dest = Path(args.json_output)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {dest}")
    print(f"\n{'ALL GATES PASS' if ok else 'GATE FAILURE'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
