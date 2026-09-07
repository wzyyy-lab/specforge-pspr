#!/usr/bin/env python3
"""Gate for ``dflash2_selector_weight_mode``.

The switch exists because stage-1 and stage-2 were silently optimising different objectives. Stage-1
(``scripts/train_pspr_accept_selector.py``) applies a uniform mean CE over covered slots. Stage-2
took ``selector_loss_weights = loss_weights * target_is_candidate``, and under any D-PACE
``loss_type`` ``loss_weights`` is ``weight_mask * dpace_weights`` -- so the moment a warm start was
loaded, the objective its weights were trained under changed. Four properties:

W1  ``base_dpace`` is byte-identical to the previous behaviour. This is the regression guard: the
    default must not move, or every earlier run and the joint-equivalence gate become incomparable.

W2  ``uniform`` weights by ``weight_mask`` instead. Checked against an independent re-derivation
    rather than against the implementation's own intermediate.

W3  ``uniform`` + ``own_denominator`` reproduces stage-1's loss EXACTLY -- the same number
    ``F.cross_entropy(sc[m], ti[m])`` returns for ``m = ti >= 0``. This is the property the whole
    change is for; W1 and W2 are only necessary conditions for it.

W4  The switch actually bites: with ``loss_weights != weight_mask`` the two modes must disagree.
    Without this a no-op wiring bug would pass W1-W3 (they would all compare equal to each other).

Usage:
    PYTHONPATH=. python scripts/gate_pspr_selector_weight_mode.py \
        --backbone /kl_infra_infer_intern/wangzhuoyu/TAPS-SP/models/Qwen3-4B-DFlash-b16 \
        --target-model /kl_infra_infer_intern/wangzhuoyu/TAPS-SP/models/Qwen3-4B
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

SPECFORGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPECFORGE))

from specforge.algorithms.common.dflash_family_model import (  # noqa: E402
    OnlineDFlashModel,
)
from specforge.modeling.draft.pspr import PSPRDraftModel  # noqa: E402
from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead  # noqa: E402
from specforge.training.model_loading import warm_start_draft_model  # noqa: E402


def build_draft(draft_config, backbone, device, dtype):
    from transformers import Qwen3Config

    raw = json.load(open(draft_config))
    cfg = Qwen3Config(**raw)
    cfg.dflash_config = raw["dflash_config"]
    cfg.block_size = raw["block_size"]
    cfg.num_target_layers = raw["num_target_layers"]
    cfg._attn_implementation = "flex_attention"
    torch.manual_seed(0)
    with torch.device("cpu"):
        draft = PSPRDraftModel(cfg)
    # A pretrained backbone is mandatory: on random weights the target almost never lands in the
    # top-16, coverage collapses to zero and every weighted sum below becomes 0 == 0.
    warm_start_draft_model(
        draft,
        backbone,
        draft_config=cfg,
        strategy="offline",
        allow_missing_embedding=True,
    )
    return draft.to(device=device, dtype=dtype).eval(), cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft-config", default=str(SPECFORGE / "configs/qwen3-4b-pspr-joint.json"))
    ap.add_argument(
        "--backbone",
        default="/kl_infra_infer_intern/wangzhuoyu/TAPS-SP/models/Qwen3-4B-DFlash-b16",
    )
    ap.add_argument(
        "--target-model",
        default="/kl_infra_infer_intern/wangzhuoyu/TAPS-SP/models/Qwen3-4B",
    )
    args = ap.parse_args()

    device, dtype = torch.device("cuda"), torch.bfloat16
    draft, cfg = build_draft(args.draft_config, args.backbone, device, dtype)
    parts = TargetEmbeddingsAndHead.from_pretrained(
        args.target_model, device="cuda", dtype=dtype
    )
    draft.bind_target_decoder(parts.embed_tokens)
    selector = draft.candidate_selector

    def build(weight_mode):
        return (
            OnlineDFlashModel(
                draft_model=draft,
                target_lm_head=parts.lm_head,
                target_embed_tokens=parts.embed_tokens,
                mask_token_id=cfg.dflash_config["mask_token_id"],
                block_size=cfg.block_size,
                attention_backend="flex_attention",
                num_anchors=64,
                loss_decay_gamma=None,
                selector_loss_alpha=1.0,
                selector_err_loss_alpha=1.0,
                selector_own_denominator=True,
                selector_weight_mode=weight_mode,
            )
            .to(device=device, dtype=dtype)
            .eval()
        )

    torch.manual_seed(11)
    b, n, block = 2, 3, cfg.block_size
    vocab = cfg.vocab_size
    hidden = torch.randn(b, n, block, cfg.hidden_size, device=device, dtype=dtype) * 0.05
    with torch.no_grad():
        logits = parts.lm_head(hidden.reshape(b, n * block, -1)).reshape(b, n, block, -1)
    logits = logits.float()
    top_log_probs, candidate_ids, _ = selector.extract_lattice(logits)

    regime = torch.randint(0, 3, (b, n, block), device=device)
    pick = torch.randint(1, selector.top_k, (b, n, block), device=device)
    target_ids = torch.where(
        regime == 0,
        candidate_ids[..., 0],
        torch.where(
            regime == 1,
            candidate_ids.gather(-1, pick.unsqueeze(-1)).squeeze(-1),
            torch.randint(0, vocab, (b, n, block), device=device),
        ),
    )
    predecessor_ids = torch.cat([target_ids[:, :, :1], target_ids[:, :, :-1]], dim=-1)

    weight_mask = (torch.rand(b, n, block, device=device) > 0.1).float()
    pos_in_block = torch.arange(block, device=device).view(1, 1, -1)
    weight_mask = weight_mask * (pos_in_block > 0).float()
    # Stand in for a D-PACE reweighting: strictly positive, never equal to the mask, so a mode that
    # reads the wrong tensor cannot coincidentally agree.
    dpace_like = 0.25 + 1.5 * torch.rand(b, n, block, device=device)
    loss_weights = weight_mask * dpace_like

    def terms_for(weight_mode):
        with torch.no_grad():
            return build(weight_mode)._selector_chunk_terms(
                candidate_selector=selector,
                objective_logits=logits,
                hidden=hidden,
                target_ids=target_ids,
                predecessor_ids=predecessor_ids,
                loss_weights=loss_weights,
                weight_mask=weight_mask,
            )

    dpace_terms = terms_for("base_dpace")
    uniform_terms = terms_for("uniform")

    with torch.no_grad():
        scores = selector.score_candidates(
            candidate_ids=candidate_ids,
            unary_logits=top_log_probs,
            hidden_states=hidden,
            predecessor_ids=predecessor_ids,
            lattice_scalars=selector.extract_lattice(logits)[2],
        )
        matches = candidate_ids.eq(target_ids.unsqueeze(-1))
        is_candidate = matches.any(dim=-1)
        target_index = matches.long().argmax(dim=-1)
        per_slot_ce = F.cross_entropy(
            scores.float().reshape(-1, scores.shape[-1]),
            target_index.reshape(-1),
            reduction="none",
        ).reshape_as(target_ids)
        covered = is_candidate.float()
        ref_dpace_num = (per_slot_ce * loss_weights * covered).sum()
        ref_uniform_num = (per_slot_ce * weight_mask * covered).sum()
        ref_uniform_den = (weight_mask * covered).sum()

        # Stage-1's objective, transcribed: mean CE over covered slots, no position weighting.
        # `weight_mask` zeroes pos_in_block == 0, which stage-1's trace data never supervises
        # either, so the comparable population is `covered & weight_mask > 0`.
        stage1_slots = (covered > 0) & (weight_mask > 0)
        stage1_loss = F.cross_entropy(
            scores.float()[stage1_slots], target_index[stage1_slots], reduction="mean"
        )

    failures = []

    gap_w1 = (dpace_terms.ce_num - ref_dpace_num).abs().item()
    if gap_w1 == 0.0:
        print(f"W1 PASS  base_dpace still weights by loss_weights (|delta| = {gap_w1:.3e})")
    else:
        failures.append(f"W1 FAIL  base_dpace ce_num drifted by {gap_w1:.3e}")

    gap_w2 = (uniform_terms.ce_num - ref_uniform_num).abs().item()
    if gap_w2 == 0.0:
        print(f"W2 PASS  uniform weights by weight_mask (|delta| = {gap_w2:.3e})")
    else:
        failures.append(f"W2 FAIL  uniform ce_num off by {gap_w2:.3e}")

    uniform_loss = (uniform_terms.ce_num / ref_uniform_den).item()
    gap_w3 = abs(uniform_loss - stage1_loss.item())
    if gap_w3 <= 1e-6:
        print(
            f"W3 PASS  uniform + own_denominator == stage-1 mean CE "
            f"({uniform_loss:.8f} vs {stage1_loss.item():.8f}, |delta| = {gap_w3:.2e})"
        )
    else:
        failures.append(
            f"W3 FAIL  uniform loss {uniform_loss:.8f} != stage-1 {stage1_loss.item():.8f} "
            f"(|delta| = {gap_w3:.3e})"
        )

    separation = (dpace_terms.ce_num - uniform_terms.ce_num).abs().item()
    if separation > 1e-3:
        print(f"W4 PASS  the two modes disagree as they must (|delta| = {separation:.3e})")
    else:
        failures.append(
            f"W4 FAIL  modes agree (|delta| = {separation:.3e}); the switch is not wired in"
        )

    print()
    if failures:
        for line in failures:
            print(line)
        print(f"RESULT: FAIL ({len(failures)} checks)")
        return 1
    print("RESULT: PASS (all checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
