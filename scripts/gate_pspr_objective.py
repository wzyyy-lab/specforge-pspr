#!/usr/bin/env python3
"""Objective gate for the PSPR selector inside SpecForge's DFlash objective.

Four properties, each of which would silently corrupt every later A/B if it broke:

1. **Base anchoring.** ``gamma`` is zero-init, so before any optimiser step the selector's scores
   must equal the base top-k log-probs bit-for-bit, and its argmax must be candidate 0. Acceptance
   is then structurally identical to plain one-shot DFlash, which is what makes any measured gain
   attributable.

2. **Frontier detector objective matches the reference.** The auxiliary BCE is re-derived here from
   the reference training script's own formulation, on the same tensors, and must agree exactly.
   This is the check that catches the SpecForge-specific trap: this objective zeroes
   ``weight_mask`` at ``pos_in_block == 0``, so a frontier scan run over the full block would hit an
   invalid slot immediately and mask out every row.

3. **Zero impact on selectors without a detector.** A selector that does not declare
   ``wants_err_objective`` must produce exactly zero for all three ``err_*`` terms, so the DFlash2
   path is bit-identical to its behaviour before the contract was widened.

4. **The gate can fail.** A perturbed delta output must break base anchoring.

Usage:
    PYTHONPATH=. python scripts/gate_pspr_objective.py \
        --features cache/hidden_states/sharegpt3k \
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
    frontier_mask,
)
from specforge.data.preprocessing import process_offline_dflash_sample  # noqa: E402
from specforge.modeling.draft.pspr import PSPRDraftModel  # noqa: E402
from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead  # noqa: E402
from specforge.runtime.data_plane.offline_reader import list_feature_files  # noqa: E402
from specforge.training.model_loading import warm_start_draft_model  # noqa: E402


def reference_err_loss(err_logits, target_index, is_candidate, weight_mask, dtype):
    """Re-derivation of the reference training script's detector loss.

    Transcribed from ``TAPS-SP/scripts/train_accept_selector.py``::

        valid = b["g"] >= 0
        fm = frontier_mask(valid, (ti == 0) & valid)
        y = (ti != 0).float()
        loss = err_w * F.binary_cross_entropy_with_logits(el[fm], y[fm])

    ``ti < 0`` there means "the target is not in the top-k", which maps to ``~is_candidate`` here.
    The reference runs on a compact ``[B, 15]`` horizon, so the SpecForge tensors are sliced to drop
    the structurally-invalid slot 0 and flattened to two dimensions before applying it verbatim.

    ``dtype`` is a parameter because the two sides reduce differently -- boolean-indexed ``mean``
    over the frontier subset versus a masked ``sum`` over the whole block, on GPU -- so they disagree
    at float32 rounding even when the logic is identical. Recomputing in float64 separates the two
    causes: a genuine difference in the mask or the label stays O(0.1) there, while a reduction-order
    artefact collapses to ~1e-16.
    """
    horizon = weight_mask.shape[-1] - 1
    ti = torch.where(is_candidate, target_index, torch.full_like(target_index, -1))
    ti = ti[..., 1:].reshape(-1, horizon)
    valid = (weight_mask[..., 1:] > 0).reshape(-1, horizon)
    el = err_logits[..., 1:].reshape(-1, horizon).to(dtype)

    fm = frontier_mask(valid, (ti == 0) & valid)
    if not bool(fm.any()):
        return None, 0
    y = (ti != 0).to(dtype)
    return (
        F.binary_cross_entropy_with_logits(el[fm], y[fm]).item(),
        int(fm.sum().item()),
    )


def _frontier_weights(weight_mask, is_candidate, target_index):
    """The frontier weights exactly as the objective builds them, for the float64 recomputation."""
    base_top1_ok = is_candidate & target_index.eq(0)
    reachable = frontier_mask(weight_mask[..., 1:] > 0, base_top1_ok[..., 1:])
    reachable = torch.cat([torch.zeros_like(reachable[..., :1]), reachable], dim=-1)
    return reachable.float()


class _DetectorlessSelector(torch.nn.Module):
    """Stand-in for the DFlash2 selector: a scorer with no ``wants_err_objective``."""

    top_k = 16

    def __init__(self, hidden_size):
        super().__init__()
        self.proj = torch.nn.Linear(hidden_size, self.top_k)

    def score_candidates(
        self, *, candidate_ids, unary_logits, hidden_states, predecessor_ids, **_
    ):
        return unary_logits + self.proj(hidden_states).float()


def build_draft(draft_config, device, dtype, backbone):
    from transformers import Qwen3Config

    raw = json.load(open(draft_config))
    # ``architectures`` is kept: ``warm_start_draft_model`` resolves the draft class from the config
    # it is handed, so stripping it makes the loader fail with "no registered architecture".
    cfg = Qwen3Config(**raw)
    cfg.dflash_config = raw["dflash_config"]
    cfg.block_size = raw["block_size"]
    cfg.num_target_layers = raw["num_target_layers"]
    cfg._attn_implementation = "flex_attention"
    torch.manual_seed(0)
    with torch.device("cpu"):
        draft = PSPRDraftModel(cfg)
    # The pretrained backbone is not optional here. On a randomly initialised backbone the target
    # almost never lands in the top-16, so selector coverage is 0 and the "selector reproduces the
    # base ranking" check becomes 0 == 0 -- it would pass whether the code works or not.
    report = warm_start_draft_model(
        draft,
        backbone,
        draft_config=cfg,
        strategy="offline",
        allow_missing_embedding=True,
    )
    print(f"warm start: {report.loaded_keys} tensors from {Path(backbone).name}")
    return draft.to(device=device, dtype=dtype), cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(SPECFORGE / "cache/hidden_states/sharegpt3k"))
    ap.add_argument("--draft-config", default=str(SPECFORGE / "configs/qwen3-4b-pspr.json"))
    ap.add_argument("--target-model", required=True)
    ap.add_argument(
        "--backbone",
        default="/kl_infra_infer_intern/wangzhuoyu/TAPS-SP/models/Qwen3-4B-DFlash-b16",
        help="pretrained DFlash backbone; required for the coverage-dependent checks to bite",
    )
    ap.add_argument("--num-samples", type=int, default=3)
    args = ap.parse_args()

    device, dtype = torch.device("cuda"), torch.bfloat16
    draft, cfg = build_draft(args.draft_config, device, dtype, args.backbone)
    parts = TargetEmbeddingsAndHead.from_pretrained(
        args.target_model, device="cuda", dtype=dtype
    )
    draft.bind_target_decoder(parts.embed_tokens)

    selector = draft.candidate_selector
    n_selector = sum(p.numel() for p in selector.parameters())
    n_all = sum(p.numel() for p in draft.parameters())
    print(
        f"params: total {n_all / 1e6:.2f}M | selector {n_selector / 1e6:.2f}M "
        f"| backbone {(n_all - n_selector) / 1e6:.2f}M"
    )
    print(f"        selector == dh2048 reference 27,059,213: {n_selector == 27059213}")

    def build_model(err_alpha, own_denominator=True):
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
                selector_err_loss_alpha=err_alpha,
                selector_own_denominator=own_denominator,
            )
            .to(device=device, dtype=dtype)
            .eval()
        )

    model = build_model(1.0)
    checks: dict[str, bool] = {"selector params match reference": n_selector == 27059213}

    # ---- 1 / 2: chunk-level terms on constructed tensors -------------------------------------
    torch.manual_seed(11)
    b, n, block = 2, 3, cfg.block_size
    vocab = cfg.vocab_size
    hidden = torch.randn(b, n, block, cfg.hidden_size, device=device, dtype=dtype) * 0.05
    with torch.no_grad():
        logits = parts.lm_head(hidden.reshape(b, n * block, -1)).reshape(b, n, block, -1)
    logits = logits.float()

    top_log_probs, candidate_ids, scalars = selector.extract_lattice(logits)
    # Mix targets so the three regimes all occur: base top-1 correct, repairable (in top-k but not
    # first) and unrepairable (outside the top-k entirely).
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

    with torch.no_grad():
        terms = model._selector_chunk_terms(
            candidate_selector=selector,
            objective_logits=logits,
            hidden=hidden,
            target_ids=target_ids,
            predecessor_ids=predecessor_ids,
            loss_weights=weight_mask,
            weight_mask=weight_mask,
        )

        matches = candidate_ids.eq(target_ids.unsqueeze(-1))
        is_candidate = matches.any(dim=-1)
        target_index = matches.long().argmax(dim=-1)
        base_ce = F.cross_entropy(
            top_log_probs.reshape(-1, selector.top_k),
            target_index.reshape(-1),
            reduction="none",
        ).reshape_as(target_ids)
        base_ce_num = (base_ce * weight_mask * is_candidate.float()).sum()

        scores, err_logits = selector.score_candidates(
            candidate_ids=candidate_ids,
            unary_logits=top_log_probs,
            hidden_states=hidden,
            predecessor_ids=predecessor_ids,
            lattice_scalars=scalars,
            return_err=True,
        )

    anchor_gap = (scores - top_log_probs).abs().max().item()
    argmax_is_zero = int((scores.argmax(-1) != 0).sum().item())
    ce_gap = (terms.ce_num - base_ce_num).abs().item()
    print("\n1. base anchoring at gamma=0")
    print(f"   max |score - base log-prob|        = {anchor_gap:.3e}")
    print(f"   slots whose argmax != candidate 0  = {argmax_is_zero} / {b * n * block}")
    print(f"   |selector_ce_num - base_ce_num|    = {ce_gap:.3e}")
    checks["scores equal base log-probs"] = anchor_gap == 0.0
    checks["selector argmax is candidate 0"] = argmax_is_zero == 0
    checks["selector CE equals base CE"] = ce_gap == 0.0

    ref_f32, ref_count = reference_err_loss(
        err_logits, target_index, is_candidate, weight_mask, torch.float32
    )
    ref_f64, _ = reference_err_loss(
        err_logits, target_index, is_candidate, weight_mask, torch.float64
    )
    got_den = terms.err_den.item()
    got_mean = terms.err_ce_num.item() / max(got_den, 1e-12)
    # Same masked sum, recomputed in float64 to strip the reduction-order artefact.
    got_f64 = (
        (
            F.binary_cross_entropy_with_logits(
                err_logits.double(),
                (~(is_candidate & target_index.eq(0))).double(),
                reduction="none",
            )
            * _frontier_weights(weight_mask, is_candidate, target_index).double()
        ).sum()
        / got_den
    ).item()
    print("\n2. frontier detector objective vs the reference formulation")
    print(f"   frontier slots : reference {ref_count} | specforge {got_den:.0f}")
    print(f"   fp32 mean BCE  : reference {ref_f32:.10f} | specforge {got_mean:.10f} "
          f"| diff {abs(ref_f32 - got_mean):.3e}  (reduction order)")
    print(f"   fp64 mean BCE  : reference {ref_f64:.16f}")
    print(f"                    specforge {got_f64:.16f} | diff {abs(ref_f64 - got_f64):.3e}")
    checks["frontier slot count matches"] = ref_count == int(got_den)
    checks["frontier BCE matches reference (fp64)"] = abs(ref_f64 - got_f64) < 1e-12
    checks["frontier mask is non-empty"] = got_den > 0

    # ---- 3: a detector-less selector must leave the err terms at exact zero -------------------
    plain = _DetectorlessSelector(cfg.hidden_size).to(device=device, dtype=dtype)
    plain_model = build_model(1.0)
    plain_model.draft_model = draft
    object.__setattr__(plain_model, "_selector_err_objective_enabled", False)
    with torch.no_grad():
        plain_terms = plain_model._selector_chunk_terms(
            candidate_selector=plain,
            objective_logits=logits,
            hidden=hidden,
            target_ids=target_ids,
            predecessor_ids=predecessor_ids,
            loss_weights=weight_mask,
            weight_mask=weight_mask,
        )
    zero_err = (
        plain_terms.err_ce_num.item() == 0.0
        and plain_terms.err_den.item() == 0.0
        and plain_terms.err_correct_num.item() == 0.0
    )
    print("\n3. selector without a detector (the DFlash2 path)")
    print(
        f"   err_ce_num={plain_terms.err_ce_num.item():.3e} "
        f"err_den={plain_terms.err_den.item():.3e} "
        f"err_correct_num={plain_terms.err_correct_num.item():.3e}"
    )
    checks["detector-less selector keeps err terms at zero"] = zero_err

    # ---- end-to-end forward on real features --------------------------------------------------
    files = list_feature_files(args.features)[: args.num_samples]
    print(f"\n4. end-to-end forward on {len(files)} real samples")

    def run_forward():
        rows = []
        for path in files:
            sample = process_offline_dflash_sample(torch.load(path, map_location="cpu"), 3072)
            torch.manual_seed(7)
            with torch.no_grad():
                loss, accuracy, metrics = model(
                    input_ids=sample["input_ids"].to(device),
                    hidden_states=sample["hidden_states"].to(device=device, dtype=dtype),
                    loss_mask=sample["loss_mask"].to(device),
                )
            ratios = metrics["ratio_metrics"]

            def ratio(key):
                num, den = ratios[key]
                return float(num) / max(float(den), 1e-12)

            rows.append(
                dict(
                    loss=float(loss),
                    acc=float(accuracy),
                    sel_acc=ratio("selector_accuracy"),
                    coverage=ratio("selector_coverage"),
                    err_loss=ratio("selector_err_loss"),
                    err_acc=ratio("selector_err_accuracy"),
                    frontier=ratio("selector_err_frontier_rate"),
                )
            )
        return rows

    rows = run_forward()
    header = f"{'#':>2s} {'loss':>8s} {'acc':>7s} {'sel_acc':>8s} {'cover':>7s} {'err_loss':>9s} {'err_acc':>8s} {'front':>7s}"
    print(f"   {header}")
    for i, row in enumerate(rows):
        print(
            f"   {i:>2d} {row['loss']:>8.4f} {row['acc']:>7.4f} {row['sel_acc']:>8.4f} "
            f"{row['coverage']:>7.4f} {row['err_loss']:>9.4f} {row['err_acc']:>8.4f} "
            f"{row['frontier']:>7.4f}"
        )
    # At gamma=0 the selector reproduces the base ranking exactly, so its accuracy over covered
    # slots must equal the backbone's accuracy divided by coverage. This is only a real check when
    # coverage is well away from zero, hence the separate coverage assertion.
    implied = [abs(r["sel_acc"] * r["coverage"] - r["acc"]) for r in rows]
    min_coverage = min(r["coverage"] for r in rows)
    print(f"   max |sel_acc * coverage - acc| = {max(implied):.3e}   (must be ~0 at gamma=0)")
    print(f"   min coverage over samples      = {min_coverage:.4f}   (must be >> 0 to have teeth)")
    checks["coverage is non-trivial"] = min_coverage > 0.3
    checks["selector reproduces base ranking end to end"] = max(implied) < 2e-3
    checks["detector metrics are reported"] = all(r["frontier"] > 0 for r in rows)

    # ---- 5: own-denominator rescale is exactly the intended one --------------------------------
    sample = process_offline_dflash_sample(torch.load(files[0], map_location="cpu"), 3072)
    inputs = dict(
        input_ids=sample["input_ids"].to(device),
        hidden_states=sample["hidden_states"].to(device=device, dtype=dtype),
        loss_mask=sample["loss_mask"].to(device),
    )

    def loss_and_terms(own):
        variant = build_model(1.0, own_denominator=own)
        torch.manual_seed(7)
        with torch.no_grad():
            loss, _, metrics = variant(**inputs)
        ce_num, covered_den = metrics["ratio_metrics"]["selector_loss"]
        _, loss_den = metrics["loss_terms"]
        return float(loss), float(ce_num), float(covered_den), float(loss_den)

    loss_shared, ce_num, covered_den, loss_den = loss_and_terms(False)
    loss_own, ce_num2, covered_den2, loss_den2 = loss_and_terms(True)
    expected_shift = ce_num / covered_den - ce_num / loss_den
    observed_shift = loss_own - loss_shared
    print("\n5. selector_own_denominator rescale")
    print(f"   covered/total slots = {covered_den:.0f}/{loss_den:.0f} = {covered_den / loss_den:.4f}")
    print(f"   expected loss shift = {expected_shift:.8f}")
    print(f"   observed loss shift = {observed_shift:.8f}")
    print(f"   |difference|        = {abs(expected_shift - observed_shift):.3e}")
    checks["own-denominator rescale is exact"] = (
        abs(expected_shift - observed_shift) < 1e-4 and expected_shift > 0
    )
    checks["own-denominator leaves the numerators alone"] = (
        ce_num == ce_num2 and covered_den == covered_den2 and loss_den == loss_den2
    )

    # ---- 6: control ---------------------------------------------------------------------------
    with torch.no_grad():
        selector.gamma.fill_(1.0)
        selector.delta_mlp[-1].weight.normal_(0, 0.02)
        control_scores = selector.score_candidates(
            candidate_ids=candidate_ids,
            unary_logits=top_log_probs,
            hidden_states=hidden,
            predecessor_ids=predecessor_ids,
            lattice_scalars=scalars,
        )
    control_gap = (control_scores - top_log_probs).abs().max().item()
    print(f"\n6. CONTROL, gamma=1 and delta perturbed: max |score - base| = {control_gap:.3e}")
    checks["control breaks base anchoring"] = control_gap > 0.0

    print()
    for name, passed in checks.items():
        print(f"  [{'ok' if passed else 'XX'}] {name}")
    ok = all(checks.values())
    print(f"\nGATE {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
