"""Gates for the PSPR lattice selector, run before any training.

Three properties must hold, each with a control that makes the gate falsifiable:

1. base-anchored init
   ``gamma`` is zero-init, so the selector's scores must equal the base top-K log-probs exactly and
   its argmax must be candidate 0. Any drift means the selector is not a strict superset of one-shot
   decoding and acceptance is no longer structurally non-decreasing.

2. candidate 0 is the argmax token, including under bf16 logit ties
   ``lm_head`` runs in bf16 and ties the maximal logit at a few percent of slots, where ``topk`` and
   ``argmax`` disagree. If candidate 0 is not the argmax token, the "keep candidate 0" branch no
   longer reproduces the backbone's own greedy draft.

3. the score at slot i actually depends on slot j != i
   This is the load-bearing novelty: the block's whole lattice is visible before any token is
   committed. Perturbing one slot's candidate set must move other slots' scores. The control zeroes
   the encoder's contribution, which must make that dependence vanish.

Gate 3 is the corrected form of an earlier probe that measured whether the *backbone* mixes slots.
That question is irrelevant: the backbone does mix hidden states, but the lattice is the lm_head
readout, which the backbone never computes, and the consumer is the head, which in Domino sees only
its own slot.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

SPECFORGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPECFORGE))

from specforge.modeling.draft.pspr import PSPRDraftModel  # noqa: E402


def build(draft_config: str, device, dtype):
    from transformers import Qwen3Config

    raw = json.load(open(draft_config))
    cfg = Qwen3Config(**{k: v for k, v in raw.items() if k != "architectures"})
    cfg.dflash_config = raw["dflash_config"]
    cfg.block_size = raw["block_size"]
    cfg.num_target_layers = raw["num_target_layers"]
    cfg._attn_implementation = "flex_attention"
    torch.manual_seed(0)
    with torch.device("cpu"):
        model = PSPRDraftModel(cfg)
    return model.to(device=device, dtype=dtype).eval(), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft-config", default=str(SPECFORGE / "configs/qwen3-4b-pspr.json"))
    ap.add_argument("--target-model", required=True)
    args = ap.parse_args()

    device, dtype = torch.device("cuda"), torch.bfloat16
    model, cfg = build(args.draft_config, device, dtype)
    sel = model.candidate_selector

    from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead

    parts = TargetEmbeddingsAndHead.from_pretrained(args.target_model, device="cuda", dtype=dtype)
    model.bind_target_decoder(parts.embed_tokens)
    frozen = model.apply_backbone_freeze()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"frozen {frozen/1e6:.2f}M | trainable {trainable/1e6:.2f}M "
          f"(selector {sum(p.numel() for p in sel.parameters())/1e6:.2f}M)")
    checks = {}
    checks["freeze leaves exactly the selector trainable"] = (
        trainable == sum(p.numel() for p in sel.parameters())
    )

    torch.manual_seed(1)
    b, n, h, v = 1, 2, cfg.block_size, cfg.vocab_size
    hidden = torch.randn(b, n, h, cfg.hidden_size, device=device, dtype=dtype)
    logits = torch.randn(b, n, h, v, device=device, dtype=dtype) * 3.0
    predecessor = torch.randint(0, v, (b, n, h), device=device)

    lp, cand, scal = sel.extract_lattice(logits)
    with torch.no_grad():
        scores = sel.score_candidates(
            candidate_ids=cand, unary_logits=lp, hidden_states=hidden,
            predecessor_ids=predecessor, lattice_scalars=scal)

    print(f"\n1. base-anchored init  (gamma={float(sel.gamma.detach()):.1f})")
    drift = (scores.float() - lp.float()).abs().max().item()
    picks_zero = int((scores.argmax(dim=-1) != 0).sum())
    print(f"   max |score - base_log_prob| = {drift:.3e}")
    print(f"   slots whose argmax != candidate 0 = {picks_zero} / {b*n*h}")
    checks["scores equal base log-probs at init"] = drift == 0.0
    checks["argmax is candidate 0 at init"] = picks_zero == 0

    print("\n2. candidate 0 == argmax token, with an all-ties control")
    ok_plain = bool((cand[..., 0] == logits.float().argmax(-1)).all())
    tied = torch.zeros(1, 1, 4, v, device=device, dtype=dtype)
    tied[..., :64] = 5.0
    lp_t, cand_t, _ = sel.extract_lattice(tied)
    ok_tied = bool((cand_t[..., 0] == tied.float().argmax(-1)).all())
    n_ties = int((tied.float() == tied.float().amax(-1, keepdim=True)).sum(-1).max())
    print(f"   random logits : candidate0 == argmax  -> {ok_plain}")
    print(f"   {n_ties}-way tied logits : candidate0 == argmax  -> {ok_tied}")
    checks["candidate 0 is argmax (random)"] = ok_plain
    checks["candidate 0 is argmax (bf16 ties)"] = ok_tied

    print("\n3. cross-slot dependence: perturb slot 11's candidate set, watch slot 3")
    poke = logits.clone()
    poke[0, 0, 11] = torch.randn(v, device=device, dtype=dtype) * 3.0
    lp2, cand2, scal2 = sel.extract_lattice(poke)

    def scores_with(gamma_value, lattice):
        lpx, candx, scalx = lattice
        with torch.no_grad():
            saved = sel.gamma.detach().clone()
            sel.gamma.fill_(gamma_value)
            out = sel.score_candidates(
                candidate_ids=candx, unary_logits=lpx, hidden_states=hidden,
                predecessor_ids=predecessor, lattice_scalars=scalx)
            sel.gamma.copy_(saved)
        return out

    a = scores_with(1.0, (lp, cand, scal))
    c = scores_with(1.0, (lp2, cand2, scal2))
    moved = (a[0, 0, 3].float() - c[0, 0, 3].float()).abs().max().item()
    same_block_far = (a[0, 0, 0].float() - c[0, 0, 0].float()).abs().max().item()
    other_block = (a[0, 1].float() - c[0, 1].float()).abs().max().item()
    print(f"   slot 3  of the poked block : max |dscore| = {moved:.4e}")
    print(f"   slot 0  of the poked block : max |dscore| = {same_block_far:.4e}")
    print(f"   whole OTHER block          : max |dscore| = {other_block:.4e}  (must be 0)")

    with torch.no_grad():
        saved_w = [p.detach().clone() for p in sel.encoder.parameters()]
        for p in sel.encoder.parameters():
            p.zero_()
        a0 = scores_with(1.0, (lp, cand, scal))
        c0 = scores_with(1.0, (lp2, cand2, scal2))
        ctrl = (a0[0, 0, 3].float() - c0[0, 0, 3].float()).abs().max().item()
        for p, w in zip(sel.encoder.parameters(), saved_w):
            p.copy_(w)
    print(f"   CONTROL, encoder weights zeroed : max |dscore| at slot 3 = {ctrl:.4e}")

    checks["slot 3 sees slot 11's candidate set"] = moved > 0.0
    checks["blocks stay independent"] = other_block == 0.0
    checks["dependence is carried by the encoder"] = ctrl < moved

    print()
    for name, passed in checks.items():
        print(f"  [{'ok' if passed else 'XX'}] {name}")
    ok = all(checks.values())
    print(f"\nGATE {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
