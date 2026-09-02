#!/usr/bin/env python3
"""Probe: is the DFlash backbone objective numerator reproducible across identical calls?

Check 4 of gate_target_greedy_labels reported the backbone CE numerator moving 5982.82 -> 6000.85
when only the *selector's* label source changed. Either the claim "only the labels move" is false,
or the forward is not bit-reproducible under a fixed seed and the check is comparing noise. This
distinguishes the two before anything is concluded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

SPECFORGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPECFORGE))

from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel  # noqa: E402
from specforge.algorithms.common.hidden_states_data import (  # noqa: E402
    TARGET_GREEDY_KEY,
    build_collator,
    build_offline_reader,
    normalize_offline_sample,
)
from specforge.modeling.draft.pspr import PSPRDraftModel  # noqa: E402
from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead  # noqa: E402
from specforge.runtime.data_plane.feature_store import LocalFeatureStore  # noqa: E402
from specforge.training.model_loading import warm_start_draft_model  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(SPECFORGE / "cache/hidden_states/sharegpt3k"))
    ap.add_argument("--draft-config", default=str(SPECFORGE / "configs/qwen3-4b-pspr.json"))
    ap.add_argument("--target-model", required=True)
    ap.add_argument(
        "--backbone",
        default="/kl_infra_infer_intern/wangzhuoyu/TAPS-SP/models/Qwen3-4B-DFlash-b16",
    )
    ap.add_argument("--max-len", type=int, default=1024)
    args = ap.parse_args()

    from transformers import Qwen3Config

    device, dtype = torch.device("cuda"), torch.bfloat16
    raw = json.load(open(args.draft_config))
    cfg = Qwen3Config(**raw)
    cfg.dflash_config = raw["dflash_config"]
    cfg.block_size = raw["block_size"]
    cfg.num_target_layers = raw["num_target_layers"]
    cfg._attn_implementation = "flex_attention"
    torch.manual_seed(0)
    with torch.device("cpu"):
        draft = PSPRDraftModel(cfg)
    warm_start_draft_model(
        draft, args.backbone, draft_config=cfg, strategy="offline",
        allow_missing_embedding=True,
    )
    draft = draft.to(device=device, dtype=dtype)
    parts = TargetEmbeddingsAndHead.from_pretrained(
        args.target_model, device="cuda", dtype=dtype
    )
    draft.bind_target_decoder(parts.embed_tokens)

    reader = build_offline_reader(
        "dflash", args.features, run_id="probe", ttt_length=7, max_len=args.max_len
    )
    store = LocalFeatureStore("probe")
    batch_features = []
    for ref in reader.read(limit=64):
        tensors, handle = store.get(ref)
        try:
            batch_features.append(normalize_offline_sample(tensors, args.max_len))
        except ValueError:
            pass
        finally:
            store.release(handle)
        if len(batch_features) == 1:
            break
    single = build_collator()(batch_features)
    inputs = dict(
        input_ids=single["input_ids"].to(device),
        hidden_states=single["hidden_states"].to(device=device, dtype=dtype),
        loss_mask=single["loss_mask"].to(device),
    )
    greedy = single[TARGET_GREEDY_KEY].to(device)

    def build(greedy_labels):
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
                selector_target_greedy_labels=greedy_labels,
            )
            .to(device=device, dtype=dtype)
            .eval()
        )

    def run(model, with_key):
        torch.manual_seed(7)
        with torch.no_grad():
            loss, acc, metrics = model(
                **inputs, target_greedy=greedy if with_key else None
            )
        num, den = metrics["loss_terms"]
        return float(num), float(den), float(loss)

    off, on = build(False), build(True)

    print("A. same model, same flag, repeated calls (pure reproducibility)")
    for label, model, key in (("off/no-key", off, False), ("on/key", on, True)):
        vals = [run(model, key)[0] for _ in range(3)]
        spread = max(vals) - min(vals)
        print(f"   {label:10s} ce_num = {vals}  spread = {spread:.6f}")

    print("\nB. two independently built models with the SAME flag")
    off2 = build(False)
    a, b = run(off, False)[0], run(off2, False)[0]
    print(f"   off vs off2 = {a:.5f} vs {b:.5f}   delta = {a - b:.6f}")

    print("\nC. flag off vs flag on (the failing comparison)")
    a = run(off, True)[0]
    b = run(on, True)[0]
    print(f"   off vs on   = {a:.5f} vs {b:.5f}   delta = {a - b:.6f}")

    print("\nD. anchors: are the sampled anchor positions identical?")
    sets = {}
    for label, model in (("off", off), ("on", on)):
        torch.manual_seed(7)
        with torch.no_grad():
            anchors, keep, hidden = model._forward_draft_blocks(
                input_ids=inputs["input_ids"],
                hidden_states=inputs["hidden_states"],
                loss_mask=inputs["loss_mask"],
                max_valid_anchors=None,
            )
        sets[label] = (anchors, keep, hidden)
    same_anchors = torch.equal(sets["off"][0], sets["on"][0])
    same_keep = torch.equal(sets["off"][1], sets["on"][1])
    hidden_delta = (sets["off"][2].float() - sets["on"][2].float()).abs().max()
    print(f"   anchors identical = {same_anchors}")
    print(f"   keep identical    = {same_keep}")
    print(f"   max|hidden delta| = {float(hidden_delta):.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
