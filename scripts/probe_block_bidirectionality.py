"""Is the DFlash backbone already bidirectional inside a draft block?

This decides whether "our head is bidirectional over the block, Domino's head is only causal"
is a real architectural difference or an empty claim. Reading the mask builder suggests the
backbone is already fully bidirectional within a block (mask_draft = q_block_id == kv_block_id,
with no q_block_offset >= kv_block_offset term unless sliding_window is set), but a claim this
load-bearing should be measured, not inferred.

Test: perturb the MASK embedding at ONE draft slot and measure how much every slot's output
hidden state moves. Causal-within-block => only slots at or after the perturbed one move.
Bidirectional => earlier slots move too.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

SPECFORGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPECFORGE))

from specforge.data.preprocessing import process_offline_dflash_sample  # noqa: E402
from specforge.runtime.data_plane.offline_reader import list_feature_files  # noqa: E402

sys.path.insert(0, str(SPECFORGE / "scripts"))
from verify_offline_features import load_official_draft  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--draft-config", default=str(SPECFORGE / "configs/qwen3-4b-domino.json"))
    ap.add_argument("--official-ckpt", required=True)
    ap.add_argument("--target-model", required=True)
    ap.add_argument("--poke-slot", type=int, default=11)
    args = ap.parse_args()

    device, dtype = torch.device("cuda"), torch.bfloat16
    draft, cfg = load_official_draft(args.draft_config, args.official_ckpt, device, dtype)
    from specforge.algorithms.common.dflash_family_model import OnlineDominoModel
    from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead

    parts = TargetEmbeddingsAndHead.from_pretrained(args.target_model, device="cuda", dtype=dtype)
    model = OnlineDominoModel(
        draft_model=draft, target_lm_head=parts.lm_head, target_embed_tokens=parts.embed_tokens,
        mask_token_id=cfg.dflash_config["mask_token_id"], block_size=cfg.block_size,
        attention_backend="flex_attention", num_anchors=8, loss_decay_gamma=7.0,
        shift_label=True).to(device=device, dtype=dtype).eval()

    raw = torch.load(list_feature_files(args.features)[0], map_location="cpu")
    s = process_offline_dflash_sample(raw, 3072)
    kw = dict(input_ids=s["input_ids"].to(device),
              hidden_states=s["hidden_states"].to(device=device, dtype=dtype),
              loss_mask=s["loss_mask"].to(device))

    bs = cfg.block_size
    original = model._create_noise_embed

    with torch.no_grad():
        torch.manual_seed(1234)
        _, _, h0 = model._forward_draft_blocks(**kw)
        def poked(input_ids, anchor_positions, block_keep_mask):
            emb = original(input_ids, anchor_positions, block_keep_mask).clone()
            emb[:, args.poke_slot, :] += 8.0
            return emb
        model._create_noise_embed = poked
        torch.manual_seed(1234)
        _, _, h1 = model._forward_draft_blocks(**kw)
        model._create_noise_embed = original

    # output_hidden is [bsz, n_blocks*block_size, hidden]; keep block 0 of sample 0.
    d = (h1.float() - h0.float()).norm(dim=-1)[0, :bs]
    ref = h0.float().norm(dim=-1)[0, :bs]
    print(f"\nperturbed draft slot {args.poke_slot} of block 0 (+8.0 on its MASK embedding)")
    print(f"{'slot':>5s} {'||dh||/||h||':>14s}  {'':s}")
    for t in range(bs):
        rel = (d[t] / ref[t]).item()
        bar = "#" * min(60, int(rel * 400))
        flag = "  <-- perturbed" if t == args.poke_slot else ""
        print(f"{t:>5d} {rel:>14.5f}  {bar}{flag}")
    before = d[: args.poke_slot]
    print()
    print(f"max relative change at slots BEFORE the perturbed one: "
          f"{(before / ref[: args.poke_slot]).max().item():.5f}")
    print("=> BIDIRECTIONAL within block" if (before > 1e-3).any()
          else "=> CAUSAL within block")


if __name__ == "__main__":
    main()
