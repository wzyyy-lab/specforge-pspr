"""Feature-fidelity gate for offline dumps produced by dump_hidden_states_hf.py.

We cannot verify the capture by inspecting shapes alone: a wrong layer choice, a wrong
hidden_states index offset, or a wrong concat order all produce a perfectly well-formed
[seq, 12800] tensor. So we verify semantically, with zero training:

  load the OFFICIAL Domino draft weights -> run the real training forward on our dumped
  features -> read accept_len.

If the capture matches what the official model was trained on, accept_len must land near
official teacher-forcing quality. If we picked the wrong layers, the backbone sees
out-of-distribution inputs and accept_len collapses. The --permute control deliberately
breaks the concat order to demonstrate the gate can actually fail.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

SPECFORGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPECFORGE))

from specforge.algorithms.common.dflash_family_model import OnlineDominoModel  # noqa: E402
from specforge.data.preprocessing import process_offline_dflash_sample  # noqa: E402
from specforge.modeling.draft.domino import DominoDraftModel  # noqa: E402
from specforge.runtime.data_plane.offline_reader import list_feature_files  # noqa: E402


def load_official_draft(draft_config: str, ckpt_dir: str, device, dtype):
    from safetensors.torch import load_file
    from transformers import Qwen3Config

    cfg_dict = json.load(open(draft_config))
    cfg = Qwen3Config(**{k: v for k, v in cfg_dict.items() if k != "architectures"})
    cfg.dflash_config = cfg_dict["dflash_config"]
    cfg.block_size = cfg_dict["block_size"]
    cfg.num_target_layers = cfg_dict["num_target_layers"]
    # OnlineDominoModel hands the backbone a BlockMask, so the layers must run flex_attention.
    cfg._attn_implementation = "flex_attention"
    # Must NOT build on meta + to_empty(): rotary_emb.inv_freq / original_inv_freq are buffers
    # that the official checkpoint does not carry, so to_empty() would leave them as
    # uninitialized garbage and silently destroy RoPE.
    with torch.device("cpu"):
        draft = DominoDraftModel(cfg)
    state = load_file(str(Path(ckpt_dir) / "model.safetensors"))
    missing, unexpected = draft.load_state_dict(state, strict=False)
    buffer_names = {n for n, _ in draft.named_buffers()}
    missing = [k for k in missing if k not in buffer_names]
    print(f"load_state_dict: missing={missing} unexpected={list(unexpected)}")
    assert not missing and not unexpected, "official weights did not map cleanly"
    inv = draft.rotary_emb.inv_freq
    assert torch.isfinite(inv).all() and inv.max() <= 1.0, f"bad rope buffer: {inv[:3]}"
    return draft.to(device=device, dtype=dtype).eval(), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--draft-config", default=str(SPECFORGE / "configs/qwen3-4b-domino.json"))
    ap.add_argument("--official-ckpt", required=True)
    ap.add_argument("--target-model", required=True)
    ap.add_argument("--num-samples", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=3072)
    ap.add_argument("--num-anchors", type=int, default=256)
    ap.add_argument("--permute", action="store_true",
                    help="control: reverse the 5 layer blocks to prove the gate has teeth")
    args = ap.parse_args()

    device, dtype = torch.device("cuda"), torch.bfloat16
    draft, cfg = load_official_draft(args.draft_config, args.official_ckpt, device, dtype)

    from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead
    parts = TargetEmbeddingsAndHead.from_pretrained(
        args.target_model, device="cuda", dtype=dtype)

    model = OnlineDominoModel(
        draft_model=draft,
        target_lm_head=parts.lm_head,
        target_embed_tokens=parts.embed_tokens,
        mask_token_id=cfg.dflash_config["mask_token_id"],
        block_size=cfg.block_size,
        attention_backend="flex_attention",
        num_anchors=args.num_anchors,
        loss_decay_gamma=7.0,
        shift_label=bool(draft.shift_label),
    ).to(device=device, dtype=dtype).eval()

    files = list_feature_files(args.features)[: args.num_samples]
    hidden = cfg.hidden_size
    agg = {}
    for i, path in enumerate(files):
        raw = torch.load(path, map_location="cpu")
        s = process_offline_dflash_sample(raw, args.max_length)
        hs = s["hidden_states"]
        if args.permute:
            blocks = [hs[..., k * hidden:(k + 1) * hidden] for k in range(5)]
            hs = torch.cat(blocks[::-1], dim=-1)
        with torch.no_grad():
            out = model(
                input_ids=s["input_ids"].to(device),
                hidden_states=hs.to(device=device, dtype=dtype),
                loss_mask=s["loss_mask"].to(device),
                lambda_base=0.0,
            )
        metrics = out[-1] if isinstance(out, tuple) else out
        row = {k: float(v) for k, v in metrics.items() if torch.as_tensor(v).numel() == 1}
        for k, v in row.items():
            agg.setdefault(k, []).append(v)
        if i < 4:
            keys = ["accept_len", "base_accept_len", "acc", "accuracy", "loss"]
            shown = {k: round(row[k], 4) for k in row if any(x in k for x in keys)}
            print(f"  [{i}] seq={s['input_ids'].shape[1]:>5d} {shown}", flush=True)

    print()
    tag = "PERMUTED CONTROL" if args.permute else "REAL CAPTURE"
    print(f"=== {tag}: mean over {len(files)} samples ===")
    for k in sorted(agg):
        print(f"  {k:<24s} {sum(agg[k]) / len(agg[k]):.4f}")


if __name__ == "__main__":
    main()
