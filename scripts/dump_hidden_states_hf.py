"""Offline feature dump for the DFlash/Domino family, using plain HF transformers.

Why this exists instead of scripts/prepare_hidden_states.py: that path captures through SGLang and
requires patches/sglang/v0.5.18/spec-capture.patch, which does not apply to the installed
sglang 0.5.15.post2.dev (multiple hunks fail). Upgrading or patching system site-packages would
pollute a shared environment. The capture contract is tiny, so we produce it directly.

Contract (verified against specforge/data/preprocessing.py::process_offline_dflash_sample):
    input_ids     [seq]                LongTensor
    loss_mask     [seq]                LongTensor, 1 = supervised
    hidden_states [seq, L*hidden]      concatenated target layers, bf16
  * all three must share the same seq length
  * loss_mask must contain two consecutive supervised tokens, else the sample is rejected

The layer selection reproduces the draft config exactly: dflash_config.target_layer_ids with the
offset=1 convention used by extract_context_feature (hidden_states[0] is the embedding output), so
target_layer_ids=[1,9,17,25,33] reads hidden_states[2,10,18,26,34].

input_ids / loss_mask come from SpecForge's own preprocess_conversations so the supervision spans
are byte-identical to the official pipeline rather than a reimplementation.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SPECFORGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPECFORGE))

from specforge.data.loss_mask import has_consecutive_supervised_tokens  # noqa: E402
from specforge.data.preprocessing import preprocess_conversations  # noqa: E402
from specforge.data.template import TEMPLATE_REGISTRY  # noqa: E402


def to_sharegpt(row: dict) -> list[dict]:
    """Normalise a prepared jsonl row into the role/content list the parser expects."""
    turns = row.get("conversations") or row.get("messages") or []
    out = []
    for t in turns:
        role = t.get("role")
        if role is None:
            src = t.get("from", "")
            role = "user" if src in ("human", "user") else "assistant"
        out.append({"role": role, "content": t.get("value") or t.get("content") or ""})
    return out


@torch.inference_mode()
def capture(target, input_ids: torch.Tensor, layer_ids: list[int], device,
            offset: int = 1) -> torch.Tensor:
    """Return [seq, len(layer_ids)*hidden] of concatenated target layer states."""
    out = target(input_ids.unsqueeze(0).to(device), output_hidden_states=True, use_cache=False)
    # offset=1: hidden_states[0] is the embedding output, so layer L lives at index L+1.
    sel = [out.hidden_states[L + offset][0] for L in layer_ids]
    feat = torch.cat(sel, dim=-1).to(torch.bfloat16).cpu()
    del out, sel
    return feat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-model", required=True)
    ap.add_argument("--draft-config", required=True,
                    help="draft json; target_layer_ids is read from dflash_config")
    ap.add_argument("--data-path", required=True, help="prepared *_train.jsonl")
    ap.add_argument("--output", required=True)
    ap.add_argument("--chat-template", default="qwen")
    ap.add_argument("--max-length", type=int, default=3072)
    ap.add_argument("--num-samples", type=int, default=None)
    ap.add_argument("--start", type=int, default=0, help="row offset, for sharding across GPUs")
    ap.add_argument("--stride", type=int, default=1, help="row stride, for sharding across GPUs")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--layer-offset", type=int, default=1,
                    help="index offset into hidden_states; 1 matches extract_context_feature. "
                         "Only change this to build a deliberate off-by-one control set.")
    args = ap.parse_args()

    device = torch.device("cuda")
    cfg = json.load(open(args.draft_config))
    layer_ids = cfg["dflash_config"]["target_layer_ids"]
    width = len(layer_ids) * cfg["hidden_size"]

    tok = AutoTokenizer.from_pretrained(args.target_model)
    target = AutoModelForCausalLM.from_pretrained(
        args.target_model, attn_implementation="sdpa", dtype=torch.bfloat16).to(device).eval()
    template = TEMPLATE_REGISTRY.get(args.chat_template)

    rows = [json.loads(l) for l in open(args.data_path)]
    rows = rows[args.start::args.stride]
    if args.num_samples:
        rows = rows[: args.num_samples]
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"layers={layer_ids} offset={args.layer_offset} width={width} "
          f"rows={len(rows)} -> {out_dir}", flush=True)

    kept = skipped_short = skipped_err = 0
    for i, row in enumerate(rows):
        try:
            conv = to_sharegpt(row)
            if not conv:
                skipped_err += 1
                continue
            res = preprocess_conversations(
                tok, [conv], template, max_length=args.max_length)
            if not res["input_ids"]:
                skipped_err += 1
                continue
            input_ids = res["input_ids"][0][0].to(torch.long)
            loss_mask = res["loss_mask"][0][0].to(torch.long)
            if not has_consecutive_supervised_tokens(loss_mask):
                skipped_short += 1
                continue
            feat = capture(target, input_ids, layer_ids, device, args.layer_offset)
            assert feat.shape[0] == input_ids.shape[0] == loss_mask.shape[0], (
                f"length mismatch {feat.shape} {input_ids.shape} {loss_mask.shape}")
            assert feat.shape[1] == width, f"width {feat.shape[1]} != {width}"
            rid = row.get("id", f"row{args.start + i * args.stride}")
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(rid))[:80]
            torch.save({"input_ids": input_ids, "loss_mask": loss_mask, "hidden_states": feat},
                       out_dir / f"{safe}_{args.start + i * args.stride:07d}.ckpt")
            kept += 1
        except torch.OutOfMemoryError:
            print(f"  OOM at row {i}, skipping", flush=True)
            torch.cuda.empty_cache(); gc.collect(); skipped_err += 1
            continue
        if i % args.log_every == 0:
            mb = sum(f.stat().st_size for f in out_dir.glob("*.ckpt")) / 2**20
            print(f"  [{i}/{len(rows)}] kept={kept} short={skipped_short} err={skipped_err} "
                  f"size={mb:.0f}MB mem={torch.cuda.memory_allocated()/2**30:.2f}G", flush=True)
            gc.collect()
    mb = sum(f.stat().st_size for f in out_dir.glob("*.ckpt")) / 2**20
    print(f"done: kept={kept} skipped_short={skipped_short} skipped_err={skipped_err} "
          f"total={mb:.0f}MB -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
