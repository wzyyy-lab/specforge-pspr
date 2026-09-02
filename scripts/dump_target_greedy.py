#!/usr/bin/env python3
"""Dump the target model's own greedy next-token at every position of an already-dumped corpus.

Motivation, measured: on ShareGPT only 0.77 of supervised positions have
``corpus_next_token == target_greedy``. The remaining 23% teach the draft to emit tokens the target
would never emit, and no such token can ever be accepted at decode time.

That mismatch is survivable for a backbone, which learns a distribution and can average the noise
out, but it is actively harmful for a re-ranking selector: the selector's whole job is deciding *when
to overrule the base top-1*, and the base top-1 is typically the target's greedy token. Supervising
it to prefer the corpus token at exactly those positions trains the destroy behaviour that was then
measured at decode (destroy 294 vs recover 84 on gsm8k, net -0.226 accepted slots per block).

So the selector needs its own labels: what the target would actually say. The backbone keeps the
corpus labels and the official objective, unchanged.

Writes one small file per corpus shard, keyed by the same filename, holding an int32 ``[seq]`` tensor
aligned so that ``greedy[i]`` is the target's argmax for position ``i+1`` -- i.e. directly comparable
to ``input_ids[1:]``. Cheap: a couple of KB per sample against ~28 MB of hidden states.

Usage (shard across GPUs like the hidden-state dump):
    PYTHONPATH=. python scripts/dump_target_greedy.py \
        --target-model .../Qwen3-4B --features cache/hidden_states/sharegpt3k \
        --output cache/target_greedy/sharegpt3k --start 0 --stride 8
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch

SPECFORGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPECFORGE))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-model", required=True)
    ap.add_argument("--features", required=True, help="existing hidden-state dump directory")
    ap.add_argument("--output", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM

    device = "cuda"
    target = (
        AutoModelForCausalLM.from_pretrained(
            args.target_model, attn_implementation="sdpa", dtype=torch.bfloat16
        )
        .to(device)
        .eval()
    )

    files = sorted(Path(args.features).glob("*.ckpt"))[args.start :: args.stride]
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"shard start={args.start} stride={args.stride}: {len(files)} files -> {out_dir}",
          flush=True)

    done = agree = supervised = 0
    for i, path in enumerate(files):
        destination = out_dir / (path.stem + ".pt")
        if destination.exists():
            done += 1
            continue
        sample = torch.load(path, map_location="cpu", weights_only=False)
        input_ids = sample["input_ids"].to(torch.long)
        loss_mask = sample["loss_mask"].to(torch.long)
        with torch.no_grad():
            logits = target(input_ids.unsqueeze(0).to(device)).logits[0]
        # logits[i] predicts position i+1, so drop the last row and store the result aligned to
        # input_ids[1:]. A trailing sentinel keeps the tensor the same length as input_ids, which
        # lets the reader index it with the same offsets it uses for input_ids.
        greedy = logits[:-1].argmax(dim=-1).to(torch.int32).cpu()
        greedy = torch.cat([greedy, greedy.new_zeros(1)])

        mask = loss_mask[1:] > 0
        supervised += int(mask.sum())
        agree += int((greedy[:-1][mask] == input_ids[1:][mask]).sum())

        torch.save({"target_greedy": greedy}, destination)
        done += 1
        del logits, sample
        if (i + 1) % args.log_every == 0:
            gc.collect()
            torch.cuda.empty_cache()
            rate = agree / max(supervised, 1)
            print(f"  [{args.start}] {i + 1}/{len(files)} agree={rate:.4f}", flush=True)

    rate = agree / max(supervised, 1)
    print(f"shard {args.start} done: {done} files, "
          f"P(target greedy == corpus) = {rate:.4f} over {supervised} supervised positions",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
