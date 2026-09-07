#!/usr/bin/env python3
"""Break one speculative-decoding block into its three real costs.

Everything so far has compared heads against each other.  That is the wrong denominator: what
matters is the head's share of a block, because that is what decides whether a better accept length
is worth paying for.  This has never been measured here.

Three segments, all at the shapes a real block uses:
  1. draft backbone: 5 layers, hidden 2560, ONE forward over 16 positions (anchor + 15 MASK)
  2. lm_head readout: [16, 2560] @ [2560, 151936]  (the lattice comes off this)
  3. target verify:  36 layers, ONE forward over 16 positions, with a KV cache already populated

Run:  PYTHONPATH=. python3 scripts/bench_block_breakdown.py --device cuda:0
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

TS = "/home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/TAPS-SP"


def timeit(fn, device, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    xs = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize(device)
        xs.append((time.perf_counter() - t0) * 1e3)
    xs.sort()
    return statistics.median(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--ctx", type=int, default=512, help="KV cache length before the block")
    args = ap.parse_args()
    device = torch.device(args.device)
    B = args.block

    from transformers import AutoModelForCausalLM

    tgt = AutoModelForCausalLM.from_pretrained(
        f"{TS}/models/Qwen3-4B", dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device).eval()
    cfg = tgt.config
    print(f"target: {cfg.num_hidden_layers} layers, hidden {cfg.hidden_size}, vocab {cfg.vocab_size}")

    ids = torch.randint(0, cfg.vocab_size, (1, args.ctx), device=device)
    with torch.no_grad():
        pre = tgt(ids, use_cache=True)
    cache = pre.past_key_values
    blk = torch.randint(0, cfg.vocab_size, (1, B), device=device)

    # `copy.copy(cache)` per iteration dominated the measurement (76 ms for what is a
    # memory-bandwidth-bound 8 GB weight read, i.e. ~1.7 ms of theoretical floor on this card).
    # Crop the SAME cache back to its pre-block length instead: no allocation, no copy.
    def verify():
        with torch.no_grad():
            tgt(blk, past_key_values=cache, use_cache=True)
            cache.crop(args.ctx)

    t_verify = timeit(verify, device)

    # A single-token decode step is the denominator that actually matters: without speculation the
    # model pays one of these per token, and it reads the same 8 GB of weights as the 16-token verify.
    one = blk[:, :1]

    def decode1():
        with torch.no_grad():
            tgt(one, past_key_values=cache, use_cache=True)
            cache.crop(args.ctx)

    t_dec1 = timeit(decode1, device)

    # Draft: the official DFlash-b16 backbone is 5 layers of the same width.  Timing the first 5
    # layers of the target is the same arithmetic and the same kernels, and avoids depending on the
    # draft package's custom attention path for a pure cost measurement.
    layers = tgt.model.layers[:5]
    hid = torch.randn(1, B, cfg.hidden_size, device=device, dtype=torch.bfloat16)
    pos = torch.arange(B, device=device).unsqueeze(0)
    rot = tgt.model.rotary_emb(hid, pos)

    def draft():
        with torch.no_grad():
            x = hid
            for lyr in layers:
                x = lyr(x, position_embeddings=rot, attention_mask=None)
                if isinstance(x, tuple):
                    x = x[0]

    t_draft = timeit(draft, device)

    lm = tgt.lm_head

    def readout():
        with torch.no_grad():
            lm(hid)

    t_read = timeit(readout, device)

    print(f"\nper-block cost breakdown (block={B}, ctx={args.ctx}) on {args.device}")
    print(f"  target decode  (36L,  1 tok)      {t_dec1:8.3f} ms   <- no-speculation cost per token")
    print(f"  target verify (36L, {B} tok)      {t_verify:8.3f} ms")
    print(f"  draft backbone (5L, {B} tok)      {t_draft:8.3f} ms")
    print(f"  lm_head readout [{B},2560]x[.,V]  {t_read:8.3f} ms")
    base = t_verify + t_draft + t_read
    print(f"  ---- speculative block, no head   {base:8.3f} ms")
    print("\n  head cost as a share of the block:")
    for name, ms in (
        ("Domino", 3.102),
        ("PSPR-v2", 6.464),
        ("PSPR-cloze", 8.240),
        ("PSPR-Decision", 11.617),
    ):
        print(f"    {name:16s} {ms:6.2f} ms -> {100*ms/(base+ms):5.1f}% of block, "
              f"block {base+ms:7.2f} ms")
    print(
        "\n  (accept length must rise by at least head_ms/base_ms x current_accept just to break "
        "even on tokens/second)"
    )


if __name__ == "__main__":
    main()
