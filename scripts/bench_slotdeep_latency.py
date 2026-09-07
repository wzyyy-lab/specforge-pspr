#!/usr/bin/env python3
"""Where does the cloze head's 8.24 ms actually go, and what would slotdeep change?

`bench_head_latency.py` measures the whole head at 8.240 ms median and the project has been reading
that as "the encoder is expensive".  A component microbenchmark says the batched z pathway is 1.8 ms
and the batched GRU plus score is 0.4 ms, so ~6 ms is unaccounted for -- and the difference between
the two measurements is that the real decode runs the GRU and the score SERIALLY, 15 Python
iterations, each with a `float(...)`/`int(...)` device-to-host sync to pick the committed token.

This script splits the deployed loop into (batched z) + (serial per-slot tail) for both arms, using
`bench_head_latency`'s own timing harness and synthetic lattice so the numbers are comparable to the
8.240 ms already on record.  It decides whether replacing the encoder is a latency lever at all, and
it converts the answer into throughput with the measured block cost of 33.593 ms (PSPR_DECISION.md
step 15) so the result is in the units the shipping decision is made in.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

SPECFORGE = Path(__file__).resolve().parents[1]
for p in (str(SPECFORGE), str(SPECFORGE / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from bench_head_latency import H, HIDDEN, K, VOCAB, synth, timeit  # noqa: E402

from specforge.modeling.draft.pspr_cloze import ClozeCorrector  # noqa: E402
from specforge.modeling.draft.pspr_slotdeep import SlotDeepCorrector  # noqa: E402

BLOCK_MS_WITHOUT_HEAD = 33.593  # PSPR_DECISION.md step 15, same machine class
ONESHOT_TOKS = 5.3043
DOMINO_MS, DOMINO_TOKS = 3.102, 5.8868


def build(kind, device, slot_layers):
    torch.manual_seed(0)
    common = dict(hidden_size=HIDDEN, vocab_size=VOCAB, top_k=K, d=512, n_layers=6, n_heads=8,
                  state_dim=512, delta_hidden=2048, max_slots=32, dropout=0.0,
                  direct_hidden=True, use_state=True, bidirectional=True, err_use_state=False)
    m = (SlotDeepCorrector(slot_layers=slot_layers, **common) if kind == "slotdeep"
         else ClozeCorrector(**common))
    return m.float().eval().to(device)


def parts(model, embed, device):
    """(batched z) and (serial per-slot tail) as two separately timeable closures.

    The tail is byte-identical to ``bench_head_latency.bench_slotz``'s inner loop with
    ``factorized=False``, including the two host syncs per slot -- those syncs are the deployed
    behaviour at ``gate_theta = 0`` (``decode_lattice.py:498`` skips ``err_head`` entirely), so
    removing them here would measure a head nobody runs.
    """
    h, cand, lp, scal, anchor = synth(device)
    cand_emb = embed[cand]

    def z_pass():
        with torch.no_grad():
            return model.cloze_states(h.unsqueeze(0), cand.unsqueeze(0), anchor,
                                      lp.unsqueeze(0), scal.unsqueeze(0))[0]

    z_cached = z_pass()

    def tail():
        with torch.no_grad():
            prev = anchor
            gru_h = None
            for i in range(H):
                step, gru_h = model.gru(embed[prev].unsqueeze(1), gru_h)
                state = step[:, 0]
                sc = model.score(z_cached[i:i + 1], h[i:i + 1], cand_emb[i], lp[i:i + 1], state)
                pr = torch.softmax(sc[0], dim=-1)
                bo, bi = pr[1:].max(dim=-1)
                j = int(bi) + 1 if float(bo) > 3.0 * float(pr[0]) else 0
                prev = cand[i, j].view(1)

    def whole():
        with torch.no_grad():
            z = model.cloze_states(h.unsqueeze(0), cand.unsqueeze(0), anchor,
                                   lp.unsqueeze(0), scal.unsqueeze(0))[0]
            prev = anchor
            gru_h = None
            for i in range(H):
                step, gru_h = model.gru(embed[prev].unsqueeze(1), gru_h)
                state = step[:, 0]
                sc = model.score(z[i:i + 1], h[i:i + 1], cand_emb[i], lp[i:i + 1], state)
                pr = torch.softmax(sc[0], dim=-1)
                bo, bi = pr[1:].max(dim=-1)
                j = int(bi) + 1 if float(bo) > 3.0 * float(pr[0]) else 0
                prev = cand[i, j].view(1)

    return z_pass, tail, whole


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--slot-layers", type=int, default=9)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--accept", type=float, default=6.0498, help="cloze@7115 accept on the 6 domains")
    args = ap.parse_args()
    device = torch.device(args.device)
    embed = torch.randn(VOCAB, HIDDEN, device=device)

    print(f"per-block head latency decomposition, H={H} K={K} hidden={HIDDEN}, {args.device}")
    print(f"{'arm':<10} {'batched z':>11} {'serial tail':>12} {'whole':>9} {'z share':>9}")
    out = {}
    for kind in ("cloze", "slotdeep"):
        m = build(kind, device, args.slot_layers)
        m.bind_target_embedding(embed)
        z_pass, tail, whole = parts(m, embed, device)
        z_ms = timeit(lambda: z_pass(), device, iters=args.iters)[0]
        t_ms = timeit(tail, device, iters=args.iters)[0]
        w_ms = timeit(whole, device, iters=args.iters)[0]
        out[kind] = dict(z=z_ms, tail=t_ms, whole=w_ms)
        print(f"{kind:<10} {z_ms:10.3f}ms {t_ms:11.3f}ms {w_ms:8.3f}ms {100 * z_ms / w_ms:8.1f}%")
        del m
        torch.cuda.empty_cache()

    saved = out["cloze"]["whole"] - out["slotdeep"]["whole"]
    print(f"\nreplacing the encoder saves {saved:.3f} ms of a {out['cloze']['whole']:.3f} ms head "
          f"({100 * saved / out['cloze']['whole']:.1f}%)")
    print(f"the serial per-slot tail is {out['cloze']['tail']:.3f} ms = "
          f"{100 * out['cloze']['tail'] / out['cloze']['whole']:.1f}% of the head and is IDENTICAL "
          f"in both arms:\n  15 Python iterations, each a GRU step, a score, a softmax and two "
          f"device-to-host syncs.\n  That, not the encoder, is what a fused kernel has to remove.")

    def toks(head_ms, accept):
        return accept / ((BLOCK_MS_WITHOUT_HEAD + head_ms) / 1e3)

    print(f"\nthroughput at accept {args.accept:.4f}, block-without-head {BLOCK_MS_WITHOUT_HEAD} ms")
    print(f"{'arm':<28} {'head ms':>8} {'tok/s':>8}")
    print(f"{'oneshot, no head':<28} {0.0:8.2f} {toks(0.0, ONESHOT_TOKS):8.1f}")
    print(f"{'Domino':<28} {DOMINO_MS:8.2f} {toks(DOMINO_MS, DOMINO_TOKS):8.1f}")
    for kind in ("cloze", "slotdeep"):
        print(f"{'PSPR-' + kind:<28} {out[kind]['whole']:8.2f} "
              f"{toks(out[kind]['whole'], args.accept):8.1f}")
    need = args.accept
    for label, head in (("cloze", out["cloze"]["whole"]), ("slotdeep", out["slotdeep"]["whole"])):
        breakeven = ONESHOT_TOKS * (BLOCK_MS_WITHOUT_HEAD + head) / BLOCK_MS_WITHOUT_HEAD
        dom_be = DOMINO_TOKS * (BLOCK_MS_WITHOUT_HEAD + head) / (BLOCK_MS_WITHOUT_HEAD + DOMINO_MS)
        print(f"  {label}: accept needed to match oneshot {breakeven:.3f}, to match Domino "
              f"{dom_be:.3f} (current {need:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
