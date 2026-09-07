#!/usr/bin/env python3
"""Measure the per-block wall-clock cost of each selector head, in isolation.

Why this script exists: the constraint "the extra module must not cost much more than Domino" has
never been measured.  Head FLOPs are a bad proxy -- the cloze head's cost was dominated by launching
H serialised small kernels, not by arithmetic (its first, prefix-conditioned design measured
45.26 ms/block against 8.63 ms for the batched redesign, at identical FLOPs).

What is timed: exactly the work a decode loop does per block AFTER the draft backbone has produced
``h`` and the lattice has been read, and BEFORE the target verifies.  That is the marginal cost the
head adds.  The draft/target forwards are excluded because they are common to every arm.

Run (needs an idle GPU):
  PYTHONPATH=. python3 scripts/bench_head_latency.py --device cuda:0 \
      --decision <selector.pt> --cloze <selector.pt> --v2 <selector.pt> --domino <best.pt>
Any arm may be omitted.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

SPECFORGE = Path(__file__).resolve().parents[1]
TAPS = SPECFORGE.parent / "TAPS-SP"
for p in (str(SPECFORGE), str(TAPS)):
    if p not in sys.path:
        sys.path.insert(0, p)

H, K, HIDDEN, VOCAB = 15, 16, 2560, 151936


def synth(device, dtype=torch.float32):
    """A lattice with the invariants ``extract_topk_lattice`` guarantees."""
    g = torch.Generator(device="cpu").manual_seed(0)
    h = torch.randn(H, HIDDEN, generator=g).to(device=device, dtype=dtype)
    raw = torch.randn(H, VOCAB, generator=g).to(device=device, dtype=dtype)
    lp_all = F.log_softmax(raw, dim=-1)
    lp, cand = lp_all.topk(K, dim=-1)
    scal = torch.stack(
        [
            -(lp_all.exp() * lp_all).sum(-1),
            (lp[:, 0] - lp[:, 1]).clamp(-20, 20),
            lp.exp().sum(-1),
        ],
        dim=-1,
    )
    anchor = torch.zeros(1, dtype=torch.long, device=device)
    return h, cand.long(), lp, scal, anchor


def timeit(fn, device, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - t0) * 1e3)
    samples.sort()
    return statistics.median(samples), samples[int(0.95 * len(samples)) - 1]


def bench_slotz(model, embed, device, factorized: bool):
    """cloze and Decision: one batched z pass per block, then per-slot GRU + score (+ detector)."""
    h, cand, lp, scal, anchor = synth(device)
    cand_emb = embed[cand]
    is_decision = hasattr(model, "slot_states")

    def run():
        with torch.no_grad():
            if is_decision:
                z = model.slot_states(
                    h.unsqueeze(0), cand.unsqueeze(0), lp.unsqueeze(0), scal.unsqueeze(0)
                )[0]
            else:
                z = model.cloze_states(
                    h.unsqueeze(0), cand.unsqueeze(0), anchor, lp.unsqueeze(0), scal.unsqueeze(0)
                )[0]
            prev = anchor
            gru_h = None
            for i in range(H):
                step, gru_h = model.gru(embed[prev].unsqueeze(1), gru_h)
                state = step[:, 0]
                sc = model.score(z[i : i + 1], h[i : i + 1], cand_emb[i], lp[i : i + 1], state)
                if factorized:
                    el = model.err_logits(
                        z[i : i + 1], cand_emb[i, 0].unsqueeze(0), lp[i : i + 1],
                        scal[i : i + 1], h[i : i + 1], state,
                    )
                    j = int(model.select_keep_repair(sc, el)[0, 0])
                else:
                    pr = torch.softmax(sc[0], dim=-1)
                    bo, bi = pr[1:].max(dim=-1)
                    j = int(bi) + 1 if float(bo) > 3.0 * float(pr[0]) else 0
                prev = cand[i, j].view(1)

    return run


def bench_v2(model, embed, device):
    """PSPR-v2: one batched cross-slot encode per block, then per-slot GRU + score + margin gate.

    Different call signatures from the two slot-z heads (``encode`` takes candidate EMBEDDINGS, and
    ``score`` takes ``(h, r, S, cand_emb, lp)``), hence a separate arm rather than a shared one.
    """
    h, cand, lp, scal, anchor = synth(device)
    cand_emb = embed[cand]

    def run():
        with torch.no_grad():
            r = model.encode(
                h.unsqueeze(0), cand_emb.unsqueeze(0), lp.unsqueeze(0), scal.unsqueeze(0)
            )[0]
            prev = anchor
            gru_h = None
            for i in range(H):
                step, gru_h = model.gru(embed[prev].unsqueeze(1), gru_h)
                state = step[:, 0]
                sc = model.score(
                    h[i : i + 1], r[i : i + 1], state, cand_emb[i], lp[i : i + 1]
                )
                pr = torch.softmax(sc[0], dim=-1)
                bo, bi = pr[1:].max(dim=-1)
                j = int(bi) + 1 if float(bo) > 3.0 * float(pr[0]) else 0
                prev = cand[i, j].view(1)

    return run


def bench_domino(dom, embed, device):
    """Domino: per-slot GRU + a low-rank correction over the FULL vocabulary."""
    h, cand, lp, scal, anchor = synth(device)
    base = torch.randn(H, VOCAB, device=device)

    def run():
        with torch.no_grad():
            hid = None
            prev = anchor
            for i in range(H):
                S_i, hid = dom.gru(embed[prev].unsqueeze(0), hid)
                corr = dom.W2(F.silu(dom.W1(torch.cat([h[i : i + 1], S_i[0]], dim=-1))))[0]
                prev = (base[i] + corr).argmax().view(1)

    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--decision", default=None)
    ap.add_argument("--cloze", default=None)
    ap.add_argument("--v2", default=None)
    ap.add_argument("--domino", default=None)
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()
    device = torch.device(args.device)

    # The candidate table / readout is the target's frozen tied embedding.  A random matrix of the
    # right shape and dtype costs exactly the same to index and matmul against.
    embed = torch.randn(VOCAB, HIDDEN, device=device)

    from scripts.decode_lattice import load_selector_checkpoint  # type: ignore

    rows = []
    for name, path, factorized in (
        ("PSPR-Decision", args.decision, True),
        ("PSPR-cloze", args.cloze, False),
        ("PSPR-v2", args.v2, False),
    ):
        if not path:
            continue
        model, _ = load_selector_checkpoint(path, device=device)
        model = model.float().eval()
        if hasattr(model, "bind_target_embedding"):
            model.bind_target_embedding(embed)
        n = sum(p.numel() for p in model.parameters() if p.dim() > 0)
        builder = (
            bench_v2(model, embed, device)
            if name == "PSPR-v2"
            else bench_slotz(model, embed, device, factorized)
        )
        med, p95 = timeit(builder, device, iters=args.iters)
        rows.append((name, n, med, p95))
        del model
        torch.cuda.empty_cache()

    if args.domino:
        from joint.domino_head import DominoHead  # type: ignore

        dom, _ = DominoHead.from_checkpoint(args.domino, device=str(device))
        dom = dom.float().eval()
        n = sum(p.numel() for p in dom.parameters())
        med, p95 = timeit(bench_domino(dom, embed, device), device, iters=args.iters)
        rows.append(("Domino", n, med, p95))

    print(f"\nper-block head latency, H={H} K={K} hidden={HIDDEN} on {args.device}")
    print(f"{'head':16s} {'params':>10s} {'median ms':>10s} {'p95 ms':>9s}  {'vs Domino':>10s}")
    dom_ms = next((m for nm, _, m, _ in rows if nm == "Domino"), None)
    for name, n, med, p95 in rows:
        rel = f"{med / dom_ms:.2f}x" if dom_ms else "-"
        print(f"{name:16s} {n/1e6:9.2f}M {med:10.3f} {p95:9.3f}  {rel:>10s}")


if __name__ == "__main__":
    main()
