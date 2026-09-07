#!/usr/bin/env python3
"""Why is the ranker worst at low slot index?  Decompose it offline from the feature dump.

The per-slot CHAIN profile shows `fix%` and `baseOK` flat in slot index while the selector's
`alt_rank_0` falls monotonically from ~87% at slot 13 to ~49% at slot 0, and slots 0-2 hold 43% of
all repair opportunity.  Three mechanisms are consistent with that, and they imply opposite fixes:

  M1  the BACKBONE is intrinsically harder at low slot index.  Then the selector is not at fault and
      the ceiling is the lattice, not the head.  Test: the draft's OWN best-alternative accuracy
      `draft_rank_1` = argmax(lp[1:]) == target, per slot.  If it falls like `alt_rank_0` does, M1.
  M2  the BASE IS MORE CONFIDENT at low slot index, so the additive rank-1 correction
      `<E_k, delta>` has a larger log-prob hill to climb and the multiplicative gate is structurally
      harder.  Test: condition on the hill `lp[0] - lp[target]`.  If the slot effect disappears
      inside a hill bucket, it is M2 and the fix is expressiveness/calibration against candidate 0,
      not more context.
  M3  the SELECTOR HAS LESS INFORMATION at low slot index (short GRU prefix, fewer seed tokens).
      Test: what survives after conditioning on the hill AND on the backbone's own accuracy.  A
      residual slot effect that is not explained by either is M3's signature.

Everything here is computed from the dumped frozen features, so it costs no forward passes:
`lp` is the 16 draft log-probs, `scores` the deployed selector scores, `target_rank` the truth's
lattice index (-1 when the truth is outside the top-16, i.e. structurally unfixable).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(path):
    d = np.load(path)
    return {k: d[k] for k in d.files if k != "files"}, [str(f) for f in d["files"]]


def alt_argmax(mat):
    """Index (in full 16-space) of the best NON-base candidate."""
    return mat[:, 1:].argmax(-1) + 1


def summarise(data, tag, out):
    lp = data["lp"].astype(np.float32)
    sc = data["scores"].astype(np.float32)
    tr = data["target_rank"].astype(np.int32)
    slot = data["slot"].astype(np.int32)

    covered = tr >= 0
    fixable = covered & (tr > 0)
    draft_alt = alt_argmax(lp)
    sel_alt = alt_argmax(sc)
    draft_ok = draft_alt == tr
    sel_ok = sel_alt == tr
    # The hill the additive correction must climb to promote the truth over the base.
    hill = np.where(covered, lp[:, 0] - lp[np.arange(len(tr)), np.clip(tr, 0, None)], np.nan)

    print(f"\n================ {tag} ================")
    print(f"slots {len(tr)}  covered {covered.sum()} ({100 * covered.mean():.2f}%)  "
          f"fixable {fixable.sum()}")

    print("\n--- M1: does the BACKBONE degrade toward slot 0 like the selector does? ---")
    print("  captured = (alt_r0 - draft_r1) / (1 - draft_r1): the share of the improvement that was")
    print("  AVAILABLE which the selector actually took.  value_add alone is bounded by the")
    print("  backbone's own accuracy, so it cannot be compared across slots; captured can.")
    print(f"  {'slot':>4} {'fixable':>8} {'draft_r1':>9} {'alt_r0':>8} {'value_add':>10} "
          f"{'captured':>9} {'hill_mean':>10} {'hill_med':>9}")
    rows = []
    for h in range(int(slot.max()) + 1):
        m = fixable & (slot == h)
        n = int(m.sum())
        if n == 0:
            continue
        d1, a0 = float(draft_ok[m].mean()), float(sel_ok[m].mean())
        captured = (a0 - d1) / max(1.0 - d1, 1e-9)
        rows.append(dict(slot=h, n=n, draft_r1=d1, alt_r0=a0, value_add=a0 - d1,
                         captured=captured,
                         hill_mean=float(np.nanmean(hill[m])),
                         hill_median=float(np.nanmedian(hill[m]))))
        print(f"  {h:>4} {n:>8} {100 * d1:>8.2f}% {100 * a0:>7.2f}% {100 * (a0 - d1):>+9.2f}pp "
              f"{100 * captured:>8.2f}% "
              f"{rows[-1]['hill_mean']:>10.3f} {rows[-1]['hill_median']:>9.3f}")
    out[f"{tag}/per_slot"] = rows

    print("\n--- M2: condition on the hill.  Does the slot effect survive? ---")
    # Quantile buckets of the hill over fixable slots, so each bucket has equal mass.
    hv = hill[fixable]
    edges = np.nanquantile(hv, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    edges[0], edges[-1] = -np.inf, np.inf
    groups = [(0, 2), (3, 5), (6, 9), (10, 14)]
    print(f"  hill quantile edges: {[round(float(e), 3) for e in edges[1:-1]]}")
    header = "  " + f"{'hill bucket':>22}" + "".join(f"{f'slot {a}-{b}':>14}" for a, b in groups)
    print(header)
    grid = []
    for bi in range(len(edges) - 1):
        lo, hi = edges[bi], edges[bi + 1]
        in_b = fixable & (hill >= lo) & (hill < hi)
        cells, line = [], f"  [{lo:>8.3f},{hi:>8.3f})"
        for a, b in groups:
            m = in_b & (slot >= a) & (slot <= b)
            n = int(m.sum())
            if n < 30:
                cells.append(None)
                line += f"{'-':>14}"
                continue
            acc = float(sel_ok[m].mean())
            cells.append(dict(n=n, alt_r0=acc, draft_r1=float(draft_ok[m].mean())))
            line += f"{100 * acc:>9.2f}%/{n:<4}"
        print(line)
        grid.append(dict(lo=float(lo), hi=float(hi), cells=cells))
    out[f"{tag}/hill_by_slot"] = grid

    print("\n--- M2b: same table for the BACKBONE, to see whether the hill explains IT too ---")
    print(header)
    for bi in range(len(edges) - 1):
        lo, hi = edges[bi], edges[bi + 1]
        in_b = fixable & (hill >= lo) & (hill < hi)
        line = f"  [{lo:>8.3f},{hi:>8.3f})"
        for a, b in groups:
            m = in_b & (slot >= a) & (slot <= b)
            n = int(m.sum())
            line += f"{'-':>14}" if n < 30 else f"{100 * float(draft_ok[m].mean()):>9.2f}%/{n:<4}"
        print(line)

    print("\n--- how the hill is distributed across slots (the M2 driver) ---")
    print(f"  {'slot':>4} {'q25':>8} {'q50':>8} {'q75':>8} {'P(hill>log3)':>13} "
          f"{'P(p0>0.25)':>11}")
    p0 = np.exp(lp[:, 0] - np.log(np.exp(lp).sum(-1) + 1e-30))
    log3 = float(np.log(3.0))
    for h in range(int(slot.max()) + 1):
        m = fixable & (slot == h)
        if not m.any():
            continue
        q = np.nanquantile(hill[m], [0.25, 0.5, 0.75])
        print(f"  {h:>4} {q[0]:>8.3f} {q[1]:>8.3f} {q[2]:>8.3f} "
              f"{100 * float((hill[m] > log3).mean()):>12.2f}% "
              f"{100 * float((p0[m] > 0.25).mean()):>10.2f}%")

    print("\n--- M3 residual: value_add (selector minus backbone) inside hill buckets ---")
    print(header)
    for bi in range(len(edges) - 1):
        lo, hi = edges[bi], edges[bi + 1]
        in_b = fixable & (hill >= lo) & (hill < hi)
        line = f"  [{lo:>8.3f},{hi:>8.3f})"
        for a, b in groups:
            m = in_b & (slot >= a) & (slot <= b)
            n = int(m.sum())
            if n < 30:
                line += f"{'-':>14}"
            else:
                va = float(sel_ok[m].mean()) - float(draft_ok[m].mean())
                line += f"{100 * va:>+9.2f}pp/{n:<3}"
        print(line)

    print("\n--- decode-frequency vs training weight: the loss weights slots UNIFORMLY ---")
    # Uniform anchors give every slot equal training mass, but only slots where the base is WRONG
    # carry repair signal, and the base is right far more often near the anchor.  So the fraction of
    # the repair gradient that lands on slot h is NOT 1/15 -- it is this table.  Decode meanwhile
    # visits slot h only if the walk got that far, which is heavily front-loaded.  The ratio of the
    # two is the importance-weighting error the current objective is making.
    tot = int(fixable.sum())
    chain = out.get("chain_fixable_share")
    print(f"  {'slot':>4} {'train share':>12} {'decode share':>13} {'ratio':>8}")
    shares = []
    for h in range(int(slot.max()) + 1):
        m = fixable & (slot == h)
        if not m.any():
            continue
        tsh = int(m.sum()) / max(tot, 1)
        if chain is not None and h < len(chain):
            dsh = chain[h]
            print(f"  {h:>4} {100 * tsh:>11.2f}% {100 * dsh:>12.2f}% {dsh / max(tsh, 1e-9):>8.2f}x")
        else:
            print(f"  {h:>4} {100 * tsh:>11.2f}% {'n/a':>12} {'n/a':>8}")
        shares.append(tsh)
    out[f"{tag}/train_share"] = shares


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", required=True,
                    help="one or more TF_FEAT_*.npz; each is summarised separately")
    ap.add_argument("--chain-ladder", default=None,
                    help="TF_LADDER_*_chain.json; its CHAIN slot table supplies the decode-side "
                         "fixable share per slot, which is what the training weight should match")
    ap.add_argument("--json-output", default=None)
    args = ap.parse_args()

    out = {}
    if args.chain_ladder:
        ladder = json.loads(Path(args.chain_ladder).read_text())
        table = ladder.get("slot_strata", {}).get("CHAIN")
        if table:
            fx = [float(table[str(h)]["fixable"]) for h in range(len(table))]
            tot = sum(fx) or 1.0
            out["chain_fixable_share"] = [v / tot for v in fx]
        else:
            print(f"warning: no CHAIN slot table found in {args.chain_ladder}; "
                  f"keys={list(ladder)[:8]}")
    for path in args.features:
        data, _ = load(path)
        summarise(data, Path(path).stem, out)
    if args.json_output:
        dest = Path(args.json_output)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(out, indent=2, default=float))
        print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
