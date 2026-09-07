#!/usr/bin/env python3
"""Effective per-slot training weight under the DEPLOYED objective, vs decode's own slot mix.

An earlier version of this accounting used UNIFORM slot weights and concluded slot 0 is
under-weighted 7.06x.  That was wrong about the deployed run: `dflash2_selector_weight_mode:
uniform_frontier_boost` with `dflash2_selector_frontier_boost: 3.0` already puts 4x weight on the
POLICY frontier (`dflash_family_model.py:1139-1177`, `first_error = reachable & ~policy_ok`), which
is precisely the decode-relevant slot.  This recomputes the mismatch with the real weights.

Two shares are reported because they answer different questions:
  * over COVERED slots      -- how the CE's total mass is distributed, i.e. capacity allocation;
  * over REPAIRABLE slots   -- how the mass that can teach a repair is distributed, which is what
                              the `captured` table is about.
The decode reference is the reachable-set decision / repair mix from the same checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def frontier_and_weights(data, boost, horizon=15):
    slot = data["slot"].astype(np.int64)
    sample = data["sample"].astype(np.int64)
    block = data["block"].astype(np.int64)
    tr = data["target_rank"].astype(np.int64)
    pick = data["pick_index"].astype(np.int64)
    policy_ok = (tr >= 0) & (pick == tr)

    order = np.lexsort((slot, block, sample))
    key = sample[order] * (block.max() + 1) + block[order]
    new_group = np.empty(len(key), dtype=bool)
    new_group[0] = True
    new_group[1:] = key[1:] != key[:-1]
    fail = (~policy_ok[order]).astype(np.int64)
    csum = np.cumsum(fail)
    gs = np.maximum.accumulate(np.where(new_group, np.arange(len(key)), 0))
    prior = csum - fail - np.where(gs > 0, csum[gs - 1], 0)
    reach = np.zeros(len(key), dtype=bool)
    reach[order] = prior == 0
    first_error = reach & ~policy_ok
    weight = 1.0 + boost * first_error.astype(np.float64)
    return reach, first_error, weight


def shares(mask, weight, slot, horizon=15):
    w = np.where(mask, weight, 0.0)
    tot = w.sum()
    return np.array([w[slot == h].sum() / max(tot, 1e-12) for h in range(horizon)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-features", required=True)
    ap.add_argument("--eval-features", required=True)
    ap.add_argument("--boost", type=float, default=3.0)
    ap.add_argument("--json-output", default=None)
    args = ap.parse_args()

    tr_d = {k: v for k, v in np.load(args.train_features).items() if k != "files"}
    ev_d = {k: v for k, v in np.load(args.eval_features).items() if k != "files"}

    out = {}
    for tag, d, boost in (("uniform (boost=0)", tr_d, 0.0),
                          (f"deployed (boost={args.boost})", tr_d, args.boost)):
        reach, first_error, weight = frontier_and_weights(d, boost)
        slot = d["slot"].astype(np.int64)
        tr = d["target_rank"].astype(np.int64)
        covered = tr >= 0
        repairable = tr > 0
        out[tag] = {
            "covered_share": shares(covered, weight, slot).tolist(),
            "repairable_share": shares(repairable, weight, slot).tolist(),
            "frontier_mass_frac": float(
                (weight[first_error]).sum() / max(weight[covered | ~covered].sum(), 1e-12)
            ),
            "mean_weight": float(weight.mean()),
        }

    # decode reference: the reachable set of the benchmark corpus, unweighted (the frontier
    # restriction itself produces decode's front-loaded mix)
    reach_e, first_e, _ = frontier_and_weights(ev_d, 0.0)
    slot_e = ev_d["slot"].astype(np.int64)
    tr_e = ev_d["target_rank"].astype(np.int64)
    dec = np.array([int(((slot_e == h) & reach_e).sum()) for h in range(15)], dtype=np.float64)
    rep = np.array([int(((slot_e == h) & reach_e & (tr_e > 0)).sum()) for h in range(15)],
                   dtype=np.float64)
    out["decode"] = {"decision_share": (dec / dec.sum()).tolist(),
                     "repair_share": (rep / rep.sum()).tolist()}

    dep = out[f"deployed (boost={args.boost})"]
    uni = out["uniform (boost=0)"]
    print(f"deployed mean weight {dep['mean_weight']:.4f}  "
          f"(policy frontier carries {100 * dep['frontier_mass_frac']:.2f}% of total mass)")
    print("\n=== share of the REPAIR-teaching mass per slot ===")
    print(f"  {'slot':>4} {'uniform':>9} {'deployed':>9} {'decode':>9} "
          f"{'ratio dep':>10} {'ratio uni':>10}")
    for h in range(15):
        u, p, c = uni["repairable_share"][h], dep["repairable_share"][h], out["decode"]["repair_share"][h]
        print(f"  {h:>4} {100 * u:>8.2f}% {100 * p:>8.2f}% {100 * c:>8.2f}% "
              f"{c / max(p, 1e-12):>9.2f}x {c / max(u, 1e-12):>9.2f}x")
    print("\n=== share of ALL covered-slot CE mass per slot ===")
    print(f"  {'slot':>4} {'uniform':>9} {'deployed':>9} {'decode':>9} {'ratio dep':>10}")
    for h in range(15):
        u, p, c = uni["covered_share"][h], dep["covered_share"][h], out["decode"]["decision_share"][h]
        print(f"  {h:>4} {100 * u:>8.2f}% {100 * p:>8.2f}% {100 * c:>8.2f}% {c / max(p, 1e-12):>9.2f}x")

    if args.json_output:
        Path(args.json_output).write_text(json.dumps(out, indent=2, default=float))
        print(f"\nwrote {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
