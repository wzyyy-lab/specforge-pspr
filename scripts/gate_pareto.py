#!/usr/bin/env python3
"""Exact (recovered, wrong_override, destroy) frontier for any gate threshold, from one forward.

The deployed rule is `softmax(scores)[1:].max() > tau + rho * softmax(scores)[0]`.  At tau=0 it is
monotone in the single scalar

    d = log softmax(scores)[best_alt] - log softmax(scores)[0]

because the normaliser cancels, so `override <=> d > log rho`.  ``measure_ranker_teacher_forced.py
--dump-decisions`` records d per reachable decision together with the label class, which makes every
counter available for ANY per-slot / confidence-dependent threshold with no further model evaluation.
The d-form was verified bit-equivalent to ``LatticePathSelector.select_margin_gate`` (gate G6,
0/19395 mismatches).

STRUCTURAL FACT THIS SCRIPT WAS WRITTEN TO CHECK
-----------------------------------------------
`p_alt <= 1 - p0` always, so `p_alt > rho * p0` is INFEASIBLE whenever `p0 >= 1/(1+rho)`.  At the
deployed rho=3 that is `p0 >= 0.25`.  Every slot whose selector posterior on KEEP exceeds 0.25 is
therefore unrepairable BY CONSTRUCTION, no matter how confident the selector is about the alternative.
This script measures how much of the repair opportunity sits inside that dead zone.

COST MODEL
----------
A repair recovered at slot h converts a block that would have ended at h into one that continues; a
destroy at slot h ends a block that would have continued.  Both are worth the expected number of
further accepted slots, w_h, estimated from the chain's own survival curve.  A "wrong override" at a
fixable slot costs ZERO: the base was already wrong there, so the block ended at h either way.  Hence
the offline objective is

    net = sum_h w_h * (recovered_h - destroy_h)

and it is used ONLY to propose candidate rules.  Every proposal is then verified by re-running the
harness, which re-simulates the chain under the new rule -- the offline objective ignores the fact
that changing the rule changes which blocks exist.

DISCIPLINE
----------
Thresholds are fitted on one half of the prompts and scored on the other half, split by prompt index
parity WITHIN each dataset so both halves cover all six domains.  Per-slot (15 free parameters) is
reported next to 1-3 parameter families precisely so the overfitting cost is visible.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

KIND_BASE_RIGHT, KIND_FIX_TRUE_ALT, KIND_FIX_WRONG_ALT, KIND_UNFIXABLE = 0, 1, 2, 3


def load(npz_path, population):
    d = np.load(npz_path)
    files = [Path(str(f)).name for f in d["files"]]
    rec = {
        "sample": d[f"{population}_sample"].astype(np.int32),
        "slot": d[f"{population}_slot"].astype(np.int32),
        "kind": d[f"{population}_kind"].astype(np.int8),
        "margin": d[f"{population}_margin"].astype(np.float64),
        "base_lp": d[f"{population}_base_lp"].astype(np.float64),
        "truth_top_alt": d[f"{population}_truth_top_alt"].astype(bool),
    }
    datasets, indices = [], []
    for name in files:
        stem = name.rsplit(".", 1)[0]
        ds, _, idx = stem.rpartition("_")
        datasets.append(ds)
        indices.append(int(idx) if idx.isdigit() else 0)
    rec["dataset"] = np.array(datasets)[rec["sample"]]
    rec["prompt_index"] = np.array(indices)[rec["sample"]]
    return rec, files


def survival_weights(rec, horizon):
    """w_h = E[further accepted slots | a block reached slot h], from the chain's own occupancy."""
    counts = np.array([int((rec["slot"] == h).sum()) for h in range(horizon)], dtype=np.float64)
    total = counts.sum()
    weights = np.zeros(horizon)
    cumulative = 0.0
    for h in range(horizon):
        cumulative += counts[h]
        weights[h] = (total - cumulative) / max(counts[h], 1.0)
    return weights, counts


def evaluate(rec, log_rho_per_row, weights, mask=None):
    keep = np.ones(len(rec["slot"]), dtype=bool) if mask is None else mask
    override = (rec["margin"] > log_rho_per_row) & keep
    kind = rec["kind"]
    out = {
        "recovered": int((override & (kind == KIND_FIX_TRUE_ALT)).sum()),
        "wrong_override": int((override & (kind == KIND_FIX_WRONG_ALT)).sum()),
        "unfixable_override": int((override & (kind == KIND_UNFIXABLE)).sum()),
        "destroy": int((override & (kind == KIND_BASE_RIGHT)).sum()),
        "fixable": int((keep & np.isin(kind, [KIND_FIX_TRUE_ALT, KIND_FIX_WRONG_ALT])).sum()),
        "base_right": int((keep & (kind == KIND_BASE_RIGHT)).sum()),
        "decisions": int(keep.sum()),
    }
    net = 0.0
    for h in range(len(weights)):
        at = keep & (rec["slot"] == h)
        ov = override & at
        rec_h = int((ov & (kind == KIND_FIX_TRUE_ALT)).sum())
        des_h = int((ov & (kind == KIND_BASE_RIGHT)).sum())
        net += weights[h] * (rec_h - des_h)
    out["net"] = net
    out["recovered_rate"] = out["recovered"] / max(out["fixable"], 1)
    out["destroy_rate"] = out["destroy"] / max(out["base_right"], 1)
    return out


def field(rec, mode, params, horizon):
    if mode == "global":
        return np.full(len(rec["slot"]), params[0])
    if mode == "per_slot":
        table = np.asarray(params, dtype=np.float64)
        return table[rec["slot"]]
    if mode == "affine_slot":
        a, b = params
        return a + b * rec["slot"]
    if mode == "affine_base":
        a, b = params
        return a + b * rec["base_lp"]
    if mode == "affine_both":
        a, b, c = params
        return a + b * rec["slot"] + c * rec["base_lp"]
    raise SystemExit(mode)


def search(rec, mode, weights, horizon, fit_mask):
    grid_a = np.arange(-2.5, 3.01, 0.05)
    best = None
    if mode == "global":
        candidates = [(a,) for a in grid_a]
    elif mode == "affine_slot":
        candidates = [(a, b) for a in grid_a for b in np.arange(-0.30, 0.301, 0.02)]
    elif mode == "affine_base":
        candidates = [(a, b) for a in grid_a for b in np.arange(-1.5, 1.501, 0.05)]
    elif mode == "affine_both":
        candidates = [
            (a, b, c)
            for a in np.arange(-2.5, 3.01, 0.1)
            for b in np.arange(-0.30, 0.301, 0.05)
            for c in np.arange(-1.5, 1.501, 0.1)
        ]
    elif mode == "per_slot":
        # separable: net is a sum of independent per-slot terms with positive weights
        table = []
        for h in range(horizon):
            at = fit_mask & (rec["slot"] == h)
            if not at.any():
                table.append(math.log(3.0))
                continue
            options = np.unique(np.concatenate([rec["margin"][at], [-50.0, 50.0]]))
            options = options[:: max(1, len(options) // 400)]
            scores = []
            for value in options:
                ov = at & (rec["margin"] > value)
                scores.append(
                    int((ov & (rec["kind"] == KIND_FIX_TRUE_ALT)).sum())
                    - int((ov & (rec["kind"] == KIND_BASE_RIGHT)).sum())
                )
            table.append(float(options[int(np.argmax(scores))]))
        return tuple(table)
    else:
        raise SystemExit(mode)
    for params in candidates:
        value = evaluate(rec, field(rec, mode, params, horizon), weights, fit_mask)["net"]
        if best is None or value > best[0]:
            best = (value, params)
    return best[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--population", default="CHAIN")
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--json-output", default=None)
    args = ap.parse_args()

    rec, files = load(args.decisions, args.population)
    weights, counts = survival_weights(rec, args.horizon)
    print(f"{args.population}: {len(rec['slot'])} decisions over {len(files)} prompts")
    print("  survival weights w_h = E[further accepted slots | reached h]")
    print("   " + " ".join(f"{w:5.2f}" for w in weights))

    # --- the structural dead zone --------------------------------------------------------------
    p0 = np.exp(rec["base_lp"])
    print("\n=== structural dead zone of `p_alt > rho * p0` ===")
    print("  a repair is INFEASIBLE at any slot with p0 >= 1/(1+rho)")
    fixable = np.isin(rec["kind"], [KIND_FIX_TRUE_ALT, KIND_FIX_WRONG_ALT])
    truth_alt = rec["kind"] == KIND_FIX_TRUE_ALT
    for rho in (1.0, 2.0, 3.0):
        limit = 1.0 / (1.0 + rho)
        dead = p0 >= limit
        print(f"  rho={rho:>3}  p0 threshold {limit:.3f}  "
              f"dead share of fixable {pctf(dead & fixable, fixable)}  "
              f"dead share of 'truth is best alt' {pctf(dead & truth_alt, truth_alt)}  "
              f"dead share of base-right {pctf(dead & (rec['kind'] == KIND_BASE_RIGHT), rec['kind'] == KIND_BASE_RIGHT)}")
    print("\n  by slot: share of 'truth is best alt' decisions inside the rho=3 dead zone")
    dead3 = p0 >= 0.25
    print(f"  {'slot':>4} {'truth_alt':>10} {'dead':>8} {'dead%':>7} {'mean p0':>8}")
    for h in range(args.horizon):
        at = truth_alt & (rec["slot"] == h)
        n = int(at.sum())
        if not n:
            continue
        print(f"  {h:>4} {n:>10} {int((at & dead3).sum()):>8} "
              f"{100.0 * int((at & dead3).sum()) / n:>6.1f}% "
              f"{p0[rec['slot'] == h].mean():>8.3f}")

    # --- honest split --------------------------------------------------------------------------
    even = (rec["prompt_index"] % 2) == 0
    halves = {"A(even prompts)": even, "B(odd prompts)": ~even}
    baseline_log_rho = math.log(3.0)
    print("\n=== baseline rho=3 ===")
    for name, mask in halves.items():
        r = evaluate(rec, np.full(len(rec["slot"]), baseline_log_rho), weights, mask)
        print(f"  {name:<16} recovered {r['recovered']}/{r['fixable']} = "
              f"{100 * r['recovered_rate']:.2f}%  destroy {r['destroy']}/{r['base_right']} = "
              f"{100 * r['destroy_rate']:.3f}%  net {r['net']:.1f}")

    results = {}
    print("\n=== fitted rules: fit on one half, score on the OTHER half ===")
    print(f"  {'mode':<12} {'fit half':<16} {'params':<44} "
          f"{'heldout recov':>14} {'heldout destroy':>16} {'heldout net':>12}")
    for mode in ("global", "affine_slot", "affine_base", "affine_both", "per_slot"):
        results[mode] = []
        for name, mask in halves.items():
            params = search(rec, mode, weights, args.horizon, mask)
            other = ~mask
            r = evaluate(rec, field(rec, mode, params, args.horizon), weights, other)
            base = evaluate(rec, np.full(len(rec["slot"]), baseline_log_rho), weights, other)
            text = ",".join(f"{v:.2f}" for v in np.atleast_1d(params))
            print(f"  {mode:<12} {name:<16} {text[:44]:<44} "
                  f"{100 * r['recovered_rate']:>13.2f}% {100 * r['destroy_rate']:>15.3f}% "
                  f"{r['net']:>12.1f}  (base net {base['net']:.1f}, "
                  f"recov {100 * base['recovered_rate']:.2f}%, "
                  f"destroy {100 * base['destroy_rate']:.3f}%)")
            results[mode].append(
                {"fit_half": name, "params": [float(v) for v in np.atleast_1d(params)],
                 "heldout": r, "heldout_baseline": base}
            )

    if args.json_output:
        dest = Path(args.json_output)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(
            {"weights": weights.tolist(), "slot_counts": counts.tolist(), "results": results},
            indent=2, default=float))
        print(f"\nwrote {dest}")
    return 0


def pctf(subset, universe):
    n = int(universe.sum())
    return f"{int((subset).sum())}/{n} = {100.0 * int(subset.sum()) / max(n, 1):.1f}%"


if __name__ == "__main__":
    raise SystemExit(main())
