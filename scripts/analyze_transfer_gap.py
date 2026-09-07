#!/usr/bin/env python3
"""Is the training-corpus -> benchmark gap real domain shift, or a composition/collider effect?

The raw comparison (PerfectBlend POLICY alt_rank_0 80.90% vs benchmark 72.24%) cannot answer this,
because conditioning on "the base is wrong but the truth is in the top-16" selects DIFFERENT residual
errors in each corpus: an easier corpus can leave rarer but more pathological mistakes.  That is a
collider, and it is the objection that must be cleared before spending a training run on more data.

The fix is direct standardization.  Stratify both corpora by the covariates that plausibly drive
ranking difficulty and are measurable without labels:
    slot group      -- position in the block
    hill quintile   -- lp[0] - lp[target], how far the base has to be overcome
    draft rank      -- where the DRAFT itself puts the truth among its alternatives
then report the benchmark-minus-training difference in alt_rank_0 reweighted onto ONE common cell
distribution.  If the standardized gap collapses, the raw gap was composition and broader data is
the wrong lever.  If it survives, the gap is in the conditional and data/objective is the lever.

Also reports the benchmark gap per domain, because "domain shift" is only actionable if it is not
simply "chat prompts are different from everything".
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

DOMAIN_RE = re.compile(r"([a-zA-Z0-9\-]+)_\d+\.ckpt$")


def load(path):
    d = np.load(path)
    return {k: d[k] for k in d.files if k != "files"}, [str(f) for f in d["files"]]


def frontier(data):
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
    return reach


def covariates(data, hill_edges=None):
    lp = data["lp"].astype(np.float32)
    sc = data["scores"].astype(np.float32)
    tr = data["target_rank"].astype(np.int64)
    slot = data["slot"].astype(np.int64)
    n = len(tr)
    safe = np.clip(tr, 0, None)
    hill = lp[:, 0] - lp[np.arange(n), safe]
    # Where the DRAFT ranks the truth among its own alternatives: the difficulty of the ranking task
    # as the backbone sees it, independent of the selector.
    draft_order = np.argsort(-lp[:, 1:], axis=1) + 1
    draft_rank = np.argmax(draft_order == safe[:, None], axis=1)
    sel_best = sc[:, 1:].argmax(1) + 1
    alt_ok = (tr > 0) & (sel_best == tr)
    if hill_edges is None:
        m = tr > 0
        hill_edges = np.quantile(hill[m], [0.2, 0.4, 0.6, 0.8])
    hill_b = np.digitize(hill, hill_edges)
    slot_b = np.digitize(slot, [3, 6, 10])
    draft_b = np.digitize(draft_rank, [1, 2, 4, 8])
    return dict(hill_b=hill_b, slot_b=slot_b, draft_b=draft_b, alt_ok=alt_ok,
                repairable=tr > 0), hill_edges


def cells(cov, mask):
    return (cov["slot_b"][mask] * 100 + cov["hill_b"][mask] * 10 + cov["draft_b"][mask])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-features", required=True)
    ap.add_argument("--eval-features", required=True)
    ap.add_argument("--min-cell", type=int, default=40)
    ap.add_argument("--json-output", default=None)
    args = ap.parse_args()

    tr_d, tr_f = load(args.train_features)
    ev_d, ev_f = load(args.eval_features)
    tr_reach, ev_reach = frontier(tr_d), frontier(ev_d)
    # One shared set of hill edges, taken from the TRAINING corpus, so the bins mean the same thing
    # in both -- quantile bins fitted separately would silently standardize the covariate away.
    cov_t, edges = covariates(tr_d)
    cov_e, _ = covariates(ev_d, edges)

    mt = tr_reach & cov_t["repairable"]
    me = ev_reach & cov_e["repairable"]
    raw_t = float(cov_t["alt_ok"][mt].mean())
    raw_e = float(cov_e["alt_ok"][me].mean())
    print(f"reachable repairable slots: train {int(mt.sum())}  benchmark {int(me.sum())}")
    print(f"raw alt_rank_0: train {100 * raw_t:.2f}%  benchmark {100 * raw_e:.2f}%  "
          f"gap {100 * (raw_e - raw_t):+.2f}pp")

    ct, ce = cells(cov_t, mt), cells(cov_e, me)
    ok_t, ok_e = cov_t["alt_ok"][mt], cov_e["alt_ok"][me]
    common = sorted(set(np.unique(ct)) & set(np.unique(ce)))
    rows, wsum, dsum, tsum, esum = [], 0.0, 0.0, 0.0, 0.0
    for c in common:
        a, b = ct == c, ce == c
        na, nb = int(a.sum()), int(b.sum())
        if na < args.min_cell or nb < args.min_cell:
            continue
        pa, pb = float(ok_t[a].mean()), float(ok_e[b].mean())
        # Standardize onto the BENCHMARK cell distribution: the question is how the selector would do
        # on benchmark-like work if it were as good there as it is on its own corpus.
        w = nb
        rows.append(dict(cell=int(c), n_train=na, n_eval=nb, train=pa, eval=pb, weight=w))
        wsum += w
        dsum += w * (pb - pa)
        tsum += w * pa
        esum += w * pb
    print(f"\ncommon cells used {len(rows)} (>= {args.min_cell} in both), "
          f"covering {int(wsum)}/{int(me.sum())} = {100 * wsum / max(int(me.sum()), 1):.1f}% "
          f"of benchmark repairable slots")
    print(f"standardized (benchmark cell mix): train {100 * tsum / wsum:.2f}%  "
          f"benchmark {100 * esum / wsum:.2f}%  gap {100 * dsum / wsum:+.2f}pp")
    print(f"  -> raw gap {100 * (raw_e - raw_t):+.2f}pp, standardized gap "
          f"{100 * dsum / wsum:+.2f}pp: "
          f"{100 * (1 - (dsum / wsum) / (raw_e - raw_t)):.0f}% of the raw gap was composition"
          if abs(raw_e - raw_t) > 1e-9 else "")

    print("\n=== per-domain, benchmark reachable repairable slots ===")
    dom = np.array([DOMAIN_RE.search(Path(f).name).group(1) if DOMAIN_RE.search(Path(f).name)
                    else "unknown" for f in ev_f])
    sample_e = ev_d["sample"].astype(np.int64)[me]
    dom_e = dom[sample_e]
    lp = ev_d["lp"].astype(np.float32)
    tr = ev_d["target_rank"].astype(np.int64)
    n = len(tr)
    draft_best = lp[:, 1:].argmax(1) + 1
    draft_ok = (tr > 0) & (draft_best == tr)
    print("  std_gap standardizes each domain onto ITS OWN cell mix and compares against the "
          "training corpus in the same cells: a deficit that survives it is conditional, i.e. "
          "attackable by data/objective, not an aleatoric ceiling.")
    print(f"  {'domain':<12} {'n':>7} {'draft_r1':>9} {'alt_r0':>8} {'captured':>9} "
          f"{'train@cells':>12} {'std_gap':>9} {'cover':>7}")
    per_dom = {}
    for d in sorted(set(dom_e)):
        k = dom_e == d
        d1 = float(draft_ok[me][k].mean())
        a0 = float(cov_e["alt_ok"][me][k].mean())
        cap = (a0 - d1) / max(1 - d1, 1e-9)
        cd = ce[k]
        ok_d = cov_e["alt_ok"][me][k]
        w, num_t, num_e = 0.0, 0.0, 0.0
        for c in np.unique(cd):
            a = ct == c
            b = cd == c
            if int(a.sum()) < args.min_cell or int(b.sum()) < args.min_cell:
                continue
            w += int(b.sum())
            num_t += int(b.sum()) * float(ok_t[a].mean())
            num_e += int(b.sum()) * float(ok_d[b].mean())
        std_t = num_t / max(w, 1e-9)
        std_e = num_e / max(w, 1e-9)
        per_dom[d] = {"n": int(k.sum()), "draft_r1": d1, "alt_r0": a0, "captured": cap,
                      "train_at_cells": std_t, "std_gap": std_e - std_t,
                      "coverage": w / max(int(k.sum()), 1)}
        print(f"  {d:<12} {int(k.sum()):>7} {100 * d1:>8.2f}% {100 * a0:>7.2f}% {100 * cap:>8.2f}% "
              f"{100 * std_t:>11.2f}% {100 * (std_e - std_t):>+8.2f}pp "
              f"{100 * w / max(int(k.sum()), 1):>6.1f}%")

    if args.json_output:
        Path(args.json_output).write_text(json.dumps(
            {"raw": {"train": raw_t, "eval": raw_e},
             "standardized": {"train": tsum / wsum, "eval": esum / wsum, "gap": dsum / wsum},
             "cells": rows, "per_domain": per_dom}, indent=2, default=float))
        print(f"\nwrote {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
