#!/usr/bin/env python3
"""Paired comparison of two decode runs over the raw per-prompt records.

Why this exists
---------------
Every architectural conclusion in this project so far was read off a difference of macro means:

    full 6.0438   vs   anchor_self 5.9993   ->   "cross-slot attention is worth -0.0445, i.e. nothing"

That inference is invalid.  Per-prompt acceptance length has a standard deviation of roughly 1.5-2.5
(prompts differ enormously in how predictable they are), so at n=20 a single domain mean carries
SEM ~= 0.45 and the six-domain macro carries SEM ~= 0.18.  A -0.0445 macro difference is a quarter of
one standard error.  Resolving it by brute force would need ~1600 prompts per domain.

But the two arms are evaluated on *the same prompts*.  Prompt difficulty is a shared additive term
and cancels exactly in the per-prompt difference, so the paired standard error is governed by
std(d_i), not std(x_i) -- typically an order of magnitude smaller here.  This is the standard
variance-reduction argument for paired designs and it costs zero additional compute: the raw records
were already written to disk by every run.

What is compared
----------------
Prompts are matched on ``(dataset, prompt_sha256)``, never on ``prompt_index``, so a run with a
different sample count or shuffle still aligns correctly and a mismatch is reported rather than
silently averaged over.

The per-prompt statistic is ``accepted_sum / len(acceptance_lengths)`` -- mean accepted tokens per
block, which is exactly the quantity the headline macro averages.

Two aggregates are printed:
  * per-domain paired mean difference with its paired SEM and t statistic;
  * the macro difference, whose variance is the sum of per-domain variances over 36 (six independent
    domain means, each divided by 6 in the macro), i.e. SEM_macro = sqrt(sum(sem_d^2))/6.

A two-sided p-value is computed from the normal approximation; with n=20 per domain the t
distribution is slightly wider, so p is reported as an optimistic bound and the t statistic is
printed alongside so the reader can apply the exact critical value if it matters.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from collections import defaultdict


DOMAINS = ["gsm8k", "math500", "humaneval", "mbpp", "alpaca", "mt-bench"]


def load(path: pathlib.Path, mode: str | None) -> dict[tuple[str, str], float]:
    payload = json.loads(path.read_text())
    out: dict[tuple[str, str], float] = {}
    modes = set()
    for rec in payload["results"]:
        modes.add(rec["mode"])
        if mode is not None and rec["mode"] != mode:
            continue
        lengths = rec["acceptance_lengths"]
        if not lengths:
            continue
        key = (rec["dataset"], rec["prompt_sha256"])
        if key in out:
            raise SystemExit(f"{path.name}: duplicate {key} for mode={rec['mode']}")
        out[key] = rec["accepted_sum"] / len(lengths)
    if not out:
        raise SystemExit(f"{path.name}: no records for mode={mode!r}; available modes: {sorted(modes)}")
    return out


def erf_p(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="baseline run json")
    ap.add_argument("--b", required=True, help="variant run json")
    ap.add_argument("--mode", default=None, help="decode mode to select in BOTH files")
    ap.add_argument("--mode-a", default=None, help="override mode for --a")
    ap.add_argument("--mode-b", default=None, help="override mode for --b")
    ap.add_argument("--label-a", default=None)
    ap.add_argument("--label-b", default=None)
    args = ap.parse_args()

    pa, pb = pathlib.Path(args.a), pathlib.Path(args.b)
    la = args.label_a or pa.stem
    lb = args.label_b or pb.stem
    a = load(pa, args.mode_a or args.mode)
    b = load(pb, args.mode_b or args.mode)

    only_a, only_b = set(a) - set(b), set(b) - set(a)
    if only_a or only_b:
        print(f"[warn] unmatched prompts: {len(only_a)} only in A, {len(only_b)} only in B "
              f"(dropped; matched {len(set(a) & set(b))})")
    shared = sorted(set(a) & set(b))
    if not shared:
        raise SystemExit("no shared prompts")

    per_domain = defaultdict(list)
    for key in shared:
        per_domain[key[0]].append(b[key] - a[key])

    order = [d for d in DOMAINS if d in per_domain] + sorted(set(per_domain) - set(DOMAINS))
    print(f"paired comparison   A={la}   B={lb}   (statistic: B - A)")
    print(f"{'domain':<10} {'n':>3} {'A mean':>8} {'B mean':>8} {'diff':>8} {'paired SEM':>11} "
          f"{'t':>7} {'unpaired SEM':>13}")
    means_a, means_b, diffs, sems = [], [], [], []
    for dom in order:
        d = per_domain[dom]
        n = len(d)
        va = [a[k] for k in shared if k[0] == dom]
        vb = [b[k] for k in shared if k[0] == dom]
        ma, mb = sum(va) / n, sum(vb) / n
        md = sum(d) / n
        if n > 1:
            var = sum((x - md) ** 2 for x in d) / (n - 1)
            sem = math.sqrt(var / n)
            var_a = sum((x - ma) ** 2 for x in va) / (n - 1)
            sem_unpaired = math.sqrt(2.0 * var_a / n)
        else:
            sem = sem_unpaired = float("nan")
        t = md / sem if sem and sem == sem and sem > 0 else float("nan")
        print(f"{dom:<10} {n:>3} {ma:8.4f} {mb:8.4f} {md:+8.4f} {sem:11.4f} {t:7.2f} {sem_unpaired:13.4f}")
        means_a.append(ma)
        means_b.append(mb)
        diffs.append(md)
        sems.append(sem)

    k = len(diffs)
    macro_a, macro_b = sum(means_a) / k, sum(means_b) / k
    macro_d = sum(diffs) / k
    macro_sem = math.sqrt(sum(s * s for s in sems)) / k
    t = macro_d / macro_sem if macro_sem > 0 else float("nan")
    p = erf_p(t)
    print("-" * 78)
    print(f"{'MACRO':<10} {'':>3} {macro_a:8.4f} {macro_b:8.4f} {macro_d:+8.4f} {macro_sem:11.4f} {t:7.2f}")
    print(f"\nmacro difference {macro_d:+.4f} +- {macro_sem:.4f} (paired SEM)   "
          f"t={t:.2f}   two-sided p~{p:.3g} (normal approx.)")
    hi = macro_d + 1.96 * macro_sem
    lo = macro_d - 1.96 * macro_sem
    print(f"95% CI [{lo:+.4f}, {hi:+.4f}]")
    if lo <= 0.0 <= hi:
        print("VERDICT: not distinguishable from zero at this sample size.")
        print(f"         the data are also consistent with an effect as large as "
              f"{max(abs(lo), abs(hi)):+.4f}, so 'no effect' is NOT established either.")
    else:
        print(f"VERDICT: significant, sign is {'positive (B better)' if macro_d > 0 else 'negative (B worse)'}.")


if __name__ == "__main__":
    main()
