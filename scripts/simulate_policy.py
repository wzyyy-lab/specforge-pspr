#!/usr/bin/env python3
"""Exact offline policy simulator: turn any decode-side rule into real accept length, no GPU.

WHY THIS REPLACES THE PARETO SCAN
---------------------------------
Every gate study so far optimised a proxy -- `recovered` at matched `destroy` -- on a population
that the rule itself changes.  Three separate objections follow, all of them fair:

  1. the reachable set is computed under rho=3, so changing the rule changes the population, and the
     "recompute frontier -> repick -> recompute" loop is not a contraction and need not converge;
  2. `recovered` and `destroy` are counted per slot with equal value, but a repair at slot 0 buys a
     whole block's continuation while one at slot 14 buys a single token, and symmetrically for
     destroy;
  3. `scripts/gate_pareto.py:77` weights a repair by `E[further accepted slots]`, which is short by
     exactly the token the repair itself earns -- slot 14 gets weight 0 instead of 1 -- so the scan
     was biased against late-slot repairs in the direction of its own conclusion.

All three vanish if the objective is simulated rather than approximated.  The feature dump contains
EVERY supervised anchor of every sequence, with each anchor's absolute token position, its 15 slots'
lattices, the selector's 16 scores and the truth's lattice index.  Because the lattice at an anchor
depends only on the anchor and the ground-truth context (never on how the walk arrived), a rule's
whole trajectory can be replayed exactly:

    accept(block) = number of leading slots whose pick equals the truth
    cursor       += accept + 1                      (the verifier's bonus token)

and the reported statistic is decode's own: mean(accept + 1) per prompt, then per domain, then macro.

Validation gate: replaying the deployed rule must reproduce the harness's CHAIN accept length and
block count on the same dump.  If it does not, the simulator is wrong and nothing below is usable.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def load(path):
    d = np.load(path)
    return {k: d[k] for k in d.files if k != "files"}, [str(f) for f in d["files"]]


def to_blocks(data, horizon=15):
    """Dense [n_blocks, horizon] views keyed by (sample, anchor position).

    Slots missing from the dump (weight_mask == 0, i.e. unsupervised or past the sequence end) are
    marked invalid; a block truncates there because decode cannot accept a token it never drafted.
    """
    sample = data["sample"].astype(np.int64)
    anchor = data["anchor"].astype(np.int64)
    slot = data["slot"].astype(np.int64)

    key = sample * (anchor.max() + 1) + anchor
    uniq, inv = np.unique(key, return_inverse=True)
    n = len(uniq)
    out = {
        "sample": np.zeros(n, dtype=np.int64),
        "anchor": np.zeros(n, dtype=np.int64),
        "valid": np.zeros((n, horizon), dtype=bool),
        "target_rank": np.full((n, horizon), -1, dtype=np.int64),
        "d": np.zeros((n, horizon), dtype=np.float32),
        "best_alt": np.zeros((n, horizon), dtype=np.int64),
        "log_p0_sel": np.zeros((n, horizon), dtype=np.float32),
        "log_p0_draft": np.zeros((n, horizon), dtype=np.float32),
        "pick_deployed": np.zeros((n, horizon), dtype=np.int64),
    }
    out["sample"][inv] = sample
    out["anchor"][inv] = anchor
    out["valid"][inv, slot] = True
    out["target_rank"][inv, slot] = data["target_rank"].astype(np.int64)
    out["pick_deployed"][inv, slot] = data["pick_index"].astype(np.int64)

    sc = data["scores"].astype(np.float32)
    lp = data["lp"].astype(np.float32)
    log_g = sc - logsumexp(sc)
    log_u = lp - logsumexp(lp)
    best_off = log_g[:, 1:].argmax(-1)
    out["d"][inv, slot] = log_g[np.arange(len(sc)), best_off + 1] - log_g[:, 0]
    out["best_alt"][inv, slot] = best_off + 1
    out["log_p0_sel"][inv, slot] = log_g[:, 0]
    out["log_p0_draft"][inv, slot] = log_u[:, 0]
    order = np.lexsort((out["anchor"], out["sample"]))
    return {k: (v[order] if v.ndim else v) for k, v in out.items()}


def logsumexp(x):
    m = x.max(-1, keepdims=True)
    return m + np.log(np.exp(x - m).sum(-1, keepdims=True))


def scatter_extra(data, values, blocks, horizon=15):
    """Place a per-row learned score onto the dense [n_blocks, horizon] grid."""
    sample = data["sample"].astype(np.int64)
    anchor = data["anchor"].astype(np.int64)
    slot = data["slot"].astype(np.int64)
    key = sample * (anchor.max() + 1) + anchor
    uniq, inv = np.unique(key, return_inverse=True)
    grid = np.full((len(uniq), horizon), -1e9, dtype=np.float32)
    grid[inv, slot] = values.astype(np.float32)
    blk_key = blocks["sample"] * (anchor.max() + 1) + blocks["anchor"]
    pos = np.searchsorted(uniq, blk_key)
    return grid[pos]


def split_prompts(files, seed=0):
    """Prompt-level halves, stratified by domain, so a fitted rule is confirmed out of sample.

    Slot-level or block-level splits would leak: blocks from one prompt share a context and a
    trajectory, so a rule tuned on some of a prompt's blocks is already tuned on the rest.
    """
    dom = domains(files)
    rng = np.random.default_rng(seed)
    fit = np.zeros(len(files), dtype=bool)
    for d in sorted(set(dom)):
        idx = np.nonzero(dom == d)[0]
        rng.shuffle(idx)
        fit[idx[: len(idx) // 2]] = True
    return fit


def macro_on(blocks, fire, files, dom, keep_prompts):
    mask = keep_prompts[blocks["sample"]]
    sub = {k: (v[mask] if getattr(v, "ndim", 0) else v) for k, v in blocks.items()}
    accept = accept_lengths(sub, fire[mask])
    mean, counts, _ = chain(sub, accept, len(files))
    _, mA, mB = macro(mean, counts, dom)
    return mB, mA


def accept_lengths(blocks, fire):
    """accept = leading run of correct picks, per block, for a boolean override decision.

    A slot is correct iff the picked lattice index equals the truth's index.  `target_rank == -1`
    (truth outside the top-16) is never correct, and an invalid slot terminates the block.
    """
    pick = np.where(fire, blocks["best_alt"], 0)
    ok = blocks["valid"] & (blocks["target_rank"] >= 0) & (pick == blocks["target_rank"])
    return np.cumprod(ok, axis=1).sum(axis=1)


def chain(blocks, accept, n_prompt):
    """Walk each sequence the way decode does and collect per-prompt accept lengths.

    Returns (per_prompt_mean, blocks_visited, missing_anchors).  A missing anchor means the rule
    walked to a position the dump never sampled, which makes that sequence's tail unsimulatable; it
    is counted rather than silently skipped.
    """
    order = np.lexsort((blocks["anchor"], blocks["sample"]))
    sample, anchor = blocks["sample"][order], blocks["anchor"][order]
    acc = accept[order]
    sums = np.zeros(n_prompt)
    counts = np.zeros(n_prompt, dtype=np.int64)
    missing = 0
    start = 0
    while start < len(sample):
        stop = start
        while stop + 1 < len(sample) and sample[stop + 1] == sample[start]:
            stop += 1
        pos = anchor[start : stop + 1]
        a = acc[start : stop + 1]
        lookup = {int(p): i for i, p in enumerate(pos)}
        s = int(sample[start])
        cursor = int(pos[0])
        while True:
            i = lookup.get(cursor)
            if i is None:
                missing += 1
                break
            sums[s] += a[i] + 1
            counts[s] += 1
            cursor += int(a[i]) + 1
        start = stop + 1
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    return mean, counts, missing


DOMAIN_RE = re.compile(r"([a-zA-Z0-9\-]+)_\d+\.ckpt$")


def domains(files):
    out = []
    for f in files:
        m = DOMAIN_RE.search(Path(f).name)
        out.append(m.group(1) if m else "unknown")
    return np.array(out)


def macro(mean, counts, dom):
    """Both project conventions, reported side by side because they disagree by ~0.5 accept.

    macroA averages per-prompt means inside a domain then across domains; macroB pools every block
    of a domain first.  The project's headline number is macroB.
    """
    rows = {}
    for d in sorted(set(dom)):
        m = (dom == d) & (counts > 0)
        if not m.any():
            continue
        rows[d] = {
            "A": float(np.nanmean(mean[m])),
            "B": float((mean[m] * counts[m]).sum() / counts[m].sum()),
            "prompts": int(m.sum()),
            "blocks": int(counts[m].sum()),
        }
    return rows, (float(np.mean([r["A"] for r in rows.values()])) if rows else float("nan")), \
        (float(np.mean([r["B"] for r in rows.values()])) if rows else float("nan"))


def report(name, blocks, fire, files, dom, out):
    accept = accept_lengths(blocks, fire)
    mean, counts, missing = chain(blocks, accept, len(files))
    rows, mA, mB = macro(mean, counts, dom)
    print(f"\n--- {name} ---")
    print("  " + "  ".join(f"{d} {r['B']:.3f}" for d, r in rows.items()))
    print(f"  macroA {mA:.4f}   macroB {mB:.4f}   blocks {int(counts.sum())}   "
          f"unsimulatable tails {missing}")
    out[name] = {"macroA": mA, "macroB": mB, "per_domain": rows,
                 "blocks": int(counts.sum()), "missing": missing}
    return mB


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--chain-ladder", default=None,
                    help="TF_LADDER_*_chain.json; used as the simulator's validation gate")
    ap.add_argument("--rho", type=float, default=3.0)
    ap.add_argument("--rho-grid", default="1,1.5,2,2.5,3,4,6",
                    help="global rho values to simulate exactly")
    ap.add_argument("--extra-scores", default=None,
                    help=".npy aligned with the dump's rows: a learned per-slot score to combine "
                         "with d.  Produced by probe_detector.py --dump-scores.")
    ap.add_argument("--search", default="",
                    help="comma list of rule families to search with a PROMPT-LEVEL split: "
                         "global, slotgroup, extra")
    ap.add_argument("--json-output", default=None)
    args = ap.parse_args()

    data, files = load(args.features)
    if "anchor" not in data:
        print("ERROR: this dump has no `anchor` column; re-run the harness with the patched "
              "--dump-features so block chaining can be simulated")
        return 1
    blocks = to_blocks(data)
    dom = domains(files)
    print(f"{len(blocks['sample'])} anchors over {len(files)} sequences, "
          f"{len(set(dom))} domains: {sorted(set(dom))}")
    if args.extra_scores:
        blocks["extra"] = scatter_extra(data, np.load(args.extra_scores), blocks)
        print(f"loaded extra scores: range [{blocks['extra'].min():.2f}, "
              f"{blocks['extra'].max():.2f}]")

    out = {}
    log_rho = float(np.log(args.rho))

    # ---- validation gate: replay the deployed rule two independent ways ------------------------
    fire_dep = blocks["d"] > log_rho
    same = int(
        (np.where(fire_dep, blocks["best_alt"], 0) == blocks["pick_deployed"])[blocks["valid"]].sum()
    )
    total = int(blocks["valid"].sum())
    print(f"\ngate: d-form pick == dumped pick_index on {same}/{total} valid slots "
          f"({100 * same / max(total, 1):.4f}%)")
    if same != total:
        print("  WARNING: the d-form rule does not reproduce the dumped picks exactly")

    base = report(f"deployed rho={args.rho}", blocks, fire_dep, files, dom, out)

    if args.chain_ladder:
        ladder = json.loads(Path(args.chain_ladder).read_text())
        want_blocks = ladder["chain"]["blocks"]
        # The harness records accepted SLOTS; decode's accept length includes the verifier's bonus
        # token, so the comparable quantity is mean_accept_slots + 1.
        want_accept = ladder["chain"]["mean_accept_slots"] + 1.0
        got_blocks = out[f"deployed rho={args.rho}"]["blocks"]
        got_accept = (
            sum(r["B"] * r["blocks"] for r in out[f"deployed rho={args.rho}"]["per_domain"].values())
            / max(got_blocks, 1)
        )
        print(f"\n=== simulator vs harness CHAIN (same dump, same rule) ===")
        print(f"  blocks        simulator {got_blocks}   harness {want_blocks}   "
              f"({100 * abs(got_blocks - want_blocks) / max(want_blocks, 1):.2f}% off)")
        print(f"  accept length simulator {got_accept:.4f}   harness {want_accept:.4f}   "
              f"({100 * abs(got_accept - want_accept) / max(want_accept, 1e-9):.2f}% off)")
        ok = (abs(got_blocks - want_blocks) / max(want_blocks, 1) < 0.02
              and abs(got_accept - want_accept) / max(want_accept, 1e-9) < 0.02)
        print(f"  [{'PASS' if ok else 'FAIL'}] simulator reproduces the harness within 2%")
        out["validation"] = {"blocks_sim": got_blocks, "blocks_harness": want_blocks,
                             "accept_sim": got_accept, "accept_harness": want_accept, "pass": ok}
        if not ok:
            print("  ABORT: fix the simulator before reading anything below")
            return 1

    # ---- exact global rho curve ---------------------------------------------------------------
    print("\n=== exact global rho curve (this is accept length, not a proxy) ===")
    curve = {}
    for r in [float(v) for v in args.rho_grid.split(",")]:
        curve[r] = report(f"rho={r}", blocks, blocks["d"] > float(np.log(r)), files, dom, out)
    best_rho = max(curve, key=curve.get)
    print(f"\n  best global rho {best_rho} at macroB {curve[best_rho]:.4f} "
          f"(deployed {base:.4f}, delta {curve[best_rho] - base:+.4f})")
    out["rho_curve"] = curve

    # ---- always-keep and oracle bookends -------------------------------------------------------
    report("never override (rho=inf)", blocks, np.zeros_like(blocks["valid"]), files, dom, out)
    report("always take best alt", blocks, blocks["valid"], files, dom, out)
    oracle = blocks["valid"] & (blocks["target_rank"] > 0) & (
        blocks["best_alt"] == blocks["target_rank"]
    )
    report("oracle gate on frozen ranker", blocks, oracle, files, dom, out)
    perfect = blocks["target_rank"].copy()
    acc_p = np.cumprod(blocks["valid"] & (perfect >= 0), axis=1).sum(axis=1)
    mean_p, cnt_p, _ = chain(blocks, acc_p, len(files))
    _, pA, pB = macro(mean_p, cnt_p, dom)
    print(f"\n--- oracle@16 (always pick the truth when it is in the lattice) ---")
    print(f"  macroA {pA:.4f}   macroB {pB:.4f}")
    out["oracle_at_16"] = {"macroA": pA, "macroB": pB}

    # ---- rule search with a prompt-level split -------------------------------------------------
    families = [f.strip() for f in args.search.split(",") if f.strip()]
    if families:
        fit_p = split_prompts(files)
        conf_p = ~fit_p
        slot_idx = np.arange(blocks["valid"].shape[1])[None, :]
        base_fit = macro_on(blocks, blocks["d"] > log_rho, files, dom, fit_p)[0]
        base_conf = macro_on(blocks, blocks["d"] > log_rho, files, dom, conf_p)[0]
        print(f"\n=== rule search, prompt-level split ({int(fit_p.sum())} fit / "
              f"{int(conf_p.sum())} confirm) ===")
        print(f"  deployed rho={args.rho}:  fit macroB {base_fit:.4f}   "
              f"confirm macroB {base_conf:.4f}")
        out["search_baseline"] = {"fit": base_fit, "confirm": base_conf}
        grid = np.concatenate([np.linspace(0.0, 3.0, 31), np.array([4.0, 6.0, 1e9])])

        def score(fire):
            return macro_on(blocks, fire, files, dom, fit_p)[0]

        if "global" in families:
            best = max(((score(blocks["d"] > t), t) for t in grid), key=lambda x: x[0])
            fire = blocks["d"] > best[1]
            cf = macro_on(blocks, fire, files, dom, conf_p)[0]
            print(f"  global    best log_rho {best[1]:.2f}  fit {best[0]:.4f}  "
                  f"confirm {cf:.4f}  ({cf - base_conf:+.4f})")
            out["search_global"] = {"log_rho": best[1], "fit": best[0], "confirm": cf,
                                    "delta": cf - base_conf}

        if "slotgroup" in families:
            # Three tied groups, as recommended: 15 free thresholds are unidentifiable at the tail
            # (slot 14 has ~134 fixable events spread over ~69 prompts).
            groups = [(slot_idx == 0), (slot_idx >= 1) & (slot_idx <= 2), (slot_idx >= 3)]
            best = (-1.0, None)
            coarse = np.linspace(0.0, 3.0, 13)
            for t0 in coarse:
                for t1 in coarse:
                    for t2 in coarse:
                        thr = t0 * groups[0] + t1 * groups[1] + t2 * groups[2]
                        s = score(blocks["d"] > thr)
                        if s > best[0]:
                            best = (s, (t0, t1, t2))
            t0, t1, t2 = best[1]
            thr = t0 * groups[0] + t1 * groups[1] + t2 * groups[2]
            cf = macro_on(blocks, blocks["d"] > thr, files, dom, conf_p)[0]
            print(f"  slotgroup best log_rho (slot0, 1-2, 3+) = ({t0:.2f}, {t1:.2f}, {t2:.2f})  "
                  f"fit {best[0]:.4f}  confirm {cf:.4f}  ({cf - base_conf:+.4f})")
            out["search_slotgroup"] = {"log_rho": [t0, t1, t2], "fit": best[0], "confirm": cf,
                                       "delta": cf - base_conf}

        if "extra" in families and "extra" in blocks:
            best = (-1.0, None)
            qs = np.quantile(blocks["extra"][blocks["extra"] > -1e8], np.linspace(0.02, 0.98, 25))
            for e in qs:
                for t in grid:
                    s = score((blocks["extra"] > e) & (blocks["d"] > t))
                    if s > best[0]:
                        best = (s, (float(e), float(t)))
            e, t = best[1]
            cf = macro_on(blocks, (blocks["extra"] > e) & (blocks["d"] > t),
                          files, dom, conf_p)[0]
            print(f"  extra AND d: extra>{e:.3f} and log_rho {t:.2f}  fit {best[0]:.4f}  "
                  f"confirm {cf:.4f}  ({cf - base_conf:+.4f})")
            out["search_extra"] = {"extra_thr": e, "log_rho": t, "fit": best[0], "confirm": cf,
                                   "delta": cf - base_conf}

    if args.json_output:
        dest = Path(args.json_output)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(out, indent=2, default=float))
        print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
