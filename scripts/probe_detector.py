#!/usr/bin/env python3
"""Detector headroom probe: at matched destroy, can a LEARNED "is the base wrong here" signal
override more often than the deployed margin rule?

WHY THIS AND NOT THE RANKER
---------------------------
The ladder blames the ranker (alt_rank_0 = 66.86%, then -22.8pp when KEEP joins, then -7.3pp at the
gate).  But the per-slot dump kills that reading for the slots that matter.  The deployed rule is
`p_alt > rho * p0`, which is INFEASIBLE whenever `p0 >= 1/(1+rho)`; at rho=3 that is `p0 >= 0.25`,
and at slot 0 -- 16.5% of all decode repair opportunity -- 95.5% of repairable slots are above it.
So a perfect ranker at slot 0 could still repair at most ~4.5% of them, against 1.28% today.  The
binding constraint there is not "which alternative", it is "dare I override a confident base".

That decision is exactly what `dflash_family_model.py:1411-1445` implements as the frontier error
detector, with label `err_target = ~base_top1_ok` (correctly including out-of-top-k slots).  It is
DISABLED in the deployed run: `selector_err_loss_alpha` defaults to 0, so `_selector_err_objective_
enabled` is False and the head is frozen; and `select_margin_gate`'s `theta` path is an AND, so even
a trained detector could only make the gate MORE conservative, never less.

This probe answers whether the cloze trunk's frozen representation carries the signal, before any
training is spent:
  * fit on the training corpus, report on held-out benchmark sequences;
  * compare against the deployed sufficient statistic `d = log p_alt_max - log p0`, which is
    algebraically equivalent to the rho rule at tau=0, so "beats d at matched destroy" is exactly
    "beats the deployed gate";
  * also fit the 2D rule (detector, d), which upper-bounds ANY monotone combination of the two;
  * report both a plain slot average and a decode-weighted one, because the uniform-anchor slot mix
    gives slot 0 2.3% of the mass and decode gives it 16.5%.

A detector that only matches `d` says the trunk has no extra information and the gate is genuinely
exhausted.  A detector that beats it says turning `selector_err_loss_alpha` on -- plus replacing the
AND-shaped `theta` with the factorised keep/repair decision -- is worth a training run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load(path):
    d = np.load(path)
    return {k: d[k] for k in d.files if k != "files"}, [str(f) for f in d["files"]]


def build(data, device):
    """Everything the gate is allowed to see, plus the labels the Pareto needs."""
    lp = torch.from_numpy(data["lp"]).float()
    sc = torch.from_numpy(data["scores"]).float()
    tr = torch.from_numpy(data["target_rank"]).long()
    slot = torch.from_numpy(data["slot"]).long()

    log_p = torch.log_softmax(lp, dim=-1)
    log_g = torch.log_softmax(sc, dim=-1)
    best_alt_g, best_alt_off = log_g[:, 1:].max(dim=-1)
    d = best_alt_g - log_g[:, 0]                       # deployed sufficient statistic, tau=0
    best_alt = best_alt_off + 1

    base_wrong = tr.ne(0)                              # includes tr == -1 (truth outside top-k)
    base_right = tr.eq(0)
    repairable = tr.gt(0)
    alt_is_truth = repairable & best_alt.eq(tr.clamp_min(0))

    ent = -(log_p.exp() * log_p).sum(-1)
    side = torch.stack(
        [
            log_g[:, 0], d, best_alt_g,
            log_p[:, 0], log_p[:, 0] - log_p[:, 1], ent,
            log_p[:, 0] - log_p[:, 3], log_p[:, 0] - log_p[:, 7],
            log_g[:, 0] - log_p[:, 0],
        ],
        dim=-1,
    )
    return {
        "z": torch.from_numpy(data["z"]),
        "state": torch.from_numpy(data["state"]),
        "scalars": torch.from_numpy(data["scalars"]).float(),
        "side": side,
        "slot": slot,
        "d": d,
        "base_wrong": base_wrong,
        "base_right": base_right,
        "alt_is_truth": alt_is_truth,
        "frontier": frontier_from_picks(data),
    }


def frontier_from_picks(data):
    """Rebuild decode's reachable set: slots up to and including each block's first policy error.

    Decoding a block stops at the first slot where the deployed policy's pick is not the target, so
    only a PREFIX of each block is ever a live decision.  Evaluating a gate on every supervised slot
    instead is not a harmless reweighting: the reachable set is 77% base-right while the full set is
    only 52%, so an aggressive rule looks far better than it is.  `pick_index` and `target_rank` were
    dumped precisely so this prefix can be rebuilt offline without another forward pass.
    """
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
    # Reachable iff every earlier slot of the same block was policy-correct.  cumulative AND, reset
    # at each group boundary, implemented as "no earlier failure in this group".
    fail = (~policy_ok[order]).astype(np.int64)
    csum = np.cumsum(fail)
    group_start = np.maximum.accumulate(np.where(new_group, np.arange(len(key)), 0))
    before_group = np.where(group_start > 0, csum[group_start - 1], 0)
    prior = csum - fail - before_group  # failures strictly before this slot, inside its own block
    reach = np.zeros(len(key), dtype=bool)
    reach[order] = prior == 0
    return torch.from_numpy(reach)


class Detector(nn.Module):
    def __init__(self, d_z, d_state, d_side, d_hidden=512, dropout=0.15, horizon=15,
                 use_trunk=True):
        super().__init__()
        self.use_trunk = use_trunk
        self.z_ln, self.s_ln = nn.LayerNorm(d_z), nn.LayerNorm(d_state)
        self.slot_emb = nn.Embedding(horizon, 32)
        d_in = 3 + d_side + 32 + (d_z + d_state if use_trunk else 0)
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, z, state, scalars, side, slot):
        parts = [scalars, side, self.slot_emb(slot)]
        if self.use_trunk:
            parts = [self.z_ln(z), self.s_ln(state)] + parts
        return self.net(torch.cat(parts, dim=-1)).squeeze(-1)


def slot_weights(fit, chain_ladder, horizon=15):
    """decode decision share / uniform-anchor share, so counts can be read the way decode sees them."""
    table = json.loads(Path(chain_ladder).read_text())["slot_strata"]["CHAIN"]
    dec = np.array([float(table[str(h)]["decision_n"]) for h in range(horizon)])
    dec = dec / dec.sum()
    tr = np.bincount(fit["slot"].numpy(), minlength=horizon).astype(np.float64)
    tr = tr / tr.sum()
    return torch.tensor(dec / np.maximum(tr, 1e-12), dtype=torch.float32)


def pareto(score, data, n_points=4000):
    """(destroy, recovered) as the firing threshold sweeps down over the reachable set.

    `recovered` requires that the base is wrong AND the truth is in the lattice AND the selector's
    own best alternative IS the truth -- the alternative choice stays frozen, so this isolates the
    override decision.  `destroy` counts reachable base-right slots that fire, which is decode's
    definition.  No slot reweighting is applied: restricting to the reachable prefix already
    reproduces decode's front-loaded slot mix natively.
    """
    order = torch.argsort(score, descending=True)
    cum_d = torch.cumsum(data["base_right"][order].float(), 0)
    cum_r = torch.cumsum(data["alt_is_truth"][order].float(), 0)
    idx = torch.linspace(0, len(order) - 1, min(n_points, len(order))).long()
    return cum_d[idx], cum_r[idx], score[order][idx]


def recovered_at_budget(score, data, budget):
    """Recovered at the tightest threshold whose destroy stays within the budget."""
    cum_d, cum_r, thr = pareto(score, data)
    ok = cum_d <= budget
    if not bool(ok.any()):
        return 0.0, 0.0, float("inf")
    j = int(torch.nonzero(ok).max())
    return float(cum_r[j]), float(cum_d[j]), float(thr[j])


def evaluate_rule(name, score, data, budget, out, n_fixable):
    rec, dest, thr = recovered_at_budget(score, data, budget)
    auc = roc_auc(score, data["base_wrong"])
    print(f"  {name:<30} recovered {rec:>7.0f} ({100 * rec / max(n_fixable, 1):>5.2f}%)  "
          f"destroy {dest:>6.0f}  thr {thr:>8.3f}  AUC(base_wrong) {auc:.4f}")
    out[name] = {"recovered": rec, "recovered_pct": rec / max(n_fixable, 1), "destroy": dest,
                 "threshold": thr, "auc": auc}
    return rec


def roc_auc(score, label):
    """Rank-based AUC; ties get their average rank, so a constant predictor scores exactly 0.5."""
    lab = label.float()
    n_pos, n_neg = float(lab.sum()), float((1 - lab).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(score)
    ranks = torch.empty_like(score, dtype=torch.float64)
    ranks[order] = torch.arange(1, len(score) + 1, dtype=torch.float64)
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1
    return float((ranks[lab.bool()].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-features", required=True)
    ap.add_argument("--eval-features", required=True)
    ap.add_argument("--chain-ladder", default="outputs/TF_LADDER_eval_chain.json")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--holdout-frac", type=float, default=0.15)
    ap.add_argument("--rho", type=float, default=3.0, help="the deployed operating point, whose "
                                                           "weighted destroy becomes the budget")
    ap.add_argument("--target", default="override_correct",
                    choices=("base_wrong", "override_correct"),
                    help="base_wrong reproduces dflash_family_model.py's err_target.  "
                         "override_correct is the label the DECISION actually needs: keeping wins "
                         "iff the base is right, overriding wins iff the frozen best alternative IS "
                         "the truth, so the comparison is P(alt correct) vs P(base correct) and the "
                         "detector should estimate the former, not 'is something wrong here'.")
    ap.add_argument("--dump-scores", default=None,
                    help="write the trained score for EVERY slot of the eval dump (not just the "
                         "current frontier), so simulate_policy.py can replay rules that change "
                         "the reachable set")
    ap.add_argument("--json-output", default=None)
    args = ap.parse_args()

    device = torch.device("cuda")
    tr_raw, tr_files = load(args.train_features)
    ev_raw, ev_files = load(args.eval_features)
    fit_all = build(tr_raw, device)
    rep_all = build(ev_raw, device)
    print(f"loaded fit {len(fit_all['slot'])} slots / {len(tr_files)} seqs   "
          f"report {len(rep_all['slot'])} slots / {len(ev_files)} seqs", flush=True)

    # ---- restrict both corpora to decode's reachable prefix -------------------------------------
    def restrict(data, sample_ids):
        m = data["frontier"]
        out = {k: v[m] for k, v in data.items() if k != "frontier"}
        out["sample"] = torch.from_numpy(sample_ids.astype(np.int64))[m]
        return out

    fit = restrict(fit_all, tr_raw["sample"])
    rep = restrict(rep_all, ev_raw["sample"])
    rep_full = None if args.dump_scores is None else {
        k: v for k, v in rep_all.items() if k != "frontier"
    }
    del fit_all, rep_all
    print(f"reachable  fit {len(fit['slot'])}  report {len(rep['slot'])}", flush=True)

    # Fidelity gate: the reconstructed prefix must reproduce the harness's own POLICY population,
    # which was computed inside the model forward by `frontier_mask`.  A silent bug in the offline
    # reconstruction would otherwise look like a finding.
    pol = json.loads(Path(args.chain_ladder).read_text())["slot_strata"]["POLICY"]
    gates = {}
    print("\n=== frontier reconstruction vs the harness's own frontier_mask ===")
    for h in range(15):
        got = int(rep["slot"].eq(h).sum())
        want = int(pol[str(h)]["decision_n"])
        rel = abs(got - want) / max(want, 1)
        gates[f"slot {h} decision_n within 2%"] = rel < 0.02
        print(f"  slot {h:>2}: reconstructed {got:>6} vs harness {want:>6}  ({100 * rel:>5.2f}% off)")
    for name, ok in sorted(gates.items()):
        if not ok:
            print(f"  [FAIL] {name}")
    print(f"  [{'PASS' if all(gates.values()) else 'FAIL'}] all 15 slots within 2%")
    if not all(gates.values()):
        print("  ABORT: frontier reconstruction does not match the harness")
        return 1

    n_seq = len(tr_files)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n_seq)
    hold_seqs = perm[: max(1, int(round(args.holdout_frac * n_seq)))]
    is_hold = np.isin(fit["sample"].numpy(), hold_seqs)
    idx_fit = torch.from_numpy(np.nonzero(~is_hold)[0])
    idx_hold = torch.from_numpy(np.nonzero(is_hold)[0])
    print(f"fit {len(idx_fit)}  holdout {len(idx_hold)}", flush=True)

    def subset(data, idx):
        return {k: v[idx] for k, v in data.items()}

    hold = subset(fit, idx_hold)
    results = {"gates": gates}

    # ---- the budget: what the deployed rho=3 rule spends on the reported reachable set ----------
    log_rho = float(np.log(args.rho))
    fire = rep["d"] > log_rho
    # `n_rankable` is the ceiling any pure GATE change can reach: the reachable base-wrong slots
    # where the selector's frozen best alternative already IS the truth.  Everything above that
    # needs a better ranker, not a better gate.
    n_fix = int(rep["alt_is_truth"].sum())
    fixable = int(rep["base_wrong"].sum())
    budget = float((rep["base_right"] & fire).sum())
    dep_rec = float((rep["alt_is_truth"] & fire).sum())
    print(f"\n=== deployed rho={args.rho} on the reported reachable set ===")
    print(f"  decisions {len(rep['slot'])}  base_right {int(rep['base_right'].sum())}  "
          f"base_wrong {fixable}  n_rankable (gate ceiling) {n_fix}")
    print(f"  destroy {budget:.0f} ({100 * budget / max(int(rep['base_right'].sum()), 1):.3f}%)  "
          f"recovered {dep_rec:.0f} ({100 * dep_rec / max(n_fix, 1):.2f}% of the gate ceiling)")
    results["deployed"] = {"destroy": budget, "recovered": dep_rec, "n_rankable": n_fix,
                           "decisions": len(rep["slot"])}

    print("\n=== rules available WITHOUT training, at the deployed destroy budget ===")
    evaluate_rule("d = log p_alt - log p0", rep["d"], rep, budget, results, n_fix)
    evaluate_rule("-log p0 (base doubt)", -rep["side"][:, 3], rep, budget, results, n_fix)
    evaluate_rule("draft entropy", rep["side"][:, 5], rep, budget, results, n_fix)
    evaluate_rule("-(lp0 - lp1) margin", -rep["side"][:, 4], rep, budget, results, n_fix)

    # ---- trained detectors ---------------------------------------------------------------------
    def infer(det, data, idx, batch=65536):
        det.eval()
        chunks = []
        with torch.no_grad():
            for s in range(0, len(idx), batch):
                sel = idx[s : s + batch]
                chunks.append(
                    det(data["z"][sel].to(device).float(), data["state"][sel].to(device).float(),
                        data["scalars"][sel].to(device), data["side"][sel].to(device),
                        data["slot"][sel].to(device)).cpu()
                )
        return torch.cat(chunks)

    hold_idx = torch.arange(len(hold["slot"]))
    rep_idx = torch.arange(len(rep["slot"]))
    hold_budget = float((hold["base_right"] & (hold["d"] > log_rho)).sum())
    hold_ceiling = int(hold["alt_is_truth"].sum())
    label_key = "base_wrong" if args.target == "base_wrong" else "alt_is_truth"
    pos = float(fit[label_key][idx_fit].float().mean())
    pos_weight = torch.tensor((1 - pos) / max(pos, 1e-6), device=device)

    trained, heads = {}, {}
    for use_trunk in (False, True):
        name = "detector (side only)" if not use_trunk else "detector (+z,S trunk)"
        torch.manual_seed(7)
        det = Detector(fit["z"].shape[1], fit["state"].shape[1], fit["side"].shape[1],
                       dropout=args.dropout, use_trunk=use_trunk).to(device)
        opt = torch.optim.AdamW(det.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        gen = torch.Generator().manual_seed(11)
        print(f"\n=== training {name} "
              f"({label_key} rate {100 * pos:.2f}%, pos_weight {float(pos_weight):.3f}) ===",
              flush=True)
        best = None
        for epoch in range(args.epochs):
            det.train()
            order = idx_fit[torch.randperm(len(idx_fit), generator=gen)]
            tot, cnt = 0.0, 0
            for s in range(0, len(order), args.batch_size):
                sel = order[s : s + args.batch_size]
                logit = det(fit["z"][sel].to(device).float(),
                            fit["state"][sel].to(device).float(),
                            fit["scalars"][sel].to(device), fit["side"][sel].to(device),
                            fit["slot"][sel].to(device))
                y = fit[label_key][sel].to(device).float()
                loss = F.binary_cross_entropy_with_logits(logit, y, pos_weight=pos_weight)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(det.parameters(), 1.0)
                opt.step()
                tot += float(loss.detach()) * len(sel)
                cnt += len(sel)
            hs = infer(det, hold, hold_idx)
            hrec, hdest, _ = recovered_at_budget(hs, hold, hold_budget)
            hauc = roc_auc(hs, hold["base_wrong"])
            print(f"  epoch {epoch} loss {tot / max(cnt, 1):.4f}  holdout AUC {hauc:.4f}  "
                  f"recovered {hrec:.0f}/{hold_ceiling} @ destroy {hdest:.0f}", flush=True)
            if best is None or hrec > best[0]:
                best = (hrec, {k: v.detach().clone() for k, v in det.state_dict().items()})
        det.load_state_dict(best[1])
        trained[name] = infer(det, rep, rep_idx)
        heads[name] = {k: v.detach().clone() for k, v in det.state_dict().items()}

    print("\n=== on the reported reachable set, at the deployed destroy budget ===")
    evaluate_rule("d = log p_alt - log p0", rep["d"], rep, budget, results, n_fix)
    for name, score in trained.items():
        evaluate_rule(name, score, rep, budget, results, n_fix)

    # The 2D rule upper-bounds any monotone combination of a detector and the deployed statistic.
    print("\n=== 2D rule: fire iff detector > t AND d > log rho', best over both ===")
    best2 = {}
    for name, score in trained.items():
        top = (-1.0, None)
        for q in np.linspace(0.0, 0.98, 33):
            t = float(torch.quantile(score.float(), float(q)))
            keep = score > t
            if int(keep.sum()) < 500:
                continue
            sub = {k: v[keep] for k, v in rep.items()}
            rec, dest, thr = recovered_at_budget(sub["d"], sub, budget)
            if rec > top[0]:
                top = (rec, dict(q=float(q), det_thr=t, d_thr=thr, destroy=dest, recovered=rec,
                                 recovered_pct=rec / max(n_fix, 1)))
        print(f"  {name:<30} {json.dumps(top[1])}")
        best2[name] = top[1]
    results["two_d"] = best2

    print("\n=== per-slot-group at the deployed budget: where does any gain come from? ===")
    ref = results["d = log p_alt - log p0"]["recovered"]
    top_name = max(best2, key=lambda k: best2[k]["recovered"])
    best_score = trained[top_name]
    fire_ref = rep["d"] > log_rho
    fire_new = (best_score > best2[top_name]["det_thr"]) & (rep["d"] > best2[top_name]["d_thr"])
    print(f"  {'slots':>8} {'decisions':>10} {'ceiling':>8} {'dep rec':>8} {'new rec':>8} "
          f"{'dep dest':>9} {'new dest':>9}")
    for lo, hi in ((0, 2), (3, 5), (6, 9), (10, 14)):
        m = (rep["slot"] >= lo) & (rep["slot"] <= hi)
        print(f"  {f'{lo}-{hi}':>8} {int(m.sum()):>10} {int((m & rep['alt_is_truth']).sum()):>8} "
              f"{int((m & rep['alt_is_truth'] & fire_ref).sum()):>8} "
              f"{int((m & rep['alt_is_truth'] & fire_new).sum()):>8} "
              f"{int((m & rep['base_right'] & fire_ref).sum()):>9} "
              f"{int((m & rep['base_right'] & fire_new).sum()):>9}")

    gain = best2[top_name]["recovered"] - ref
    print("\n=== verdict ===")
    print(f"  gate ceiling (frozen ranker, perfect gate) {n_fix}  "
          f"deployed reaches {dep_rec:.0f} = {100 * dep_rec / max(n_fix, 1):.2f}%")
    print(f"  best learned rule ({top_name}) beats the deployed statistic by {gain:+.0f} repairs "
          f"({100 * gain / max(ref, 1e-9):+.2f}%) at equal destroy")
    results["verdict"] = {"best_rule": top_name, "gain_repairs": gain,
                          "gain_pct": gain / max(ref, 1e-9)}

    if args.dump_scores:
        # Scores for EVERY slot, in the dump's own row order, so a rule that lengthens the accepted
        # run can be simulated at anchors the current policy never reaches.
        det = Detector(fit["z"].shape[1], fit["state"].shape[1], fit["side"].shape[1],
                       dropout=0.0, use_trunk="trunk" in top_name).to(device)
        det.load_state_dict(heads[top_name])
        full = infer(det, rep_full, torch.arange(len(rep_full["slot"])))
        np.save(args.dump_scores, full.numpy().astype(np.float32))
        print(f"wrote {args.dump_scores} ({len(full)} slots, head={top_name})")

    if args.json_output:
        dest = Path(args.json_output)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(results, indent=2, default=float))
        print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
