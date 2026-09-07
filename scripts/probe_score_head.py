#!/usr/bin/env python3
"""Headroom probe: can a bigger SCORE HEAD on the FROZEN trunk fix the ranker, and does it generalise?

WHY THIS IS THE RIGHT PROBE
---------------------------
Step 23 established that the repair shortfall is not exposure (~2pp) and not gate calibration (every
threshold family has a held-out net gain of ~0), but two model-side quantities measured on decode's
own population: `alternative_rank_0` = 65.5% and the further 21.8pp lost when candidate 0 joins.

It also matters that MAP is the CORRECT decision rule here.  With E[accept] = sum_h prod_{j<=h} p_j
and p_j = P(pick_j == target_j), maximising each p_j independently maximises every factor and hence
the sum.  So the deployed rho=3 beating rho=1 (argmax) by 0.111 macro is not evidence of an asymmetric
cost -- it is direct evidence that the 16-way posterior is MISCALIBRATED, with candidate 0 systematically
under-scored.  The target of any fix is therefore the posterior itself, and the honest success criterion
is: does argmax of the new posterior beat the current rho=3 policy?

The current head is
    score[k] = top_log_probs[k] + <E[cand_k], delta>,  delta = W2 silu(W1 [LN(h); LN(z); LN(S)])
i.e. a RANK-1 correction in embedding space: for a given slot the only freedom is one direction
`delta`, and the score is LINEAR in the candidate embedding.  It cannot express any candidate-specific
non-linear preference, and it has no dedicated freedom to calibrate candidate 0 against the other 15.
This probe adds exactly those two things on top of the frozen `z`/`S`/`lp`, so a positive result
lower-bounds what a head-only architecture change buys -- and a negative result says the trunk, not the
head, is the binding constraint.

Trained on the TRAINING corpus features, model-selected on a held-out slice of them, and finally
reported on the BENCHMARK corpus features.  That last split is the axis that actually failed in step 23
(selector value-add 34pp in-corpus vs 19pp at deploy), so a head that only helps in-corpus will be
visible as such rather than being mistaken for a win.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScoreHead(nn.Module):
    """Residual per-candidate score plus a dedicated KEEP calibration term.

    Both output layers are zero-initialised, so at step 0 the head reproduces the deployed scores
    exactly and any measured change is attributable to what it learned.
    """

    def __init__(self, d_z, d_state, d_embed, d_cand=128, d_hidden=512, horizon=15, dropout=0.1,
                 d_h=0):
        super().__init__()
        self.z_ln = nn.LayerNorm(d_z)
        self.state_ln = nn.LayerNorm(d_state)
        self.d_h = d_h
        if d_h:
            # The deployed head reads LN(h) alongside LN(z) and LN(S).  Omitting h made the earlier
            # negative result unfalsifiable: the probe could not represent anything h-dependent.
            self.h_ln = nn.LayerNorm(d_h)
        self.slot_emb = nn.Embedding(horizon, 32)
        self.ctx = nn.Sequential(
            nn.Linear(d_z + d_state + d_h + 3 + 32, d_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_cand),
        )
        self.embed_ln = nn.LayerNorm(d_embed)
        self.embed_proj = nn.Linear(d_embed, d_cand, bias=False)
        # per-candidate features: ctx (dc), e_k (dc), ctx*e_k (dc), lp_k, lp_k - lp_0, rank/H
        self.per_candidate = nn.Sequential(
            nn.Linear(3 * d_cand + 3, d_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )
        self.keep_bias = nn.Sequential(
            nn.Linear(d_cand + 4, d_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )
        nn.init.zeros_(self.per_candidate[-1].weight)
        nn.init.zeros_(self.per_candidate[-1].bias)
        nn.init.zeros_(self.keep_bias[-1].weight)
        nn.init.zeros_(self.keep_bias[-1].bias)

    def forward(self, z, state, scalars, slot, lp, scores, candidate_embeddings, use_keep=True,
                use_candidate=True, h=None):
        k = lp.shape[-1]
        parts = [self.z_ln(z), self.state_ln(state)]
        if self.d_h:
            parts.append(self.h_ln(h))
        parts += [scalars, self.slot_emb(slot)]
        ctx = self.ctx(torch.cat(parts, dim=-1))
        out = scores
        if use_candidate:
            e = self.embed_proj(self.embed_ln(candidate_embeddings))
            ctx_k = ctx.unsqueeze(-2).expand_as(e)
            rank = (
                torch.arange(k, device=lp.device, dtype=lp.dtype).view(1, k).expand_as(lp) / k
            )
            per_candidate = self.per_candidate(
                torch.cat(
                    [
                        ctx_k,
                        e,
                        ctx_k * e,
                        lp.unsqueeze(-1),
                        (lp - lp[..., :1]).unsqueeze(-1),
                        rank.unsqueeze(-1),
                    ],
                    dim=-1,
                )
            ).squeeze(-1)
            out = out + per_candidate
        if use_keep:
            bias = self.keep_bias(
                torch.cat([ctx, scalars, lp[..., :1]], dim=-1)
            ).squeeze(-1)
            out = out + F.pad(bias.unsqueeze(-1), (0, k - 1))
        return out


def load(path):
    d = np.load(path)
    out = {k: d[k] for k in d.files if k != "files"}
    out["files"] = [str(f) for f in d["files"]]
    return out


def metrics(scores, target_rank, slot, tag, draft_lp=None):
    """Argmax accuracy plus the fixable-subset ladder, on covered slots only."""
    covered = target_rank >= 0
    pick = scores.argmax(-1)
    order = torch.argsort(scores, dim=-1, descending=True, stable=True)
    inverse = torch.zeros_like(order)
    inverse.scatter_(-1, order, torch.arange(order.shape[-1], device=order.device).expand_as(order))
    safe = target_rank.clamp_min(0)
    sel_rank = inverse.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    alt_rank = sel_rank - (inverse[..., 0] < sel_rank).long()
    fixable = covered & (target_rank > 0)
    base_right = covered & (target_rank == 0)
    n_cov = int(covered.sum())
    out = {
        "tag": tag,
        "covered": n_cov,
        "argmax_acc_covered": float((pick[covered] == safe[covered]).float().mean()),
        "argmax_acc_all": float(
            ((pick == safe) & covered).float().sum() / max(len(pick), 1)
        ),
        "fixable": int(fixable.sum()),
        "alt_rank_0": float(alt_rank[fixable].eq(0).float().mean()),
        "selector_rank_0": float(sel_rank[fixable].eq(0).float().mean()),
        "argmax_repairs": float(pick[fixable].eq(safe[fixable]).float().mean()),
        "base_right_kept_argmax": float(pick[base_right].eq(0).float().mean()),
    }
    # Per-slot rows.  `captured` normalises the selector's gain by the gain that was AVAILABLE given
    # the draft's own alternative ranking, which is the only cross-slot-comparable statistic: at
    # slot 0 the draft is already right 57% of the time so a raw value-add of +0.3pp is not
    # "the selector is 0.3pp better", it is "the selector took 0.7% of what was there".
    rows = []
    if draft_lp is not None:
        draft_alt = draft_lp[:, 1:].argmax(-1) + 1
        draft_ok = draft_alt == safe
        for h in range(int(slot.max()) + 1):
            m = fixable & slot.eq(h)
            n = int(m.sum())
            if n == 0:
                continue
            d1 = float(draft_ok[m].float().mean())
            a0 = float(alt_rank[m].eq(0).float().mean())
            rows.append({"slot": h, "n": n, "draft_r1": d1, "alt_rank_0": a0,
                         "captured": (a0 - d1) / max(1.0 - d1, 1e-9)})
    out["per_slot"] = rows
    return out


def show(m):
    return (
        f"argmax_acc {100 * m['argmax_acc_covered']:.2f}%  "
        f"alt_r0 {100 * m['alt_rank_0']:.2f}%  sel_r0 {100 * m['selector_rank_0']:.2f}%  "
        f"argmax_repair {100 * m['argmax_repairs']:.2f}%  "
        f"baseOK_kept {100 * m['base_right_kept_argmax']:.2f}%  (fixable {m['fixable']})"
    )


def decode_weighted(m, share):
    """alt_rank_0 reweighted onto decode's chained slot distribution.

    Plain alt_rank_0 averages over the uniform-anchor slot mix, in which slot 0 is 2.3% of the
    repairable mass; at deploy it is 16.5%.  Every model-selection decision here is made on this
    number instead, so a head that trades slot-14 accuracy for slot-0 accuracy is scored the way
    decode will actually experience it.
    """
    rows = m.get("per_slot") or []
    if not rows:
        return float("nan")
    num = sum(share[r["slot"]] * r["alt_rank_0"] for r in rows if r["slot"] < len(share))
    den = sum(share[r["slot"]] for r in rows if r["slot"] < len(share))
    return num / max(den, 1e-12)


def show_slots(m, share=None):
    rows = m.get("per_slot") or []
    if not rows:
        return
    print(f"      {'slot':>4} {'n':>7} {'draft_r1':>9} {'alt_r0':>8} {'captured':>9}")
    for r in rows:
        print(f"      {r['slot']:>4} {r['n']:>7} {100 * r['draft_r1']:>8.2f}% "
              f"{100 * r['alt_rank_0']:>7.2f}% {100 * r['captured']:>8.2f}%")
    if share is not None:
        print(f"      decode-weighted alt_r0 {100 * decode_weighted(m, share):.2f}%")


def batches(data, embed, device, index, batch_size, shuffle, generator=None):
    order = (
        index[torch.randperm(len(index), generator=generator)] if shuffle else index
    )
    for start in range(0, len(order), batch_size):
        sel = order[start : start + batch_size]
        cand = torch.from_numpy(data["cand"][sel.numpy()]).to(device).long()
        yield (
            (torch.from_numpy(data["h"][sel.numpy()]).to(device).float()
             if "h" in data else None),
            torch.from_numpy(data["z"][sel.numpy()]).to(device).float(),
            torch.from_numpy(data["state"][sel.numpy()]).to(device).float(),
            torch.from_numpy(data["scalars"][sel.numpy()]).to(device).float(),
            torch.from_numpy(data["slot"][sel.numpy()]).to(device).long(),
            torch.from_numpy(data["lp"][sel.numpy()]).to(device).float(),
            torch.from_numpy(data["scores"][sel.numpy()]).to(device).float(),
            embed[cand].float(),
            torch.from_numpy(data["target_rank"][sel.numpy()]).to(device).long(),
        )


@torch.no_grad()
def evaluate(head, data, embed, device, index, batch_size, tag, use_keep=True, use_candidate=True):
    head.eval()
    all_scores, all_target, all_slot, all_lp = [], [], [], []
    for h, z, state, scalars, slot, lp, scores, cand_emb, target in batches(
        data, embed, device, index, batch_size, shuffle=False
    ):
        new = head(z, state, scalars, slot, lp, scores, cand_emb,
                   use_keep=use_keep, use_candidate=use_candidate, h=h)
        all_scores.append(new.cpu())
        all_target.append(target.cpu())
        all_slot.append(slot.cpu())
        all_lp.append(lp.cpu())
    return metrics(torch.cat(all_scores), torch.cat(all_target), torch.cat(all_slot), tag,
                   draft_lp=torch.cat(all_lp))


def slot_loss_weights(fit_data, chain_ladder, device, horizon=15):
    """Importance weights that make the training slot distribution match decode's.

    Uniform anchor sampling gives every slot equal mass, but decode only reaches slot h if the walk
    got that far, so decode's decision mass is heavily front-loaded.  Measured on this checkpoint the
    mismatch on repairable slots is 7.06x at slot 0 and 0.27x at slot 14 -- i.e. the objective spends
    a quarter of the attention decode needs on the slot that carries 16% of all repair opportunity.
    The weight is decode_share(h) / train_share(h), computed on DECISION counts (not fixable counts)
    because the loss runs over every covered slot, and normalised to mean 1 so the effective learning
    rate is unchanged.
    """
    ladder = json.loads(Path(chain_ladder).read_text())
    table = ladder["slot_strata"]["CHAIN"]
    decode = np.array([float(table[str(h)]["decision_n"]) for h in range(horizon)])
    decode = decode / decode.sum()
    slot = fit_data["slot"].astype(np.int64)
    covered = fit_data["target_rank"].astype(np.int64) >= 0
    train = np.bincount(slot[covered], minlength=horizon).astype(np.float64)
    train = train / train.sum()
    w = decode / np.maximum(train, 1e-12)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32, device=device), decode, train


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-features", required=True)
    ap.add_argument("--eval-features", required=True)
    ap.add_argument("--target-model",
                    default="/kl_infra_infer_intern/wangzhuoyu/TAPS-SP/models/Qwen3-4B")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--holdout-frac", type=float, default=0.15)
    ap.add_argument("--ablate", default="full,candidate_only,keep_only",
                    help="which head variants to fit")
    ap.add_argument("--fit-on", default="train", choices=("train", "eval"),
                    help="train: fit on the training corpus and report transfer to the benchmark "
                         "corpus (the axis that failed at deploy).  eval: fit on half the benchmark "
                         "SEQUENCES and report the other half, which removes corpus shift and so "
                         "upper-bounds what on-policy/DAgger data collection could buy.")
    ap.add_argument("--fit-report-slots", type=int, default=60000,
                    help="how many fitted slots to re-score for the train-accuracy readout; this "
                         "separates capacity (fit acc high, holdout low) from a representation "
                         "ceiling (fit acc also low)")
    ap.add_argument("--slot-weight", default="uniform", choices=("uniform", "decode"),
                    help="decode: importance-weight the loss so the slot distribution matches "
                         "decode's chained decision distribution instead of the uniform-anchor one")
    ap.add_argument("--fixable-weight", type=float, default=1.0,
                    help="extra multiplier on base-WRONG slots.  77%% of slots are base-right and "
                         "the selector is already 99.65%% correct on them, so under a uniform loss "
                         "most of the gradient carries no decision-relevant information")
    ap.add_argument("--chain-ladder", default="outputs/TF_LADDER_eval_chain.json",
                    help="source of decode's per-slot decision distribution for --slot-weight decode")
    ap.add_argument("--json-output", default=None)
    args = ap.parse_args()

    device = torch.device("cuda")
    train = load(args.train_features)
    ev = load(args.eval_features)
    print(f"train {len(train['slot'])} slots / {len(train['files'])} seqs   "
          f"eval {len(ev['slot'])} slots / {len(ev['files'])} seqs", flush=True)

    from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead

    parts = TargetEmbeddingsAndHead.from_pretrained(
        args.target_model, device="cuda", dtype=torch.float16
    )
    embed = parts.embed_tokens.weight.detach()
    d_embed = embed.shape[1]
    d_h = int(train["h"].shape[1]) if "h" in train else 0
    print(f"backbone hidden channel: {'ON (%d)' % d_h if d_h else 'OFF'}", flush=True)

    # Sequence-level splits, so model selection never sees a sequence it was fitted on.  With
    # --fit-on eval the fitted and reported corpora are the SAME distribution, which is what makes
    # the pair of runs interpretable: transfer gap = (in-corpus gain) - (cross-corpus gain).
    if args.fit_on == "train":
        fit_data, hold_data, report_data = train, train, ev
        n_seq = len(train["files"])
        rng = np.random.default_rng(0)
        perm = rng.permutation(n_seq)
        n_hold = max(1, int(round(args.holdout_frac * n_seq)))
        is_hold = np.isin(train["sample"], perm[:n_hold].tolist())
        idx_fit = torch.from_numpy(np.nonzero(~is_hold)[0])
        idx_hold = torch.from_numpy(np.nonzero(is_hold)[0])
        idx_report = torch.arange(len(ev["slot"]))
    else:
        fit_data, hold_data, report_data = ev, ev, ev
        n_seq = len(ev["files"])
        rng = np.random.default_rng(0)
        perm = rng.permutation(n_seq)
        n_report = n_seq // 2
        report_seqs = perm[:n_report]
        rest = perm[n_report:]
        n_hold = max(1, int(round(args.holdout_frac * len(rest))))
        is_report = np.isin(ev["sample"], report_seqs.tolist())
        is_hold = np.isin(ev["sample"], rest[:n_hold].tolist())
        idx_fit = torch.from_numpy(np.nonzero(~is_report & ~is_hold)[0])
        idx_hold = torch.from_numpy(np.nonzero(is_hold)[0])
        idx_report = torch.from_numpy(np.nonzero(is_report)[0])
    idx_eval = idx_report
    fit_probe = idx_fit[
        torch.randperm(len(idx_fit), generator=torch.Generator().manual_seed(3))[
            : args.fit_report_slots
        ]
    ]
    print(f"fit slots {len(idx_fit)}  model-selection holdout {len(idx_hold)}  "
          f"reported {len(idx_report)}  (fit-on {args.fit_on})", flush=True)

    baseline = {
        "train_holdout": evaluate(
            ScoreHead(train["z"].shape[1], train["state"].shape[1], d_embed,
                      dropout=0.0, d_h=d_h).to(device),
            hold_data, embed, device, idx_hold, args.batch_size, "baseline/train_holdout"),
        "eval_corpus": evaluate(
            ScoreHead(train["z"].shape[1], train["state"].shape[1], d_embed,
                      dropout=0.0, d_h=d_h).to(device),
            report_data, embed, device, idx_eval, args.batch_size, "baseline/eval_corpus"),
        "fit_slots": evaluate(
            ScoreHead(train["z"].shape[1], train["state"].shape[1], d_embed,
                      dropout=0.0, d_h=d_h).to(device),
            fit_data, embed, device, fit_probe, args.batch_size, "baseline/fit_slots"),
    }
    print("\n=== baseline (zero-init head == deployed scores) ===")
    for key, m in baseline.items():
        print(f"  {key:<16} {show(m)}")

    results = {"baseline": baseline, "variants": {}, "args": vars(args)}
    weights, dec_share, tr_share = slot_loss_weights(fit_data, args.chain_ladder, device)
    slot_w = weights if args.slot_weight == "decode" else None
    if slot_w is not None:
        print("\n=== slot importance weights (decode decision share / train share) ===")
        print(f"  {'slot':>4} {'train':>8} {'decode':>8} {'weight':>8}")
        for h in range(len(dec_share)):
            print(f"  {h:>4} {100 * tr_share[h]:>7.2f}% {100 * dec_share[h]:>7.2f}% "
                  f"{float(slot_w[h]):>8.3f}")
    print("\n=== baseline per-slot capture on the reported split ===")
    show_slots(baseline["eval_corpus"], dec_share)
    for variant in args.ablate.split(","):
        variant = variant.strip()
        use_keep = variant in ("full", "keep_only")
        use_candidate = variant in ("full", "candidate_only")
        torch.manual_seed(7)
        head = ScoreHead(train["z"].shape[1], train["state"].shape[1], d_embed,
                         dropout=args.dropout, d_h=d_h).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        generator = torch.Generator().manual_seed(11)
        best = None
        print(f"\n=== variant {variant} "
              f"(candidate head {use_candidate}, keep bias {use_keep}) ===", flush=True)
        for epoch in range(args.epochs):
            head.train()
            total, count = 0.0, 0
            for h, z, state, scalars, slot, lp, scores, cand_emb, target in batches(
                fit_data, embed, device, idx_fit, args.batch_size, True, generator
            ):
                covered = target >= 0
                if not bool(covered.any()):
                    continue
                new = head(z, state, scalars, slot, lp, scores, cand_emb,
                           use_keep=use_keep, use_candidate=use_candidate, h=h)
                per = F.cross_entropy(new[covered], target[covered].clamp_min(0),
                                      reduction="none")
                wt = torch.ones_like(per)
                if slot_w is not None:
                    wt = wt * slot_w[slot[covered]]
                if args.fixable_weight != 1.0:
                    wt = wt * torch.where(
                        target[covered] > 0,
                        torch.full_like(per, args.fixable_weight),
                        torch.ones_like(per),
                    )
                loss = (per * wt).sum() / wt.sum().clamp_min(1e-6)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                opt.step()
                total += float(loss) * int(covered.sum())
                count += int(covered.sum())
            hold = evaluate(head, hold_data, embed, device, idx_hold, args.batch_size,
                            f"{variant}/train_holdout", use_keep, use_candidate)
            score = decode_weighted(hold, dec_share)
            print(f"  epoch {epoch} loss {total / max(count, 1):.4f}   "
                  f"dw_alt_r0 {100 * score:.2f}%   holdout {show(hold)}", flush=True)
            if best is None or score > best[0]:
                best = (score, {k: v.detach().clone()
                                for k, v in head.state_dict().items()}, hold)
        head.load_state_dict(best[1])
        out = evaluate(head, report_data, embed, device, idx_eval, args.batch_size,
                       f"{variant}/eval_corpus", use_keep, use_candidate)
        fit_acc = evaluate(head, fit_data, embed, device, fit_probe, args.batch_size,
                           f"{variant}/fit_slots", use_keep, use_candidate)
        print(f"  SELECTED epoch holdout {show(best[2])}")
        print(f"  FIT SLOTS (memorisation ceiling) {show(fit_acc)}")
        print(f"  REPORTED (held-out) {show(out)}")
        print("  REPORTED per-slot:")
        show_slots(out, dec_share)
        results["variants"][variant] = {"train_holdout": best[2], "eval_corpus": out,
                                       "fit_slots": fit_acc,
                                       "dw_alt_r0_holdout": best[0],
                                       "dw_alt_r0_reported": decode_weighted(out, dec_share),
                                       "params": sum(p.numel() for p in head.parameters())}

    print("\n=== summary: argmax_acc / alt_rank_0 (uniform) / alt_rank_0 (decode-weighted) ===")
    print(f"  {'variant':<16} {'fit':>16} {'holdout':>16} {'reported':>16} {'reported dw':>12}")

    def cell(m):
        return f"{100 * m['argmax_acc_covered']:6.2f}%/{100 * m['alt_rank_0']:6.2f}%"

    print(f"  {'baseline':<16} {cell(baseline['fit_slots']):>16} "
          f"{cell(baseline['train_holdout']):>16} {cell(baseline['eval_corpus']):>16} "
          f"{100 * decode_weighted(baseline['eval_corpus'], dec_share):>11.2f}%")
    for name, r in results["variants"].items():
        print(f"  {name:<16} {cell(r['fit_slots']):>16} "
              f"{cell(r['train_holdout']):>16} {cell(r['eval_corpus']):>16} "
              f"{100 * r['dw_alt_r0_reported']:>11.2f}%")

    if args.json_output:
        dest = Path(args.json_output)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(results, indent=2, default=float))
        print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())
