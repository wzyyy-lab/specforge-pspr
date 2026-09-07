"""Does the FUTURE lattice carry information about the token that repairs the FIRST error?

Why this probe exists
---------------------
Three separate experiments have reported that cross-slot information is worth ~+0.03:
the slot-level ``cross_slot=full/none`` ablation (twice), and the H-by-K ``lattice2d``
family (``EXPERIMENT_LOG.md:123``, <= +0.326 against a +0.440 slot-level baseline).
None of them settles the question, because each confounds two things:

  (i)  is there extractable information about the answer in the other slots' candidate sets?
  (ii) can a model of that size, trained on 276k anchors, extract and exploit it?

A full-model comparison that loses can lose for reason (ii) while (i) is true -- and the head is
known to be data-limited, which penalises exactly the bigger models.  ``lattice2d`` in particular
has no cross-cell on/off switch, so its number is a total, not an ablation.

This probe isolates (i).  It trains a small scorer on ONE population -- the repairable first error --
and changes exactly one thing: whether the scorer may look at the following slots' candidate sets.

Population
----------
For each anchor, the frontier ``t*`` is the first slot whose true token is not the draft's top-1
(``ti != 0``).  Accept length is decided there and nowhere else: every slot before it is correct by
definition, and every slot after it is already outside the accepted prefix.  We keep the frontiers
where the true token is inside the top-16 (``ti >= 1``), i.e. exactly the population whose repair
rate is currently 29.8%.

Arms (identical data, init, schedule; only the feature set differs)
------------------------------------------------------------------
  A  causal      h_t*, the 16 candidates, their log-probs, GRU state over the committed prefix
  B  +future     A + slots t*+1..t*+3 candidate ids and log-probs
  C  +shuffled   A + the same future features taken from a DIFFERENT random anchor

C is the control that makes the result interpretable.  B carries more parameters and more input
dimensions than A, so ``B > A`` alone proves nothing.  Only ``B > C`` is evidence that the gain comes
from the future lattice's *content* rather than from capacity or from the optimiser having more knobs.

Reading the result
------------------
  B >> C  ->  the coupling information is real and extractable; the ~+0.03 ceiling is an
              architecture/optimisation failure and an H-by-K design deserves a proper attempt.
  B ~= C  ->  the information is not there; no bidirectional structure over these marginals can
              raise the repair rate, and the lever is elsewhere.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from specforge.data.lattice_trace_data import (  # noqa: E402
    dataset_files,
    load_files,
    load_vocab_embeds,
)

FUT = 3  # overwritten by --fut
FUTK = 4  # overwritten by --futk


def log(*a):
    print(*a, flush=True)


def extract(records, H, K):
    """-> flat per-frontier tensors. Stores ids, not embeddings: 16 x 2560 floats per example
    would be 160 KB and blow up host memory at 100k examples."""
    h, top, lp, pref, lab, ftop, flp, fmask, tstar, group = (
        [], [], [], [], [], [], [], [], [], []
    )
    group_ids = {}
    for r in records:
        ti = r["target_idx"].long()
        g = r["g_block"].long()
        valid = g >= 0
        if not bool(valid[0]):
            continue
        # frontier = first valid slot whose true token is not the draft's top-1
        wrong = valid & (ti != 0)
        if not bool(wrong.any()):
            continue
        t = int(wrong.float().argmax())
        # every slot before the frontier must be valid and correct, else this anchor's
        # accept run ended for a different reason and t is not a frontier
        if t > 0 and not bool((valid[:t] & (ti[:t] == 0)).all()):
            continue
        if int(ti[t]) < 1:
            continue  # true token outside top-16: not repairable, excluded from the 29.8%
        anchor = int(r["anchor_tok"])
        p = torch.cat([torch.tensor([anchor]), g[:-1].clamp(min=0)])
        ft = torch.zeros(FUT, FUTK, dtype=torch.long)
        fl = torch.zeros(FUT, FUTK, dtype=torch.float16)
        fm = torch.zeros(FUT, dtype=torch.float16)
        for j in range(FUT):
            s = t + 1 + j
            if s < H and bool(valid[s]):
                ft[j] = r["top_token_ids"][s, :FUTK].long()
                fl[j] = r["top_log_probs"][s, :FUTK].half()
                fm[j] = 1.0
        h.append(r["draft_hidden"][t].half())
        top.append(r["top_token_ids"][t].long())
        lp.append(r["top_log_probs"][t].half())
        pref.append(p)
        lab.append(int(ti[t]))
        ftop.append(ft)
        flp.append(fl)
        fmask.append(fm)
        tstar.append(t)
        # Windows from the same prompt are highly correlated.  A frontier-level random split can
        # therefore put neighbouring continuations in train and validation and overstate the
        # absolute ceiling.  Keep a stable integer group so the caller can split prompt-disjoint.
        group_key = (str(r.get("dataset", "")), str(r.get("prompt_id", "")))
        group.append(group_ids.setdefault(group_key, len(group_ids)))
    if not h:
        return None
    return dict(
        h=torch.stack(h),
        top=torch.stack(top),
        lp=torch.stack(lp),
        pref=torch.stack(pref),
        lab=torch.tensor(lab, dtype=torch.long),
        ftop=torch.stack(ftop),
        flp=torch.stack(flp),
        fmask=torch.stack(fmask),
        # ``pref`` is padded to H for tensorisation, but the causal state for frontier t is the
        # state after [anchor, g_0, ..., g_{t-1}], i.e. GRU output index t.  Keeping the true index
        # is essential: taking the final padded output leaks labels from *after* the frontier.
        tstar=torch.tensor(tstar, dtype=torch.long),
        group=torch.tensor(group, dtype=torch.long),
    )


class Scorer(nn.Module):
    """Scores the 16 candidates at one frontier slot.

    ``future_mode`` controls how (and whether) the following slots' candidate cells reach the score:

      none  the candidate never sees them
      bag   they are pooled into ONE vector shared by all 16 candidates, concatenated into the MLP.
            This is the weak form: expressing "candidate A here fits candidate B there" is left to a
            2-layer MLP over a concatenation, which has no query-key structure to do it with.
      attn  each candidate is a query attending over every future candidate cell, so the pairwise
            compatibility the lattice framing is about is a first-class operation.
    """

    def __init__(self, hidden=2560, d=512, future_mode="none", heads=8):
        super().__init__()
        self.future_mode = future_mode
        self.h_in = nn.Linear(hidden, d)
        self.c_in = nn.Linear(hidden, d)
        self.gru = nn.GRU(hidden, d, batch_first=True)
        extra = 0
        if future_mode == "bag":
            self.f_in = nn.Linear(hidden * FUTK + FUTK, d)
            self.f_pos = nn.Parameter(torch.zeros(FUT, d))
            nn.init.normal_(self.f_pos, std=0.02)
            extra = d
        elif future_mode == "attn":
            self.fq = nn.Linear(hidden, d)
            self.fkv = nn.Linear(hidden + 1, d)
            self.f_pos = nn.Parameter(torch.zeros(FUT, d))
            nn.init.normal_(self.f_pos, std=0.02)
            # A always-valid null key: a frontier in the last slot has no future at all, and an
            # all-masked key set makes softmax attention emit NaN.
            self.null_kv = nn.Parameter(torch.zeros(1, 1, d))
            nn.init.normal_(self.null_kv, std=0.02)
            self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
            self.attn_ln = nn.LayerNorm(d)
            extra = d
        in_dim = 3 * d + 1 + extra
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1),
        )

    def forward(self, h, cand, lp, pref_emb, tstar, fut_emb, fut_lp, fut_mask):
        B, K, _ = cand.shape
        hp = self.h_in(h)
        s, _ = self.gru(pref_emb)
        s = s[torch.arange(B, device=s.device), tstar]
        cproj = self.c_in(cand)
        ctx = torch.cat([hp, s], dim=-1).unsqueeze(1).expand(B, K, -1)
        z = [cproj, ctx, lp.unsqueeze(-1)]

        if self.future_mode == "bag":
            f = self.f_in(torch.cat([fut_emb.flatten(-2, -1), fut_lp], dim=-1)) + self.f_pos
            f = (f * fut_mask.unsqueeze(-1)).sum(1)
            z.insert(2, f.unsqueeze(1).expand(B, K, -1))
        elif self.future_mode == "attn":
            q = self.fq(cand) + (hp + s).unsqueeze(1)
            kv = self.fkv(torch.cat([fut_emb, fut_lp.unsqueeze(-1)], dim=-1))
            kv = kv + self.f_pos.view(1, FUT, 1, -1)
            kv = kv.flatten(1, 2)
            pad = (fut_mask.unsqueeze(-1).expand(B, FUT, FUTK).flatten(1) < 0.5)
            kv = torch.cat([self.null_kv.expand(B, 1, -1), kv], dim=1)
            pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=pad.device), pad], dim=1)
            a, _ = self.attn(q, kv, kv, key_padding_mask=pad, need_weights=False)
            z.insert(2, self.attn_ln(a))

        return self.mlp(torch.cat(z, dim=-1)).squeeze(-1)


def run_arm(name, data, val, embed, device, args, future_mode, shuffle_future):
    torch.manual_seed(args.seed)
    model = Scorer(hidden=embed.shape[1], d=args.d, future_mode=future_mode).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    n = data["lab"].numel()
    steps = (n // args.batch_size) * args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1))
    gen = torch.Generator().manual_seed(args.seed)

    def batch(d, idx, shuf):
        h = d["h"][idx].to(device).float()
        cand = embed[d["top"][idx].to(device)]
        lp = d["lp"][idx].to(device).float()
        pref = embed[d["pref"][idx].to(device)]
        tstar = d["tstar"][idx].to(device)
        lab = d["lab"][idx].to(device)
        fi = idx[torch.randperm(idx.numel(), generator=gen)] if shuf else idx
        fut = embed[d["ftop"][fi].to(device)]
        flp = d["flp"][fi].to(device).float()
        fm = d["fmask"][fi].to(device).float()
        return h, cand, lp, pref, tstar, lab, fut, flp, fm

    step = 0
    for _ in range(args.epochs):
        order = torch.randperm(n, generator=gen)
        for i in range(0, n - args.batch_size + 1, args.batch_size):
            idx = order[i : i + args.batch_size]
            h, cand, lp, pref, tstar, lab, fut, flp, fm = batch(data, idx, shuffle_future)
            loss = F.cross_entropy(model(h, cand, lp, pref, tstar, fut, flp, fm), lab)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1

    model.eval()

    def accuracy(split):
        hit = tot = 0
        with torch.no_grad():
            m = split["lab"].numel()
            for i in range(0, m, 512):
                idx = torch.arange(i, min(i + 512, m))
                h, cand, lp, pref, tstar, lab, fut, flp, fm = batch(
                    split, idx, shuffle_future
                )
                pred = model(h, cand, lp, pref, tstar, fut, flp, fm).argmax(-1)
                hit += int((pred == lab).sum())
                tot += int(lab.numel())
        return hit / max(tot, 1), hit, tot

    train_acc, train_hit, train_tot = accuracy(data)
    val_acc, val_hit, val_tot = accuracy(val)
    log(
        f"  {name:14s} train {train_acc:.4f} ({train_hit}/{train_tot})  "
        f"val {val_acc:.4f} ({val_hit}/{val_tot})  steps={step}"
    )
    return val_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", default=str(ROOT.parent / "TAPS-SP/outputs/seqtrace_train"))
    ap.add_argument("--target-model", default=str(ROOT.parent / "TAPS-SP/models/Qwen3-4B"))
    ap.add_argument("--shards", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--data-seed", type=int, default=None,
                    help="file/split seed; defaults to --seed so independent model seeds can share data")
    ap.add_argument("--split-unit", choices=["prompt", "frontier"], default="prompt")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--fut", type=int, default=3)
    ap.add_argument("--futk", type=int, default=4)
    ap.add_argument("--mode", choices=["bag", "attn"], default="attn")
    args = ap.parse_args()
    global FUT, FUTK
    FUT, FUTK = args.fut, args.futk

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    by_ds = dataset_files(args.trace_dir)
    files = []
    data_seed = args.seed if args.data_seed is None else args.data_seed
    g = torch.Generator().manual_seed(data_seed)
    per = max(1, args.shards // max(len(by_ds), 1))
    for d in sorted(by_ds):
        fs = by_ds[d]
        perm = torch.randperm(len(fs), generator=g).tolist()
        files.extend([fs[i] for i in perm[:per]])
    log(f"shards {len(files)} over {len(by_ds)} datasets")
    recs = load_files(files)
    H, K = recs[0]["top_token_ids"].shape
    log(f"anchors {len(recs)} | H={H} K={K}")

    data = extract(recs, H, K)
    del recs
    n = data["lab"].numel()
    log(f"repairable frontiers {n}")
    split_gen = torch.Generator().manual_seed(data_seed)
    if args.split_unit == "prompt":
        groups = torch.unique(data["group"])
        group_perm = groups[torch.randperm(groups.numel(), generator=split_gen)]
        nv_groups = max(1, int(groups.numel() * args.val_frac))
        val_groups = group_perm[:nv_groups]
        is_val = torch.isin(data["group"], val_groups)
        vi, ti_ = torch.where(is_val)[0], torch.where(~is_val)[0]
        log(
            f"prompt-disjoint split: {groups.numel() - nv_groups} train prompts | "
            f"{nv_groups} val prompts"
        )
    else:
        perm = torch.randperm(n, generator=split_gen)
        nv = int(n * args.val_frac)
        vi, ti_ = perm[:nv], perm[nv:]
    val = {k: v[vi] for k, v in data.items()}
    train = {k: v[ti_] for k, v in data.items()}
    log(
        f"split={args.split_unit} data_seed={data_seed} | "
        f"train {train['lab'].numel()} | val {val['lab'].numel()}"
    )

    embed = load_vocab_embeds(args.target_model).to(device).float()
    log(f"embed {tuple(embed.shape)}")

    log(f"arms (identical data / init / schedule; only the feature set differs) "
        f"| future {FUT} slots x {FUTK} cands | mode={args.mode}")
    base_seed = args.seed
    accs = {"A": [], "B": [], "C": []}
    for s_i in range(args.seeds):
        args.seed = base_seed + s_i
        log(f" seed {args.seed}")
        accs["A"].append(run_arm("A causal", train, val, embed, device, args, "none", False))
        accs["C"].append(run_arm("C attn+shuf", train, val, embed, device, args, args.mode, True))
        accs["B"].append(run_arm("B attn+real", train, val, embed, device, args, args.mode, False))
    m = {k: sum(v) / len(v) for k, v in accs.items()}
    log("")
    log(f"  mean over {args.seeds} seed(s):  A {m['A']:.4f}  C {m['C']:.4f}  B {m['B']:.4f}")
    log(f"  B - A = {m['B'] - m['A']:+.4f}   (capacity + information, not interpretable alone)")
    log(f"  C - A = {m['C'] - m['A']:+.4f}   (capacity only)")
    log(f"  B - C = {m['B'] - m['C']:+.4f}   <-- the information in the future lattice")
    per = [b - c for b, c in zip(accs["B"], accs["C"])]
    log(f"  per-seed B - C: {' '.join(f'{x:+.4f}' for x in per)}")


if __name__ == "__main__":
    main()
