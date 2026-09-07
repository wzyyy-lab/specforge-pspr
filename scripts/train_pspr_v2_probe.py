"""Architecture probe for PSPR-v2 on pre-extracted lattice traces.

Purpose and scope
-----------------
This script exists to *choose an architecture cheaply*, not to produce a shippable head.  A full
PerfectBlend online pass costs ~7.5 GPU-hours; one probe here costs ~20 minutes because the backbone
is never run -- every anchor record already carries ``draft_hidden``, the top-K lattice and the
committed continuation ``g_block``.

The selection signal is ``report_gate``'s calibration macro accept-length gain over plain DFlash
top-1, the same statistic that selected the stage-1 head (``cal_gain=+0.4068`` at epoch 2, which then
measured ``+0.4375`` in real six-domain decode -- the proxy tracks decode closely).

Teacher forcing is exact here, not an approximation: acceptance stops at the first divergence, so a
run measured against the target's own greedy continuation ``g`` equals the decode run, and the GRU's
ground-truth prefix agrees with the committed prefix on every slot that can still affect it.

Probe axes
----------
``--direct-hidden/--no-direct-hidden``
    Whether ``h_t`` reaches the corrector at full 2560 width or only through the encoder's 512-d
    output.  v1 had only the latter (its ``score()`` consumes ``hidden_states`` solely in the
    disabled ``transition_term``); this is the wire the deep audit missed.
``--loss {ce16,full}``
    ``ce16`` is stage-1's masked 16-way cross-entropy, which drops the 20.93% of slots whose target
    is outside top-16.  ``full`` is dense cross-entropy over the corrected readout ``E (h + delta)``,
    which is available at zero architectural cost because Qwen3-4B ties ``lm_head`` to ``E``.
``--cross-slot {full,none}``
    ``none`` keeps every parameter and FLOP and masks off only the off-diagonal attention, so its
    delta isolates cross-slot information from encoder capacity.

Both a lattice-restricted and a full-vocabulary accept length are reported every evaluation, because
restricting the argmax to top-16 costs a measured 8.4% of first errors and the two numbers answer
different questions.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from specforge.algorithms.pspr.accept_selector import (  # noqa: E402
    dpace_weights,
    eval_gate,
    frontier_mask,
    report_gate,
    slot_probs,
)
from specforge.data.lattice_trace_data import (  # noqa: E402
    collate,
    dataset_files,
    is_cal,
    load_files,
    load_vocab_embeds,
)
from specforge.modeling.draft.pspr_v2 import LatticeCorrector  # noqa: E402


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def log(*a):
    if is_main():
        print(*a, flush=True)


def rebase_records(records, embed, device, bs=64):
    """Overwrite each record's ``top_log_probs`` and uncertainty scalars with ``h @ E.T``.

    The trace stores ``draft_hidden`` in fp16 and ``top_log_probs`` from a bf16 ``lm_head``, so the
    two disagree by ~0.03 nats.  A full-vocabulary objective necessarily reads the base logits off
    ``h``, while the lattice-restricted score reads them off the stored ``lp``; leaving both in play
    would train the correction against one base and evaluate it on another.  Recomputing once at load
    time makes the probe internally consistent and matches online training, where the lattice and the
    correction come from the same forward.

    Candidate ids and ``target_idx`` are left exactly as traced, so the DFlash top-1 reference accept
    length -- the denominator of every gain reported here -- is untouched and stays comparable to
    every stage-1 number.
    """
    for i in range(0, len(records), bs):
        chunk = records[i : i + bs]
        h = torch.stack([r["draft_hidden"].float() for r in chunk]).to(device)
        top = torch.stack([r["top_token_ids"].long() for r in chunk]).to(device)
        B, H, _ = h.shape
        logits = F.linear(h.reshape(B * H, -1), embed).float()
        lz = torch.logsumexp(logits, dim=-1, keepdim=True)
        logp = logits - lz
        lp = logp.gather(-1, top.reshape(B * H, -1)).reshape(B, H, -1)
        ent = -(logp.exp() * logp).sum(-1).reshape(B, H)
        del logits, logp, lz
        margin = (lp[..., 0] - lp[..., 1]).clamp(-20.0, 20.0)
        mass = lp.exp().sum(-1).clamp(max=1.0)
        for j, r in enumerate(chunk):
            r["top_log_probs"] = lp[j].cpu()
            r["position_entropy"] = ent[j].cpu()
            r["top1_top2_margin"] = margin[j].cpu()
            r["topk_mass"] = mass[j].cpu()
    return records


@torch.no_grad()
def eval_full_vocab(core, records, embed, device, bs=128):
    """Macro accept length when the argmax ranges over the whole vocabulary, not just the lattice.

    Reported alongside the gated lattice number because the two bound different things: the lattice
    version is directly comparable to every stage-1 result, while this one shows what the same
    corrector is worth once the top-16 restriction is lifted.
    """
    core.eval()
    run = defaultdict(float)
    base = defaultdict(float)
    n = defaultdict(float)
    for i in range(0, len(records), bs):
        chunk = records[i : i + bs]
        b = collate(chunk, embed, device)
        emb = core._embedding()
        cand = F.embedding(b["top"], emb)
        pref = F.embedding(b["prefix_ids"], emb)
        ctx = core.encode(b["dh"], cand, b["lp"], b["scal"])
        st = core.causal_states(
            pref,
            core.prefix_ranks(b["top"], b["prefix_ids"]) if core.prefix_rank else None,
        )
        logits = core.full_vocab_logits(b["dh"], ctx, st)
        pick = logits.argmax(dim=-1)
        g, ti = b["g"], b["ti"]
        valid = g >= 0
        ok = (pick == g) & valid
        top_ok = (ti == 0) & valid
        lead = (torch.cumsum((~ok).long(), dim=1) == 0).sum(dim=1) + 1
        lead0 = (torch.cumsum((~top_ok).long(), dim=1) == 0).sum(dim=1) + 1
        for j, r in enumerate(chunk):
            d = r["dataset"]
            run[d] += lead[j].item()
            base[d] += lead0[j].item()
            n[d] += 1
    dss = sorted(n)
    macro = sum(run[d] / n[d] for d in dss) / len(dss)
    macro_b = sum(base[d] / n[d] for d in dss) / len(dss)
    per = " ".join(f"{d[:4]}={run[d]/n[d]:.2f}" for d in dss)
    return macro, macro_b, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--trace-dir", default=str(ROOT.parent / "TAPS-SP/outputs/seqtrace_train")
    )
    ap.add_argument("--target-model", default=str(ROOT.parent / "TAPS-SP/models/Qwen3-4B"))
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--weight-decay", type=float, default=2e-3)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--batch-size", type=int, default=128, help="PER RANK")
    ap.add_argument("--dim", "--d", dest="d", type=int, default=512)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--delta-h", type=int, default=2048)
    ap.add_argument("--dstate", "--ds", dest="ds", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--cross-slot", choices=["full", "none"], default="full")
    ap.add_argument("--direct-hidden", dest="direct_hidden", action="store_true", default=True)
    ap.add_argument("--no-direct-hidden", dest="direct_hidden", action="store_false")
    ap.add_argument("--cand-slots", type=int, default=4)
    ap.add_argument("--prefix-rank", action="store_true")
    ap.add_argument("--loss", choices=["ce16", "full", "hybrid"], default="full")
    ap.add_argument(
        "--hybrid-lambda",
        type=float,
        default=0.5,
        help="weight on the dense full-vocab term when --loss hybrid; the 16-way term, which is the "
        "one that matches the lattice-restricted serving decision, always carries weight 1",
    )
    ap.add_argument(
        "--data-frac",
        type=float,
        default=1.0,
        help="PROMPT-level subsample of the TRAIN split (cal untouched). Used to measure the "
        "gain-vs-data slope, which is what decides between variants when the head is data-limited",
    )
    ap.add_argument("--dpace", action="store_true")
    ap.add_argument("--dpace-floor", type=float, default=0.02)
    ap.add_argument("--err-w", type=float, default=1.0)
    ap.add_argument("--ce-chunk", type=int, default=2048, help="rows per full-vocab CE chunk")
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--cal-per-ds", type=int, default=1200)
    ap.add_argument("--eval-every", type=int, default=300)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank >= 0:
        from datetime import timedelta

        dist.init_process_group("nccl", timeout=timedelta(hours=6))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        world, rank = dist.get_world_size(), dist.get_rank()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world, rank = 1, 0
    torch.manual_seed(args.seed)

    by_ds = dataset_files(args.trace_dir)
    all_files = [f for d in sorted(by_ds) for f in by_ds[d]]
    vp = args.val_frac * 100
    log(f"files: {len(all_files)} shards over {len(by_ds)} datasets | world={world}")

    cal = []
    if is_main():
        need = int(args.cal_per_ds / max(0.01, args.val_frac) / 512) + 4
        pool = []
        for d in sorted(by_ds):
            fs = by_ds[d]
            perm = torch.randperm(
                len(fs), generator=torch.Generator().manual_seed(args.seed)
            ).tolist()
            pool.extend(
                load_files([fs[i] for i in perm[:need]], keep=lambda p: is_cal(p, vp, args.seed))
            )
        bucket = defaultdict(list)
        for r in pool:
            bucket[r["dataset"]].append(r)
        for d in sorted(bucket):
            got = bucket[d]
            if len(got) > args.cal_per_ds:
                sel = torch.randperm(
                    len(got), generator=torch.Generator().manual_seed(args.seed)
                )
                got = [got[i] for i in sel[: args.cal_per_ds].tolist()]
            log(f"  cal[{d}] {len(got)} anchors / {len({r['prompt_id'] for r in got})} prompts")
            cal.extend(got)

    if args.data_frac >= 1.0:
        keep_tr = lambda p: not is_cal(p, vp, args.seed)  # noqa: E731
    else:
        from zlib import crc32

        hi = vp * 10 + args.data_frac * (1000 - vp * 10)
        keep_tr = lambda p: (not is_cal(p, vp, args.seed)) and crc32(  # noqa: E731
            f"{args.seed}:{p}".encode()
        ) % 1000 < hi
    train = load_files(all_files[rank::world], keep=keep_tr)
    assert not (
        {r["prompt_id"] for r in train} & {r["prompt_id"] for r in cal}
    ), "train/cal leak!"
    H, K = train[0]["top_token_ids"].shape
    n_tot = torch.tensor([len(train)], device=device)
    if world > 1:
        dist.all_reduce(n_tot)
    log(f"  train anchors {n_tot.item()} (this rank {len(train)}) | cal {len(cal)} | H={H} K={K}")

    embed = load_vocab_embeds(args.target_model).float().to(device)
    log("  rebasing lattice log-probs from h @ E.T ...")
    rebase_records(train, embed, device)
    if cal:
        rebase_records(cal, embed, device)
    core = LatticeCorrector(
        hidden_size=train[0]["draft_hidden"].shape[-1],
        vocab_size=int(embed.shape[0]),
        top_k=K,
        d=args.d,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        state_dim=args.ds,
        delta_hidden=args.delta_h,
        max_slots=max(32, H),
        dropout=args.dropout,
        cross_slot=args.cross_slot,
        direct_hidden=args.direct_hidden,
        cand_slots=args.cand_slots,
        prefix_rank=args.prefix_rank,
    ).to(device)
    core.bind_target_embedding(embed)
    log(
        f"  cfg: cross_slot={args.cross_slot} direct_hidden={args.direct_hidden} "
        f"cand_slots={args.cand_slots} prefix_rank={args.prefix_rank} loss={args.loss} "
        f"lam={args.hybrid_lambda} data_frac={args.data_frac} dpace={args.dpace}"
    )
    log(f"  params: {sum(p.numel() for p in core.parameters())}")
    # Plain data parallelism with an explicit gradient all-reduce rather than DDP: the full-vocab
    # objective loops a chunked readout, so which parameters are touched varies per step and DDP's
    # bucket/ready bookkeeping would either stall or need find_unused_parameters on every step.  All
    # ranks seed identically above, so the replicas start equal and stay equal.
    model = core

    lr = args.lr * (world**0.5)
    decay, no_decay = [], []
    for n_, p in core.named_parameters():
        (no_decay if (p.ndim <= 1 or n_ == "pos_emb") else decay).append(p)
    opt = torch.optim.AdamW(
        [
            {"params": decay, "lr": lr, "weight_decay": args.weight_decay},
            {"params": no_decay, "lr": lr, "weight_decay": 0.0},
        ]
    )
    steps_t = torch.tensor([len(train) // args.batch_size], device=device)
    if world > 1:
        dist.all_reduce(steps_t, op=dist.ReduceOp.MIN)
    steps_per_ep = max(1, int(steps_t.item()))
    total_steps = max(1, args.epochs * steps_per_ep)
    warmup_steps = int(total_steps * max(0.0, args.warmup_ratio))
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                opt, start_factor=1e-8, end_factor=1.0, total_iters=max(1, warmup_steps)
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(1, total_steps - warmup_steps)
            ),
        ],
        milestones=[max(1, warmup_steps)],
    )
    log(f"  lr={lr:.2e} steps/epoch={steps_per_ep} total={total_steps} warmup={warmup_steps}")

    state = {"best": -1e9}

    def checkpoint(tag):
        if not is_main():
            return
        base, per = eval_gate(core, cal, embed, device)
        log(f"--- {tag}")
        tau, gain = report_gate("cal", base, per, log=log, top=4)
        fv, fv_base, fv_per = eval_full_vocab(core, cal, embed, device)
        log(f"  full-vocab macroAcc={fv:.3f} (base={fv_base:.3f}, delta={fv-fv_base:+.3f}) {fv_per}")
        score = max(gain, fv - fv_base)
        if score > state["best"]:
            state["best"] = score
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_type": "LatticeCorrector",
                    "config": core.reference_config(),
                    "state_dict": {
                        k: v.detach().cpu().clone()
                        for k, v in core.state_dict().items()
                        if k != "target_embedding"
                    },
                    "cal_gain": gain,
                    "full_vocab_gain": fv - fv_base,
                    "tau": tau,
                },
                args.output,
            )
            log(f"  SAVED (lattice {gain:+.3f} | full-vocab {fv-fv_base:+.3f}, tau={tau})")

    checkpoint("init (delta=0 => identical to DFlash)")
    gsp = torch.Generator().manual_seed(args.seed + rank)
    gstep = 0
    for ep in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train), generator=gsp).tolist()
        tl = 0.0
        nb = 0
        for s in range(steps_per_ep):
            i = s * args.batch_size
            b = collate([train[j] for j in order[i : i + args.batch_size]], embed, device)
            emb = core._embedding()
            cand = F.embedding(b["top"], emb)
            pref = F.embedding(b["prefix_ids"], emb)
            ctx = core.encode(b["dh"], cand, b["lp"], b["scal"])
            st = core.causal_states(
                pref,
                core.prefix_ranks(b["top"], b["prefix_ids"]) if core.prefix_rank else None,
            )
            g, ti = b["g"], b["ti"]
            valid = g >= 0
            loss = b["dh"].sum() * 0.0

            if args.loss in ("full", "hybrid"):
                hc = core.corrected_hidden(b["dh"], ctx, st)
                rows = hc.reshape(-1, hc.shape[-1])[valid.reshape(-1)]
                tgt = g.reshape(-1)[valid.reshape(-1)]
                if args.dpace:
                    with torch.no_grad():
                        sc0 = core.score(b["dh"], ctx, st, cand, b["lp"])
                        p0 = torch.softmax(sc0, -1).gather(
                            -1, ti.clamp(min=0).unsqueeze(-1)
                        ).squeeze(-1)
                        p0 = torch.where(ti >= 0, p0, torch.zeros_like(p0))
                        w = dpace_weights(p0).clamp(min=args.dpace_floor).reshape(-1)[
                            valid.reshape(-1)
                        ]
                else:
                    w = None
                # Chunked so a 151936-way readout over ~1900 rows does not spike memory.
                tot = rows.new_zeros(())
                den = 0.0
                for j in range(0, rows.shape[0], args.ce_chunk):
                    lg = F.linear(rows[j : j + args.ce_chunk], emb)
                    ce = F.cross_entropy(lg, tgt[j : j + args.ce_chunk], reduction="none")
                    if w is None:
                        tot = tot + ce.sum()
                        den += ce.shape[0]
                    else:
                        ww = w[j : j + args.ce_chunk]
                        tot = tot + (ce * ww).sum()
                        den += ww.sum().item()
                loss = loss + (
                    args.hybrid_lambda if args.loss == "hybrid" else 1.0
                ) * tot / max(den, 1e-6)
            if args.loss in ("ce16", "hybrid"):
                sc = core.score(b["dh"], ctx, st, cand, b["lp"])
                m = ti >= 0
                if m.any():
                    if args.dpace:
                        with torch.no_grad():
                            p0 = torch.softmax(sc, -1).gather(
                                -1, ti.clamp(min=0).unsqueeze(-1)
                            ).squeeze(-1)
                            p0 = torch.where(m, p0, torch.zeros_like(p0))
                            wt = dpace_weights(p0).clamp(min=args.dpace_floor)
                        ce = F.cross_entropy(sc[m], ti[m], reduction="none")
                        loss = loss + (ce * wt[m]).sum() / wt[m].sum().clamp(min=1e-6)
                    else:
                        loss = loss + F.cross_entropy(sc[m], ti[m])

            if args.err_w > 0:
                el = core.err_logits(ctx, st, b["lp"], b["scal"], b["dh"])
                fm = frontier_mask(valid, (ti == 0) & valid)
                if fm.any():
                    y = (ti != 0).float()
                    loss = loss + args.err_w * F.binary_cross_entropy_with_logits(
                        el[fm], y[fm]
                    )

            opt.zero_grad()
            loss.backward()
            if world > 1:
                for p in core.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0)
            opt.step()
            sched.step()
            tl += loss.item()
            nb += 1
            gstep += 1
            if args.eval_every and gstep % args.eval_every == 0:
                checkpoint(f"ep{ep} step{gstep} loss={tl/nb:.4f}")
                model.train()
                if world > 1:
                    dist.barrier()
        checkpoint(f"ep{ep} END loss={tl/max(nb,1):.4f}")
        if world > 1:
            dist.barrier()
    log(f"done. best gain: {state['best']:+.3f}")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
