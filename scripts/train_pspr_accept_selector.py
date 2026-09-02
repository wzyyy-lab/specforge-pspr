"""Train SpecForge's PSPR lattice path selector with the dh2048 recipe, ported verbatim.

This is a line-for-line port of ``TAPS-SP/scripts/train_accept_selector.py`` -- the script that
produced ``TAPS-SP/outputs/dh2048/best.pt`` (``cal_gain=+0.4401``, ``tau=(0.0, 4.5, False, 0.35)``,
``params=27059213``).  Everything that shapes the result is carried over unchanged:

  data      anchor-level lattice traces from ``outputs/seqtrace_train``; the backbone is NEVER run
  loss      ``ce_aux * CE(sc[ti>=0], ti[ti>=0]) + err_w * BCE(err_logits[frontier], ti != 0)``
  optimizer AdamW, ``gamma`` in its own group at 5x lr and weight_decay 0, cosine schedule
  lr        ``args.lr * sqrt(world)``, batch size PER RANK
  split     prompt-hash calibration split, macro-averaged over the six domains
  selection best CALIBRATION macro accept-length gain over DFlash top-1, evaluated every
            ``--eval-every`` steps and at every epoch end (dh2048's peak was at epoch 3 of 8)

The only substitutions are the model class (SpecForge ``LatticePathSelector``, verified bit-identical
to the reference under ``scripts/gate_pspr_reference_equivalence.py``) and the checkpoint writer,
which emits the reference's own ``{model_type, config, state_dict, cal_gain, tau}`` layout with
reference key names so ``TAPS-SP/scripts/decode_lattice.py`` loads it without a conversion step.

Reference invocation reproduced from ``TAPS-SP/outputs/dh2048/ddp.log``::

    torchrun --nproc_per_node=2 scripts/train_accept_selector.py --delta-h 2048 \
        --output outputs/dh2048/best.pt
    # files: 610 shards over 6 datasets | world=2 | prompt-hash split val=10%
    # train anchors 276284 (this rank 138064) | cal 5642 | H=15 K=16
    # params: 27059213 | lr=5.66e-04 steps/epoch=1078
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
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from specforge.algorithms.pspr.accept_selector import (  # noqa: E402
    GATES,
    THETAS,
    dpace_weights,
    eval_gate,
    expected_accept,
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
from specforge.modeling.draft.pspr import LatticePathSelector  # noqa: E402

# ``delta_mlp`` is the port's name for the reference's ``dh_mlp``; ``gamma`` is 1-D here (FSDP
# rejects zero-dim parameters) and a 0-D scalar in the reference.
PORT_TO_REFERENCE = {"delta_mlp.": "dh_mlp."}
SCALAR_IN_REFERENCE = {"gamma"}


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def log(*a):
    if is_main():
        print(*a, flush=True)


def reference_state_dict(core: LatticePathSelector) -> dict:
    """Rename/reshape the port's state dict into the reference ``LatticeSelector``'s vocabulary."""
    out = {}
    for k, v in core.state_dict().items():
        if k == "target_embedding":
            continue  # non-persistent in the port, absent in the reference
        for src, dst in PORT_TO_REFERENCE.items():
            if k.startswith(src):
                k = dst + k[len(src) :]
                break
        if k in SCALAR_IN_REFERENCE:
            v = v.reshape(())
        out[k] = v.detach().cpu().clone()
    return out


def save_checkpoint(core: LatticePathSelector, path, **extra):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_type": "LatticeSelector",
            "config": core.reference_config(),
            "state_dict": reference_state_dict(core),
            **extra,
        },
        path,
    )


def load_reference_state(core: LatticePathSelector, sd: dict, device) -> tuple[list, list]:
    """Inverse of ``reference_state_dict``: load a reference checkpoint into the port."""
    inv = {v: k for k, v in PORT_TO_REFERENCE.items()}
    mapped = {}
    for k, v in sd.items():
        for src, dst in inv.items():
            if k.startswith(src):
                k = dst + k[len(src) :]
                break
        if k in SCALAR_IN_REFERENCE:
            v = v.reshape(1)
        mapped[k] = v.to(device)
    return core.load_state_dict(mapped, strict=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--trace-dir",
        default=str(ROOT.parent / "TAPS-SP/outputs/seqtrace_train"),
        help="one or more comma-separated trace dirs (aggregated)",
    )
    ap.add_argument("--target-model", default=str(ROOT.parent / "TAPS-SP/models/Qwen3-4B"))
    ap.add_argument(
        "--cal-trace-dir",
        default=None,
        help="draw the calibration split from these dirs only (default: --trace-dir). Set this to "
        "the on-policy trace when aggregating DAgger rounds, so gate selection is measured on the "
        "distribution decode actually walks.",
    )
    ap.add_argument("--output", default=str(ROOT / "outputs/pspr_dh2048/best.pt"))
    ap.add_argument("--init-from", default=None, help="warm-start checkpoint (optional)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--weight-decay", type=float, default=2e-3)
    ap.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.0,
        help="linear lr warmup fraction before the cosine decay. The reference recipe has none; "
        "official DFlash2 uses 0.04.",
    )
    ap.add_argument("--batch-size", type=int, default=128, help="PER RANK")
    ap.add_argument("--dim", "--d", dest="d", type=int, default=512)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument(
        "--data-frac",
        type=float,
        default=1.0,
        help="fraction of the TRAIN prompts to keep (cal split untouched)",
    )
    ap.add_argument(
        "--delta-h",
        type=int,
        default=2048,
        help="hidden width of the draft-hidden-correction head, decoded through the FROZEN target "
        "embedding (see LatticePathSelector.delta_term).  dh2048 == this value.",
    )
    ap.add_argument("--dstate", "--ds", dest="ds", type=int, default=512)
    ap.add_argument(
        "--trans-rank",
        type=int,
        default=0,
        help="rank of official DFlash2's learned bilinear transition "
        "<succ_codebook[cand], pred_codebook[prev] * W h> (0 = off). Official uses 256. Adds "
        "2*vocab*rank parameters and is the one part of DFlash2's selector this port lacks: both "
        "token tables are learned for transition scoring, whereas every existing candidate/prefix "
        "vector here comes from the frozen target embedding.",
    )
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument(
        "--output-zero-init",
        action="store_true",
        help="put the init-time no-op on the correction heads' output layers (gamma=1) instead of on "
        "the gamma scalar (gamma=0), following official DFlash2's CandidateSelector. Identical "
        "forward algebra and identical predictions at step 0, but the output layers get gradient "
        "immediately rather than being multiplied by a scalar that starts at zero.",
    )
    ap.add_argument("--ce-aux", type=float, default=1.0)
    ap.add_argument(
        "--err-w",
        type=float,
        default=1.0,
        help="weight on the binary 'top-1 is wrong' detector (trained on frontier slots)",
    )
    ap.add_argument(
        "--dpace",
        action="store_true",
        help="use dynamic D-PACE position weights on the CE (overrides --reach-w)",
    )
    ap.add_argument(
        "--dpace-floor",
        type=float,
        default=0.02,
        help="floor on the dynamic weight so deep slots keep a little signal",
    )
    ap.add_argument(
        "--wrong-w",
        type=float,
        default=1.0,
        help="CE weight multiplier on slots where the draft top-1 is WRONG (the only slots where a "
        "repair is possible).  Needs --reach-w < 1 to take effect.",
    )
    ap.add_argument(
        "--reach-w",
        type=float,
        default=1.0,
        help="CE weight for slots PAST the first top-1 error (1.0 = uniform CE, the old behaviour; "
        "<1 concentrates capacity on the reachable frontier)",
    )
    ap.add_argument(
        "--accept-w",
        type=float,
        default=0.0,
        help="weight on -E[accept]; 0 = pure CE (better: gate handles risk)",
    )
    ap.add_argument("--val-frac", type=float, default=0.10, help="calibration split (for tau)")
    ap.add_argument("--cal-per-ds", type=int, default=1200, help="cap cal anchors PER DATASET")
    ap.add_argument(
        "--eval-every", type=int, default=400, help="gate eval every N steps (peak is early)"
    )
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank >= 0:
        # file shards give each rank a different anchor count; a long rank-0-only gate eval plus a
        # ragged step count is what makes NCCL time out, so the timeout is generous AND the step
        # count is reduced to the global minimum below.
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
    log(
        f"files: {len(all_files)} shards over {len(by_ds)} datasets | world={world} | "
        f"prompt-hash split val={vp:.0f}%"
    )

    # calibration (rank 0 only): scan a deterministic subset of each shard group, then rebalance by
    # the record's TRUE dataset field -- shard dir names stop being reliable labels once on-policy
    # dirs (which can hold two datasets) are aggregated in.
    #
    # --cal-trace-dir exists because calibration's ONLY job is to predict decode-time accept length
    # and pick the gate for it.  Decode walks a purely on-policy trajectory, so once an off-policy and
    # an on-policy trace are aggregated for training, drawing cal from the mixture makes the proxy
    # measure a distribution nothing is ever evaluated on.  Training still uses the full aggregate;
    # only the selection signal is restricted.  is_cal is a prompt-level hash, so the same prompts
    # land in cal under either dir and the train/cal split stays leak-free.
    cal_by_ds = dataset_files(args.cal_trace_dir) if args.cal_trace_dir else by_ds
    cal = []
    if is_main():
        need = int(args.cal_per_ds / max(0.01, args.val_frac) / 512) + 4  # shards to cover the cap
        pool = []
        for d in sorted(cal_by_ds):
            fs = cal_by_ds[d]
            perm = torch.randperm(
                len(fs), generator=torch.Generator().manual_seed(args.seed)
            ).tolist()
            pool.extend(
                load_files(
                    [fs[i] for i in perm[:need]], keep=lambda p: is_cal(p, vp, args.seed)
                )
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

    fr = float(args.data_frac)
    if fr >= 1.0:
        keep_tr = lambda p: not is_cal(p, vp, args.seed)  # noqa: E731
    else:
        # PROMPT-level subsample (never file-level: one shard spans many prompts), disjoint from cal
        # by construction.  Used to answer "is the head data-limited or feature-limited?".
        from zlib import crc32

        hi = vp * 10 + fr * (1000 - vp * 10)
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
    log(
        f"  train anchors {n_tot.item()} (this rank {len(train)}) | cal {len(cal)} | H={H} K={K}"
    )

    embed = load_vocab_embeds(args.target_model).float().to(device)
    core = LatticePathSelector(
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
        output_zero_init=args.output_zero_init,
        trans_rank=args.trans_rank,
    ).to(device)
    # The port reads candidate/prefix embeddings from the frozen target table it is bound to, rather
    # than from collate's pre-gathered tensors; same matrix, same values.
    core.bind_target_embedding(embed)
    if args.init_from:
        sd = torch.load(args.init_from, map_location=device, weights_only=False)["state_dict"]
        miss, unexp = load_reference_state(core, sd, device)
        miss = [k for k in miss if not k.startswith(("tok_", "err_head.", "target_embedding"))]
        if miss or unexp:
            raise RuntimeError(f"init-from mismatch: missing={miss} unexpected={unexp}")
        log(f"  warm-started from {args.init_from} (new err_head params stay zero-init)")
    log(f"  params: {sum(p.numel() for p in core.parameters())}")
    # err_head receives no gradient when --err-w 0, which DDP rejects unless told to expect it
    model = (
        DDP(core, device_ids=[local_rank], find_unused_parameters=(args.err_w <= 0))
        if world > 1
        else core
    )

    lr = args.lr * (world**0.5)
    # ``gamma`` keeps the reference's 5x lr / no-decay group.  The DFlash2 transition codebooks get
    # their own no-decay group: they are embedding-like and SPARSELY read (one row per predecessor,
    # K rows per slot), but AdamW's weight decay is applied to the whole tensor every step, so a
    # decaying group would shrink the vectors of every token that did not appear in the batch. That
    # is the standard reason embedding tables are excluded from weight decay. Empty when
    # --trans-rank 0, so this cannot perturb any existing comparison.
    codebook_names = ("predecessor_codebook", "successor_codebook")
    fast = [p for n_, p in core.named_parameters() if n_ == "gamma"]
    books = [p for n_, p in core.named_parameters() if n_ in codebook_names]
    slow = [
        p
        for n_, p in core.named_parameters()
        if n_ != "gamma" and n_ not in codebook_names
    ]
    groups = [
        {"params": slow, "lr": lr, "weight_decay": args.weight_decay},
        {"params": fast, "lr": lr * 5, "weight_decay": 0.0},
    ]
    if books:
        groups.append({"params": books, "lr": lr, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups)
    if is_main():
        log(
            f"optimizer: {len(slow)} decayed @1x | {len(fast)} gamma @5x wd=0 | "
            f"{len(books)} codebook @1x wd=0"
        )
    steps_t = torch.tensor([len(train) // args.batch_size], device=device)
    if world > 1:
        dist.all_reduce(steps_t, op=dist.ReduceOp.MIN)  # DDP needs an identical step count per rank
    steps_per_ep = max(1, int(steps_t.item()))
    total_steps = max(1, args.epochs * steps_per_ep)
    # The reference recipe is bare cosine with no warmup; official DFlash2 trains with
    # warmup_ratio 0.04.  Off by default so the reference path is untouched.
    warmup_steps = int(total_steps * max(0.0, args.warmup_ratio))
    if warmup_steps > 0:
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    opt, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=max(1, total_steps - warmup_steps)
                ),
            ],
            milestones=[warmup_steps],
        )
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    log(f"  lr={lr:.2e} steps/epoch={steps_per_ep} warmup_steps={warmup_steps}")

    state = {"best": -1e9}

    def checkpoint(tag):
        if not is_main():
            return
        base, per = eval_gate(core, cal, embed, device)
        log(f"--- {tag} gamma={core.gamma.item():.2f}")
        tau, gain = report_gate("cal", base, per, log=log)
        if gain > state["best"]:
            state["best"] = gain
            save_checkpoint(core, args.output, cal_gain=gain, tau=tau)
            log(f"  SAVED (macro delta={gain:+.3f}, tau={tau})")

    checkpoint("init (gamma=0 => identical to DFlash)")
    gsp = torch.Generator().manual_seed(args.seed + rank)
    gstep = 0
    for ep in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train), generator=gsp).tolist()
        tl = ta = 0.0
        nb = 0
        for s in range(steps_per_ep):
            i = s * args.batch_size
            b = collate([train[j] for j in order[i : i + args.batch_size]], embed, device)
            use_err = args.err_w > 0
            out_ = slot_probs(model, b, with_err=use_err)
            sc, p, el = out_ if use_err else (out_[0], out_[1], None)
            ea = expected_accept(p)
            loss = -args.accept_w * ea.mean()
            ti = b["ti"]
            if args.ce_aux > 0:
                m = ti >= 0
                if m.any():
                    if args.dpace:
                        # dynamic, self-adapting position weights (see dpace_weights)
                        w = dpace_weights(p).detach().clamp(min=args.dpace_floor)
                        ce = F.cross_entropy(sc[m], ti[m], reduction="none")
                        loss = loss + args.ce_aux * (ce * w[m]).sum() / w[m].sum().clamp(min=1e-6)
                    elif args.reach_w < 1.0:
                        # Uniform CE spends ~2/3 of its gradient on slots that are almost never
                        # reached (reach rate decays 100%,80%,61%,...,8.8%), and the frontier probe
                        # showed the module actually LOSES 1pp at slot 0 while gaining a useless
                        # +20pp at slot 14.  Down-weight slots past the first top-1 error, which is
                        # where the accept run has already ended.
                        valid = b["g"] >= 0
                        w = torch.where(
                            frontier_mask(valid, (ti == 0) & valid), 1.0, args.reach_w
                        )
                        if args.wrong_w != 1.0:
                            # ~76% of slots have a CORRECT top-1, so uniform CE spends 3/4 of its
                            # capacity on decisions the base log-probs already make right, and only
                            # 1/4 on the repair decisions that actually move accept length.  The
                            # on-policy diagnosis is unambiguous about where the loss is: of the
                            # FIXABLE errors only 42.7% are recovered, and 89% of the misses are the
                            # selector failing to rank g first (not the gate blocking it).
                            w = w * torch.where(ti != 0, args.wrong_w, 1.0)
                        ce = F.cross_entropy(sc[m], ti[m], reduction="none")
                        loss = loss + args.ce_aux * (ce * w[m]).sum() / w[m].sum().clamp(min=1)
                    else:
                        loss = loss + args.ce_aux * F.cross_entropy(sc[m], ti[m])
            if args.err_w > 0:
                valid = b["g"] >= 0
                fm = frontier_mask(valid, (ti == 0) & valid)
                if fm.any():
                    y = (ti != 0).float()  # top-1 wrong (incl. g not in top-K)
                    loss = loss + args.err_w * F.binary_cross_entropy_with_logits(el[fm], y[fm])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tl += loss.item()
            ta += ea.mean().item()
            nb += 1
            gstep += 1
            if args.eval_every and gstep % args.eval_every == 0:
                checkpoint(f"ep{ep} step{gstep} loss={tl/nb:.4f} E_acc={ta/nb:.3f}")
                model.train()
                if world > 1:
                    dist.barrier()
        checkpoint(f"ep{ep} END loss={tl/max(nb,1):.4f} E_acc={ta/max(nb,1):.3f}")
        if world > 1:
            dist.barrier()
    log(f"done. best calibration macro accept-length gain over DFlash: {state['best']:+.3f}")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
