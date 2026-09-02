"""Sweep a depth-aware override gate on the calibration split.

The shipped gate is uniform over slots: override candidate 0 iff ``p_alt > tau + rho * p_top1``,
with a single boolean ``skip0`` that forbids overriding slot 0 entirely. The repair diagnosis says
that boolean is too coarse:

    repair rate by frontier depth (dh2048-exact selector, cal split, 871 fixable blocks)
      slot  0:  21/204 = 10.3%     <- most cases, worst repair rate
      slot  1:  39/145 = 26.9%
      slot  2:  39/139 = 28.1%
      slot  4:  29/75  = 38.7%
      slot 10:   6/13  = 46.2%

Slot 0 is both the most frequent frontier and the least repairable one: its prefix is a single anchor
token, so the causal state carries almost no information there. It is also the most expensive slot to
break, because every block reaches it. ``skip0=True`` is the rho=infinity corner of that observation
and ``skip0=False`` is the rho=uniform corner; nothing in between was ever measured.

This sweep gives slot 0 its own rho, so the two shipped settings become the endpoints of a
one-parameter family, and reports whether any interior point beats both. Pure post-processing: no
retraining, no weight change.

Run::

    cd SpecForge && PYTHONPATH=. python scripts/sweep_depth_gate.py \
        --selector outputs/pspr_dh2048_exact/best.pt
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT.parent / "TAPS-SP"
sys.path.insert(0, str(ROOT))

from specforge.algorithms.pspr.accept_selector import slot_probs  # noqa: E402
from specforge.data.lattice_trace_data import (  # noqa: E402
    collate,
    dataset_files,
    is_cal,
    load_files,
    load_vocab_embeds,
)
from specforge.modeling.draft.pspr import LatticePathSelector  # noqa: E402


def load_selector(path: str, target_model: str, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    model = LatticePathSelector(
        hidden_size=cfg["hidden_dim"],
        vocab_size=cfg["vocab_size"],
        top_k=cfg["K"],
        d=cfg["d"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        state_dim=cfg["ds"],
        delta_hidden=cfg["delta_h"],
        max_slots=cfg["max_slots"],
        dropout=cfg["dropout"],
    ).to(device)
    state = {}
    for k, v in payload["state_dict"].items():
        if k.startswith("dh_mlp."):
            k = "delta_mlp." + k[len("dh_mlp.") :]
        state[k] = v.reshape(1) if k == "gamma" else v
    model.load_state_dict(state, strict=False)
    model.eval()
    embed = load_vocab_embeds(target_model).float().to(device)
    model.bind_target_embedding(embed)
    return model, embed, payload


@torch.no_grad()
def sweep(model, records, embed, device, rho_rest, rho_slot0, batch=256):
    """Accept length per (rho_rest, rho_slot0), macro-averaged over datasets.

    Vectorised over the whole grid: prob/best are computed once per chunk, then broadcast against the
    two rho axes, so the cost is one selector forward per chunk regardless of grid size.
    """
    rr = torch.tensor(rho_rest, device=device).view(-1, 1, 1, 1)
    r0 = torch.tensor(rho_slot0, device=device).view(1, -1, 1, 1)
    runs = defaultdict(lambda: torch.zeros(len(rho_rest), len(rho_slot0), device=device))
    base = defaultdict(lambda: torch.zeros((), device=device))
    counts = defaultdict(int)
    for i in range(0, len(records), batch):
        chunk = records[i : i + batch]
        b = collate(chunk, embed, device)
        scores, _ = slot_probs(model, b)
        prob = torch.softmax(scores, dim=-1)
        p_top1 = prob[..., 0]
        best_alt, best_idx = prob[..., 1:].max(dim=-1)
        best_idx = best_idx + 1
        ti, g = b["ti"], b["g"]
        valid = g >= 0
        # rho threshold per slot: slot 0 gets its own axis, every other slot the shared one.
        H = ti.shape[1]
        is_slot0 = torch.zeros(H, dtype=torch.bool, device=device).index_fill_(
            0, torch.tensor([0], device=device), True
        )
        thresh = torch.where(is_slot0.view(1, 1, 1, H), r0, rr) * p_top1.unsqueeze(0).unsqueeze(0)
        fires = best_alt.unsqueeze(0).unsqueeze(0) > thresh  # [R, R0, B, H]
        pick = torch.where(fires, best_idx.unsqueeze(0).unsqueeze(0), 0)
        ok = (pick == ti.unsqueeze(0).unsqueeze(0)) & valid.unsqueeze(0).unsqueeze(0)
        run = (torch.cumsum((~ok).long(), dim=-1) == 0).sum(dim=-1) + 1  # [R, R0, B]
        base_run = (
            torch.cumsum((~((ti == 0) & valid)).long(), dim=-1) == 0
        ).sum(dim=-1) + 1  # [B]
        names = [r["dataset"] for r in chunk]
        for d in sorted(set(names)):
            sel = torch.tensor([n == d for n in names], device=device)
            runs[d] += run[:, :, sel].sum(dim=-1)
            base[d] += base_run[sel].sum()
            counts[d] += int(sel.sum())
    macro = torch.zeros(len(rho_rest), len(rho_slot0), device=device)
    macro_base = 0.0
    for d in runs:
        macro += runs[d] / max(counts[d], 1)
        macro_base += float(base[d]) / max(counts[d], 1)
    n = len(runs)
    return macro / n, macro_base / n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selector", default=str(ROOT / "outputs/pspr_dh2048_exact/best.pt"))
    ap.add_argument("--target-model", default=str(REF / "models/Qwen3-4B"))
    ap.add_argument("--trace-dir", default=str(REF / "outputs/seqtrace_train"))
    ap.add_argument("--shards-per-domain", type=int, default=6)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, embed, payload = load_selector(args.selector, args.target_model, device)
    shipped = payload.get("tau")
    print(f"selector: {args.selector}")
    print(f"  cal_gain={payload.get('cal_gain'):+.4f}  shipped gate tau={shipped}")

    by_ds = dataset_files(args.trace_dir)
    records = []
    for d in sorted(by_ds):
        records += load_files(
            by_ds[d][: args.shards_per_domain],
            keep=lambda p: is_cal(p, args.val_frac * 100, args.seed),
        )
    print(f"  cal anchors: {len(records)} over {len({r['dataset'] for r in records})} domains\n")

    rho_rest = [1.3, 2.2, 3.0, 4.5, 6.5, 9.0, 14.0, 22.0]
    rho_slot0 = [1.3, 2.2, 3.0, 4.5, 6.5, 9.0, 14.0, 22.0, 35.0, 60.0, float("inf")]
    macro, base = sweep(model, records, embed, device, rho_rest, rho_slot0)

    print(f"DFlash top-1 macro accept length = {base:.4f}")
    print("macro accept length, rows = rho(slots 1+), cols = rho(slot 0)")
    head = "".join(f"{('inf' if r == float('inf') else f'{r:g}'):>8}" for r in rho_slot0)
    print(f"  {'rho\\r0':>8}{head}")
    for i, r in enumerate(rho_rest):
        row = "".join(f"{macro[i, j].item() - base:>+8.3f}" for j in range(len(rho_slot0)))
        print(f"  {r:>8g}{row}")

    flat = macro.flatten()
    order = flat.argsort(descending=True)
    print("\ntop 8 (delta over DFlash top-1):")
    for k in order[:8].tolist():
        i, j = divmod(k, len(rho_slot0))
        r0 = rho_slot0[j]
        tag = ""
        if abs(rho_rest[i] - rho_slot0[j]) < 1e-9:
            tag = "  <- uniform (shipped skip0=False family)"
        elif r0 == float("inf"):
            tag = "  <- skip0=True (shipped skip0=True family)"
        print(
            f"  rho_rest={rho_rest[i]:>6g}  rho_slot0={('inf' if r0 == float('inf') else f'{r0:g}'):>6}"
            f"   macro={flat[k].item():.4f}  delta={flat[k].item() - base:+.4f}{tag}"
        )

    # Compare the best interior point against the two shipped corners it generalises.
    uni = max(
        (macro[i, j].item(), rho_rest[i])
        for i in range(len(rho_rest))
        for j in range(len(rho_slot0))
        if abs(rho_rest[i] - rho_slot0[j]) < 1e-9
    )
    inf_j = rho_slot0.index(float("inf"))
    skip = max((macro[i, inf_j].item(), rho_rest[i]) for i in range(len(rho_rest)))
    best = flat[order[0]].item()
    print(
        f"\nbest uniform (skip0=False): {uni[0]:.4f} at rho={uni[1]:g}"
        f"\nbest skip0=True           : {skip[0]:.4f} at rho={skip[1]:g}"
        f"\nbest depth-aware          : {best:.4f}"
        f"\n  gain over the better shipped corner: {best - max(uni[0], skip[0]):+.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
