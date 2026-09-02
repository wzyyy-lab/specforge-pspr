"""Gate: the SpecForge port of the dh2048 recipe is bit-identical to the reference script.

Compares, on the SAME real anchor records from ``outputs/seqtrace_train`` and under the SAME dh2048
weights, the port in ``scripts/train_pspr_accept_selector.py`` against the reference
``TAPS-SP/scripts/train_accept_selector.py``:

  1. INIT       same ``torch.manual_seed`` -> bit-identical initial state dicts (module construction
                order is what fixes this, so it catches a reordered ``__init__``)
  2. DATA       ``precompute`` / ``align_top1`` / ``collate`` produce identical tensors
  3. FORWARD    ``slot_probs`` scores, ground-truth probabilities and err logits
  4. LOSS       the exact dh2048 objective: ``CE(sc[ti>=0], ti[ti>=0]) + BCE(el[frontier], ti != 0)``
  5. GATE EVAL  the full ``GATES x THETAS`` calibration table, and the selected ``(tau, rho, skip0,
                theta)`` plus macro gain that drive checkpoint selection
  6. PARAMS     27,059,213, closing exactly against dh2048
  7. ROUNDTRIP  the port's checkpoint writer loads into the reference class with ``strict=True``

Run::

    cd SpecForge && PYTHONPATH=. python scripts/gate_accept_selector_equivalence.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT.parent / "TAPS-SP"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REF))

from specforge.algorithms.pspr.accept_selector import (  # noqa: E402
    GATES,
    THETAS,
    eval_gate,
    expected_accept,
    frontier_mask,
    report_gate,
    slot_probs,
)
from specforge.data.lattice_trace_data import collate, load_vocab_embeds, precompute  # noqa: E402
from specforge.modeling.draft.pspr import LatticePathSelector  # noqa: E402

sys.path.insert(0, str(REF))
from joint.lattice_selector import LatticeSelector  # noqa: E402
from scripts.train_accept_selector import eval_gate as ref_eval_gate  # noqa: E402
from scripts.train_accept_selector import frontier_mask as ref_frontier_mask  # noqa: E402
from scripts.train_accept_selector import report_gate as ref_report_gate  # noqa: E402
from scripts.train_accept_selector import slot_probs as ref_slot_probs  # noqa: E402
from scripts.train_lattice_selector import collate as ref_collate  # noqa: E402
from scripts.train_pairwise import precompute as ref_precompute  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from train_pspr_accept_selector import (  # noqa: E402
    PORT_TO_REFERENCE,
    SCALAR_IN_REFERENCE,
    reference_state_dict,
    save_checkpoint,
)

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""), flush=True)


def maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.double() - b.double()).abs().max().item()


def port_to_ref_keys(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        if k == "target_embedding":
            continue
        for src, dst in PORT_TO_REFERENCE.items():
            if k.startswith(src):
                k = dst + k[len(src) :]
                break
        if k in SCALAR_IN_REFERENCE:
            v = v.reshape(())
        out[k] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(REF / "outputs/dh2048/best.pt"))
    ap.add_argument("--trace-dir", default=str(REF / "outputs/seqtrace_train"))
    ap.add_argument("--target-model", default=str(REF / "models/Qwen3-4B"))
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--records", type=int, default=512)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    ref_sd = payload["state_dict"]
    print(f"dh2048 config: {cfg}")
    print(f"dh2048 cal_gain={payload['cal_gain']:.6f} tau={payload['tau']}")

    # ---------------------------------------------------------------- 1. INIT (same seed) --------
    torch.manual_seed(args.seed)
    port_init = LatticePathSelector(
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
    )
    torch.manual_seed(args.seed)
    ref_init = LatticeSelector(**cfg)
    a, b = port_to_ref_keys(port_init.state_dict()), ref_init.state_dict()
    check("init: key sets identical", set(a) == set(b), f"{len(a)} keys")
    worst = max(((maxdiff(a[k], b[k]), k) for k in b), default=(0.0, "-"))
    check(
        "init: every tensor bit-identical under one seed",
        worst[0] == 0.0,
        f"max|delta| = {worst[0]:.3e} (worst {worst[1]})",
    )
    check(
        "params == dh2048 (27,059,213)",
        sum(p.numel() for p in port_init.parameters()) == 27059213,
        f"{sum(p.numel() for p in port_init.parameters())}",
    )
    del port_init, ref_init

    # ---------------------------------------------------------------- models with dh2048 ---------
    port = LatticePathSelector(
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
    inv = {v: k for k, v in PORT_TO_REFERENCE.items()}
    mapped = {}
    for k, v in ref_sd.items():
        for src, dst in inv.items():
            if k.startswith(src):
                k = dst + k[len(src) :]
                break
        mapped[k] = v.reshape(1) if k in SCALAR_IN_REFERENCE else v
    miss, unexp = port.load_state_dict(mapped, strict=False)
    miss = [k for k in miss if k != "target_embedding"]
    check(
        "dh2048 loads strict into the port",
        not miss and not unexp,
        f"{len(mapped)} keys, missing={miss} unexpected={unexp}",
    )
    ref = LatticeSelector(**cfg).to(device)
    ref.load_state_dict(ref_sd, strict=True)
    port.eval()
    ref.eval()

    # ---------------------------------------------------------------- 2. DATA -------------------
    files = sorted(Path(args.trace_dir).rglob("*.pt"))
    picked, recs_port, recs_ref = [], [], []
    for f in files:
        if len(recs_port) >= args.records:
            break
        raw = torch.load(f, weights_only=False)
        picked.append(f.parent.name)
        for r in raw:
            if len(recs_port) >= args.records:
                break
            if "draft_hidden" not in r:
                continue
            import copy

            recs_port.append(precompute(copy.deepcopy(r)))
            recs_ref.append(ref_precompute(copy.deepcopy(r)))
        if len(picked) >= args.shards and len(recs_port) >= args.records:
            break
    print(f"records: {len(recs_port)} from shards {sorted(set(picked))}")
    keys = ("target_idx", "top_token_ids", "top_log_probs", "oracle_len", "L0", "OK16")
    bad = []
    for i, (rp, rr) in enumerate(zip(recs_port, recs_ref)):
        for k in keys:
            vp, vr = rp[k], rr[k]
            if torch.is_tensor(vp):
                if not torch.equal(vp, vr):
                    bad.append((i, k))
            elif vp != vr:
                bad.append((i, k))
    check("precompute/align_top1 identical", not bad, f"{len(recs_port)} records, {len(keys)} fields")

    embed = load_vocab_embeds(args.target_model).float().to(device)
    port.bind_target_embedding(embed)
    chunk = recs_port[: args.batch]
    bp = collate(chunk, embed, device)
    br = ref_collate(recs_ref[: args.batch], embed, device)
    dbad = [k for k in ("top", "lp", "dh", "g", "ti", "scal", "prefix_ids") if not torch.equal(bp[k], br[k])]
    check("collate identical", not dbad, f"batch={len(chunk)} mismatched={dbad}")
    for k in ("prefix_emb", "cand_emb"):
        check(f"collate {k}", maxdiff(bp[k], br[k]) == 0.0, f"max|delta| = {maxdiff(bp[k], br[k]):.3e}")

    # ---------------------------------------------------------------- 3. FORWARD ----------------
    with torch.no_grad():
        sc_p, p_p, el_p = slot_probs(port, bp, with_err=True)
        sc_r, p_r, el_r = ref_slot_probs(ref, br, with_err=True)
    check("slot_probs scores", maxdiff(sc_p, sc_r) == 0.0, f"max|delta| = {maxdiff(sc_p, sc_r):.3e}")
    check("slot_probs p(g)", maxdiff(p_p, p_r) == 0.0, f"max|delta| = {maxdiff(p_p, p_r):.3e}")
    check("slot_probs err logits", maxdiff(el_p, el_r) == 0.0, f"max|delta| = {maxdiff(el_p, el_r):.3e}")
    check(
        "expected_accept",
        maxdiff(expected_accept(p_p), expected_accept(p_r)) == 0.0,
        f"max|delta| = {maxdiff(expected_accept(p_p), expected_accept(p_r)):.3e}",
    )
    valid_p, valid_r = bp["g"] >= 0, br["g"] >= 0
    fm_p = frontier_mask(valid_p, (bp["ti"] == 0) & valid_p)
    fm_r = ref_frontier_mask(valid_r, (br["ti"] == 0) & valid_r)
    check("frontier_mask", torch.equal(fm_p, fm_r), f"{int(fm_p.sum())} frontier slots")

    # ---------------------------------------------------------------- 4. LOSS -------------------
    def dh2048_loss(sc, el, b, fm):
        ti = b["ti"]
        m = ti >= 0
        loss = F.cross_entropy(sc[m].double(), ti[m])
        y = (ti != 0).float()
        return loss + F.binary_cross_entropy_with_logits(el[fm].double(), y[fm])

    lp = dh2048_loss(sc_p, el_p, bp, fm_p).item()
    lr = dh2048_loss(sc_r, el_r, br, fm_r).item()
    check("dh2048 loss (CE + err BCE)", lp == lr, f"{lp:.12f} vs {lr:.12f}  |delta| = {abs(lp-lr):.3e}")

    # ---------------------------------------------------------------- 5. GATE EVAL --------------
    cal = recs_port[: args.records]
    torch.manual_seed(0)
    base_p, per_p = eval_gate(port, cal, embed, device)
    torch.manual_seed(0)
    base_r, per_r = ref_eval_gate(ref, recs_ref[: args.records], embed, device)
    bbad = [
        (d, f)
        for d in base_p
        for f in ("n", "run")
        if base_p[d][f] != base_r.get(d, {}).get(f)
    ]
    check("eval_gate base (DFlash top-1 run)", not bbad, f"{len(base_p)} datasets, mismatched={bbad}")
    gbad = []
    for k in ((t, r, s0, o) for (t, r, s0) in GATES for o in THETAS):
        for d in per_p[k]:
            for f in ("run", "right_n", "destroyed", "wrong_n", "recovered"):
                if per_p[k][d][f] != per_r[k][d][f]:
                    gbad.append((k, d, f))
    check(
        "eval_gate full GATESxTHETAS table",
        not gbad,
        f"{len(GATES)*len(THETAS)} gates x {len(base_p)} datasets x 5 counters, mismatched={len(gbad)}",
    )
    tau_p, gain_p = report_gate("port", base_p, per_p, verbose=False)
    tau_r, gain_r = ref_report_gate("ref", base_r, per_r, verbose=False)
    check("selected gate (tau, rho, skip0, theta)", tau_p == tau_r, f"{tau_p} vs {tau_r}")
    check(
        "selection metric (macro accept-length gain)",
        gain_p == gain_r,
        f"{gain_p:+.12f} vs {gain_r:+.12f}",
    )

    # ---------------------------------------------------------------- 6. CKPT ROUNDTRIP ---------
    tmp = ROOT / "outputs/_gate_accept_selector_roundtrip.pt"
    save_checkpoint(port, tmp, cal_gain=gain_p, tau=tau_p)
    rt = torch.load(tmp, map_location="cpu", weights_only=False)
    check("ckpt model_type/config match dh2048", rt["model_type"] == "LatticeSelector" and rt["config"] == cfg, f"{rt['config'] == cfg}")
    probe = LatticeSelector(**rt["config"])
    probe.load_state_dict(rt["state_dict"], strict=True)
    live = port_to_ref_keys(port.state_dict())
    worst = max(maxdiff(rt["state_dict"][k], live[k].cpu()) for k in rt["state_dict"])
    check("ckpt loads into reference class strict=True", worst == 0.0, f"{len(rt['state_dict'])} keys, max|delta| = {worst:.3e}")
    tmp.unlink(missing_ok=True)

    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n{n_pass}/{len(RESULTS)} PASS")
    if n_pass != len(RESULTS):
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"  FAILED: {name}  {detail}")
        return 1
    print("The port reproduces the dh2048 recipe bit for bit: init, data, forward, loss, gate "
          "selection and checkpoint format.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
