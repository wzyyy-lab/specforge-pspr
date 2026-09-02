"""Gate: the PSPR joint-training path is the dh2048 recipe with an unfrozen backbone.

Stage 1 trains the selector on pre-extracted lattices with the backbone never running. Stage 2 runs
the backbone online, so five things that stage 1 got for free have to be re-proved:

  1. LATTICE     the online ``extract_lattice`` reproduces the trace features dh2048 trained on
                 (candidate ids incl. the bf16-tie argmax swap, top log-probs, entropy, margin, mass)
  2. OBJECTIVE   the joint path's selector CE + err BCE equal dh2048's
                 ``CE(sc[ti>=0], ti[ti>=0]) + BCE(el[frontier], ti != 0)``
  3. GRADIENT    with ``stop_gradient=false`` the selector loss alone produces nonzero backbone
                 gradients, and with ``stop_gradient=true`` it produces exactly zero -- i.e. the
                 joint coupling is real and the switch controls it
  4. GROUPS      the optimizer builds dh2048's parameter groups: backbone at 0.05x, ``gamma`` at 5x
                 with weight_decay 0, everything else at 1x with weight_decay 2e-3
  5. ROUNDTRIP   import -> model -> export returns the stage-1 selector bit for bit

Run::

    cd SpecForge && PYTHONPATH=. python scripts/gate_pspr_joint_equivalence.py \
        --target-model ../TAPS-SP/models/Qwen3-4B
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT.parent / "TAPS-SP"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(REF))

from specforge.algorithms.common.dflash_family_model import frontier_mask  # noqa: E402
from specforge.data.lattice_trace_data import collate, load_vocab_embeds, precompute  # noqa: E402
from specforge.modeling.draft.pspr import LatticePathSelector  # noqa: E402

from joint.lattice import extract_topk_lattice  # noqa: E402

from export_pspr_for_decode import selector_config, split_state  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""), flush=True)


def maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.double() - b.double()).abs().max().item()


# Reductions of the same quantity in a different order differ in the last bits of the format they
# run in. The comparisons below are all fp64, so anything at or below this relative tolerance is
# reduction order, not a difference in the objective.
FP64_REDUCTION_TOL = 1e-12


def close(a: float, b: float) -> bool:
    return abs(a - b) <= FP64_REDUCTION_TOL * max(1.0, abs(a), abs(b))


def build_selector(cfg: dict, device) -> LatticePathSelector:
    return LatticePathSelector(
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-model", default=str(REF / "models/Qwen3-4B"))
    ap.add_argument("--selector-ckpt", default=str(ROOT / "outputs/pspr_dh2048_exact/best.pt"))
    ap.add_argument("--draft-config", default=str(ROOT / "configs/qwen3-4b-pspr-joint.json"))
    ap.add_argument("--joint-init", default=str(ROOT / "outputs/pspr_joint_init"))
    ap.add_argument(
        "--joint-yaml",
        default=str(ROOT / "examples/configs/offline/colocated/qwen3-4b-pspr-joint-offline.yaml"),
    )
    ap.add_argument("--trace-dir", default=str(REF / "outputs/seqtrace_train"))
    ap.add_argument("--records", type=int, default=128)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    draft_config = json.loads(Path(args.draft_config).read_text())
    payload = torch.load(args.selector_ckpt, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    check(
        "draft config selector shape == stage-1 checkpoint",
        selector_config(draft_config) == cfg,
        f"K={cfg['K']} d={cfg['d']} delta_h={cfg['delta_h']} ds={cfg['ds']}",
    )
    # dh2048 H=15; SpecForge slot 0 is the anchor itself and is masked out, so block_size must be 16.
    check(
        "block_size == dh2048 horizon + 1",
        draft_config["block_size"] == 16,
        f"block_size={draft_config['block_size']} (slot 0 = anchor, 15 supervised slots)",
    )

    selector = build_selector(cfg, device)
    inv = {"dh_mlp.": "delta_mlp."}
    mapped = {}
    for k, v in payload["state_dict"].items():
        for src, dst in inv.items():
            if k.startswith(src):
                k = dst + k[len(src) :]
                break
        mapped[k] = v.reshape(1) if k == "gamma" else v
    miss, unexp = selector.load_state_dict(mapped, strict=False)
    miss = [k for k in miss if k != "target_embedding"]
    check("stage-1 selector loads into the joint model", not miss and not unexp, f"{len(mapped)} keys")

    embed = load_vocab_embeds(args.target_model).float().to(device)
    selector.bind_target_embedding(embed)
    selector.eval()

    # ------------------------------------------------------------------ 1. LATTICE --------------
    # dh2048's trace stored the lattice extracted from the draft's own logits. Joint training must
    # recompute it online, so reconstruct the logits the trace was built from (draft_hidden @ E^T,
    # the frozen tied lm_head) and require the online extractor to reproduce the stored features.
    files = sorted(Path(args.trace_dir).rglob("*.pt"))
    recs = []
    for f in files:
        raw = torch.load(f, weights_only=False)
        for r in raw:
            if len(recs) >= args.records:
                break
            if "draft_hidden" in r:
                recs.append(precompute(r))
        if len(recs) >= args.records:
            break
    dh = torch.stack([r["draft_hidden"].float() for r in recs]).to(device)
    with torch.no_grad():
        logits = dh @ embed.t()
    # Compare per anchor, not batched: logsumexp over a [H, V] slab and over a [B, H, V] slab reduce
    # in a different order, and that difference would show up here as a fake mismatch. The reference
    # extractor is per anchor, so the port is called the same way.
    online_ids, online_lp, online_scalars = [], [], []
    ref_ids, ref_lp, ref_ent, ref_margin, ref_mass = [], [], [], [], []
    with torch.no_grad():
        for i in range(len(recs)):
            lp_i, ids_i, scal_i = selector.extract_lattice(logits[i])
            online_ids.append(ids_i)
            online_lp.append(lp_i)
            online_scalars.append(scal_i)
            lat = extract_topk_lattice(logits[i], cfg["K"])
            ref_ids.append(lat.top_token_ids)
            ref_lp.append(lat.top_log_probs)
            ref_ent.append(lat.position_entropy)
            ref_margin.append(lat.top1_top2_margin)
            ref_mass.append(lat.topk_mass)
    online_ids = torch.stack(online_ids)
    online_lp = torch.stack(online_lp)
    online_scalars = torch.stack(online_scalars)
    ref_ids = torch.stack(ref_ids)
    ref_lp = torch.stack(ref_lp)
    check(
        "extract_lattice candidate_ids == reference extract_topk_lattice",
        torch.equal(online_ids, ref_ids),
        f"{online_ids.numel()} cells, {len(recs)} anchors",
    )
    check(
        "extract_lattice top_log_probs",
        maxdiff(online_lp, ref_lp) == 0.0,
        f"max|delta| = {maxdiff(online_lp, ref_lp):.3e}",
    )
    for j, (name, ref) in enumerate(
        (("position_entropy", torch.stack(ref_ent)),
         ("top1_top2_margin", torch.stack(ref_margin).clamp(-20.0, 20.0)),
         ("topk_mass", torch.stack(ref_mass)))
    ):
        d = maxdiff(online_scalars[..., j], ref)
        check(f"extract_lattice scalar[{j}] {name}", d == 0.0, f"max|delta| = {d:.3e}")
    # Candidate 0 must be what the backbone would greedily draft, or "keep candidate 0" no longer
    # reproduces one-shot decoding and acceptance stops being structurally non-decreasing.
    argmax_ids = logits.argmax(dim=-1)
    check(
        "candidate 0 == argmax (bf16-tie swap survives online extraction)",
        torch.equal(online_ids[..., 0], argmax_ids),
        f"{argmax_ids.numel()} slots",
    )

    # ------------------------------------------------------------------ 2. OBJECTIVE ------------
    # Build the joint path's tensors from the trace records and compare the loss against dh2048's.
    b = collate(recs, embed, device)
    H = b["ti"].shape[1]
    with torch.no_grad():
        sc, el = selector.score_candidates(
            candidate_ids=b["top"],
            unary_logits=b["lp"],
            hidden_states=b["dh"],
            predecessor_ids=b["prefix_ids"],
            lattice_scalars=b["scal"],
            return_err=True,
        )
    ti = b["ti"]
    valid = b["g"] >= 0

    # dh2048's own objective.
    m = ti >= 0
    ref_ce = F.cross_entropy(sc[m].double(), ti[m])
    ref_fm = frontier_mask(valid, (ti == 0) & valid)
    ref_err = F.binary_cross_entropy_with_logits(
        el[ref_fm].double(), (ti != 0).double()[ref_fm]
    )

    # The joint path's arithmetic, transcribed from _selector_chunk_terms: a weighted sum over a 0/1
    # weight_mask divided by its own denominator, plus the frontier scan shifted to skip slot 0.
    # Slot indices are offset by one there (slot 0 is the anchor), so emulate that by prepending a
    # masked-out column: dh2048 slot k  <->  joint slot k+1.
    pad = lambda t, v: torch.cat([torch.full_like(t[:, :1], v), t], dim=1)  # noqa: E731
    j_ti = pad(ti, -1)
    j_valid = pad(valid, False)
    j_sc = torch.cat([torch.zeros_like(sc[:, :1]), sc], dim=1)
    j_el = torch.cat([torch.zeros_like(el[:, :1]), el], dim=1)
    j_weight_mask = j_valid.float()
    j_weight_mask[:, 0] = 0.0  # pos_in_block == 0 is zeroed unconditionally
    target_is_candidate = j_ti >= 0
    per_slot_ce = F.cross_entropy(
        j_sc.double().reshape(-1, j_sc.shape[-1]),
        j_ti.clamp(min=0).reshape(-1),
        reduction="none",
    ).reshape_as(j_ti)
    j_w = j_weight_mask * target_is_candidate.float()
    joint_ce = (per_slot_ce * j_w).sum() / j_w.sum()
    check(
        "selector CE: joint path == dh2048",
        close(joint_ce.item(), ref_ce.item()),
        f"{joint_ce.item():.12f} vs {ref_ce.item():.12f}",
    )

    base_top1_ok = target_is_candidate & j_ti.eq(0)
    reachable = frontier_mask(j_weight_mask[:, 1:] > 0, base_top1_ok[:, 1:])
    reachable = torch.cat([torch.zeros_like(reachable[:, :1]), reachable], dim=1)
    check(
        "err frontier mask: joint path == dh2048 (shifted by the anchor slot)",
        torch.equal(reachable[:, 1:], ref_fm),
        f"{int(reachable.sum())} frontier slots",
    )
    j_err_w = reachable.float()
    j_err = (
        F.binary_cross_entropy_with_logits(
            j_el.double(), (~base_top1_ok).double(), reduction="none"
        )
        * j_err_w
    ).sum() / j_err_w.sum()
    check(
        "err BCE: joint path == dh2048",
        close(j_err.item(), ref_err.item()),
        f"{j_err.item():.12f} vs {ref_err.item():.12f}",
    )
    check(
        "total dh2048 loss (alpha=1, err_alpha=1)",
        close((joint_ce + j_err).item(), (ref_ce + ref_err).item()),
        f"{(joint_ce + j_err).item():.12f} vs {(ref_ce + ref_err).item():.12f}",
    )

    # ------------------------------------------------------------------ 3. GRADIENT -------------
    # The whole point of joint training: the selector's loss must reach the backbone. Emulate the
    # boundary with a leaf standing in for the backbone's output, since that is exactly where
    # selector_stop_gradient cuts.
    def selector_loss_backbone_grad(stop_gradient: bool) -> float:
        # cuDNN refuses to backward through the GRU in eval mode, and joint training runs in train
        # mode anyway; dropout only perturbs magnitudes, and this check is about zero vs nonzero.
        selector.train()
        h = b["dh"][:8].detach().clone().requires_grad_(True)
        lg = (h @ embed.t()).float()
        lg_in, h_in = (lg.detach(), h.detach()) if stop_gradient else (lg, h)
        lp2, ids2, scal2 = selector.extract_lattice(lg_in)
        s2 = selector.score_candidates(
            candidate_ids=ids2,
            unary_logits=lp2,
            hidden_states=h_in,
            predecessor_ids=b["prefix_ids"][:8],
            lattice_scalars=scal2,
        )
        tt = ids2.eq(b["g"][:8].clamp(min=0).unsqueeze(-1))
        idx = tt.long().argmax(dim=-1)
        keep = tt.any(dim=-1) & (b["g"][:8] >= 0)
        loss = F.cross_entropy(s2[keep], idx[keep])
        selector.zero_grad(set_to_none=True)
        loss.backward()
        grad = 0.0 if h.grad is None else h.grad.abs().max().item()
        selector.eval()
        return grad

    coupled = selector_loss_backbone_grad(stop_gradient=False)
    cut = selector_loss_backbone_grad(stop_gradient=True)
    check(
        "stop_gradient=false: selector loss reaches the backbone",
        coupled > 0.0,
        f"max|d loss / d backbone_hidden| = {coupled:.3e}",
    )
    check(
        "stop_gradient=true: backbone gradient is exactly zero",
        cut == 0.0,
        f"max|grad| = {cut:.3e}",
    )
    selector.zero_grad(set_to_none=True)

    # ------------------------------------------------------------------ 4. GROUPS ---------------
    import yaml

    ycfg = yaml.safe_load(Path(args.joint_yaml).read_text())
    t = ycfg["training"]
    check(
        "joint yaml has no duplicate deployment block (8 GPUs actually parsed)",
        ycfg["deployment"]["trainer"]["nproc_per_node"] == 8,
        f"nproc_per_node={ycfg['deployment']['trainer']['nproc_per_node']}",
    )
    check(
        "joint yaml: dh2048 objective weights and coupling",
        t["dflash2_selector_loss_alpha"] == 1.0
        and t["dflash2_selector_err_loss_alpha"] == 1.0
        and t["dflash2_selector_own_denominator"] is True
        and t["dflash2_selector_stop_gradient"] is False
        and t["dflash2_selector_target_greedy_labels"] is False,
        f"alpha={t['dflash2_selector_loss_alpha']} err={t['dflash2_selector_err_loss_alpha']} "
        f"own_den={t['dflash2_selector_own_denominator']} "
        f"stop_grad={t['dflash2_selector_stop_gradient']} "
        f"greedy_labels={t['dflash2_selector_target_greedy_labels']}",
    )
    check(
        "joint yaml: dh2048 weight decay 2e-3",
        float(t["weight_decay"]) == 2e-3,
        f"weight_decay={t['weight_decay']}",
    )

    from transformers import AutoConfig

    from specforge.modeling.draft.pspr import PSPRDraftModel
    from specforge.optimizer import BF16Optimizer

    # The optimizer contract must be checked against the config TRAINING actually loads -- the one
    # the yaml names in model.draft_model_config -- NOT the importer's output directory. Those two
    # drift apart the moment a selector option is added (the init dir was written by an earlier
    # importer run), and checking the stale one silently exempts every new parameter tensor from the
    # group assertions below.
    hf_config = AutoConfig.from_pretrained(args.draft_config, trust_remote_code=False)
    draft = PSPRDraftModel(hf_config)
    lr = float(t["learning_rate"])
    opt = BF16Optimizer(
        draft,
        lr=lr,
        weight_decay=float(t["weight_decay"]),
        total_steps=100,
        lr_scale_rules=tuple(sorted(t["lr_scale_rules"].items(), key=lambda kv: -len(kv[0]))),
        weight_decay_rules=tuple(
            sorted(t["weight_decay_rules"].items(), key=lambda kv: -len(kv[0]))
        ),
    )
    named = [(n, p) for n, p in draft.named_parameters() if p.requires_grad]
    # Assert by ROLE, never by copying the config's prefixes: PSPRDraftModel names its backbone
    # `layers.*` / `norm.*` / `fc.*` / `hidden_norm.*` with no `model.` prefix, and ties `lm_head`
    # away entirely, so a rule written as {"model.": 0.05} matches nothing. A gate that reused the
    # config's own prefix list would be wrong in exactly the same way and pass anyway.
    want = {}
    for n, p in named:
        if n == "candidate_selector.gamma":
            want[n] = (5.0, 0.0)
        elif n.endswith(("predecessor_codebook", "successor_codebook")):
            # DFlash2's transition tables are embeddings: sparsely read, so decaying the whole
            # tensor every step would shrink rows absent from the batch.
            want[n] = (1.0, 0.0)
        elif n.startswith("candidate_selector."):
            want[n] = (1.0, 2e-3)
        else:
            want[n] = (0.05, 2e-3)  # everything that is not the selector is the backbone
    got = {}
    masters = {id(m): i for i, m in enumerate(opt.fp32_params)}
    index_to_name = {i: n for i, (n, _) in enumerate(named)}
    for group in opt.optimizer.param_groups:
        for master in group["params"]:
            got[index_to_name[masters[id(master)]]] = (
                group["lr"] / lr,
                group["weight_decay"],
            )
    bad = [
        (n, want[n], got.get(n))
        for n in want
        if got.get(n) is None or abs(got[n][0] - want[n][0]) > 1e-9 or got[n][1] != want[n][1]
    ]
    check(
        "optimizer groups == dh2048 (backbone 0.05x, gamma 5x/wd0, rest 1x/wd2e-3)",
        not bad,
        f"{len(opt.optimizer.param_groups)} groups over {len(named)} tensors, mismatched={len(bad)}"
        + (f" e.g. {bad[:2]}" if bad else ""),
    )
    n_backbone = sum(1 for n, _ in named if not n.startswith("candidate_selector."))
    check(
        "backbone is actually scaled down (not silently left at 1x)",
        n_backbone > 0
        and all(
            abs(got[n][0] - 0.05) < 1e-9
            for n, _ in named
            if not n.startswith("candidate_selector.")
        ),
        f"{n_backbone} backbone tensors at lr*0.05",
    )
    check(
        "selector non-gamma tensors at 1x",
        all(
            abs(got[n][0] - 1.0) < 1e-9
            for n, _ in named
            if n.startswith("candidate_selector.") and n != "candidate_selector.gamma"
        ),
        f"{sum(1 for n, _ in named if n.startswith('candidate_selector.')) - 1} tensors",
    )
    check(
        "gamma is in a group of its own",
        sum(1 for g in opt.optimizer.param_groups if g["weight_decay"] == 0.0
            and abs(g["lr"] / lr - 5.0) < 1e-9 and len(g["params"]) == 1) == 1,
        f"gamma group lr={5.0 * lr:.3e} wd=0",
    )

    # ------------------------------------------------------------------ 5. ROUNDTRIP ------------
    from safetensors.torch import load_file

    fused = load_file(str(Path(args.joint_init) / "model.safetensors"))
    _, sel_back = split_state(fused)
    stage1 = payload["state_dict"]
    keys_ok = set(sel_back) == set(stage1)
    worst = max(maxdiff(sel_back[k], stage1[k]) for k in stage1) if keys_ok else float("nan")
    check(
        "import -> export roundtrip returns the stage-1 selector",
        keys_ok and worst == 0.0,
        f"{len(sel_back)} keys, max|delta| = {worst:.3e}",
    )

    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n{n_pass}/{len(RESULTS)} PASS")
    if n_pass != len(RESULTS):
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"  FAILED: {name}  {detail}")
        return 1
    print(
        "Joint training is wired to the dh2048 recipe: same online lattice, same objective, "
        "backbone actually coupled, same optimizer groups, closed checkpoint loop."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
