#!/usr/bin/env python3
"""Export a PSPR-cloze training checkpoint into the artefacts the six-domain decoder consumes.

A separate file from ``export_pspr_v2_for_decode.py`` on purpose: that script is what produced
every v2 artefact currently on disk, and a shared script that grew a branch could silently change
how an existing checkpoint exports.  The two differ only in the selector config and constructor.

Why convert instead of writing a new decoder: every number this run has to beat -- the stage-1
selector's 5.7217 macro, the 5.2842 one-shot base, the official Domino 6.4255 -- was produced by
``TAPS-SP/scripts/decode_lattice.py``.  Re-implementing block construction, KV-cache handling and
acceptance counting would put an unvalidated decoder between the model and the comparison.

Two artefacts:

``<out>/backbone/``
    A DFlash HF directory.  Written even though this run freezes the backbone, precisely so the
    export can *prove* it is unchanged: ``--verify-frozen`` compares every backbone tensor against
    the source release and fails on any difference.

``<out>/selector.pt``
    ``{"model_type": "CascadeCorrector", "config", "state_dict", "training_policy"}``.
    ``decode_lattice.load_selector_checkpoint`` dispatches on ``model_type``, so a v1 and a v2
    artefact can never be silently loaded into each other's class.

``--verify-roundtrip`` reloads the written selector with ``strict=True`` into a freshly constructed
module and asserts every tensor matches bit for bit, which is what licenses attributing any decode
difference to training rather than to this conversion.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

SPECFORGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPECFORGE))

SELECTOR_PREFIX = "candidate_selector."


def selector_config(draft_config: dict) -> dict:
    """Rebuild the ``CascadeCorrector`` constructor config from the draft-config JSON."""
    d = draft_config["dflash_config"]
    return {
        "model_type": "CascadeCorrector",
        "hidden_dim": int(draft_config["hidden_size"]),
        "vocab_size": int(draft_config["vocab_size"]),
        "K": int(d.get("selector_top_k", 16)),
        "d": int(d.get("selector_dim", 512)),
        "n_layers": int(d.get("selector_layers", 6)),
        "n_heads": int(d.get("selector_heads", 8)),
        "ds": int(d.get("selector_state_dim", 512)),
        "delta_h": int(d.get("selector_delta_hidden", 2048)),
        "max_slots": int(d.get("selector_max_slots", 32)),
        "n_scal": 3,
        "dropout": float(d.get("selector_dropout", 0.1)),
        "direct_hidden": bool(d.get("selector_direct_hidden", True)),
        "use_state": bool(d.get("selector_use_state", True)),
        "bidirectional": bool(d.get("selector_bidirectional", True)),
        "err_use_state": bool(d.get("selector_err_use_state", False)),
        # Recorded so decode reconstructs the same graph.  It only gates whether the gate loss can
        # flow back into the ranker, which is training-time behaviour, but it is part of the module
        # signature and a mismatch would make strict loading the only thing standing between a
        # silently different model and a "successful" evaluation.
        "err_score_detach": bool(d.get("selector_err_score_detach", True)),
        # Unlike err_score_detach these three change the DECISION.  ``err_score_enabled=False`` is the
        # blind control arm and ``err_score_anchor_rho`` is the parameter-free champion-equivalent gate
        # logit, so a decoder that defaulted either one would evaluate a different policy than was
        # trained while every tensor still loaded strictly.
        "err_score_features": str(d.get("selector_err_score_features", "margins_summary")),
        "err_score_enabled": bool(d.get("selector_err_score_enabled", True)),
        "err_score_anchor_rho": float(d.get("selector_err_score_anchor_rho", 3.0)),
    }


def build(cfg: dict):
    from specforge.modeling.draft.pspr_cascade import CascadeCorrector

    return CascadeCorrector(
        hidden_size=cfg["hidden_dim"], vocab_size=cfg["vocab_size"], top_k=cfg["K"],
        d=cfg["d"], n_layers=cfg["n_layers"], n_heads=cfg["n_heads"], state_dim=cfg["ds"],
        delta_hidden=cfg["delta_h"], max_slots=cfg["max_slots"], dropout=cfg["dropout"],
        direct_hidden=cfg["direct_hidden"], use_state=cfg["use_state"],
        bidirectional=cfg["bidirectional"],
        err_use_state=cfg.get("err_use_state", False),
        err_score_detach=cfg.get("err_score_detach", True),
        err_score_features=cfg["err_score_features"],
        err_score_enabled=cfg["err_score_enabled"],
        err_score_anchor_rho=cfg["err_score_anchor_rho"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="path to training_state.pt")
    ap.add_argument("--draft-config", required=True, help="the draft-config JSON used to train")
    ap.add_argument(
        "--backbone-template", required=True,
        help="HF dir supplying config.json/modeling files for the exported backbone",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify-roundtrip", action="store_true", default=True)
    ap.add_argument(
        "--verify-frozen", default=None,
        help="assert every backbone tensor equals this release's (use for frozen-backbone runs)",
    )
    args = ap.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = payload["draft_state_dict"]
    step = int(payload.get("global_step", -1))
    draft_config = json.loads(Path(args.draft_config).read_text())

    # Provenance, asserted BEFORE a single artefact is written.  The failure this prevents is not
    # hypothetical: every cascade tensor name that the champion also has matches, so pointing this
    # script at a cloze checkpoint (or pointing it at a cascade checkpoint with the cloze
    # draft-config) produces an artefact labelled ``CascadeCorrector`` that is missing exactly the
    # tensors the arm is about.  ``strict=True`` on the roundtrip does eventually catch it, but only
    # after the backbone has been written and with an error that names tensors instead of the mistake.
    architectures = list(draft_config.get("architectures") or [])
    if architectures != ["PSPRCascadeDraftModel"]:
        raise SystemExit(
            f"--draft-config declares architectures={architectures!r}; this exporter only writes "
            "CascadeCorrector artefacts. Use export_pspr_cloze_for_decode.py for a cloze run."
        )
    cascade_tensors = {
        k
        for k in state
        if k.startswith(SELECTOR_PREFIX)
        and k.removeprefix(SELECTOR_PREFIX).startswith(("err_score_", "err_base_gain"))
    }
    score_head_expected = bool(
        draft_config["dflash_config"].get("selector_err_score_enabled", True)
    )
    has_score_head = any("err_score_" in k for k in cascade_tensors)
    if not any("err_base_gain" in k for k in cascade_tensors):
        raise SystemExit(
            "checkpoint carries no candidate_selector.err_base_gain.*: it was not trained by "
            "PSPRCascadeDraftModel, so labelling it CascadeCorrector would misreport provenance"
        )
    if has_score_head != score_head_expected:
        raise SystemExit(
            f"draft-config says selector_err_score_enabled={score_head_expected} but the checkpoint "
            f"{'has' if has_score_head else 'lacks'} err_score_* tensors; the config and the "
            "checkpoint describe different arms"
        )

    selector_state = {
        k[len(SELECTOR_PREFIX):]: v.detach().cpu().clone()
        for k, v in state.items()
        if k.startswith(SELECTOR_PREFIX) and not k.endswith("target_embedding")
    }
    backbone_state = {
        k: v.detach().cpu().clone()
        for k, v in state.items()
        if not k.startswith(SELECTOR_PREFIX)
    }
    if not selector_state:
        raise SystemExit("checkpoint carries no candidate_selector.* tensors")

    cfg = selector_config(draft_config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # --- backbone -------------------------------------------------------------------------------
    bb = out / "backbone"
    bb.mkdir(exist_ok=True)
    template = Path(args.backbone_template)
    for name in ("config.json", "dflash.py", "modeling_dflash.py", "utils.py"):
        src = template / name
        if src.exists():
            shutil.copy2(src, bb / name)
    from safetensors.torch import save_file

    save_file(backbone_state, str(bb / "model.safetensors"))
    print(f"backbone: {len(backbone_state)} tensors -> {bb}")

    if args.verify_frozen:
        from safetensors.torch import load_file

        ref = load_file(str(Path(args.verify_frozen) / "model.safetensors"))
        drift = []
        for k, v in backbone_state.items():
            r = ref.get(k)
            if r is None:
                drift.append((k, "absent in reference"))
            elif not torch.equal(v.float(), r.float()):
                drift.append((k, f"max|d|={(v.float()-r.float()).abs().max():.3e}"))
        missing = sorted(set(ref) - set(backbone_state))
        if drift or missing:
            raise SystemExit(
                f"backbone is NOT frozen: {len(drift)} changed, {len(missing)} missing\n"
                + "\n".join(f"  {k}: {why}" for k, why in drift[:8])
            )
        print(f"verify-frozen: all {len(ref)} backbone tensors bit-identical to {args.verify_frozen}")

    # --- selector -------------------------------------------------------------------------------
    sel_path = out / "selector.pt"
    dcfg = draft_config["dflash_config"]
    selector_objective = str(payload.get("dflash2_selector_objective", ""))
    allowed_objectives = {
        # covered_conditional serves argmax over the same K conditional scores, i.e. margin_gate
        # at rho=1/tau=0/theta=0.  Without this entry a 9-hour training run exports nothing:
        # the whitelist here is separate from the host's train/serve contract and from
        # decode_lattice.validate_selector_policy (which only checks decision_mode), so the
        # objective name has to be registered in all of them.
        "margin_gate": {"multiclass", "profitable_action", "covered_conditional"},
        "keep_repair": {"keep_repair", "profitable_repair", "accept_repair"},
        "full_vocab": {"full_vocab"},
    }.get(str(dcfg.get("selector_decision_mode", "margin_gate")))
    if not selector_objective:
        raise SystemExit(
            "checkpoint does not record dflash2_selector_objective; refusing to invent training "
            "provenance"
        )
    if allowed_objectives is None or selector_objective not in allowed_objectives:
        raise SystemExit(
            "selector train/serve policy mismatch: "
            f"objective={selector_objective!r}, decision_mode="
            f"{dcfg.get('selector_decision_mode', 'margin_gate')!r}, "
            f"allowed={sorted(allowed_objectives) if allowed_objectives else []}"
        )
    # ``decode_lattice.validate_selector_policy`` is fail-closed: it refuses to score a checkpoint
    # whose serving rule is not recorded, because top-k and gate values change the decision without
    # changing any tensor shape, so neither strict loading nor a clean forward proves the evaluator
    # is running the policy the head was exported for.  Emit the full contract rather than reaching
    # for --allow-policy-mismatch, which would silently drop that check for every future run too.
    decode_policy = {
        "selector_decision_mode": str(dcfg.get("selector_decision_mode", "margin_gate")),
        "selector_compute_dtype": str(dcfg.get("selector_compute_dtype", "float32")),
        "selector_gate_rho": float(dcfg.get("selector_gate_rho", 1.0)),
        "selector_gate_tau": float(dcfg.get("selector_gate_tau", 0.0)),
        "selector_gate_theta": float(dcfg.get("selector_gate_theta", 0.0)),
        "selector_gate_skip_first": bool(dcfg.get("selector_gate_skip_first", False)),
        "selector_keep_repair_margin": float(dcfg.get("selector_keep_repair_margin", 0.0)),
        "selector_top_k": cfg["K"],
    }
    torch.save(
        {
            "model_type": "CascadeCorrector",
            "config": cfg,
            "state_dict": selector_state,
            "global_step": step,
            "decode_policy": decode_policy,
            "training_policy": {
                "verified": True,
                "selector_objective": selector_objective,
                "selector_top_k": cfg["K"],
                "backbone": "frozen" if args.verify_frozen else "trained",
                **decode_policy,
            },
        },
        sel_path,
    )
    n_params = sum(v.numel() for v in selector_state.values())
    print(f"selector: {len(selector_state)} tensors / {n_params/1e6:.2f}M -> {sel_path} (step {step})")

    if args.verify_roundtrip:
        reloaded = torch.load(sel_path, map_location="cpu", weights_only=False)
        model = build(reloaded["config"])
        model.load_state_dict(reloaded["state_dict"], strict=True)
        back = {k: v for k, v in model.state_dict().items() if k != "target_embedding"}
        if set(back) != set(selector_state):
            raise SystemExit(
                f"roundtrip key mismatch: only-in-model={sorted(set(back)-set(selector_state))[:5]} "
                f"only-in-file={sorted(set(selector_state)-set(back))[:5]}"
            )
        worst = max(float((back[k] - selector_state[k]).abs().max()) for k in back)
        if worst != 0.0:
            raise SystemExit(f"roundtrip value drift: max|delta|={worst:.3e}")
        print(f"verify-roundtrip: {len(back)} tensors, strict=True, max|delta|=0")


if __name__ == "__main__":
    main()
