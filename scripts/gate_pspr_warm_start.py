"""G2: prove the warm start actually survives into the model the trainer will optimize.

``PSPRDraftModel._init_weights`` calls ``LatticePathSelector.restore_zero_init_contract()``, which
either zeroes ``gamma`` (default) or sets it to 1.0 and zeroes ``mlp[-1]`` (``output_zero_init``).
HF runs ``_init_weights`` over every submodule AFTER loading a state dict, so this is a live hazard,
not a hypothetical one: gamma back to 0 makes the selector a no-op, gamma forced to 1.0 changes the
correction magnitude the calibration was tuned for.

This gate has already earned its keep once. ``restore_zero_init_contract`` wrote ``gamma`` with raw
in-place tensor ops (``.zero_()``/``.fill_()``) while writing everything else through ``nn.init``.
Only the latter is safe: while ``_init_weights`` runs, HF swaps the ``nn.init`` primitives for
guarded versions (``transformers.initialization.guard_torch_init_functions``) that skip any tensor the
loader tagged ``_is_hf_initialized``. So ``delta_mlp[-1].weight`` survived and ``gamma`` was zeroed --
inside ``_load_pretrained_draft_state``'s own ``from_pretrained``, whose loading report still said
``missing=0, unexpected=0``. Ordering does not save you; only running the real path and looking does.

The check is stronger than "gamma has the right value": every selector tensor must be bit-identical
to the stage-1 checkpoint, and every backbone tensor bit-identical to the released DFlash weights.
That leaves no room for a partially-applied warm start, nor for a warm start that silently rounds the
head to the config dtype on its way through the loader.

Run::

    PYTHONPATH=. python scripts/gate_pspr_warm_start.py \
        --draft-config configs/qwen3-4b-pspr-joint.json \
        --checkpoint outputs/pspr_joint_init \
        --selector outputs/pspr_dh2048_exact/best.pt \
        --backbone /home/wangzhuoyu/sp-decoding-iclr/assets/models/Qwen3-4B-DFlash-b16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

SELECTOR_PREFIX = "candidate_selector."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft-config", default="configs/qwen3-4b-pspr-joint.json")
    ap.add_argument("--checkpoint", default="outputs/pspr_joint_init")
    ap.add_argument("--selector", default="outputs/pspr_dh2048_exact/best.pt")
    ap.add_argument(
        "--backbone",
        default="/home/wangzhuoyu/sp-decoding-iclr/assets/models/Qwen3-4B-DFlash-b16",
    )
    ap.add_argument(
        "--dtype",
        default=None,
        help="construction dtype; defaults to the draft config's, which is what the trainer uses",
    )
    args = ap.parse_args()

    from transformers import AutoConfig

    from export_pspr_for_decode import load_draft_state  # noqa: E402
    from specforge.modeling.auto import AutoDraftModel
    from specforge.training.model_loading import warm_start_draft_model

    draft_config = json.loads(Path(args.draft_config).read_text())
    dflash = draft_config["dflash_config"]
    print(
        f"config: output_zero_init={dflash.get('selector_output_zero_init', False)} "
        f"trans_rank={dflash.get('selector_trans_rank', 0)} "
        f"freeze_backbone={dflash.get('freeze_backbone', False)}"
    )

    failures: list[str] = []

    # Construct exactly the way ``build_dflash_draft`` does -- ``from_config`` with the run's dtype,
    # THEN warm start (model_providers.py:180-185 -> _finish_registered_draft -> _warm_start). The
    # dtype is not cosmetic: it decides whether the destination can hold what the checkpoint stores,
    # and checking a fp32 model would not be checking what the trainer runs.
    hf_config = AutoConfig.from_pretrained(args.checkpoint, trust_remote_code=False)
    dtype = getattr(torch, args.dtype or str(getattr(hf_config, "dtype", "float32")).split(".")[-1])
    print(f"construction dtype = {dtype} (trainer path: AutoDraftModel.from_config)")

    def build():
        return AutoDraftModel.from_config(hf_config, torch_dtype=dtype)

    # --- what the construction path alone produces, before any checkpoint --------------------
    fresh_gamma = float(build().candidate_selector.gamma.detach())
    print(f"freshly constructed gamma = {fresh_gamma:.6f}  (contract value, pre-warm-start)")

    # --- the real loading path the trainer uses ----------------------------------------------
    model = build()
    report = warm_start_draft_model(
        model,
        args.checkpoint,
        draft_config=hf_config,
        strategy="dflash",
        allow_missing_embedding=True,
    )
    print(f"warm start: {report.source} format={report.checkpoint_format}")

    loaded = dict(model.state_dict())
    gamma = float(loaded[f"{SELECTOR_PREFIX}gamma"].detach())
    print(f"gamma after warm start   = {gamma:.6f}")

    # --- selector must survive the warm start intact ------------------------------------------
    payload = torch.load(args.selector, map_location="cpu", weights_only=False)
    if payload.get("model_type") != "LatticeSelector":
        raise SystemExit(f"not a LatticeSelector checkpoint: {payload.get('model_type')!r}")
    stage1 = payload["state_dict"]
    print(
        f"stage-1 reference: {len(stage1)} tensors, cal_gain={payload.get('cal_gain'):.6f}, "
        f"tau={payload.get('tau')}"
    )

    from export_pspr_for_decode import split_state  # noqa: E402

    backbone_got, selector_got = split_state(loaded)

    # The bar is "exact after the one cast the destination dtype forces, and no other loss". Casting
    # the fp32 reference down once is the whole budget: a loader that rounds twice (e.g. by staging
    # through a bf16 intermediate model) or that re-initialises anything fails this, while genuine
    # bf16 storage passes. For an fp32 model the cast is the identity and this is bit-identity.
    def reference_for(name: str) -> torch.Tensor:
        return stage1[name].to(selector_got[name].dtype)

    missing = sorted(set(stage1) - set(selector_got))
    extra = sorted(set(selector_got) - set(stage1))
    if missing:
        failures.append(f"selector tensors absent after warm start: {missing[:6]}")
    if extra:
        failures.append(f"unexpected selector tensors: {extra[:6]}")

    worst_name, worst = None, 0.0
    for name in sorted(set(stage1) & set(selector_got)):
        delta = (selector_got[name].double() - reference_for(name).double()).abs().max().item()
        if delta > worst:
            worst_name, worst = name, delta
    print(f"selector vs stage-1: {len(set(stage1) & set(selector_got))} tensors, "
          f"max|delta|={worst:.3e}" + (f" at {worst_name}" if worst else ""))
    if worst != 0.0:
        failures.append(f"selector tensor {worst_name} drifted by {worst:.3e} (must be exact)")

    # --- backbone must be bit-identical to the released DFlash weights -----------------------
    released = load_draft_state(args.backbone)
    released = {k: v for k, v in released.items() if "embed_tokens" not in k}
    shared = sorted(set(released) & set(backbone_got))
    b_worst_name, b_worst = None, 0.0
    for name in shared:
        delta = (backbone_got[name].double() - released[name].double()).abs().max().item()
        if delta > b_worst:
            b_worst_name, b_worst = name, delta
    print(f"backbone vs released: {len(shared)}/{len(released)} tensors, max|delta|={b_worst:.3e}"
          + (f" at {b_worst_name}" if b_worst else ""))
    if len(shared) != len(released):
        failures.append(
            f"backbone tensor count mismatch: matched {len(shared)} of {len(released)}"
        )
    if b_worst != 0.0:
        failures.append(f"backbone tensor {b_worst_name} drifted by {b_worst:.3e}")

    # --- the specific tensors restore_zero_init_contract would clobber -----------------------
    # Indexed in reference format via ``split_state`` rather than by slicing the port-format key:
    # ``delta_mlp`` is the port's name for the reference's ``dh_mlp``, so a raw slice looks up a name
    # stage-1 never had, ``stage1.get`` returns None, and the check silently degrades to a print.
    print("\nzero-init-contract targets (these are what a late _init_weights would destroy):")
    watched = ["gamma"] + sorted(
        name for name in selector_got if name.startswith(("mlp.", "dh_mlp."))
    )
    for name in watched:
        tensor = selector_got[name]
        if name not in stage1:
            failures.append(
                f"zero-init-contract target {name!r} has no stage-1 counterpart: the check for the "
                "one thing this gate exists to catch is vacuous"
            )
            print(f"  {name:44s} absmax={tensor.abs().max().item():.6f}  NO REFERENCE")
            continue
        exact = torch.equal(tensor, reference_for(name))
        print(
            f"  {name:44s} absmax={tensor.abs().max().item():.6f}"
            f"{' == stage-1' if exact else '  != STAGE-1'}"
        )
        if not exact:
            failures.append(f"{name} differs from stage-1")

    # gamma is the one value whose corruption is silent rather than loud, so name it explicitly.
    if abs(gamma) < 1e-9:
        failures.append("gamma is 0 after warm start: the selector is a no-op (contract clobbered)")
    if abs(gamma - 1.0) < 1e-9 and abs(fresh_gamma - 1.0) < 1e-9:
        failures.append(
            "gamma is exactly the output_zero_init contract value 1.0, not the stage-1 value: "
            "the warm start was overwritten by _init_weights"
        )

    print()
    if failures:
        print("G2 FAIL:")
        for line in failures:
            print(f"  - {line}")
        return 1
    print(
        f"G2 PASS: warm start is exact -- gamma={gamma:.6f}, all {len(stage1)} selector tensors "
        f"identical to stage-1 under the single {dtype} cast, all {len(shared)} backbone tensors "
        f"bit-identical to the released weights"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
