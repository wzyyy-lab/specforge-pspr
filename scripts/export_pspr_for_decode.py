#!/usr/bin/env python3
"""Export a SpecForge PSPR checkpoint into the two artefacts the reference decoder consumes.

Why not write a new decoder: the 5.705 macro this work has to beat was produced by
``TAPS-SP/scripts/decode_lattice.py``, and the 6.435 official-Domino baseline by that repo's own
harness. Re-implementing block construction, KV-cache handling and acceptance counting would put an
unvalidated decoder between the model and every number. So the model is converted to that decoder's
input format instead, and the same audited decoding harness is used.

Two artefacts are produced:

``<out>/backbone/``
    A DFlash HF directory loadable by ``TAPS-SP/model.py::DFlashDraftModel.from_pretrained``. Needed
    even when the backbone is frozen, because joint training changes it.

``<out>/selector.pt``
    A ``LatticeSelector`` checkpoint, i.e. ``{"model_type", "config", "state_dict", ...}``.

The mapping is asserted to be lossless in both directions by
``--verify-roundtrip``, which is what licenses attributing any decode difference to training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

SPECFORGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPECFORGE))

SELECTOR_PREFIX = "candidate_selector."
# The two deliberate renames, documented at their definition sites in pspr.py.
PORT_TO_REFERENCE = {
    "delta_mlp.": "dh_mlp.",
    "predecessor_codebook": "trans_pred",
    "successor_codebook": "trans_succ",
}
SCALAR_IN_REFERENCE = {"gamma"}


def selector_config(draft_config: dict) -> dict:
    """Rebuild the reference ``LatticeSelector`` config from the PSPR draft config."""
    dflash = draft_config["dflash_config"]
    config = {
        "hidden_dim": draft_config["hidden_size"],
        "vocab_embed_dim": draft_config["hidden_size"],
        "K": dflash.get("selector_top_k", 16),
        "d": dflash.get("selector_dim", 512),
        "n_layers": dflash.get("selector_layers", 3),
        "n_heads": dflash.get("selector_heads", 8),
        "ds": dflash.get("selector_state_dim", 512),
        "max_slots": dflash.get("selector_max_slots", 32),
        "n_scal": 3,
        "dropout": dflash.get("selector_dropout", 0.1),
        "tok_rank": 0,
        "vocab_size": draft_config["vocab_size"],
        "pair_dim": 0,
        "delta_h": dflash.get("selector_delta_hidden", 2048),
        "thidden_dim": 0,
    }
    # Old dh2048 checkpoints intentionally have no ``trans_rank`` key in the
    # reference config, so preserve exact dictionary equality for their import
    # gate.  A PSPR+DFlash2-transition checkpoint does contain extra tensors,
    # however, and must instantiate the matching reference submodule before a
    # strict load/roundtrip can succeed.
    trans_rank = int(dflash.get("selector_trans_rank", 0))
    if trans_rank > 0:
        config["trans_rank"] = trans_rank
    return config


def split_state(state: Dict[str, torch.Tensor]) -> Tuple[dict, dict]:
    """Split a PSPR draft state dict into (backbone, reference-format selector)."""
    backbone, selector = {}, {}
    for key, value in state.items():
        if not key.startswith(SELECTOR_PREFIX):
            backbone[key] = value
            continue
        inner = key[len(SELECTOR_PREFIX) :]
        for port, reference in PORT_TO_REFERENCE.items():
            if inner.startswith(port):
                inner = reference + inner[len(port) :]
        # ``gamma`` is 1-D in the port (FSDP rejects zero-dim parameters) and a scalar in the
        # reference; reshape rather than squeeze so an unexpected width fails loudly.
        selector[inner] = value.reshape(()) if inner in SCALAR_IN_REFERENCE else value
    return backbone, selector


def join_state(backbone: dict, selector: dict) -> Dict[str, torch.Tensor]:
    """Inverse of ``split_state``, used by the roundtrip check and by reference imports."""
    reference_to_port = {v: k for k, v in PORT_TO_REFERENCE.items()}
    state = dict(backbone)
    for key, value in selector.items():
        inner = key
        for reference, port in reference_to_port.items():
            if key.startswith(reference):
                inner = port + key[len(reference) :]
        state[SELECTOR_PREFIX + inner] = (
            value.reshape(1) if key in SCALAR_IN_REFERENCE else value
        )
    return state


def load_draft_state_and_metadata(
    checkpoint: str,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Read draft tensors plus non-tensor checkpoint provenance.

    An HF directory contains weights but no training-policy contract, so its
    metadata result is empty.  A SpecForge ``training_state.pt`` records the
    resolved selector objective/weight mode in its resume contract; retaining
    those fields is what lets export fail closed on a train/serve mismatch.
    """
    path = Path(checkpoint)
    if path.is_dir():
        from safetensors.torch import load_file

        shards = sorted(path.glob("*.safetensors"))
        if not shards:
            raise ValueError(f"{path} contains no .safetensors shard")
        state: Dict[str, torch.Tensor] = {}
        for shard in shards:
            state.update(load_file(str(shard)))
        return state, {}
    # Checkpoints also contain multi-gigabyte optimizer state. mmap keeps those
    # untouched pages out of RAM while the exporter reads only draft tensors.
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    policy_metadata_keys = {
        "dflash2_selector_objective",
        "dflash2_selector_weight_mode",
        "dflash2_selector_top_k",
        "dflash2_selector_own_denominator",
        "dflash2_selector_survival_floor",
        "dflash2_selector_decision_mode",
        "dflash2_selector_keep_repair_margin",
        "dflash2_selector_gate_rho",
        "dflash2_selector_gate_tau",
        "dflash2_selector_gate_theta",
        "dflash2_selector_gate_skip_first",
        "dflash2_selector_compute_dtype",
        "specforge_model_provenance",
    }
    metadata = (
        {key: payload[key] for key in policy_metadata_keys if key in payload}
        if isinstance(payload, dict)
        else {}
    )
    for key in ("draft_state_dict", "state_dict", "model"):
        if isinstance(payload, dict) and isinstance(payload.get(key), dict):
            return payload[key], metadata
    if isinstance(payload, dict) and all(
        torch.is_tensor(v) for v in payload.values()
    ):
        return payload, {}
    raise ValueError(f"cannot find a draft state dict inside {path}")


def load_draft_state(checkpoint: str) -> Dict[str, torch.Tensor]:
    """Backward-compatible weights-only loader used by the conversion gates."""
    return load_draft_state_and_metadata(checkpoint)[0]


def validate_policy_provenance(
    metadata: Dict[str, Any],
    decode_policy: Dict[str, Any],
    *,
    selector_top_k: int,
    draft_config_path: str,
    allow_missing: bool,
) -> Dict[str, Any]:
    """Prove that the exported serving rule matches the trained likelihood."""
    objective = metadata.get("dflash2_selector_objective")
    expected_objectives = {
        # covered_conditional serves argmax over the same K conditional scores, i.e. margin_gate
        # at rho=1/tau=0/theta=0.  Without this entry a 9-hour training run exports nothing:
        # the whitelist here is separate from the host's train/serve contract and from
        # decode_lattice.validate_selector_policy (which only checks decision_mode), so the
        # objective name has to be registered in all of them.
        "margin_gate": {"multiclass", "profitable_action", "covered_conditional"},
        "keep_repair": {"keep_repair", "profitable_repair", "accept_repair"},
    }.get(decode_policy["selector_decision_mode"])
    if expected_objectives is None:
        raise ValueError(
            "unsupported selector_decision_mode in draft config: "
            f"{decode_policy['selector_decision_mode']!r}"
        )
    missing = []
    if objective is None:
        missing.append("dflash2_selector_objective")
    elif str(objective) not in expected_objectives:
        raise ValueError(
            "checkpoint/config train-serve policy mismatch: checkpoint trained "
            f"dflash2_selector_objective={objective!r}, but draft config serves "
            f"selector_decision_mode={decode_policy['selector_decision_mode']!r} "
            f"(allows {sorted(expected_objectives)!r})"
        )

    recorded_top_k = metadata.get("dflash2_selector_top_k")
    if recorded_top_k is None:
        missing.append("dflash2_selector_top_k")
    elif int(recorded_top_k) != int(selector_top_k):
        raise ValueError(
            "checkpoint/config top-k mismatch: checkpoint records "
            f"K={recorded_top_k}, draft config requests K={selector_top_k}; "
            "strict state_dict loading cannot detect this semantic mismatch"
        )

    weight_mode = metadata.get("dflash2_selector_weight_mode")
    if weight_mode is None:
        missing.append("dflash2_selector_weight_mode")
    own_denominator = metadata.get("dflash2_selector_own_denominator")
    if own_denominator is None:
        missing.append("dflash2_selector_own_denominator")
    elif bool(own_denominator):
        raise ValueError(
            "checkpoint used dflash2_selector_own_denominator=true, whose local selector mean is "
            "partition-dependent under DDP; it cannot be exported as a verified training policy"
        )
    survival_floor = metadata.get("dflash2_selector_survival_floor")
    if str(weight_mode) == "smoothed_accept":
        if survival_floor is None:
            missing.append("dflash2_selector_survival_floor")
        elif not 0.0 < float(survival_floor) <= 1.0:
            raise ValueError(
                "invalid dflash2_selector_survival_floor recorded by checkpoint: "
                f"{survival_floor!r}"
            )

    config_sha256 = hashlib.sha256(Path(draft_config_path).read_bytes()).hexdigest()
    recorded_config_sha256 = metadata.get("draft_config_sha256")
    provenance = metadata.get("specforge_model_provenance")
    if recorded_config_sha256 is None and provenance is not None:
        try:
            draft_identity = dict(provenance)["draft_config"]
            record = draft_identity[2]
            if record and record[0] == "sha256" and record[1] in ("", "config.json"):
                recorded_config_sha256 = record[3]
        except (KeyError, TypeError, ValueError, IndexError):
            recorded_config_sha256 = None
    if recorded_config_sha256 is None:
        missing.append("draft_config_sha256")
    elif str(recorded_config_sha256) != config_sha256:
        raise ValueError(
            "draft config content does not match the config recorded by the checkpoint: "
            f"checkpoint sha256={recorded_config_sha256}, supplied sha256={config_sha256}"
        )

    # New checkpoints also persist every resolved serving field explicitly.
    # Older current-format checkpoints remain verifiable through the config
    # content hash; compare explicit fields whenever they are available.
    explicit_policy_fields = {
        "dflash2_selector_decision_mode": "selector_decision_mode",
        "dflash2_selector_keep_repair_margin": "selector_keep_repair_margin",
        "dflash2_selector_gate_rho": "selector_gate_rho",
        "dflash2_selector_gate_tau": "selector_gate_tau",
        "dflash2_selector_gate_theta": "selector_gate_theta",
        "dflash2_selector_gate_skip_first": "selector_gate_skip_first",
        "dflash2_selector_compute_dtype": "selector_compute_dtype",
    }
    for checkpoint_key, policy_key in explicit_policy_fields.items():
        if checkpoint_key in metadata and metadata[checkpoint_key] != decode_policy[policy_key]:
            raise ValueError(
                f"checkpoint/config policy mismatch for {policy_key}: checkpoint records "
                f"{metadata[checkpoint_key]!r}, draft config requests {decode_policy[policy_key]!r}"
            )

    if missing and not allow_missing:
        raise ValueError(
            "checkpoint lacks policy provenance fields "
            f"{missing}, so export cannot prove that serving matches training; use a current "
            "SpecForge training_state.pt or pass --allow-missing-policy-provenance explicitly "
            "for a legacy checkpoint"
        )
    return {
        "verified": not missing,
        "missing": missing,
        "selector_objective": None if objective is None else str(objective),
        "selector_weight_mode": None if weight_mode is None else str(weight_mode),
        "selector_top_k": None if recorded_top_k is None else int(recorded_top_k),
        "selector_own_denominator": (
            None if own_denominator is None else bool(own_denominator)
        ),
        "selector_survival_floor": (
            None if survival_floor is None else float(survival_floor)
        ),
        "draft_config_sha256": config_sha256,
    }


def write_backbone(
    backbone: dict, template: Path, output: Path, drop_embedding: bool
) -> int:
    """Write a DFlash HF directory: template metadata plus the trained backbone tensors."""
    from safetensors.torch import save_file

    output.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "dflash.py", "tokenizer.json", "tokenizer_config.json",
                 "generation_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"):
        source = template / name
        if source.exists():
            shutil.copy2(source, output / name)

    template_config = json.loads((template / "config.json").read_text())
    if template_config.get("architectures") != ["DFlashDraftModel"]:
        raise ValueError(
            "backbone template must be a DFlash draft directory; got "
            f"architectures={template_config.get('architectures')!r}"
        )

    tensors = {
        key: value.contiguous()
        for key, value in backbone.items()
        if not (drop_embedding and "embed_tokens" in key)
    }
    save_file(tensors, str(output / "model.safetensors"), metadata={"format": "pt"})
    for stale in output.glob("model-*.safetensors"):
        stale.unlink()
    index = output / "model.safetensors.index.json"
    if index.exists():
        index.unlink()
    return len(tensors)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        help="SpecForge training state, or an HF directory exported from one",
    )
    ap.add_argument(
        "--from-reference",
        help="instead of a SpecForge checkpoint, import a reference LatticeSelector ckpt "
        "(used by the neutrality check: reference selector + official backbone must reproduce 5.705)",
    )
    ap.add_argument("--draft-config", default=str(SPECFORGE / "configs/qwen3-4b-pspr.json"))
    ap.add_argument(
        "--backbone-template",
        default="/kl_infra_infer_intern/wangzhuoyu/TAPS-SP/models/Qwen3-4B-DFlash-b16",
        help="supplies config.json/tokenizer and, with --from-reference, the backbone weights",
    )
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--keep-embedding", action="store_true")
    ap.add_argument("--verify-roundtrip", action="store_true", default=True)
    ap.add_argument(
        "--allow-missing-policy-provenance",
        action="store_true",
        help="allow a legacy weights-only/reference checkpoint whose training objective cannot "
        "be proven; the exported selector is marked policy_provenance.verified=false",
    )
    args = ap.parse_args()

    if bool(args.checkpoint) == bool(args.from_reference):
        ap.error("pass exactly one of --checkpoint / --from-reference")

    draft_config = json.loads(Path(args.draft_config).read_text())
    config = selector_config(draft_config)
    dflash_config = draft_config["dflash_config"]
    decode_policy = {
        "selector_decision_mode": dflash_config.get(
            "selector_decision_mode", "margin_gate"
        ),
        "selector_keep_repair_margin": float(
            dflash_config.get("selector_keep_repair_margin", 0.0)
        ),
        "selector_gate_rho": float(dflash_config.get("selector_gate_rho", 1.0)),
        "selector_gate_tau": float(dflash_config.get("selector_gate_tau", 0.0)),
        "selector_gate_theta": float(dflash_config.get("selector_gate_theta", 0.0)),
        "selector_gate_skip_first": bool(
            dflash_config.get("selector_gate_skip_first", False)
        ),
        "selector_compute_dtype": str(
            dflash_config.get("selector_compute_dtype", "model")
        ),
    }
    template = Path(args.backbone_template)
    # The printed decode command changes into the sibling TAPS-SP repository.
    # Resolve here so a caller that supplied a SpecForge-relative output path
    # still receives a directly runnable command rather than a broken path
    # interpreted relative to TAPS-SP.
    output = Path(args.output_dir).resolve()
    extras: dict = {}
    checkpoint_metadata: Dict[str, Any] = {}

    if args.from_reference:
        payload = torch.load(args.from_reference, map_location="cpu", weights_only=False)
        if payload.get("model_type") != "LatticeSelector":
            raise ValueError(f"not a LatticeSelector ckpt: {payload.get('model_type')!r}")
        if payload["config"] != config:
            raise ValueError(
                "reference selector config does not match the PSPR draft config:\n"
                f"  reference: {payload['config']}\n  draft:     {config}"
            )
        selector = payload["state_dict"]
        backbone = load_draft_state(str(template))
        extras = {k: v for k, v in payload.items()
                  if k not in ("model_type", "config", "state_dict")}
        checkpoint_metadata = dict(payload.get("training_policy") or {})
        if "selector_objective" in checkpoint_metadata:
            checkpoint_metadata["dflash2_selector_objective"] = checkpoint_metadata[
                "selector_objective"
            ]
        if "selector_weight_mode" in checkpoint_metadata:
            checkpoint_metadata["dflash2_selector_weight_mode"] = checkpoint_metadata[
                "selector_weight_mode"
            ]
        if "selector_top_k" in checkpoint_metadata:
            checkpoint_metadata["dflash2_selector_top_k"] = checkpoint_metadata[
                "selector_top_k"
            ]
        if "selector_own_denominator" in checkpoint_metadata:
            checkpoint_metadata["dflash2_selector_own_denominator"] = checkpoint_metadata[
                "selector_own_denominator"
            ]
        if "selector_survival_floor" in checkpoint_metadata:
            checkpoint_metadata["dflash2_selector_survival_floor"] = checkpoint_metadata[
                "selector_survival_floor"
            ]
        source_decode_policy = payload.get("decode_policy")
        if source_decode_policy is not None and source_decode_policy != decode_policy:
            raise ValueError(
                "reference checkpoint decode_policy does not match the supplied draft config: "
                f"reference={source_decode_policy!r} supplied={decode_policy!r}"
            )
        print(f"imported reference selector from {Path(args.from_reference).name}")
    else:
        state, checkpoint_metadata = load_draft_state_and_metadata(args.checkpoint)
        backbone, selector = split_state(state)
        print(f"loaded {len(state)} draft tensors from {args.checkpoint}")

    training_policy = validate_policy_provenance(
        checkpoint_metadata,
        decode_policy,
        selector_top_k=int(dflash_config.get("selector_top_k", 16)),
        draft_config_path=args.draft_config,
        allow_missing=args.allow_missing_policy_provenance,
    )
    print(
        "  policy provenance: "
        f"verified={training_policy['verified']} "
        f"objective={training_policy['selector_objective']} "
        f"weight_mode={training_policy['selector_weight_mode']}"
    )

    n_selector = sum(v.numel() for v in selector.values())
    n_backbone = sum(v.numel() for v in backbone.values())
    print(f"  backbone {len(backbone)} tensors / {n_backbone / 1e6:.2f}M")
    print(f"  selector {len(selector)} tensors / {n_selector / 1e6:.2f}M")

    if args.verify_roundtrip:
        rejoined = join_state(backbone, selector)
        reference_split = split_state(rejoined)
        ok = True
        for name, before, after in (
            ("backbone", backbone, reference_split[0]),
            ("selector", selector, reference_split[1]),
        ):
            if set(before) != set(after):
                print(f"  [XX] {name} key set changed across the roundtrip")
                ok = False
                continue
            worst = max(
                (before[k].float() - after[k].float()).abs().max().item() for k in before
            )
            shapes_ok = all(before[k].shape == after[k].shape for k in before)
            print(f"  [{'ok' if worst == 0 and shapes_ok else 'XX'}] {name} roundtrip: "
                  f"max|delta|={worst:.3e} shapes_preserved={shapes_ok}")
            ok = ok and worst == 0.0 and shapes_ok
        if not ok:
            print("\nEXPORT ABORTED: the key mapping is not lossless")
            return 1

        # The selector must also load into the reference class with strict=True: a silently
        # renamed or missing tensor would otherwise only surface as a bad decode number.
        sys.path.insert(0, "/kl_infra_infer_intern/wangzhuoyu/TAPS-SP")
        from joint.lattice_selector import LatticeSelector

        probe = LatticeSelector(**config)
        missing, unexpected = probe.load_state_dict(selector, strict=True)
        print(f"  [ok] reference LatticeSelector accepts it with strict=True "
              f"(missing={list(missing)} unexpected={list(unexpected)})")

    n_written = write_backbone(backbone, template, output / "backbone", not args.keep_embedding)
    torch.save(
        {
            **extras,
            "model_type": "LatticeSelector",
            "config": config,
            "state_dict": selector,
            "decode_policy": decode_policy,
            "training_policy": training_policy,
        },
        output / "selector.pt",
    )
    print(f"\nwrote {output / 'backbone'} ({n_written} tensors) and {output / 'selector.pt'}")
    print("\ndecode with:")
    decision_mode = decode_policy["selector_decision_mode"]
    if decision_mode == "keep_repair":
        policy_args = (
            "--modes oneshot,latrepair "
            f"--repair-margin {decode_policy['selector_keep_repair_margin']}"
        )
    else:
        policy_args = (
            "--modes oneshot,latgate "
            f"--gate-tau {decode_policy['selector_gate_tau']} "
            f"--gate-rho {decode_policy['selector_gate_rho']} "
            f"--gate-theta {decode_policy['selector_gate_theta']}"
        )
        if decode_policy["selector_gate_skip_first"]:
            policy_args += " --gate-skip0"
    print(
        f"  cd /kl_infra_infer_intern/wangzhuoyu/TAPS-SP && PYTHONPATH=. "
        f"python scripts/decode_lattice.py \\\n"
        f"      --draft-model {output / 'backbone'} --lattice-head {output / 'selector.pt'} \\\n"
        f"      {policy_args} \\\n"
        f"      --eval-reserved --max-samples 20 --max-new-tokens 256"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
