#!/usr/bin/env python3
"""Export a SpecForge PSPR checkpoint into the two artefacts the reference decoder consumes.

Why not write a new decoder: the 5.705 macro this work has to beat was produced by
``TAPS-SP/scripts/decode_lattice.py``, and the 6.435 official-Domino baseline by that repo's own
harness. Re-implementing block construction, KV-cache handling and acceptance counting would put an
unvalidated decoder between the model and every number. So the model is converted to that decoder's
input format instead, and the decoder is used unmodified.

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
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch

SPECFORGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPECFORGE))

SELECTOR_PREFIX = "candidate_selector."
# The two deliberate renames, documented at their definition sites in pspr.py.
PORT_TO_REFERENCE = {"delta_mlp.": "dh_mlp."}
SCALAR_IN_REFERENCE = {"gamma"}


def selector_config(draft_config: dict) -> dict:
    """Rebuild the reference ``LatticeSelector`` config from the PSPR draft config."""
    dflash = draft_config["dflash_config"]
    return {
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


def load_draft_state(checkpoint: str) -> Dict[str, torch.Tensor]:
    """Read a draft state dict from a SpecForge checkpoint or an exported HF directory."""
    path = Path(checkpoint)
    if path.is_dir():
        from safetensors.torch import load_file

        shards = sorted(path.glob("*.safetensors"))
        if not shards:
            raise ValueError(f"{path} contains no .safetensors shard")
        state: Dict[str, torch.Tensor] = {}
        for shard in shards:
            state.update(load_file(str(shard)))
        return state
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("draft_state_dict", "state_dict", "model"):
        if isinstance(payload, dict) and isinstance(payload.get(key), dict):
            return payload[key]
    if isinstance(payload, dict) and all(
        torch.is_tensor(v) for v in payload.values()
    ):
        return payload
    raise ValueError(f"cannot find a draft state dict inside {path}")


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
    args = ap.parse_args()

    if bool(args.checkpoint) == bool(args.from_reference):
        ap.error("pass exactly one of --checkpoint / --from-reference")

    draft_config = json.loads(Path(args.draft_config).read_text())
    config = selector_config(draft_config)
    template = Path(args.backbone_template)
    output = Path(args.output_dir)
    extras: dict = {}

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
        print(f"imported reference selector from {Path(args.from_reference).name}")
    else:
        state = load_draft_state(args.checkpoint)
        backbone, selector = split_state(state)
        print(f"loaded {len(state)} draft tensors from {args.checkpoint}")

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
            "model_type": "LatticeSelector",
            "config": config,
            "state_dict": selector,
            **extras,
        },
        output / "selector.pt",
    )
    print(f"\nwrote {output / 'backbone'} ({n_written} tensors) and {output / 'selector.pt'}")
    print("\ndecode with:")
    print(
        f"  cd /kl_infra_infer_intern/wangzhuoyu/TAPS-SP && python scripts/decode_lattice.py \\\n"
        f"      --draft-model {output / 'backbone'} --lattice-head {output / 'selector.pt'} \\\n"
        f"      --modes oneshot,latgate --gate-tau 0 --gate-rho 9 --gate-skip0 \\\n"
        f"      --eval-reserved --max-samples 40 --max-new-tokens 256"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
