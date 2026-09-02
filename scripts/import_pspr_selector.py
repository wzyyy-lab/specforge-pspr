"""Import a dh2048-format selector plus a DFlash backbone into one PSPR draft checkpoint.

Stage 1 (``scripts/train_pspr_accept_selector.py``) trains the selector alone and writes it in the
reference ``LatticeSelector`` layout -- ``{model_type, config, state_dict, cal_gain, tau}`` with
reference key names -- so the reference decoder can read it without conversion. Stage 2 (joint
training) instead needs a single HF directory that ``PSPRDraftModel`` warm-starts from: the DFlash
backbone tensors and the selector tensors together, under port key names.

This is the exact inverse of ``scripts/export_pspr_for_decode.py`` and reuses its mapping, so
``import -> train -> export`` is a closed loop.

Run::

    PYTHONPATH=. python scripts/import_pspr_selector.py \
        --selector outputs/pspr_dh2048_exact/best.pt \
        --backbone ../TAPS-SP/models/Qwen3-4B-DFlash-b16 \
        --draft-config configs/qwen3-4b-pspr-joint.json \
        --output-dir outputs/pspr_joint_init
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from export_pspr_for_decode import (  # noqa: E402
    SELECTOR_PREFIX,
    join_state,
    load_draft_state,
    selector_config,
    split_state,
)

COPY_FROM_BACKBONE = (
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--selector",
        required=True,
        help="reference-format selector checkpoint, e.g. outputs/pspr_dh2048_exact/best.pt",
    )
    ap.add_argument(
        "--backbone",
        default=str(ROOT.parent / "TAPS-SP/models/Qwen3-4B-DFlash-b16"),
        help="DFlash draft HF directory supplying the backbone tensors",
    )
    ap.add_argument("--draft-config", default=str(ROOT / "configs/qwen3-4b-pspr-joint.json"))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument(
        "--keep-embedding",
        action="store_true",
        help="keep embed_tokens in the output; by default it is dropped because the training host "
        "supplies the target embedding and a [vocab, hidden] tensor must not enter the draft ckpt",
    )
    args = ap.parse_args()

    from safetensors.torch import save_file

    draft_config = json.loads(Path(args.draft_config).read_text())
    if draft_config.get("architectures") != ["PSPRDraftModel"]:
        raise SystemExit(
            f"--draft-config must declare PSPRDraftModel, got {draft_config.get('architectures')!r}"
        )

    payload = torch.load(args.selector, map_location="cpu", weights_only=False)
    if payload.get("model_type") != "LatticeSelector":
        raise SystemExit(f"not a LatticeSelector checkpoint: {payload.get('model_type')!r}")
    expected = selector_config(draft_config)
    if payload["config"] != expected:
        diff = {
            k: (payload["config"].get(k), expected.get(k))
            for k in set(payload["config"]) | set(expected)
            if payload["config"].get(k) != expected.get(k)
        }
        raise SystemExit(
            "selector checkpoint shape does not match --draft-config "
            f"(key: (checkpoint, config)): {diff}"
        )
    selector = payload["state_dict"]
    print(f"selector: {len(selector)} keys, cal_gain={payload.get('cal_gain')} tau={payload.get('tau')}")

    backbone_state = load_draft_state(args.backbone)
    stray = sorted(k for k in backbone_state if k.startswith(SELECTOR_PREFIX))
    if stray:
        raise SystemExit(f"--backbone already carries selector tensors: {stray[:4]}")
    print(f"backbone: {len(backbone_state)} keys from {args.backbone}")

    state = join_state(backbone_state, selector)
    # Closed loop against the exporter: splitting the joined state must return both inputs bit for
    # bit, which is what makes stage-1 -> stage-2 -> decode a roundtrip rather than a re-derivation.
    back, sel = split_state(state)
    if set(back) != set(backbone_state) or set(sel) != set(selector):
        raise SystemExit("roundtrip key mismatch between join_state and split_state")
    worst = max(
        max((sel[k].double() - selector[k].double()).abs().max().item() for k in selector),
        max((back[k].double() - backbone_state[k].double()).abs().max().item() for k in back),
    )
    if worst != 0.0:
        raise SystemExit(f"roundtrip value mismatch: max|delta| = {worst:.3e}")
    print(f"roundtrip through split_state: {len(state)} keys, max|delta| = {worst:.3e}")

    # The model must accept the joined state with nothing unexpected and nothing required missing --
    # the same contract warm_start_draft_model enforces at training time, checked here where the
    # failure is cheap.
    from specforge.modeling.draft.pspr import PSPRDraftModel
    from transformers import AutoConfig

    cfg_path = Path(args.output_dir)
    cfg_path.mkdir(parents=True, exist_ok=True)
    (cfg_path / "config.json").write_text(json.dumps(draft_config, indent=2) + "\n")
    hf_config = AutoConfig.from_pretrained(str(cfg_path), trust_remote_code=False)
    probe = PSPRDraftModel(hf_config)
    result = probe.load_state_dict(state, strict=False)
    optional = tuple(getattr(probe, "warm_start_optional_prefixes", ()))
    required_missing = [
        k for k in result.missing_keys if not (k.startswith(optional) or "embed" in k.lower())
    ]
    if result.unexpected_keys or required_missing:
        raise SystemExit(
            f"joined state does not match PSPRDraftModel: "
            f"unexpected={sorted(result.unexpected_keys)[:6]} missing={required_missing[:6]}"
        )
    print(
        f"PSPRDraftModel load: {len(state)} keys, unexpected=0, "
        f"missing={len(result.missing_keys)} (all optional/embedding)"
    )
    del probe

    tensors = {
        k: v.contiguous()
        for k, v in state.items()
        if args.keep_embedding or "embed_tokens" not in k
    }
    save_file(tensors, str(cfg_path / "model.safetensors"), metadata={"format": "pt"})
    for stale in cfg_path.glob("model-*.safetensors"):
        stale.unlink()
    index = cfg_path / "model.safetensors.index.json"
    if index.exists():
        index.unlink()
    backbone_dir = Path(args.backbone)
    if backbone_dir.is_dir():
        for name in COPY_FROM_BACKBONE:
            source = backbone_dir / name
            if source.exists():
                shutil.copy2(source, cfg_path / name)

    n_sel = sum(1 for k in tensors if k.startswith(SELECTOR_PREFIX))
    print(
        f"wrote {cfg_path}/model.safetensors: {len(tensors)} tensors "
        f"({len(tensors) - n_sel} backbone + {n_sel} selector)"
    )
    print(f"set model.draft_checkpoint_path: {cfg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
