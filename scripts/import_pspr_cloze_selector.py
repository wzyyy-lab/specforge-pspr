#!/usr/bin/env python3
"""Import a cloze/SlotDeep selector plus the released DFlash backbone into one PSPR draft checkpoint.

This is the exact inverse of ``scripts/export_pspr_cloze_for_decode.py`` and reuses its
``selector_config``/``build``, so ``train -> export -> import -> train`` is a closed loop.

``scripts/import_pspr_selector.py`` covers only the reference ``LatticeSelector`` layout used by
``PSPRDraftModel``; the cloze family stores a different reference config (``ClozeCorrector`` /
``SlotDeepCorrector``) and cannot go through it.

Stage 1 freezes the backbone, so the stage-1 export already verified its 58 backbone tensors are
bit-identical to the release. Stage 2 therefore starts from that same release backbone plus the
trained selector, and this script asserts both halves rather than assuming them.

Run::

    PYTHONPATH=. python scripts/import_pspr_cloze_selector.py \\
        --selector outputs/<run>_step7115_decode/selector.pt \\
        --backbone ../TAPS-SP/models/Qwen3-4B-DFlash-b16 \\
        --draft-config configs/qwen3-4b-pspr-slotdeep-joint.json \\
        --output-dir outputs/slotdeep_joint_init
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

from export_pspr_cloze_for_decode import (  # noqa: E402
    SELECTOR_PREFIX,
    build,
    selector_config,
)

CLOZE_FAMILY = ("PSPRClozeDraftModel", "PSPRSlotDeepDraftModel")

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
    ap.add_argument("--selector", required=True, help="cloze/SlotDeep selector.pt from the exporter")
    ap.add_argument(
        "--backbone",
        default=str(ROOT.parent / "TAPS-SP/models/Qwen3-4B-DFlash-b16"),
        help="DFlash draft HF directory supplying the backbone tensors",
    )
    ap.add_argument("--draft-config", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument(
        "--expect-frozen-backbone-from",
        default=None,
        help="stage-1 decode export dir; assert its backbone equals --backbone bit for bit",
    )
    ap.add_argument(
        "--keep-embedding",
        action="store_true",
        help="keep embed_tokens in the output; by default it is dropped because the training host "
        "supplies the target embedding and a [vocab, hidden] tensor must not enter the draft ckpt",
    )
    args = ap.parse_args()

    from safetensors.torch import load_file, save_file

    draft_config = json.loads(Path(args.draft_config).read_text())
    arch = (draft_config.get("architectures") or [None])[0]
    if arch not in CLOZE_FAMILY:
        raise SystemExit(
            f"--draft-config must declare one of {CLOZE_FAMILY}, got {arch!r}. "
            "Use scripts/import_pspr_selector.py for PSPRDraftModel."
        )

    payload = torch.load(args.selector, map_location="cpu", weights_only=False)
    expected = selector_config(draft_config)
    if payload.get("model_type") != expected["model_type"]:
        raise SystemExit(
            f"selector model_type {payload.get('model_type')!r} does not match the config's "
            f"{expected['model_type']!r}"
        )
    if payload["config"] != expected:
        diff = {
            k: (payload["config"].get(k), expected.get(k))
            for k in set(payload["config"]) | set(expected)
            if payload["config"].get(k) != expected.get(k)
        }
        raise SystemExit(
            f"selector checkpoint shape does not match --draft-config (key: (ckpt, config)): {diff}"
        )
    selector = {k: v.clone() for k, v in payload["state_dict"].items()}
    step = payload.get("global_step")
    print(
        f"selector: {len(selector)} tensors, model_type={payload['model_type']}, step={step}, "
        f"objective={payload.get('training_policy', {}).get('selector_objective')}"
    )

    # The selector must instantiate strictly under its own reference config before anything is
    # written, so a stale config fails here rather than after a multi-hour run.
    probe_sel = build(expected)
    probe_sel.load_state_dict(selector, strict=True)
    del probe_sel
    print(f"selector strict-load against {expected['model_type']}: OK")

    backbone_state = load_file(str(Path(args.backbone) / "model.safetensors"))
    stray = sorted(k for k in backbone_state if k.startswith(SELECTOR_PREFIX))
    if stray:
        raise SystemExit(f"--backbone already carries selector tensors: {stray[:4]}")
    print(f"backbone: {len(backbone_state)} tensors from {args.backbone}")

    if args.expect_frozen_backbone_from:
        ref = load_file(str(Path(args.expect_frozen_backbone_from) / "backbone" / "model.safetensors"))
        drift = [
            k for k, v in backbone_state.items()
            if k not in ref or not torch.equal(v.float(), ref[k].float())
        ]
        missing = sorted(set(ref) - set(backbone_state))
        if drift or missing:
            raise SystemExit(
                f"stage-1 backbone is not the release backbone: {len(drift)} changed, "
                f"{len(missing)} missing; first={drift[:4]}"
            )
        print(
            f"expect-frozen-backbone: all {len(ref)} tensors bit-identical to "
            f"{args.expect_frozen_backbone_from}"
        )

    state = dict(backbone_state)
    for k, v in selector.items():
        state[SELECTOR_PREFIX + k] = v

    # Closed loop against the exporter: re-splitting with the exporter's own rule must return both
    # inputs bit for bit, which is what makes export -> import a roundtrip, not a re-derivation.
    back = {k: v for k, v in state.items() if not k.startswith(SELECTOR_PREFIX)}
    sel = {
        k[len(SELECTOR_PREFIX):]: v
        for k, v in state.items()
        if k.startswith(SELECTOR_PREFIX) and not k.endswith("target_embedding")
    }
    if set(back) != set(backbone_state) or set(sel) != set(selector):
        raise SystemExit("roundtrip key mismatch against the exporter's split rule")
    worst = max(
        max((sel[k].double() - selector[k].double()).abs().max().item() for k in selector),
        max((back[k].double() - backbone_state[k].double()).abs().max().item() for k in back),
    )
    if worst != 0.0:
        raise SystemExit(f"roundtrip value mismatch: max|delta| = {worst:.3e}")
    print(f"roundtrip through the exporter's split rule: {len(state)} tensors, max|delta| = 0")

    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite a non-empty output dir: {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(draft_config, indent=2) + "\n")

    # The model must accept the joined state with nothing unexpected and nothing required missing --
    # the same contract warm_start_draft_model enforces at training time, checked here where the
    # failure is cheap.
    from transformers import AutoConfig

    from specforge.modeling.draft.registry import resolve_draft

    cls = resolve_draft(arch)
    hf_config = AutoConfig.from_pretrained(str(out), trust_remote_code=False)
    probe = cls(hf_config)
    result = probe.load_state_dict(state, strict=False)
    optional = tuple(getattr(probe, "warm_start_optional_prefixes", ()))
    required_missing = [
        k for k in result.missing_keys if not (k.startswith(optional) or "embed" in k.lower())
    ]
    if result.unexpected_keys or required_missing:
        raise SystemExit(
            f"joined state does not match {arch}: "
            f"unexpected={sorted(result.unexpected_keys)[:6]} missing={required_missing[:6]}"
        )
    print(
        f"{arch} load: {len(state)} tensors, unexpected=0, "
        f"missing={len(result.missing_keys)} (all optional/embedding)"
    )
    del probe

    if not args.keep_embedding:
        state = {k: v for k, v in state.items() if "embed_tokens" not in k}
    save_file(state, str(out / "model.safetensors"))
    for name in COPY_FROM_BACKBONE:
        src = Path(args.backbone) / name
        if src.exists():
            shutil.copy2(src, out / name)
    n_params = sum(v.numel() for v in state.values())
    print(f"wrote {len(state)} tensors / {n_params/1e6:.2f}M -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
