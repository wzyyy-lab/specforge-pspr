#!/usr/bin/env python3
"""Measure whether PSPR's base, selector, and detector losses fight in the backbone.

The diagnostic uses one *identical* offline feature sample and resets every RNG before each
forward.  ``OnlineDFlashModel.forward`` already accepts loss-weight overrides, so the three
pre-clipping gradient contributions can be isolated by linearity:

    g_base = grad L(0, 0)
    g_sel  = grad L(1, 0) - g_base
    g_err  = grad L(0, 1) - g_base

The selector is still evaluated in all three forwards; this keeps random-number consumption and
the sampled anchor set identical.  Reported cosine values are therefore about the actual objective,
not about three different mini-batches.

Example:

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/diagnose_pspr_gradient_conflict.py \
      --draft outputs/pspr_s1_step2000_init \
      --target-model ../TAPS-SP/models/Qwen3-4B \
      --output outputs/pspr_gradient_conflict_s1.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import Qwen3Config

from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
from specforge.data.preprocessing import process_offline_dflash_sample
from specforge.modeling.draft.pspr import PSPRDraftModel
from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead
from specforge.runtime.data_plane.offline_reader import list_feature_files


def _group(name: str) -> str:
    if name.startswith("layers."):
        return ".".join(name.split(".")[:2])
    return name.split(".", 1)[0]


def _accumulate(stats: dict, group: str, key: str, value: torch.Tensor) -> None:
    stats.setdefault(group, {}).setdefault(key, 0.0)
    stats[group][key] += float(value.double().sum().item())


def _finish(stats: dict) -> dict:
    out = {}
    for group, row in stats.items():
        bn = math.sqrt(max(row.get("bb", 0.0), 0.0))
        sn = math.sqrt(max(row.get("ss", 0.0), 0.0))
        en = math.sqrt(max(row.get("ee", 0.0), 0.0))

        def cos(dot: float, x: float, y: float) -> float | None:
            return dot / (x * y) if x > 0 and y > 0 else None

        out[group] = {
            "base_norm": bn,
            "selector_norm": sn,
            "err_norm": en,
            "selector_over_base": sn / bn if bn else None,
            "err_over_base": en / bn if bn else None,
            "cos_base_selector": cos(row.get("bs", 0.0), bn, sn),
            "cos_base_err": cos(row.get("be", 0.0), bn, en),
            "cos_selector_err": cos(row.get("se", 0.0), sn, en),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", required=True, help="PSPR HF directory containing model.safetensors")
    ap.add_argument("--target-model", required=True)
    ap.add_argument("--features", default="cache/hidden_states/sharegpt3k")
    ap.add_argument("--feature-index", type=int, default=0)
    ap.add_argument("--max-length", type=int, default=768)
    ap.add_argument("--anchors", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("this diagnostic needs one CUDA device")
    device, dtype = torch.device("cuda"), torch.bfloat16
    draft_dir = Path(args.draft)
    raw_config = json.loads((draft_dir / "config.json").read_text())
    config = Qwen3Config(**raw_config)
    config.dflash_config = raw_config["dflash_config"]
    config.block_size = raw_config["block_size"]
    config.num_target_layers = raw_config["num_target_layers"]
    config._attn_implementation = "sdpa"

    with torch.device("cpu"):
        draft = PSPRDraftModel(config)
    state = load_file(str(draft_dir / "model.safetensors"))
    missing, unexpected = draft.load_state_dict(state, strict=False)
    missing = [key for key in missing if key != "candidate_selector.target_embedding"]
    if missing or unexpected:
        raise RuntimeError(f"draft checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    draft = draft.to(device=device, dtype=dtype)

    target = TargetEmbeddingsAndHead.from_pretrained(
        args.target_model, device=str(device), dtype=dtype
    )
    draft.bind_target_decoder(target.embed_tokens)
    model = OnlineDFlashModel(
        draft_model=draft,
        target_lm_head=target.lm_head,
        target_embed_tokens=target.embed_tokens,
        mask_token_id=config.dflash_config["mask_token_id"],
        block_size=config.block_size,
        attention_backend="sdpa",
        num_anchors=args.anchors,
        objective_chunk_blocks=args.anchors,
        loss_type="dpace",
        dpace_alpha=0.5,
        selector_loss_alpha=1.0,
        selector_err_loss_alpha=1.0,
        # A local selector denominator is partition-dependent once gradients
        # are accumulated/reduced. Sharing the base denominator preserves the
        # cosine direction this diagnostic measures and matches the fixed
        # distributed training path.
        selector_own_denominator=False,
        selector_weight_mode="uniform",
        selector_objective="multiclass",
        selector_err_weight_mode="base_frontier",
        selector_stop_gradient=False,
    ).to(device=device, dtype=dtype)
    model.train()

    files = list_feature_files(args.features)
    path = None
    sample = None
    skipped = 0
    for candidate_path in files[args.feature_index :]:
        raw = torch.load(candidate_path, map_location="cpu", weights_only=False)
        try:
            sample = process_offline_dflash_sample(raw, args.max_length)
        except ValueError as exc:
            # Feature shards may begin with a long prompt whose supervised
            # response starts after ``max_length``.  Such a shard is unusable
            # for this diagnostic but is not a corrupt training example; find
            # the first deterministic usable shard instead of making the
            # default command depend on filesystem ordering.
            if "two consecutive supervised tokens" not in str(exc):
                raise
            skipped += 1
            continue
        path = candidate_path
        break
    if path is None or sample is None:
        raise ValueError(
            f"no usable offline feature at or after index {args.feature_index} "
            f"with max_length={args.max_length}"
        )
    inputs = {
        "input_ids": sample["input_ids"].to(device),
        "hidden_states": sample["hidden_states"].to(device=device, dtype=dtype),
        "loss_mask": sample["loss_mask"].to(device),
        "max_valid_anchors": args.anchors,
    }
    backbone = [
        (name, parameter)
        for name, parameter in draft.named_parameters()
        if parameter.requires_grad and not name.startswith("candidate_selector.")
    ]

    def run(selector_alpha: float, err_alpha: float) -> float:
        model.zero_grad(set_to_none=True)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        loss, _, _ = model(
            **inputs,
            selector_loss_alpha=selector_alpha,
            selector_err_loss_alpha=err_alpha,
        )
        loss.backward()
        return float(loss.detach())

    losses = {"base": run(0.0, 0.0)}
    base_grads = {
        name: torch.zeros_like(parameter, dtype=torch.float32)
        if parameter.grad is None
        else parameter.grad.detach().float().clone()
        for name, parameter in backbone
    }

    losses["base_plus_selector"] = run(1.0, 0.0)
    selector_grads = {}
    stats: dict = {}
    for name, parameter in backbone:
        total = torch.zeros_like(base_grads[name]) if parameter.grad is None else parameter.grad.float()
        base = base_grads[name]
        selector = total - base
        selector_grads[name] = selector.clone()
        for group in ("all", _group(name)):
            _accumulate(stats, group, "bb", base * base)
            _accumulate(stats, group, "ss", selector * selector)
            _accumulate(stats, group, "bs", base * selector)

    losses["base_plus_err"] = run(0.0, 1.0)
    for name, parameter in backbone:
        total = torch.zeros_like(base_grads[name]) if parameter.grad is None else parameter.grad.float()
        base = base_grads[name]
        selector = selector_grads[name]
        err = total - base
        for group in ("all", _group(name)):
            _accumulate(stats, group, "ee", err * err)
            _accumulate(stats, group, "be", base * err)
            _accumulate(stats, group, "se", selector * err)

    result = {
        "method": "same-sample pre-clipping gradient decomposition by alpha override",
        "checkpoint": str(draft_dir.resolve()),
        "feature": str(Path(path).resolve()),
        "feature_shards_skipped": skipped,
        "max_length": args.max_length,
        "anchors": args.anchors,
        "seed": args.seed,
        "dtype": str(dtype),
        "losses": losses,
        "groups": _finish(stats),
        "notes": {
            "effective_backbone_lr_scale": 0.05,
            "cosine_is_unchanged_by_uniform_lr_scale": True,
            "selector_objective": "multiclass",
            "selector_weight_mode": "uniform",
            "selector_normalization": "shared_global_base_denominator",
            "err_weight_mode": "base_frontier",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
