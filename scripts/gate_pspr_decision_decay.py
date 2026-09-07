#!/usr/bin/env python3
"""Verify the weight-decay partition of the PSPR-Decision head EXHAUSTIVELY.

The optimiser matches on ``name.startswith(prefix) or f".{prefix}" in name`` over rules sorted by
descending prefix length (specforge/optimizer.py:94-98).  A missing rule silently drops decay from a
weight matrix; a too-broad rule silently applies it to LayerNorms and biases.  Both are invisible in
the loss curve, so they are checked here instead.

Run:  PYTHONPATH=. python3 scripts/gate_pspr_decision_decay.py
"""

from __future__ import annotations

import sys

import yaml

sys.path.insert(0, ".")

from specforge.modeling.draft.pspr_decision import DecisionCorrector  # noqa: E402

YAML_PATH = (
    "examples/configs/online/disaggregated/managed-local/qwen3-4b-pspr-decision.yaml"
)
JSON_PATH = "configs/qwen3-4b-pspr-decision.json"

cfg = yaml.safe_load(open(YAML_PATH))
rules = cfg["training"]["weight_decay_rules"]
default_wd = float(cfg["training"]["weight_decay"])
sorted_rules = sorted(rules.items(), key=lambda kv: -len(kv[0]))

import json  # noqa: E402

dflash = json.load(open(JSON_PATH))["dflash_config"]
head = DecisionCorrector(
    hidden_size=2560,
    vocab_size=151936,
    top_k=int(dflash["selector_top_k"]),
    d=int(dflash["selector_dim"]),
    n_layers=int(dflash["selector_layers"]),
    expansion=int(dflash["selector_expansion"]),
    state_dim=int(dflash["selector_state_dim"]),
    delta_hidden=int(dflash["selector_delta_hidden"]),
    det_hidden=int(dflash["selector_det_hidden"]),
    dropout=float(dflash["selector_dropout"]),
    direct_hidden=bool(dflash["selector_direct_hidden"]),
    use_state=bool(dflash["selector_use_state"]),
    err_use_state=bool(dflash["selector_err_use_state"]),
    block_context=bool(dflash["selector_block_context"]),
)


def match(name: str):
    for prefix, value in sorted_rules:
        if name.startswith(prefix) or f".{prefix}" in name:
            return prefix, float(value)
    return None, default_wd


decayed, undecayed, problems = [], [], []
for name, p in head.named_parameters():
    # The host prefixes every selector tensor.
    full = f"candidate_selector.{name}"
    prefix, wd = match(full)
    entry = (full, tuple(p.shape), p.numel(), prefix, wd)
    (decayed if wd > 0 else undecayed).append(entry)
    is_matrix = p.dim() >= 2
    if is_matrix and wd == 0.0:
        problems.append(f"MATRIX WITHOUT DECAY: {full} {tuple(p.shape)}")
    if not is_matrix and wd > 0.0:
        problems.append(f"NON-MATRIX WITH DECAY: {full} {tuple(p.shape)} via {prefix!r}")

print(f"rules: {len(rules)}   default weight_decay={default_wd}")
print(f"decayed   : {len(decayed):3d} tensors  {sum(e[2] for e in decayed)/1e6:7.3f}M")
print(f"undecayed : {len(undecayed):3d} tensors  {sum(e[2] for e in undecayed)/1e6:7.3f}M")
print(f"total     : {len(decayed)+len(undecayed):3d} tensors  "
      f"{sum(p.numel() for p in head.parameters())/1e6:7.3f}M")

unused = [r for r, _ in sorted_rules if not any(e[3] == r for e in decayed)]
if unused:
    problems.append(f"RULES THAT MATCHED NOTHING: {unused}")

print("\ndecayed tensors:")
for full, shape, n, prefix, _ in sorted(decayed):
    print(f"  {full:52s} {str(shape):20s} via {prefix!r}")

print("\nundecayed tensor kinds:")
kinds: dict[str, int] = {}
for full, shape, n, _, _ in undecayed:
    kind = "LayerNorm/bias" if len(shape) <= 1 else f"dim{len(shape)}"
    kinds[kind] = kinds.get(kind, 0) + 1
for kind, count in sorted(kinds.items()):
    print(f"  {kind}: {count}")

print()
if problems:
    print("PROBLEMS:")
    for problem in problems:
        print(f"  {problem}")
    sys.exit(1)
print("DECAY PARTITION OK: every matrix decays, nothing else does, no dead rules")
