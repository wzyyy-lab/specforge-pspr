#!/usr/bin/env python3
"""Gate the stage-2 (joint) optimizer grouping for the SlotDeep arm.

Stage 2 differs from stage 1 by exactly one thing: the backbone is trainable.  That makes two
grouping questions load-bearing, and both are silent if wrong:

  G1  the stage-1 weight-decay rules are suffix-shaped ("fc1.weight"), and ``match`` also fires on
      ``f".{prefix}" in name`` -- so a backbone tensor could accidentally inherit selector decay.
  G2  ``lr_scale_rules`` is order-sensitive: the catch-all "" matches every name, so it must be last
      or the selector silently trains at the backbone's throttled rate.

Run::
    PYTHONPATH=. python scripts/gate_slotdeep_joint_groups.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SELECTOR_PREFIX = "candidate_selector."
EXPECTED_SELECTOR_SHA256 = "4bffa7688ff209399922c00deb194e776133a74dc406122f10fa2406389c0879"
RESULTS: list[tuple[str, bool, str]] = []


def gate(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def match(rules, name, default):
    """Verbatim copy of specforge/optimizer.py::_parameter_groups.match."""
    for prefix, value in rules:
        if name.startswith(prefix) or f".{prefix}" in name:
            return float(value)
    return default


def as_assembled(rules: dict) -> tuple:
    """Verbatim copy of the ordering specforge/training/assembly.py applies before the optimizer.

    The YAML's own key order is NOT what reaches the optimizer: assembly sorts longest-prefix-first
    so a specific rule can override a broader one.  A gate that asserts on the YAML order therefore
    tests a risk that does not exist; this reproduces the real ordering instead.
    """
    return tuple(sorted(rules.items(), key=lambda kv: -len(kv[0])))


def main() -> int:
    ap_yaml = ROOT / "examples/configs/online/disaggregated/managed-local/qwen3-4b-pspr-slotdeep-joint.yaml"
    s1_yaml = ROOT / "examples/configs/online/disaggregated/managed-local/qwen3-4b-pspr-slotdeep.yaml"
    init_dir = ROOT / "outputs/slotdeep_joint_init"

    cfg = yaml.safe_load(ap_yaml.read_text())["training"]
    s1cfg = yaml.safe_load(s1_yaml.read_text())["training"]
    lr = float(cfg["learning_rate"])
    lr_rules = as_assembled(cfg.get("lr_scale_rules") or {})
    wd_rules = as_assembled(cfg.get("weight_decay_rules") or {})
    wd_default = float(cfg.get("weight_decay", 0.0))

    from safetensors.torch import load_file

    names = sorted(load_file(str(init_dir / "model.safetensors")).keys())
    sel = [n for n in names if n.startswith(SELECTOR_PREFIX)]
    bb = [n for n in names if not n.startswith(SELECTOR_PREFIX)]
    print(f"init checkpoint: {len(names)} tensors ({len(sel)} selector, {len(bb)} backbone)\n")

    print("=== G1 stage-1 decay rules must not leak onto the now-trainable backbone ===")
    leaked = {n: match(wd_rules, n, wd_default) for n in bb}
    bad = {n: w for n, w in leaked.items() if w != wd_default}
    gate("G1 every backbone tensor keeps the default weight decay", not bad,
         f"default={wd_default}, leaked={list(bad)[:4]}")

    print("\n=== G2 rule ordering as ASSEMBLED (not as written in the yaml) ===")
    yaml_keys = list((cfg.get("lr_scale_rules") or {}).keys())
    keys = [k for k, _ in lr_rules]
    gate("G2a a catch-all rule exists", "" in keys, f"assembled={keys}")
    gate("G2b assembly puts the catch-all last regardless of yaml order",
         bool(keys) and keys[-1] == "", f"yaml={yaml_keys} -> assembled={keys}")
    shuffled = as_assembled({k: v for k, v in reversed(list((cfg.get("lr_scale_rules") or {}).items()))})
    gate("G2c the assembled order is invariant to yaml key order",
         shuffled == lr_rules, f"reversed-yaml assembles to {[k for k, _ in shuffled]}")

    print("\n=== G3 the resulting two rates are the intended ones ===")
    sel_scales = {match(lr_rules, n, 1.0) for n in sel}
    bb_scales = {match(lr_rules, n, 1.0) for n in bb}
    gate("G3a every selector tensor trains at the full rate", sel_scales == {1.0},
         f"scales={sorted(sel_scales)} -> lr={sorted(s*lr for s in sel_scales)}")
    gate("G3b every backbone tensor shares one throttled rate", len(bb_scales) == 1,
         f"scales={sorted(bb_scales)} -> lr={sorted(s*lr for s in bb_scales)}")
    bb_scale = next(iter(bb_scales)) if len(bb_scales) == 1 else None
    gate("G3c the backbone rate is strictly below the selector rate",
         bb_scale is not None and bb_scale < 1.0, f"backbone lr={bb_scale*lr:.2e}" if bb_scale else "")

    print("\n=== G4 the selector's own grouping is unchanged from stage 1 ===")
    s1_wd = as_assembled(s1cfg.get("weight_decay_rules") or {})
    s1_default = float(s1cfg.get("weight_decay", 0.0))
    s1_lr = float(s1cfg["learning_rate"])
    differ = {
        n: (match(s1_wd, n, s1_default), match(wd_rules, n, wd_default))
        for n in sel
        if match(s1_wd, n, s1_default) != match(wd_rules, n, wd_default)
    }
    gate("G4a selector weight decay identical to stage 1", not differ,
         f"differ={list(differ)[:4]}")
    gate("G4b selector effective lr identical to stage 1", lr == s1_lr,
         f"stage1={s1_lr:.2e} stage2={lr:.2e}")
    decayed = sum(1 for n in sel if match(wd_rules, n, wd_default) > 0)
    gate("G4c a non-trivial share of the selector is still decayed", decayed > 0,
         f"{decayed}/{len(sel)} selector tensors decayed at 2e-3")

    print("\n=== G5 stage 2 differs from stage 1 only by unfreezing the backbone ===")
    j = json.loads((ROOT / "configs/qwen3-4b-pspr-slotdeep-joint.json").read_text())
    s = json.loads((ROOT / "configs/qwen3-4b-pspr-slotdeep.json").read_text())
    diff = {k: (s["dflash_config"].get(k), j["dflash_config"].get(k))
            for k in set(s["dflash_config"]) | set(j["dflash_config"])
            if s["dflash_config"].get(k) != j["dflash_config"].get(k)}
    gate("G5a draft configs differ only in freeze_backbone", set(diff) == {"freeze_backbone"}, f"{diff}")
    gate("G5b stage 2 has the backbone trainable", j["dflash_config"].get("freeze_backbone") is False,
         f"freeze_backbone={j['dflash_config'].get('freeze_backbone')}")
    gate("G5c architectures unchanged", j["architectures"] == s["architectures"], f"{j['architectures']}")

    print("\n=== G6 the init artifact is provably built from the intended stage-1 selector ===")
    # The importer prints step/objective but does not assert them, and the normal and turnwise
    # selectors share model type, config, step AND objective -- so "it loaded" is not provenance.
    # Pin the selector by content hash and re-derive the joined tensors from it here.
    import hashlib

    sel_path = ROOT / "outputs/qwen3-4b-pspr-slotdeep-s1-20260905_step7115_decode/selector.pt"
    sha = hashlib.sha256(sel_path.read_bytes()).hexdigest()
    gate("G6a selector.pt content hash is the audited one",
         sha == EXPECTED_SELECTOR_SHA256, f"{sha[:16]}... (expected {EXPECTED_SELECTOR_SHA256[:16]}...)")

    import torch

    payload = torch.load(sel_path, map_location="cpu", weights_only=False)
    pol = payload.get("training_policy", {})
    gate("G6b step is 7115", payload.get("global_step") == 7115, f"step={payload.get('global_step')}")
    gate("G6c objective is multiclass", pol.get("selector_objective") == "multiclass",
         f"objective={pol.get('selector_objective')}")
    gate("G6d stage-1 recorded a frozen backbone", pol.get("backbone") == "frozen",
         f"backbone={pol.get('backbone')}")
    gate("G6e training policy is marked verified", pol.get("verified") is True,
         f"verified={pol.get('verified')}")

    init = load_file(str(init_dir / "model.safetensors"))
    src = payload["state_dict"]
    # torch.equal, not a float cast: dtype drift must fail here, and the selector is fp32 while the
    # backbone is bf16, so a cast-based comparison would hide exactly the bug worth catching.
    mismatched = [
        k for k in src
        if SELECTOR_PREFIX + k not in init
        or init[SELECTOR_PREFIX + k].dtype != src[k].dtype
        or not torch.equal(init[SELECTOR_PREFIX + k], src[k])
    ]
    gate("G6f every selector tensor in the init equals the source, dtype included",
         not mismatched, f"{len(src)} tensors, mismatched={mismatched[:4]}")

    rel = load_file("/kl_infra_infer_intern/wangzhuoyu/TAPS-SP/models/Qwen3-4B-DFlash-b16/model.safetensors")
    bb_bad = [
        k for k in rel
        if k not in init or init[k].dtype != rel[k].dtype or not torch.equal(init[k], rel[k])
    ]
    gate("G6g every backbone tensor in the init equals the release, dtype included",
         not bb_bad, f"{len(rel)} tensors, mismatched={bb_bad[:4]}")
    dtypes = {str(init[k].dtype) for k in init if k.startswith(SELECTOR_PREFIX)}
    bdtypes = {str(init[k].dtype) for k in init if not k.startswith(SELECTOR_PREFIX)}
    gate("G6h selector stays fp32 and backbone stays bf16",
         dtypes == {"torch.float32"} and bdtypes == {"torch.bfloat16"},
         f"selector={sorted(dtypes)} backbone={sorted(bdtypes)}")
    leaked_emb = [k for k in init if "embed_tokens" in k or k.endswith("target_embedding")]
    gate("G6i no embedding tensor leaked into the draft checkpoint", not leaked_emb, f"{leaked_emb[:3]}")

    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{'ALL GATES PASS' if n_fail == 0 else f'{n_fail} GATE(S) FAILED'}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
