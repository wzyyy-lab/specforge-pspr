#!/usr/bin/env python3
"""Correctness gates for ``SlotDeepCorrector`` before any GPU time is spent on it.

Every claim the module's docstring makes is checked here against the built module, not against the
intention.  The gates are deliberately adversarial about the two things that would make a training
run uninterpretable: a head that is not a no-op at step 0 (so a gain cannot be attributed) and a
head that leaks the token it is asked to predict (so a gain is not real).

  G1  the dense encoder is gone and nothing references it
  G2  parameter count of the z pathway matches the 6-layer encoder it replaces (same capacity)
  G3  delta == 0 at initialisation  =>  scores are the bare lattice log-probs, bit for bit
  G4  z_i does not depend on its own seed token  (structural no-leak)
  G5  z_i does not depend on any other slot's hidden state, seed or confidence  (no cross-slot)
  G6  z_i DOES depend on the anchor token, on its own hidden state, and on its own confidence
  G7  z_i depends on slot position
  G8  attention_pattern ablations are rejected instead of silently no-oping
  G9  the parent cloze head still passes G3-G7's applicable parts under the same harness, so a
      failure here is attributable to this module rather than to the harness
  G10 forward/backward runs and every selector parameter receives a gradient
  G11 wall-clock of cloze_states, slotdeep vs cloze, on the deployed block geometry
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from specforge.modeling.draft.pspr_cloze import ClozeCorrector  # noqa: E402
from specforge.modeling.draft.pspr_slotdeep import SlotDeepCorrector  # noqa: E402

HID = 2560
VOCAB = 151936
D = 512
K = 16
H = 15

RESULTS: list[tuple[str, bool, str]] = []


def gate(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""), flush=True)


def build(kind: str, slot_layers: int = 9, seed: int = 0):
    torch.manual_seed(seed)
    common = dict(
        hidden_size=HID, vocab_size=VOCAB, top_k=K, d=D, n_layers=6, n_heads=8,
        state_dim=512, delta_hidden=2048, max_slots=32, dropout=0.0,
        direct_hidden=True, use_state=True, bidirectional=True, err_use_state=False,
    )
    module = (SlotDeepCorrector(slot_layers=slot_layers, **common) if kind == "slotdeep"
              else ClozeCorrector(**common))
    module = module.double().eval()
    torch.manual_seed(1234)
    module.bind_target_embedding(torch.nn.Embedding(VOCAB, HID).double())
    return module


def inputs(batch: int = 2, blocks: int = 3, seed: int = 7, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    lead = (batch, blocks)
    hidden = torch.randn(*lead, H, HID, generator=g, dtype=dtype)
    cand = torch.randint(0, VOCAB, (*lead, H, K), generator=g)
    anchor = torch.randint(0, VOCAB, lead, generator=g)
    lp = torch.log_softmax(torch.randn(*lead, H, K, generator=g, dtype=dtype), dim=-1)
    lp, _ = lp.sort(dim=-1, descending=True)
    scalars = torch.rand(*lead, H, 3, generator=g, dtype=dtype)
    prefix = torch.randn(*lead, H, HID, generator=g, dtype=dtype)
    return dict(hidden_states=hidden, candidate_ids=cand, anchor_ids=anchor,
                log_probs=lp, scalars=scalars, prefix=prefix)


def z_of(module, x, **override):
    kw = dict(hidden_states=x["hidden_states"], candidate_ids=x["candidate_ids"],
              anchor_ids=x["anchor_ids"], log_probs=x["log_probs"], scalars=x["scalars"])
    kw.update(override)
    return module.cloze_states(**kw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot-layers", type=int, default=9)
    ap.add_argument("--json-output", default=None)
    ap.add_argument("--bench-device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print("=== building ===", flush=True)
    sd = build("slotdeep", args.slot_layers)
    cz = build("cloze")
    x = inputs()

    print("\n=== G1/G2 structure ===")
    gate("G1 dense encoder removed", not hasattr(sd, "encoder") and "encoder" not in dict(sd.named_modules()))
    enc = sum(p.numel() for n, p in cz.named_parameters() if n.startswith("encoder."))
    blk = sum(p.numel() for n, p in sd.named_parameters()
              if n.startswith("slot_blocks.") or n.startswith("slot_out_ln."))
    ratio = blk / max(enc, 1)
    gate("G2 z-pathway parameters matched within 5%", abs(ratio - 1.0) < 0.05,
         f"cloze encoder {enc/1e6:.2f}M vs slotdeep blocks {blk/1e6:.2f}M (ratio {ratio:.3f})")
    tot_sd = sum(p.numel() for p in sd.parameters())
    tot_cz = sum(p.numel() for p in cz.parameters())
    print(f"      total selector params: cloze {tot_cz/1e6:.2f}M  slotdeep {tot_sd/1e6:.2f}M")

    print("\n=== G3 step-0 no-op ===")
    for name, m in (("slotdeep", sd), ("cloze", cz)):
        z = z_of(m, x)
        state = m.causal_states(x["prefix"])
        delta = m.hidden_delta(z, hidden_states=x["hidden_states"], state=state)
        cand_emb = torch.nn.functional.embedding(x["candidate_ids"], m._embedding())
        scores = m.score(z, x["hidden_states"], cand_emb, x["log_probs"], state=state)
        gate(f"G3 delta == 0 at init ({name})", bool(delta.abs().max().item() == 0.0),
             f"max|delta| = {delta.abs().max().item():.3e}")
        if True:
            worst = (scores - x["log_probs"].float()).abs().max().item()
            gate(f"G3b scores == lattice log-probs ({name})", worst < 1e-12,
                 f"max|score - lp| = {worst:.3e}")

    print("\n=== G4-G7 information flow (float64, exact-equality where the claim is structural) ===")
    z0 = z_of(sd, x)
    zc0 = z_of(cz, x)

    seed_perturbed = x["candidate_ids"].clone()
    seed_perturbed[..., 0] = (seed_perturbed[..., 0] + 7919) % VOCAB
    d_seed = (z_of(sd, x, candidate_ids=seed_perturbed) - z0).abs().max().item()
    gate("G4 z is exactly invariant to its own seed token", d_seed == 0.0, f"max|dz| = {d_seed:.3e}")
    d_seed_cz = (z_of(cz, x, candidate_ids=seed_perturbed) - zc0).abs().max().item()
    print(f"      (parent cloze under the same perturbation: max|dz| = {d_seed_cz:.3e} -- nonzero is"
          f" expected, other rows read the perturbed seeds as context)")

    hid_p = x["hidden_states"].clone()
    g = torch.Generator().manual_seed(4242)
    hid_p[..., 5, :] += torch.randn(HID, generator=g, dtype=hid_p.dtype)
    dz = (z_of(sd, x, hidden_states=hid_p) - z0).abs()
    other = torch.cat([dz[..., :5, :], dz[..., 6:, :]], dim=-2).max().item()
    own = dz[..., 5, :].max().item()
    gate("G5 z_i is exactly invariant to slot j!=i's hidden state", other == 0.0,
         f"max|dz| off-slot {other:.3e}, on-slot {own:.3e}")
    dzc = (z_of(cz, x, hidden_states=hid_p) - zc0).abs()
    other_cz = torch.cat([dzc[..., :5, :], dzc[..., 6:, :]], dim=-2).max().item()
    gate("G9 the same probe shows the parent DOES mix across slots", other_cz > 1e-6,
         f"parent off-slot max|dz| = {other_cz:.3e} (harness can detect cross-slot flow)")

    anchor_p = (x["anchor_ids"] + 104729) % VOCAB
    d_anchor = (z_of(sd, x, anchor_ids=anchor_p) - z0).abs().max().item()
    gate("G6a z depends on the anchor token", d_anchor > 1e-6, f"max|dz| = {d_anchor:.3e}")
    gate("G6b z depends on its own hidden state", own > 1e-6, f"max|dz| = {own:.3e}")
    lp_p = x["log_probs"].clone()
    lp_p[..., 3, :] = torch.log_softmax(lp_p[..., 3, :] * 0.3, dim=-1)
    dlp = (z_of(sd, x, log_probs=lp_p) - z0).abs()
    gate("G6c z depends on its own confidence", dlp[..., 3, :].max().item() > 1e-6,
         f"on-slot max|dz| = {dlp[..., 3, :].max().item():.3e}")
    gate("G6d confidence does not leak across slots",
         torch.cat([dlp[..., :3, :], dlp[..., 4:, :]], dim=-2).max().item() == 0.0)

    # Position: feed the SAME per-slot content to every slot and check the outputs still differ.
    flat = dict(x)
    flat["hidden_states"] = x["hidden_states"][..., :1, :].expand_as(x["hidden_states"]).contiguous()
    flat["log_probs"] = x["log_probs"][..., :1, :].expand_as(x["log_probs"]).contiguous()
    flat["scalars"] = x["scalars"][..., :1, :].expand_as(x["scalars"]).contiguous()
    zf = z_of(sd, flat)
    spread = (zf - zf[..., :1, :]).abs().max().item()
    gate("G7 z depends on slot position", spread > 1e-6, f"spread across slots = {spread:.3e}")

    print("\n=== G8 ablation switches must fail loudly ===")
    for pattern in ("causal", "anchor_self"):
        sd.attention_pattern = pattern
        try:
            z_of(sd, x)
            gate(f"G8 attention_pattern={pattern} rejected", False, "no error raised")
        except ValueError as exc:
            gate(f"G8 attention_pattern={pattern} rejected", True, str(exc)[:60] + "...")
    del sd.attention_pattern

    print("\n=== G10 gradients: dead-at-step-0 set, and live after one update ===")
    x32 = inputs(dtype=torch.float32)

    def dead_set(kind, steps):
        m = build(kind, args.slot_layers).float()
        opt = torch.optim.SGD(m.parameters(), lr=1e-2)
        for _ in range(steps + 1):
            opt.zero_grad(set_to_none=True)
            z = z_of(m, x32)
            state = m.causal_states(x32["prefix"])
            emb = m._embedding()
            ce = torch.nn.functional.embedding(x32["candidate_ids"], emb)
            se = torch.nn.functional.embedding(x32["candidate_ids"][..., 0], emb)
            sc = m.score(z, x32["hidden_states"], ce, x32["log_probs"], state=state)
            er = m.err_logits(z, se, x32["log_probs"], x32["scalars"],
                              hidden_states=x32["hidden_states"])
            (sc.square().mean() + er.square().mean()).backward()
            dead = {n for n, p in m.named_parameters()
                    if p.requires_grad and (p.grad is None or p.grad.abs().max().item() == 0.0)}
            if steps:
                opt.step()
        return dead

    dead0_sd, dead0_cz = dead_set("slotdeep", 0), dead_set("cloze", 0)
    only_new = {n for n in dead0_sd if not n.startswith(("slot_blocks.", "slot_out_ln."))} - dead0_cz
    gate("G10a step-0 dead set is inherited, not introduced", not only_new,
         f"{len(dead0_sd)} dead at step 0 (parent {len(dead0_cz)}); new: "
         + (", ".join(sorted(only_new)[:5]) if only_new else "none"))
    gate("G10b the new blocks are live at step 0",
         not any(n.startswith(("slot_blocks.", "slot_out_ln.")) for n in dead0_sd),
         "dead blocks: " + ", ".join(n for n in sorted(dead0_sd)
                                     if n.startswith(("slot_blocks.", "slot_out_ln."))))
    dead1 = dead_set("slotdeep", 1)
    gate("G10c every selector parameter is live after one update", not dead1,
         "still dead: " + ", ".join(sorted(dead1)[:6]) if dead1 else "")

    print(f"\n=== G11 head wall-clock breakdown on {args.bench_device}, deployment geometry ===")
    # The 8.24 ms figure that makes PSPR-cloze lose end-to-end (144.6 tok/s against Domino's 160.4)
    # is the WHOLE head in the decode path, not the encoder.  Replacing the encoder only pays if the
    # encoder is where the time goes, so measure the components separately instead of assuming it.
    # fp32 because `selector_compute_dtype: float32`; B=1, one block, H=15 because that is what the
    # serving walk runs.
    bench = {}
    if args.bench_device != "cpu":
        for kind in ("cloze", "slotdeep"):
            m = build(kind, args.slot_layers).float().to(args.bench_device)
            xb = inputs(batch=1, blocks=1, dtype=torch.float32)
            xb = {k: (v.to(args.bench_device) if torch.is_tensor(v) else v) for k, v in xb.items()}
            emb = m._embedding()
            ce = torch.nn.functional.embedding(xb["candidate_ids"], emb)

            def timed(fn, n=100):
                with torch.no_grad():
                    for _ in range(10):
                        fn()
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(n):
                        fn()
                    torch.cuda.synchronize()
                return (time.perf_counter() - t0) / n * 1e3

            with torch.no_grad():
                zc = z_of(m, xb)
                st = m.causal_states(xb["prefix"])
            parts = {
                "z pathway": timed(lambda: z_of(m, xb)),
                "GRU (15 serial steps)": timed(lambda: m.causal_states(xb["prefix"])),
                "delta + score": timed(
                    lambda: m.score(zc, xb["hidden_states"], ce, xb["log_probs"], state=st)),
            }
            parts["total"] = sum(parts.values())
            bench[kind] = parts
            print(f"      {kind}")
            for k, v in parts.items():
                print(f"        {k:<24} {v:.3f} ms")
            del m
            torch.cuda.empty_cache()
        z_speedup = bench["cloze"]["z pathway"] / max(bench["slotdeep"]["z pathway"], 1e-9)
        saved = bench["cloze"]["total"] - bench["slotdeep"]["total"]
        gate("G11a z pathway is cheaper", z_speedup > 1.0, f"{z_speedup:.1f}x")
        gate("G11b whole head is cheaper", saved > 0.0,
             f"{bench['cloze']['total']:.3f} -> {bench['slotdeep']['total']:.3f} ms "
             f"({saved:.3f} ms saved, {100 * saved / bench['cloze']['total']:.0f}%)")
        print("      NOTE these are unfused eager-mode microbenchmarks at B=1: at this size every")
        print("      component is kernel-launch bound, so the ratio understates the 16x FLOP cut and")
        print("      the absolute numbers are NOT the deployed 8.24 ms.  The shipping claim needs a")
        print("      real decode measurement, which this gate deliberately does not attempt.")
    else:
        print("      skipped (no cuda)")

    print("\n=== G12 the two arms' weight-decay policy is identical where they share tensors ===")
    # Matching is `name.startswith(prefix) or f".{prefix}" in name` (specforge/optimizer.py:94).
    # The first version of this gate asked "does every dim>=2 tensor match a rule" and failed on
    # `pos_emb` / `role_emb.weight` -- which the cloze recipe also leaves undecayed on purpose
    # ("every LayerNorm / bias / embedding-like tensor matches none").  The claim that actually
    # matters for a one-variable experiment is not an absolute policy but an IDENTICAL one, so both
    # yamls are parsed and the two arms are compared tensor by tensor.
    import re as _re

    def yaml_rules(name):
        body = (Path(__file__).resolve().parents[1]
                / f"examples/configs/online/disaggregated/managed-local/{name}").read_text()
        block = body[body.index("weight_decay_rules:"):body.index("attention_backend:")]
        pairs = _re.findall(r'^\s+"([^"]+)":\s*([0-9.e+-]+)', block, _re.MULTILINE)
        return tuple(sorted(((k, float(v)) for k, v in pairs), key=lambda kv: -len(kv[0])))

    rules_sd = yaml_rules("qwen3-4b-pspr-slotdeep.yaml")
    rules_cz = yaml_rules("qwen3-4b-pspr-cloze.yaml")
    gate("G12a both rule lists parsed", len(rules_sd) >= 14 and len(rules_cz) >= 12,
         f"slotdeep {len(rules_sd)} rules, cloze {len(rules_cz)} rules")

    def decay_of(rules, name, default=0.0):
        hits = [v for k, v in rules if name.startswith(k) or f".{k}" in name]
        return (hits[0] if hits else default), len(hits)

    def policy(module, rules, replaced_prefixes):
        shared, own, alias = {}, {}, []
        for n, prm in module.named_parameters():
            full = f"candidate_selector.{n}"
            wd, hits = decay_of(rules, full)
            if hits > 1:
                alias.append(full)
            (own if n.startswith(replaced_prefixes) else shared)[n] = (wd, prm.dim())
        return shared, own, alias

    shared_sd, own_sd, alias_sd = policy(sd, rules_sd, ("slot_blocks.", "slot_out_ln."))
    shared_cz, own_cz, alias_cz = policy(cz, rules_cz, ("encoder.",))
    gate("G12b no tensor matches two rules in either arm", not alias_sd and not alias_cz,
         f"slotdeep {alias_sd[:3]} cloze {alias_cz[:3]}")
    gate("G12c shared tensors have identical names in both arms",
         set(shared_sd) == set(shared_cz),
         f"only-slotdeep {sorted(set(shared_sd) - set(shared_cz))[:4]} "
         f"only-cloze {sorted(set(shared_cz) - set(shared_sd))[:4]}")
    mismatch = {n: (shared_sd[n][0], shared_cz[n][0]) for n in set(shared_sd) & set(shared_cz)
                if shared_sd[n][0] != shared_cz[n][0]}
    gate("G12d shared tensors get identical weight decay", not mismatch,
         f"differ: {list(mismatch.items())[:4]}")
    # And the replaced pathway itself must be decayed the same way in both arms: every dim>=2 matrix
    # at 2e-3, every norm and bias at 0.
    def pathway_ok(own):
        return (all(wd == 2.0e-3 for wd, dim in own.values() if dim >= 2)
                and all(wd == 0.0 for wd, dim in own.values() if dim < 2)
                and any(dim >= 2 for _, dim in own.values()))
    gate("G12e replaced pathway decayed identically (2e-3 on matrices, 0 on norms/biases)",
         pathway_ok(own_sd) and pathway_ok(own_cz),
         f"slotdeep {len(own_sd)} tensors, cloze {len(own_cz)} tensors")
    print("      undecayed dim>=2 tensors, both arms (embedding-like, by design): "
          + ", ".join(sorted(n for n, (wd, dim) in shared_sd.items() if dim >= 2 and wd == 0.0)))

    print("\n=== G13 the arm is constructible from its config json through the registry ===")
    try:
        from specforge.modeling.draft.registry import resolve_draft
        cfg = json.loads((Path(__file__).resolve().parents[1]
                          / "configs/qwen3-4b-pspr-slotdeep.json").read_text())
        cls = resolve_draft(cfg["architectures"][0])
        gate("G13 architectures[0] resolves to PSPRSlotDeepDraftModel",
             cls.__name__ == "PSPRSlotDeepDraftModel", cls.__name__)
        gate("G13b config carries the new knobs",
             cfg["dflash_config"].get("selector_slot_layers") == args.slot_layers,
             f"selector_slot_layers={cfg['dflash_config'].get('selector_slot_layers')}")
    except Exception as exc:  # noqa: BLE001
        gate("G13 architectures[0] resolves to PSPRSlotDeepDraftModel", False, repr(exc)[:80])

    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{'ALL GATES PASS' if n_fail == 0 else f'{n_fail} GATE(S) FAILED'}")
    if args.json_output:
        Path(args.json_output).write_text(json.dumps(
            {"gates": [{"name": n, "pass": ok, "detail": d} for n, ok, d in RESULTS],
             "params": {"cloze_encoder": enc, "slotdeep_blocks": blk,
                        "total_cloze": tot_cz, "total_slotdeep": tot_sd},
             "bench_ms": bench}, indent=2))
        print(f"wrote {args.json_output}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
