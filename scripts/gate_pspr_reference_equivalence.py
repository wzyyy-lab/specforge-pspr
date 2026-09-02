#!/usr/bin/env python3
"""Numerical equivalence gate: SpecForge's PSPR selector vs the reference LatticeSelector.

The port claim is that ``specforge.modeling.draft.pspr.LatticePathSelector`` computes exactly the
same function as the frozen-DFlash module that measured macro 5.705, so that any later change in
acceptance is attributable to training rather than to a re-implementation drift. Parameter counts
matching is necessary but nowhere near sufficient, so this gate loads the *actual* reference
checkpoint into the port and compares every intermediate tensor on identical inputs.

Every reported max-abs difference must be exactly 0. The two modules run the same ops in the same
order on the same dtype, so anything non-zero is a real behavioural difference, not float noise.

Usage:
    PYTHONPATH=. python scripts/gate_pspr_reference_equivalence.py \
        --reference-ckpt /kl_infra_infer_intern/wangzhuoyu/TAPS-SP/outputs/dh2048/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REFERENCE_REPO = "/kl_infra_infer_intern/wangzhuoyu/TAPS-SP"

# ``dh_mlp`` was renamed to ``delta_mlp`` (it is the delta term, and the port has no other MLP that
# would make the abbreviation unambiguous). ``gamma`` became 1-D because FSDP rejects zero-dim
# parameters. Nothing else may differ, and the gate asserts that.
RENAMES = {"dh_mlp.": "delta_mlp."}
RESHAPED = {"gamma"}


def build_reference(config: dict, state_dict: dict, device: str):
    sys.path.insert(0, REFERENCE_REPO)
    from joint.lattice_selector import LatticeSelector

    module = LatticeSelector(**config).to(device=device, dtype=torch.float32)
    missing, unexpected = module.load_state_dict(state_dict, strict=True)
    assert not missing and not unexpected
    return module.eval()


def build_port(config: dict, state_dict: dict, device: str):
    from specforge.modeling.draft.pspr import LatticePathSelector

    module = LatticePathSelector(
        hidden_size=config["hidden_dim"],
        vocab_size=config["vocab_size"],
        top_k=config["K"],
        d=config["d"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        state_dim=config["ds"],
        delta_hidden=config["delta_h"],
        max_slots=config["max_slots"],
        dropout=config["dropout"],
        n_scalars=config["n_scal"],
    ).to(device=device, dtype=torch.float32)

    mapped = {}
    for key, value in state_dict.items():
        new_key = key
        for old, new in RENAMES.items():
            if key.startswith(old):
                new_key = new + key[len(old) :]
        mapped[new_key] = value.reshape(1) if key in RESHAPED else value

    missing, unexpected = module.load_state_dict(mapped, strict=True)
    assert not missing and not unexpected
    return module.eval()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference-ckpt",
        default=f"{REFERENCE_REPO}/outputs/dh2048/best.pt",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--vocab-probe", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    payload = torch.load(args.reference_ckpt, map_location="cpu", weights_only=False)
    assert payload["model_type"] == "LatticeSelector", payload["model_type"]
    config, state_dict = payload["config"], payload["state_dict"]
    print(f"reference: {Path(args.reference_ckpt)}")
    print(f"  config  = {config}")
    print(f"  gamma   = {state_dict['gamma'].item():.4f}  (trained strength of the correction)")

    reference = build_reference(config, state_dict, args.device)
    port = build_port(config, state_dict, args.device)

    n_reference = sum(p.numel() for p in reference.parameters())
    n_port = sum(p.numel() for p in port.parameters())
    print(f"\nparameters: reference {n_reference:,} | port {n_port:,}")

    torch.manual_seed(args.seed)
    b, h, k = args.batch, args.horizon, config["K"]
    hidden_dim, embed_dim = config["hidden_dim"], config["vocab_embed_dim"]
    device = args.device

    # A shared embedding table lets the port look candidate ids up while the reference is handed the
    # resulting vectors directly, so both see bit-identical candidate embeddings.
    table = torch.randn(args.vocab_probe, embed_dim, device=device, dtype=torch.float32)
    candidate_ids = torch.randint(0, args.vocab_probe, (b, h, k), device=device)
    predecessor_ids = torch.randint(0, args.vocab_probe, (b, h), device=device)
    candidate_embeddings = F.embedding(candidate_ids, table)
    prefix_embeddings = F.embedding(predecessor_ids, table)
    hidden = torch.randn(b, h, hidden_dim, device=device, dtype=torch.float32)
    log_probs = torch.log_softmax(
        torch.randn(b, h, k, device=device, dtype=torch.float32) * 3.0, dim=-1
    )
    scalars = torch.randn(b, h, config["n_scal"], device=device, dtype=torch.float32)
    port.bind_target_embedding(torch.nn.Embedding.from_pretrained(table).to(device))

    checks: list[tuple[str, float]] = []
    with torch.no_grad():
        r_reference = reference.encode(hidden, candidate_embeddings, log_probs, scalars)
        r_port = port.encode(hidden, candidate_embeddings, log_probs, scalars)
        checks.append(("encode r (bidirectional lattice)", (r_reference - r_port).abs().max().item()))

        s_reference = reference.causal_states(prefix_embeddings)
        s_port = port.causal_states(prefix_embeddings)
        checks.append(("causal S (committed-prefix GRU)", (s_reference - s_port).abs().max().item()))

        d_reference = reference.delta_term(r_reference, s_reference, candidate_embeddings)
        d_port = port.delta_term(r_port, s_port, candidate_embeddings)
        checks.append(("delta_term (target-geometry correction)", (d_reference - d_port).abs().max().item()))

        e_reference = reference.err_logits(r_reference, s_reference, log_probs, scalars)
        e_port = port.err_logits(r_port, s_port, log_probs, scalars)
        checks.append(("err_logits (frontier detector)", (e_reference - e_port).abs().max().item()))

        sc_reference = reference.scores(r_reference, s_reference, candidate_embeddings, log_probs)
        sc_port = port.score(r_port, s_port, candidate_embeddings, log_probs)
        checks.append(("scores (per-candidate)", (sc_reference - sc_port).abs().max().item()))

        # End to end, through each side's own public entry point, including lattice lookup.
        end_reference, end_err_reference = reference(
            hidden, candidate_embeddings, log_probs, scalars, prefix_embeddings, with_err=True
        )
        end_port, end_err_port = port.score_candidates(
            candidate_ids=candidate_ids,
            unary_logits=log_probs,
            hidden_states=hidden,
            predecessor_ids=predecessor_ids,
            lattice_scalars=scalars,
            return_err=True,
        )
        checks.append(("end-to-end scores", (end_reference - end_port).abs().max().item()))
        checks.append(("end-to-end err_logits", (end_err_reference - end_err_port).abs().max().item()))

        argmax_agreement = (
            (end_reference.argmax(-1) == end_port.argmax(-1)).sum().item(),
            end_reference.shape[0] * end_reference.shape[1],
        )

    print("\ncomponent-wise max |reference - port| (must be exactly 0):")
    ok = n_reference == n_port
    if not ok:
        print(f"  [FAIL] parameter counts differ")
    for name, delta in checks:
        flag = "ok" if delta == 0.0 else "FAIL"
        if delta != 0.0:
            ok = False
        print(f"  [{flag}] {name:42s} {delta:.3e}")
    print(f"  [{'ok' if argmax_agreement[0] == argmax_agreement[1] else 'FAIL'}] "
          f"argmax agreement {argmax_agreement[0]}/{argmax_agreement[1]}")
    ok = ok and argmax_agreement[0] == argmax_agreement[1]

    print(f"\nGATE {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
