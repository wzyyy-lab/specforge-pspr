# coding=utf-8
"""Gate for ``OracleFutureProbeSelector``: the probe is only interpretable if it leaks nothing.

Three properties, each of which invalidates the experiment on its own if it fails:

G1  ``oracle_gate == 0`` reproduces ``LatticePathSelector`` elementwise, so warm-starting from the
    5.7217 checkpoint really does start at 5.7217 and any measured gain is attributable to the
    future channel alone.

G2  No leakage of ``y_t`` into slot ``t``. Perturbing the oracle token at slot ``t`` must leave
    slot ``t``'s scores bit-identical. If this fails the probe is silently measuring oracle@16
    ("was the target in top-K") rather than the value of future context, which would look like a
    huge win and mean nothing.

G3  The channel is alive: perturbing the oracle token at a *future* slot must change the current
    slot's scores. Without this, a null result would be indistinguishable from a dead wire, and a
    dead wire is exactly the failure mode a zero-init gate produces silently.

Plus a NaN check on the final slot, whose future set is empty.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from specforge.modeling.draft.pspr import LatticePathSelector  # noqa: E402
from specforge.modeling.draft.pspr_oracle_probe import (  # noqa: E402
    OracleFutureProbeSelector,
)

HIDDEN = 64
VOCAB = 512
TOP_K = 8
HORIZON = 6
BATCH = 4
CONFIG = dict(
    hidden_size=HIDDEN,
    vocab_size=VOCAB,
    top_k=TOP_K,
    d=32,
    n_layers=2,
    n_heads=4,
    state_dim=32,
    delta_hidden=64,
    max_slots=16,
    dropout=0.0,
)


def build(seed: int):
    torch.manual_seed(seed)
    reference = LatticePathSelector(**CONFIG)
    torch.manual_seed(seed)
    probe = OracleFutureProbeSelector(**CONFIG)
    embedding = torch.nn.Embedding(VOCAB, HIDDEN)
    torch.nn.init.normal_(embedding.weight, std=0.02)
    reference.bind_target_embedding(embedding)
    probe.bind_target_embedding(embedding)
    # `gamma` ships zero-initialised, which makes `scores = log_probs + 0 * correction` and routes
    # the entire context pathway around the decision. Under that setting G1 and G2 pass vacuously
    # (nothing can change any score) and G3 is indistinguishable from a real dead wire. Open the
    # same gamma on both models so G1 stays a fair comparison.
    with torch.no_grad():
        reference.gamma.fill_(1.0)
        probe.gamma.fill_(1.0)
    reference.eval()
    probe.eval()
    return reference, probe


def batch(seed: int):
    generator = torch.Generator().manual_seed(seed)
    candidate_ids = torch.randint(0, VOCAB, (BATCH, HORIZON, TOP_K), generator=generator)
    unary = torch.randn(BATCH, HORIZON, TOP_K, generator=generator).log_softmax(dim=-1)
    hidden = torch.randn(BATCH, HORIZON, HIDDEN, generator=generator)
    predecessor = torch.randint(0, VOCAB, (BATCH, HORIZON), generator=generator)
    oracle = torch.randint(0, VOCAB, (BATCH, HORIZON), generator=generator)
    oracle[0, -1] = -1
    return dict(
        candidate_ids=candidate_ids,
        unary_logits=unary,
        hidden_states=hidden,
        predecessor_ids=predecessor,
    ), oracle


def main() -> int:
    failures = []
    reference, probe = build(0)
    inputs, oracle = batch(1)

    probe.set_oracle(oracle)
    with torch.no_grad():
        reference_scores = reference.score_candidates(**inputs)
        probe_scores = probe.score_candidates(**inputs)
    if torch.equal(reference_scores, probe_scores):
        print("G1 PASS  oracle_gate=0 is elementwise identical to LatticePathSelector")
    else:
        gap = (reference_scores - probe_scores).abs().max().item()
        failures.append(f"G1 FAIL  max|delta| = {gap:.3e}")

    with torch.no_grad():
        probe.oracle_gate.fill_(1.0)
        base = probe.score_candidates(**inputs)
    if torch.isnan(base).any():
        failures.append("NaN FAIL  scores contain NaN with the gate open")
    else:
        print("NaN PASS  final slot (empty future set) produces no NaN")

    leak = []
    alive = []
    for slot in range(HORIZON):
        perturbed = oracle.clone()
        perturbed[:, slot] = (perturbed[:, slot] + 137) % VOCAB
        probe.set_oracle(perturbed)
        with torch.no_grad():
            moved = probe.score_candidates(**inputs)
        delta = (moved - base).abs()
        own = delta[:, slot].max().item()
        if own > 0.0:
            leak.append((slot, own))
        if slot > 0:
            before = delta[:, :slot].max().item()
            alive.append((slot, before))

    if leak:
        failures.append(
            "G2 FAIL  perturbing y_t changed slot t's own scores: "
            + ", ".join(f"slot {s} by {v:.3e}" for s, v in leak)
        )
    else:
        print(f"G2 PASS  y_t unreachable from slot t for all {HORIZON} slots")

    dead = [s for s, v in alive if v == 0.0]
    if dead:
        failures.append(f"G3 FAIL  future channel dead: perturbing y_s moved nothing for s={dead}")
    else:
        smallest = min(v for _, v in alive)
        print(f"G3 PASS  future channel live (smallest response {smallest:.3e})")

    probe.set_oracle(None)
    with torch.no_grad():
        detached = probe.score_candidates(**inputs)
    if torch.equal(detached, reference_scores):
        print("G4 PASS  clear_oracle() restores the unmodified selector even with the gate open")
    else:
        failures.append("G4 FAIL  score_candidates without a staged oracle is not the baseline")

    print()
    if failures:
        for line in failures:
            print(line)
        print(f"RESULT: FAIL ({len(failures)} checks)")
        return 1
    print("RESULT: PASS (all checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
