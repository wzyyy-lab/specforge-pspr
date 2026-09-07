# coding=utf-8
"""Run the stage-1 recipe with ``OracleFutureProbeSelector`` substituted in, changing nothing else.

This is a wrapper, not a fork: ``train_pspr_accept_selector.py``, ``accept_selector.py``,
``pspr.py`` and ``decode_lattice.py`` are all left untouched, so the number this produces is
comparable to the 5.7217 run by construction. Two module attributes are rebound before the stage-1
module executes:

``LatticePathSelector`` -> ``OracleFutureProbeSelector``
``slot_probs``          -> a version that stages the batch's ground-truth ids on the selector

``eval_gate`` resolves ``slot_probs`` through its own module globals at call time, so patching the
attribute reaches the calibration path as well as the training loop. Every argument of the original
script (``--init-from``, ``--data-frac``, ``--eval-every``, ...) is forwarded untouched.

Which future tokens are handed over
-----------------------------------
Controlled by ``PSPR_PROBE_MASK``, and the choice turned out to decide whether the experiment is
valid at all. See the comment on ``ORACLE_MASK``: restricting the oracle to ``frontier_mask`` slots
looks like the careful option but opens a visibility-pattern side channel that hands ``err_head``
the answer it is supposed to predict, inflating the result. Default is ``valid``, which exposes
every labelled slot and is biased downward instead.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import specforge.algorithms.pspr.accept_selector as accept_selector  # noqa: E402
from specforge.modeling.draft.pspr_oracle_probe import (  # noqa: E402
    OracleFutureProbeSelector,
)

# The control arm. Same class, same parameter count, same data order, same seed -- the oracle is
# simply never staged, so `_inject_future` returns the context untouched and the run measures what a
# few hundred extra steps on an already-converged checkpoint do on their own. Without this arm a
# positive probe result is uninterpretable.
ORACLE_DISABLED = os.environ.get("PSPR_PROBE_DISABLE", "") == "1"

# Which slots may expose their ground-truth token.
#
# "valid" (default) exposes every slot the trace labels at all. "frontier" additionally hides slots
# past the draft's first top-1 error, whose labels were produced under a wrong target context.
#
# "frontier" looks like the more careful choice and is in fact CONTAMINATED. Frontier ends at the
# first top-1 error, so "how many non-hidden oracle tokens follow slot t" almost perfectly encodes
# "is slot t itself the first error" -- exactly the quantity `err_head` exists to predict, and the
# quantity the decode gate thresholds. The leak is in the visibility pattern, not in any token
# value, so the G2 gate (which perturbs token values) cannot see it. Its signature is the optimal
# gate moving off theta=0.0.
#
# "valid" removes that side channel at the cost of feeding noisier labels as future context, which
# biases the probe DOWNWARD. A conservative upper bound is the useful kind.
ORACLE_MASK = os.environ.get("PSPR_PROBE_MASK", "valid")
if ORACLE_MASK not in ("valid", "frontier"):
    raise SystemExit(f"PSPR_PROBE_MASK must be 'valid' or 'frontier', got {ORACLE_MASK!r}")

_original_slot_probs = accept_selector.slot_probs


def slot_probs_with_oracle(model, b, with_err=False):
    core = model.module if hasattr(model, "module") else model
    setter = getattr(core, "set_oracle", None)
    if setter is None or ORACLE_DISABLED:
        return _original_slot_probs(model, b, with_err=with_err)
    ids, index = b["g"], b["ti"]
    visible = ids >= 0
    if ORACLE_MASK == "frontier":
        visible = accept_selector.frontier_mask(visible, (index == 0) & visible)
    setter(torch.where(visible, ids, torch.full_like(ids, -1)))
    try:
        return _original_slot_probs(model, b, with_err=with_err)
    finally:
        core.clear_oracle()


accept_selector.slot_probs = slot_probs_with_oracle

_spec = importlib.util.spec_from_file_location(
    "_pspr_stage1", ROOT / "scripts/train_pspr_accept_selector.py"
)
_stage1 = importlib.util.module_from_spec(_spec)
sys.modules["_pspr_stage1"] = _stage1
_spec.loader.exec_module(_stage1)

_stage1.LatticePathSelector = OracleFutureProbeSelector
_stage1.slot_probs = slot_probs_with_oracle

_original_load_reference_state = _stage1.load_reference_state


def load_reference_state_allowing_probe(core, state_dict, device):
    """Let the 5.7217 checkpoint warm-start a strictly larger module.

    The stage-1 script treats ``init-from`` as all-or-nothing on purpose, so a renamed or dropped
    tensor cannot silently degrade a warm start. Only the probe's own additions are excused; every
    other missing or unexpected key still aborts the run, and the probe's additions are zero-gated
    so excusing them cannot change the starting point.
    """
    missing, unexpected = _original_load_reference_state(core, state_dict, device)
    residual = [key for key in missing if not key.startswith("oracle_")]
    return residual, list(unexpected)


_stage1.load_reference_state = load_reference_state_allowing_probe

if __name__ == "__main__":
    raise SystemExit(_stage1.main())
