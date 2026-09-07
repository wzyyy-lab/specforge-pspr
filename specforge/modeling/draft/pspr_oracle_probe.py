# coding=utf-8
"""Diagnostic-only probe: how much is candidate-level cross-slot information WORTH?

Motivation
----------
``LatticePathSelector.encode`` collapses each slot's K candidates into one softmax-weighted
centroid (``pspr.py:321``) before the transformer mixes slots (``pspr.py:329``), and every one of a
slot's K candidates then reads the same context vector (``pspr.py:408``). The module is therefore
*candidate-level causal* (candidate identity flows forward only through the GRU over committed
predecessors) and only *slot-level bidirectional*. It cannot represent "candidate A at slot 2 is
compatible with candidate B at slot 7".

Whether that limitation is worth removing is an empirical question, and removing it for real is
expensive: axial attention over [H, K] invalidates the stage-1 warm start, changes the decode path
that produced macro 5.7217, and adds per-block cost to a module whose entire purpose is saving
wall-clock. So this file buys an upper bound first.

What this measures
------------------
Each slot's context is allowed to attend to the *ground-truth future tokens* ``{y_s : s > t}``.
That is the most optimistic possible version of "compare the complete future paths": the future
path is not merely representable, it is handed over for free and exactly.

Interpretation is asymmetric, and this is the whole point:

* If the probe does **not** improve macro accept length, the limitation is refuted as a bottleneck.
  No architecture can beat free exact future information, so a real implementation that must
  *infer* the future from a top-K lattice cannot do better. This is a decisive negative.
* If the probe **does** improve it, only a necessary condition has been met. A real implementation
  sees 16 noisy candidates per future slot, not the answer, and would recover some unknown
  fraction of the probe's gain.

Leakage discipline
------------------
The slot's own answer ``y_t`` must never become reachable, otherwise the probe measures
"was the target in top-K" (i.e. oracle@16) instead of the value of future context. Two paths have
to be closed, and a naive "append oracle tokens to the encoder input" closes neither:

1. Direct: ``t`` reading ``y_t``. Closed by the strict ``s > t`` mask.
2. Two-hop: slot ``u < t`` legitimately reads ``y_t`` (it is ``u``'s future), and slot ``t`` then
   reads slot ``u``. This is why the oracle tokens are NOT added to the encoder's token sequence.
   They are keys/values of a single cross-attention applied AFTER the encoder has finished, and
   they participate in no self-attention at all, so no oracle embedding is ever mixed into a slot
   representation that another slot can read.

Knowing ``y_{t+1}`` is genuinely informative about ``y_t`` (inflection, agreement) and that is not
leakage -- it is exactly the path-compatibility signal under test.

Zero-init contract
------------------
``oracle_gate`` starts at zero, so at initialisation this class is elementwise identical to
``LatticePathSelector``. The probe warm-starts from the 5.7217 checkpoint and only has to learn how
to *use* the future channel, which is what makes a short run informative.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .pspr import LatticePathSelector


class OracleFutureProbeSelector(LatticePathSelector):
    """``LatticePathSelector`` plus a masked cross-attention over ground-truth future tokens."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        d = self.d
        embed_dim = self.cand_proj.in_features
        self.oracle_heads = int(kwargs.get("n_heads", 8))
        self.oracle_in = nn.Linear(embed_dim, d)
        self.oracle_attn = nn.MultiheadAttention(
            d, self.oracle_heads, dropout=0.0, batch_first=True
        )
        self.oracle_ln = nn.LayerNorm(d)
        # Every query row keeps one always-visible key so the last slot, which has no future, does
        # not produce an all-masked softmax (NaN).
        self.oracle_sink = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.oracle_sink, std=0.02)
        self.oracle_gate = nn.Parameter(torch.zeros(1))
        self._oracle_ids: Optional[torch.Tensor] = None

    def restore_zero_init_contract(self) -> None:
        super().restore_zero_init_contract()
        nn.init.zeros_(self.oracle_gate)

    def set_oracle(self, oracle_ids: Optional[torch.Tensor]) -> None:
        """Stage the batch's ground-truth token ids. Negative entries mark invalid slots."""
        self._oracle_ids = oracle_ids

    def clear_oracle(self) -> None:
        self._oracle_ids = None

    def _inject_future(self, context: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        ids = self._oracle_ids
        if ids is None:
            return context
        horizon = context.shape[-2]
        ctx = context.reshape(-1, horizon, self.d)
        ids = ids.reshape(-1, horizon)
        if ids.shape[0] != ctx.shape[0]:
            raise RuntimeError(
                f"staged oracle batch {ids.shape[0]} does not match context batch {ctx.shape[0]}"
            )
        param_dtype = self.oracle_in.weight.dtype
        batch = ctx.shape[0]
        valid = ids >= 0
        keys = self.oracle_in(F.embedding(ids.clamp(min=0), embedding).to(param_dtype))
        sink = self.oracle_sink.to(param_dtype).expand(batch, 1, self.d)
        keys = torch.cat([sink, keys], dim=1)

        index = torch.arange(horizon, device=ctx.device)
        strictly_future = index.view(1, horizon, 1) < index.view(1, 1, horizon)
        allow = strictly_future & valid.view(batch, 1, horizon)
        sink_allow = torch.ones(batch, horizon, 1, dtype=torch.bool, device=ctx.device)
        allow = torch.cat([sink_allow, allow], dim=-1)
        mask = (~allow).repeat_interleave(self.oracle_heads, dim=0)

        attended, _ = self.oracle_attn(
            ctx.to(param_dtype), keys, keys, attn_mask=mask, need_weights=False
        )
        gated = self.oracle_gate.to(ctx.dtype) * self.oracle_ln(attended).to(ctx.dtype)
        return (ctx + gated).reshape(context.shape)

    def score_candidates(
        self,
        *,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        predecessor_ids: torch.Tensor,
        lattice_scalars: Optional[torch.Tensor] = None,
        return_err: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        embedding = self._embedding(hidden_states.dtype)
        candidate_embeddings = F.embedding(candidate_ids, embedding)
        prefix_embeddings = F.embedding(predecessor_ids, embedding)
        unary_logits = unary_logits.float()
        if lattice_scalars is None:
            margin = (unary_logits[..., 0] - unary_logits[..., 1]).clamp(-20.0, 20.0)
            mass = unary_logits.exp().sum(dim=-1).clamp(max=1.0)
            lattice_scalars = torch.stack(
                [torch.zeros_like(margin), margin, mass], dim=-1
            )
        context = self.encode(
            hidden_states, candidate_embeddings, unary_logits, lattice_scalars
        )
        context = self._inject_future(context, embedding)
        state = self.causal_states(prefix_embeddings)
        scores = self.score(
            context,
            state,
            candidate_embeddings,
            unary_logits,
            hidden_states=hidden_states,
            predecessor_ids=predecessor_ids,
            candidate_ids=candidate_ids,
        )
        if return_err:
            return scores, self.err_logits(context, state, unary_logits, lattice_scalars)
        return scores
