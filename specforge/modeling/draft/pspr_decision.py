# coding=utf-8
"""PSPR-Decision: a factorised keep/repair head on the frozen DFlash-b16 lattice.

Why a fourth module and not an edit to ``pspr_cloze.py``
-------------------------------------------------------
``pspr.py`` / ``pspr_v2.py`` / ``pspr_cloze.py`` each carry a shipped checkpoint and reported
numbers.  This file changes the *decision structure*, which is a different model, so it gets its own
class, its own ``model_type`` string and its own registry entry.  Nothing in the three existing
modules is touched.

What the measurements said, and what this file does about it
-----------------------------------------------------------
Three inference-time ablations of the same trained cloze@7115 weights (identical parameters, FLOPs
and lattice, six domains, n=20; see ``TAPS-SP/PSPR_DECISION.md``):

    full bidirectional                       macro 6.0438
    causal      (future removed)             macro 6.0427   -0.0012
    anchor_self (ALL cross-slot removed)     macro 5.9993   -0.0445
    zero_context(encoder output zeroed)      macro 5.5352   -0.5086

So of the encoder path's ~0.509, cross-slot information is worth 0.045 and future information is
worth 0.001.  The remaining ~91% is a per-slot non-linear transform.  A six-layer bidirectional
Transformer over H*(H+1) cells is a very expensive way to buy a per-slot MLP: cloze costs
8.63 ms/block against v2's 6.16 ms.  **This module therefore replaces the cross-slot attention with
a per-slot residual tower and recovers the 0.045 with a single attention-free block-mean broadcast.**

The reason the bidirectional pass could never pay off is structural, not a tuning failure.  The
backbone is full-attention inside a block (``dflash.py:245``, ``is_causal=False``) so ``h_i`` has
already aggregated the whole block, and the lattice is read off the frozen tied ``lm_head``
(``dflash.py:785``) so every seed token, log-prob and margin is a deterministic function of
``{h_1..h_H}``.  The cloze sequence carries no information beyond ``{h_j}``.  Worse, with
``direct_hidden=True`` the query row still receives ``h_i`` at full width, and
``argmax(lm_head(h_i))`` *is* the token being masked -- so the query position was never actually
masked.  It was not a cloze.

The real bottleneck: KEEP-vs-REPAIR, not "which alternative"
------------------------------------------------------------
Measured on reachable decisions (accepted prefix + frontier slot) of cloze@7115, ``fixable_n=5789``
(backbone top-1 wrong AND the target inside top-16):

    target ranked #1 among the 15 ALTERNATIVES                    66.86%
    target ranked #1 among all 16 candidates (KEEP in the softmax) 44.05%
    actually repaired after the rho=3 margin gate                  36.71%

The ordering ability is already there; a joint 16-way softmax throws half of it away.  The cause is
the training distribution: multiclass CE is supervised only where the target is inside top-16, and
77.4% of those labels are class 0, so the model learns an overwhelming KEEP prior -- a prior that is
irrelevant at serving time, because the decision only matters where the base is suspect.

Hence the factorisation::

    p_i = sigmoid(D(feat_i))                 P(a usable repair exists at slot i)
    r_i = softmax(R(feat_i)) in R^{K-1}      P(target = cand_j | a usable repair exists)

    P(target = cand_0) = 1 - p_i             absorbs BOTH "base is right" and "target not in top-16"
    P(target = cand_j) = p_i * r_i[j-1]

    !! ``1 - p_i`` is therefore ``P(target = cand_0) + P(target not in the lattice)``, NOT
    ``P(target = cand_0)``.  Both events have KEEP as their optimal action so the greedy argmax
    stays well defined, but the KEEP score is systematically INFLATED by the miss mass (~0.10-0.16
    where the base is wrong), i.e. this decoder is biased conservative.  It is also why these
    outputs must not be fed to anything that treats them as prefix-survival probabilities
    (``latrepairpath``).  A 17-way head with an explicit "none of the above" class would remove the
    inflation; that is deliberately left to the next iteration.

Decision: ``argmax_j P(target = cand_j)``.  No tau, no rho, no theta -- the three hand-tuned decode
thresholds disappear, and the rule is exactly the training likelihood's MAP action
(``LatticePathSelector.keep_repair_log_probs``).

The detector's label is ``repair_is_covered`` (target is one of candidates 1..K-1), **not**
``~base_top1_ok``.  That single change fixes two defects at once: a top-16 miss is no longer a
positive target for a detector that cannot act on it (~15.6% of the old positives), and those ~21%
of slots stop being unsupervised -- under multiclass they were dropped entirely
(``selector_supervised = target_is_candidate``).  ``1 - p_i`` legitimately merges "base is right"
with "unrepairable", because KEEP is the optimal action in both.

The detector is a first-class decision maker here, not a veto.  In the deployed ``latgate`` rule the
detector sat under ``if ok and gate_theta > 0`` (``decode_lattice.py:498``), so it could only ever
remove an override the margin rule had already admitted -- never add one -- which is why every theta
sweep came back negative (``EXPERIMENT_LOG.md:2386-2393``).  Rewiring the *same untouched* detector
as a trigger instead (``latfront``, theta=0.6) moved macro 6.0438 -> 6.0853 at zero training cost.
That is the evidence this module is built on, and that detector still had none of the fixes below.

It also gets the inputs it was missing.  ``selector_err_use_state`` was absent from the cloze config,
so the detector never saw the committed-prefix GRU state; here it is on by default.  It additionally
reads all K log-probs rather than four hand-picked scalars, and the token it is judging, ``E(d0_i)``,
which no ``z``-style feature can contain.

Zero-init contract
------------------
``delta_w2`` and the detector's output layer are both zero-initialised, so at step 0
``score == log_probs`` and ``keep_repair_log_probs`` argmax is identically 0: the module is
bit-identical to plain DFlash.  Concretely at init KEEP has probability ``1 - det_prior = 0.78``
while every alternative is at most ``det_prior * r_j <= 0.22``, so candidate 0 wins strictly for any
finite lattice, including ties, equal logits and K=2.

Note on what zero-init costs: on step 1 ONLY ``delta_w2.weight`` and the detector's final
weight/bias receive nonzero gradient -- every upstream tower, fusion, projection and GRU parameter is
gradient-blocked until those two outputs grow away from zero.  (An earlier version of this docstring
claimed all parameters train from step 1; that was wrong.)  This is the reason ``warmup_ratio`` is
0.06 rather than 0.03, and there are now two such blocked paths rather than one.

The detector's output bias starts at ``logit(det_prior)`` rather than 0, which is still a strict KEEP
at step 0 but starts calibration near the base rate instead of 0.5.  Caveat: 0.22 is the UNWEIGHTED
rate of ``repair_is_covered``; under ``frontier_boost=3`` the weighted rate is nearer 0.33, so this
bias is a mild under-estimate at init.

What this does NOT do
---------------------
It does not re-run the draft backbone, does not touch the target, and learns no ``[vocab, *]``
matrix (the readout is the frozen tied embedding; Domino spends 38.9M of its 50.82M there).  Serving
cost per block is one batched per-slot tower call plus, per slot, one GRU step, one delta MLP and one
detector MLP.
"""

from __future__ import annotations

from math import log
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .dflash import DFlashDraftModel
from .pspr import LatticePathSelector
from .registry import register_draft


class _ResidualBlock(nn.Module):
    """Pre-LN residual MLP block: the per-slot transform the cloze encoder was really providing."""

    def __init__(self, d: int, expansion: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.up = nn.Linear(d, expansion * d)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.down = nn.Linear(expansion * d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.down(self.drop(self.act(self.up(self.norm(x)))))


class DecisionCorrector(nn.Module):
    """Factorised keep/repair decision over a fixed top-K lattice."""

    # The host adds the detector term.  Under a factorised objective the detector is trained inside
    # ``selector_ce`` itself (binary_ce + conditional alt CE), so the standalone err objective must
    # be switched off in the recipe or two conflicting labels would be applied to the same logit.
    wants_err_objective = True
    # Modules used exclusively by err_logits().  The GRU is shared with the ranker and is therefore
    # deliberately absent.
    err_only_module_names = ("err_seed_in", "err_head")
    # ``OnlineDFlashModel`` lays a training block out as ``[anchor, proposal_1..proposal_H]``; the
    # anchor has no candidate list.  The host slices it off and hands us its token id separately.
    selector_excludes_anchor = True
    supports_full_vocab = True

    def __init__(
        self,
        *,
        hidden_size: int,
        vocab_size: int,
        top_k: int = 16,
        d: int = 512,
        n_layers: int = 4,
        expansion: int = 4,
        state_dim: int = 512,
        delta_hidden: int = 2048,
        det_hidden: int = 1024,
        max_slots: int = 32,
        dropout: float = 0.05,
        n_scalars: int = 3,
        direct_hidden: bool = True,
        use_state: bool = True,
        err_use_state: bool = True,
        block_context: bool = True,
        det_prior: float = 0.22,
    ) -> None:
        super().__init__()
        if top_k < 2:
            raise ValueError("PSPR-Decision needs top_k >= 2 so an alternative can win")
        if not 0.0 < det_prior < 1.0:
            raise ValueError(f"det_prior must be in (0, 1), got {det_prior}")
        self.top_k = int(top_k)
        self.d = int(d)
        self.hidden_size = int(hidden_size)
        self.state_dim = int(state_dim)
        # NOT a width knob: `confidence()` reads `scalars[..., 0]` (entropy) and `scalars[..., 2]`
        # (top-k mass) by position, so this records the `extract_lattice` output contract and any
        # other value would silently mean something different.  Fail instead of pretending it tunes.
        if int(n_scalars) != 3:
            raise ValueError(
                "n_scalars is the extract_lattice contract (entropy, margin, topk_mass) and is read "
                f"positionally by confidence(); only 3 is meaningful, got {n_scalars}"
            )
        self.n_scalars = int(n_scalars)
        self.direct_hidden = bool(direct_hidden)
        self.use_state = bool(use_state)
        if err_use_state and not self.use_state:
            raise ValueError("err_use_state=True requires use_state=True")
        if not use_state:
            # `decode_lattice.py`'s latrepair/latgate loops step `lat.gru` unconditionally, so a
            # state-free head would load fine and then crash mid-decode.  Refuse at construction
            # rather than at serving time.
            raise ValueError(
                "use_state=False is not servable: the external decoder steps the GRU "
                "unconditionally. Remove the causal stream there first if you want this ablation."
            )
        self.err_use_state = bool(err_use_state)
        # The measured value of ALL cross-slot information is 0.045 macro.  One attention-free
        # block-mean broadcast is the proportionate way to buy it; an ablation switch keeps that
        # attributable.
        self.block_context = bool(block_context)
        self.max_slots = int(max_slots)
        self.det_prior = float(det_prior)

        # ---- per-slot input streams.  Each is normalised in its own space before projection: ``h``
        # is a bf16 backbone activation and ``E(token)`` is an embedding row, and their scales differ
        # enough that summing raw projections lets one of them set the cell.
        self.h_ln = nn.LayerNorm(hidden_size)
        self.h_in = nn.Linear(hidden_size, d, bias=False)
        self.tok_ln = nn.LayerNorm(hidden_size)
        self.tok_in = nn.Linear(hidden_size, d, bias=False)
        # Four trust scalars (top-1 log prob, top1-top2 margin, entropy, top-k mass) ...
        self.conf_ln = nn.LayerNorm(4)
        self.conf_in = nn.Linear(4, d)
        # ... plus the full log-prob profile.  The shape of the top-K tail is exactly the evidence a
        # keep/repair decision needs, and four hand-picked scalars discard most of it.
        self.lp_ln = nn.LayerNorm(self.top_k)
        self.lp_in = nn.Linear(self.top_k, d)
        self.pos_emb = nn.Parameter(torch.zeros(max_slots, d))
        nn.init.normal_(self.pos_emb, std=0.02)
        if self.block_context:
            self.ctx_ln = nn.LayerNorm(hidden_size)
            self.ctx_in = nn.Linear(hidden_size, d, bias=False)

        self.tower = nn.ModuleList(
            _ResidualBlock(d, expansion, dropout) for _ in range(int(n_layers))
        )
        self.z_ln = nn.LayerNorm(d)

        # ---- causal stream: what the walk ACTUALLY committed at j < i.  This is the one input the
        # backbone provably lacks -- its block forward sees MASK at every proposal slot.  One GRU
        # step per slot keeps the walk O(1) per slot.
        if self.use_state:
            self.gru = nn.GRU(hidden_size, self.state_dim, batch_first=True)

        # ---- ranker.  Late fusion of separately-normalised streams, then a hidden-space delta read
        # back through the frozen tied embedding, so no [vocab, *] matrix is learned.
        self.fuse_h_ln = nn.LayerNorm(hidden_size)
        self.fuse_z_ln = nn.LayerNorm(d)
        self.fuse_s_ln = nn.LayerNorm(self.state_dim)
        delta_in = (
            d
            + (hidden_size if self.direct_hidden else 0)
            + (self.state_dim if self.use_state else 0)
        )
        self.delta_w1 = nn.Linear(delta_in, delta_hidden)
        self.delta_act = nn.SiLU()
        self.delta_drop = nn.Dropout(dropout)
        self.delta_w2 = nn.Linear(delta_hidden, hidden_size, bias=False)
        self.delta_w2._pspr_zero_init = True
        nn.init.zeros_(self.delta_w2.weight)

        # ---- detector.  A first-class decision maker, so it gets the committed-prefix state, the
        # token it is judging, the whole log-prob profile and full-width h.
        self.err_seed_in = nn.Linear(hidden_size, d, bias=False)
        err_in = (
            d  # z
            + d  # E(d0_i) projected
            + 4  # trust scalars
            + self.top_k  # log-prob profile
            + (hidden_size if self.direct_hidden else 0)
            + (self.state_dim if self.err_use_state else 0)
        )
        self.err_head = nn.Sequential(
            nn.LayerNorm(err_in),
            nn.Linear(err_in, det_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(det_hidden, det_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(det_hidden, 1),
        )
        self.err_head[-1]._pspr_zero_init = True
        self._init_detector_output()

        self.register_buffer(
            "target_embedding", torch.empty(0, dtype=torch.float32), persistent=False
        )
        self._config = {
            "model_type": "DecisionCorrector",
            "hidden_dim": int(hidden_size),
            "vocab_size": int(vocab_size),
            "K": self.top_k,
            "d": self.d,
            "n_layers": int(n_layers),
            "expansion": int(expansion),
            "ds": self.state_dim,
            "delta_h": int(delta_hidden),
            "det_h": int(det_hidden),
            "max_slots": self.max_slots,
            "n_scal": self.n_scalars,
            "dropout": float(dropout),
            "direct_hidden": self.direct_hidden,
            "use_state": self.use_state,
            "err_use_state": self.err_use_state,
            "block_context": self.block_context,
            "det_prior": self.det_prior,
        }

    def _init_detector_output(self) -> None:
        """Zero weights plus a base-rate bias: a strict KEEP at step 0, already calibrated."""
        nn.init.zeros_(self.err_head[-1].weight)
        nn.init.constant_(
            self.err_head[-1].bias, log(self.det_prior / (1.0 - self.det_prior))
        )

    def reference_config(self) -> dict:
        return dict(self._config)

    @property
    def K(self) -> int:
        """Alias the reference selector uses; the external decoder reads it to size the lattice."""
        return self.top_k

    # ---------------------------------------------------------------- binding

    def bind_target_embedding(self, embed_tokens: nn.Module) -> None:
        weight = getattr(embed_tokens, "weight", embed_tokens)
        self.target_embedding = weight.detach()

    def _embedding(self) -> torch.Tensor:
        weight = self.target_embedding
        if weight.numel() == 0:
            raise RuntimeError(
                "DecisionCorrector needs bind_target_embedding() before the first forward"
            )
        return weight.to(dtype=self.h_in.weight.dtype)

    # ---------------------------------------------------------------- lattice

    # Byte-identical lattice extraction and decision rules, so a Decision-vs-cloze comparison
    # isolates the head and nothing else.
    extract_lattice = LatticePathSelector.extract_lattice
    keep_repair_log_probs = staticmethod(LatticePathSelector.keep_repair_log_probs)
    select_margin_gate = staticmethod(LatticePathSelector.select_margin_gate)
    select_keep_repair = classmethod(LatticePathSelector.select_keep_repair.__func__)
    restore_zero_init_contract = None  # replaced below

    # ------------------------------------------------------------- per-slot z

    def confidence(
        self, log_probs: torch.Tensor, scalars: torch.Tensor
    ) -> torch.Tensor:
        """``[..., H, 4]`` how far the base top-1 at each slot can be trusted."""
        param_dtype = self.h_in.weight.dtype
        log_probs = log_probs.to(param_dtype)
        margin = (log_probs[..., 0] - log_probs[..., 1]).clamp(-20.0, 20.0)
        scalars = scalars.to(param_dtype)
        return torch.stack(
            [log_probs[..., 0], margin, scalars[..., 0], scalars[..., 2]], dim=-1
        )

    def slot_states(
        self,
        hidden_states: torch.Tensor,
        candidate_ids: torch.Tensor,
        log_probs: torch.Tensor,
        scalars: torch.Tensor,
    ) -> torch.Tensor:
        """``[..., H, d]`` per-slot features. One batched call per block.

        Every input comes from the single all-MASK draft forward, so this is called identically in
        training and in the serving walk: there is no teacher-forced variant to keep in sync, and
        nothing here depends on the committed prefix (that enters through the GRU), which is what
        keeps the walk O(1) per slot.
        """
        param_dtype = self.h_in.weight.dtype
        embedding = self._embedding()
        hidden = hidden_states.to(param_dtype)
        H = candidate_ids.shape[-2]
        if H > self.max_slots:
            raise ValueError(f"block has {H} slots, max_slots={self.max_slots}")

        cell = (
            self.h_in(self.h_ln(hidden))
            + self.tok_in(self.tok_ln(F.embedding(candidate_ids[..., 0], embedding)))
            + self.conf_in(self.conf_ln(self.confidence(log_probs, scalars)))
            + self.lp_in(self.lp_ln(log_probs.to(param_dtype)))
            + self.pos_emb[:H].to(param_dtype)
        )
        if self.block_context:
            # The whole measured value of cross-slot information is 0.045 macro.  A mean over the
            # block's hidden states, broadcast to every slot, buys it without any attention: cost is
            # O(H*d) and, crucially, it is computed once per block and is independent of the walk.
            summary = self.ctx_in(self.ctx_ln(hidden.mean(dim=-2, keepdim=True)))
            cell = cell + summary
        for block in self.tower:
            cell = block(cell)
        return self.z_ln(cell)

    def causal_states(self, prefix_embeddings: torch.Tensor) -> torch.Tensor:
        """GRU over the committed prefix. ``[..., H, E]`` -> ``[..., H, state_dim]``."""
        if not self.use_state:
            raise RuntimeError("use_state=False has no causal stream")
        horizon, width = prefix_embeddings.shape[-2], prefix_embeddings.shape[-1]
        flat = prefix_embeddings.reshape(-1, horizon, width).to(
            self.gru.weight_ih_l0.dtype
        )
        states, _ = self.gru(flat)
        return states.reshape(*prefix_embeddings.shape[:-1], self.state_dim)

    # --------------------------------------------------------------- ranker

    def hidden_delta(
        self,
        z: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``[..., hidden]`` additive correction, late-fused from separately-normalised streams."""
        param_dtype = self.h_in.weight.dtype
        parts = [self.fuse_z_ln(z.to(param_dtype))]
        if self.direct_hidden:
            if hidden_states is None:
                raise RuntimeError("direct_hidden=True requires hidden_states")
            parts.insert(0, self.fuse_h_ln(hidden_states.to(param_dtype)))
        if self.use_state:
            if state is None:
                raise RuntimeError("use_state=True requires state")
            parts.append(self.fuse_s_ln(state.to(param_dtype)))
        fused = self.delta_w1(torch.cat(parts, dim=-1))
        return self.delta_w2(self.delta_drop(self.delta_act(fused)))

    def corrected_hidden(
        self,
        z: torch.Tensor,
        hidden_states: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        param_dtype = self.h_in.weight.dtype
        return hidden_states.to(param_dtype) + self.hidden_delta(z, hidden_states, state)

    def full_vocab_logits(
        self,
        z: torch.Tensor,
        hidden_states: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``[..., vocab]`` corrected logits. Training-only: dense CE needs every class."""
        return F.linear(self.corrected_hidden(z, hidden_states, state), self._embedding())

    def score(
        self,
        z: torch.Tensor,
        hidden_states: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        log_probs: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``[..., K]`` corrected scores restricted to the lattice.

        Anchored on ``log_probs`` and reduces to plain DFlash when ``delta == 0``.  Under a
        factorised objective only ``[..., 1:]`` is normalised, so candidate 0's score is never asked
        to compete against the alternatives inside one softmax.
        """
        param_dtype = self.h_in.weight.dtype
        delta = self.hidden_delta(z, hidden_states, state)
        correction = (candidate_embeddings.to(param_dtype) * delta.unsqueeze(-2)).sum(-1)
        return log_probs.float() + correction.float()

    # -------------------------------------------------------------- detector

    def err_logits(
        self,
        z: torch.Tensor,
        seed_embeddings: torch.Tensor,
        log_probs: torch.Tensor,
        scalars: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Keep/repair gate logit ``[..., H]``; label semantics come from the host objective.

        Under ``profitable_repair`` this is ``logit P(a usable repair exists here)``, which is the
        quantity the decode rule divides by -- so it is trained and served with one semantics.
        """
        if self.direct_hidden and hidden_states is None:
            raise RuntimeError("direct_hidden=True requires hidden_states in err_logits")
        if self.err_use_state and state is None:
            raise RuntimeError("err_use_state=True requires committed-prefix state")
        param_dtype = self.h_in.weight.dtype
        parts = [
            z.to(param_dtype),
            self.err_seed_in(self.tok_ln(seed_embeddings.to(param_dtype))),
            self.confidence(log_probs, scalars),
            log_probs.to(param_dtype),
        ]
        if self.direct_hidden:
            parts.insert(0, hidden_states.to(param_dtype))
        if self.err_use_state:
            parts.append(state.to(param_dtype))
        return self.err_head(torch.cat(parts, dim=-1)).squeeze(-1).float()

    # ------------------------------------------------------------- host entry

    def score_candidates(
        self,
        *,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        predecessor_ids: torch.Tensor,
        lattice_scalars: Optional[torch.Tensor] = None,
        return_err: bool = False,
        return_full: bool = False,
    ):
        """Signature-compatible with the DFlash family's selector contract."""
        embedding = self._embedding()
        candidate_embeddings = F.embedding(candidate_ids, embedding)
        unary_logits = unary_logits.float()
        if lattice_scalars is None:
            margin = (unary_logits[..., 0] - unary_logits[..., 1]).clamp(-20.0, 20.0)
            mass = unary_logits.exp().sum(dim=-1).clamp(max=1.0)
            lattice_scalars = torch.stack(
                [torch.zeros_like(margin), margin, mass], dim=-1
            )
        z = self.slot_states(hidden_states, candidate_ids, unary_logits, lattice_scalars)
        state = (
            self.causal_states(F.embedding(predecessor_ids, embedding))
            if self.use_state
            else None
        )
        scores = self.score(
            z, hidden_states, candidate_embeddings, unary_logits, state
        )
        if not (return_err or return_full):
            return scores
        out = [scores]
        if return_err:
            out.append(
                self.err_logits(
                    z,
                    candidate_embeddings[..., 0, :],
                    unary_logits,
                    lattice_scalars,
                    hidden_states,
                    state,
                )
            )
        if return_full:
            out.append(self.full_vocab_logits(z, hidden_states, state))
        return tuple(out)

    def forward(self, **kwargs):
        """Alias for :meth:`score_candidates` so DDP registers every parameter."""
        return self.score_candidates(**kwargs)


def _restore_zero_init_contract(self: DecisionCorrector) -> None:
    """Re-assert the zero-init that the step-0 KEEP identity depends on.

    ``DFlashDraftModel.__init__`` runs ``post_init()`` after ``_init_draft_head``, and ``post_init``
    re-applies ``_init_weights`` to every submodule, so zeroing inside ``__init__`` alone does not
    survive.  The write goes through ``torch.nn.init`` on purpose: while HF runs ``_init_weights`` it
    swaps those primitives for guarded versions that skip tensors the loader already filled from a
    checkpoint, whereas a raw in-place ``zero_()`` is invisible to the guard and would silently
    destroy a warm start after ``from_pretrained`` re-runs init.
    """
    nn.init.zeros_(self.delta_w2.weight)
    self._init_detector_output()


DecisionCorrector.restore_zero_init_contract = _restore_zero_init_contract


@register_draft
class PSPRDecisionDraftModel(DFlashDraftModel):
    """DFlash backbone with a factorised keep/repair decision head."""

    warm_start_optional_prefixes = ("candidate_selector.",)

    def _init_draft_head(self, config, dflash_config: dict) -> None:
        self.candidate_selector = DecisionCorrector(
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            top_k=int(dflash_config.get("selector_top_k", 16)),
            d=int(dflash_config.get("selector_dim", 512)),
            n_layers=int(dflash_config.get("selector_layers", 4)),
            expansion=int(dflash_config.get("selector_expansion", 4)),
            state_dim=int(dflash_config.get("selector_state_dim", 512)),
            delta_hidden=int(dflash_config.get("selector_delta_hidden", 2048)),
            det_hidden=int(dflash_config.get("selector_det_hidden", 1024)),
            max_slots=int(dflash_config.get("selector_max_slots", 32)),
            dropout=float(dflash_config.get("selector_dropout", 0.05)),
            direct_hidden=bool(dflash_config.get("selector_direct_hidden", True)),
            use_state=bool(dflash_config.get("selector_use_state", True)),
            err_use_state=bool(dflash_config.get("selector_err_use_state", True)),
            block_context=bool(dflash_config.get("selector_block_context", True)),
            det_prior=float(dflash_config.get("selector_det_prior", 0.22)),
        )
        self.freeze_backbone = bool(dflash_config.get("freeze_backbone", False))
        self.selector_train_scope = str(
            dflash_config.get("selector_train_scope", "all")
        )
        if self.selector_train_scope not in {"all", "err_only"}:
            raise ValueError(
                "selector_train_scope must be 'all' or 'err_only', got "
                f"{self.selector_train_scope!r}"
            )
        self.selector_compute_dtype = str(
            dflash_config.get("selector_compute_dtype", "model")
        )
        if self.selector_compute_dtype not in {"model", "float32"}:
            raise ValueError(
                "selector_compute_dtype must be 'model' or 'float32', got "
                f"{self.selector_compute_dtype!r}"
            )
        # ``keep_repair`` is the native decision for this head: it is the MAP action of the training
        # likelihood.  ``margin_gate`` is retained only so the historical rule can be run as an
        # ablation on the same weights.
        self.selector_decision_mode = str(
            dflash_config.get("selector_decision_mode", "keep_repair")
        )
        if self.selector_decision_mode not in {"full_vocab", "margin_gate", "keep_repair"}:
            raise ValueError(
                "selector_decision_mode must be 'full_vocab', 'margin_gate' or 'keep_repair', "
                f"got {self.selector_decision_mode!r}"
            )
        self.selector_gate_rho = float(dflash_config.get("selector_gate_rho", 1.0))
        self.selector_gate_tau = float(dflash_config.get("selector_gate_tau", 0.0))
        self.selector_gate_theta = float(dflash_config.get("selector_gate_theta", 0.0))
        self.selector_gate_skip_first = bool(
            dflash_config.get("selector_gate_skip_first", False)
        )
        self.selector_keep_repair_margin = float(
            dflash_config.get("selector_keep_repair_margin", 0.0)
        )

    def _init_weights(self, module) -> None:
        super()._init_weights(module)
        if getattr(module, "_pspr_zero_init", False):
            nn.init.zeros_(module.weight)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, DecisionCorrector):
            module.restore_zero_init_contract()

    def transform_unary_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Identity in float32: the lattice is read straight off ``lm_head(h)``."""
        return logits.float()

    def bind_target_decoder(self, embed_tokens: nn.Module) -> None:
        self.candidate_selector.bind_target_embedding(embed_tokens)
        self.enforce_selector_compute_dtype()

    def enforce_selector_compute_dtype(self) -> None:
        if self.selector_compute_dtype == "float32":
            self.candidate_selector.to(dtype=torch.float32)

    def apply_backbone_freeze(self) -> int:
        frozen = 0
        for name, parameter in self.named_parameters():
            is_selector = name.startswith("candidate_selector.")
            should_freeze = self.freeze_backbone and not is_selector
            if is_selector and self.selector_train_scope == "err_only":
                relative = name.removeprefix("candidate_selector.")
                train_prefixes = tuple(
                    f"{module_name}."
                    for module_name in self.candidate_selector.err_only_module_names
                )
                should_freeze = not relative.startswith(train_prefixes)
            if should_freeze and parameter.requires_grad:
                parameter.requires_grad_(False)
                frozen += parameter.numel()
        return frozen

    def _sample_draft_tokens(
        self,
        target: nn.Module,
        draft_hidden: torch.Tensor,
        block_output_ids: torch.LongTensor,
    ) -> torch.LongTensor:
        """Native fixed-lattice walk, identical in semantics to the external decoder.

        The per-slot tower runs ONCE for the block -- it depends only on quantities fixed the moment
        the lattice is read -- and the walk then costs one GRU step, one delta MLP and one detector
        MLP per slot.
        """
        hidden = draft_hidden[:, -self.block_size + 1 :, :]
        selector = self.candidate_selector
        if selector.target_embedding.numel() == 0:
            self.bind_target_decoder(target.model.embed_tokens)
        unary_logits, candidate_ids, lattice_scalars = selector.extract_lattice(
            target.lm_head(hidden)
        )
        embedding = selector._embedding()
        candidate_embeddings = F.embedding(candidate_ids, embedding)

        z = selector.slot_states(hidden, candidate_ids, unary_logits, lattice_scalars)

        predecessor_ids = block_output_ids[:, 0]
        gru_hidden = None
        path = []
        for position in range(candidate_ids.shape[1]):
            state = None
            if selector.use_state:
                step, gru_hidden = selector.gru(
                    F.embedding(predecessor_ids, embedding)
                    .unsqueeze(1)
                    .to(selector.gru.weight_ih_l0.dtype),
                    gru_hidden,
                )
                state = step[:, 0]
            zi = z[:, position]
            hi = hidden[:, position]
            if self.selector_decision_mode == "full_vocab":
                predecessor_ids = selector.full_vocab_logits(zi, hi, state).argmax(dim=-1)
                path.append(predecessor_ids)
                continue
            scores = selector.score(
                zi, hi, candidate_embeddings[:, position], unary_logits[:, position], state
            )
            err_logits = None
            if self.selector_decision_mode == "keep_repair" or self.selector_gate_theta > 0:
                err_logits = selector.err_logits(
                    zi,
                    candidate_embeddings[:, position, 0],
                    unary_logits[:, position],
                    lattice_scalars[:, position],
                    hi,
                    state,
                )
            if self.selector_decision_mode == "keep_repair":
                selected_index = selector.select_keep_repair(
                    scores, err_logits, repair_margin=self.selector_keep_repair_margin
                )
            else:
                force_keep = torch.full(
                    scores.shape[:-1],
                    self.selector_gate_skip_first and position == 0,
                    dtype=torch.bool,
                    device=scores.device,
                )
                selected_index = selector.select_margin_gate(
                    scores,
                    err_logits=err_logits,
                    rho=self.selector_gate_rho,
                    tau=self.selector_gate_tau,
                    theta=self.selector_gate_theta,
                    force_keep_mask=force_keep,
                )
            predecessor_ids = candidate_ids[:, position].gather(1, selected_index)[:, 0]
            path.append(predecessor_ids)
        return torch.stack(path, dim=1)


__all__ = ["DecisionCorrector", "PSPRDecisionDraftModel"]
