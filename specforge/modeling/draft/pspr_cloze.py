# coding=utf-8
"""PSPR-cloze: one-shot masked-LM refinement of a draft block's seed path.

Why a third module and not an edit to ``pspr_v2.py``
---------------------------------------------------
``pspr_v2.py`` carries every number currently reported (stage-1 macro 5.7743, v2 step4000 macro
5.9700) and the only checkpoint worth keeping.  This file changes what the bidirectional pass
*consumes*, which is a different model, so it gets its own class, its own ``model_type`` string, and
its own registry entry.  ``LatticeCorrector`` and its artifacts stay byte-identical and loadable.

What v2's bidirectional encoder actually sees
---------------------------------------------
``LatticeCorrector.encode`` builds one token per slot::

    centroid_t = sum_k softmax(lp_t)_k * E(c_{t,k})
    slot_t     = dh_in(h_t) + candset_in(centroid_t) + scal_in(scalars_t)

so the K candidates at slot ``t`` are collapsed into a single probability-weighted average *before*
any attention runs.  Two consequences:

1. The object the lattice framing is about -- "candidate A at slot t is right BECAUSE candidate B
   sits at slot t+1" -- is destroyed before the encoder sees it and reconstructed nowhere after.
   A contradiction such as ``1 + 1 = 3 ... so the answer is 2`` exists only between *concrete*
   tokens; the centroid of 16 candidates smears it away, hardest at exactly the high-entropy slots
   where the first error lives.
2. Nothing in the design is a cloze.  Slot ``t``'s prediction is never conditioned on "slot t is
   unknown, every other slot is a definite token"; it is conditioned on a soup in which slot ``t``'s
   own top-1 is already the dominant term, so the module's cheapest solution is to copy it.

Measured on the stage-1 trace: of 4,499 first errors, 3,824 (85.0%) have the correct token somewhere
in top-16, yet the selector ranks it first in only ~40%.  That 40-vs-85 gap is an *availability*
gap that this module was designed to target; by itself it does not prove that the draft-observable
features identify which of the 16 tokens is the target token.  The completed PerfectBlend run and
the bidirectional/causal ablation must be used to answer that separate identifiability question.

Seed refinement: why the context is the seed path and not the committed prefix
------------------------------------------------------------------------------
The obvious cloze construction fills positions ``j < i`` with what the walk has already committed.
It is strictly more informative and it is fatal, because it makes row ``i`` depend on rows
``0..i-1``: serving would have to run the encoder ``H`` times, once per slot, and ``H`` serialised
launch-bound kernels cost more wall clock than the target tokens they are supposed to save.  A
speculative-decoding head that is only cheap in FLOPs is not cheap.

So the context is the **seed path** ``d0[j] = cand[j, 0]``, the greedy path of the same single
all-MASK draft forward that produced the lattice.  Every row reads the same ``d0``, so all ``H`` rows
are one batched encoder call, and -- more importantly -- the construction is *literally identical* in
training and in serving.  There is no teacher-forced future, no ground-truth prefix, and therefore
nothing to argue about: no distribution shift to bound, no leakage channel to audit.  Positions are
still told which side of the query they are on (``LEFT_SEED`` vs ``RIGHT_SEED``), because their
relationship to the query differs even when their content does not.

Two orthogonal questions, two mechanisms
----------------------------------------
Dropping the committed prefix from the cloze rows would lose real information: at slot ``i`` the walk
genuinely knows what it committed at ``0..i-1``, and the seed only guesses.  That is recovered by the
GRU, kept from v2:

    cloze rows (bidirectional, batched once per block)  ->  "what does the whole block look like?"
    GRU over committed tokens (causal, one step per slot) -> "what did we actually commit so far?"

The GRU step is a single small matmul per slot, so the walk stays cheap while the expensive
bidirectional pass runs exactly once.  This is the split that lets the module answer both questions
without paying ``H`` encoder passes.

Late fusion
-----------
``delta`` is built from three separately-normalised streams concatenated, not summed::

    delta_i = W2 . silu( W1 [ LN_h(h_i) ; LN_z(z_i) ; LN_s(S_i) ] )

v2 concatenated first and applied one LayerNorm to the result, which lets the 2560-wide ``h`` block
dominate the shared statistics and drown the 512-wide streams.  ``h_i`` also enters at full width
rather than only through the encoder's ``2560 -> d`` squeeze.  Trace-scale probe B measured full-width
``h`` as *worse* (+0.373 against +0.418), but that probe used the shared-LayerNorm form, so the
comparison is confounded; ``direct_hidden`` stays a switch so it can be ablated at PerfectBlend scale.

``W2`` is zero-initialised, so ``delta == 0`` and the module is bit-identical to plain DFlash at step
0, while every parameter still receives gradient from step 1 (unlike v1's ``gamma=0`` scalar gate).

The readout is the frozen tied embedding, so no ``[vocab, *]`` matrix is learned (Domino spends 38.9M
of its 50.82M there).  ``E (h + delta)`` is a full-vocabulary distribution at no architectural cost.

What this does NOT do
---------------------
It does not re-run the draft backbone and it does not touch the target.  The lattice and the seed
path both come from the single all-MASK parallel forward.  Serving adds one batched encoder call per
block plus one GRU step and one MLP per slot.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .dflash import DFlashDraftModel
from .pspr import LatticePathSelector
from .registry import register_draft

# Role of a position inside a cloze row.  ``LEFT``/``RIGHT`` carry the same content (the seed path)
# but not the same relationship to the query: the left side is what the walk will have committed by
# the time slot ``i`` is decided, the right side is what the draft is still guessing.
_ROLE_ANCHOR = 0
_ROLE_LEFT_SEED = 1
_ROLE_QUERY = 2
_ROLE_RIGHT_SEED = 3
_N_ROLES = 4


class ClozeCorrector(nn.Module):
    """Refines a draft block by masking one slot and reading the seed path around it."""

    # The host adds the frontier detector's BCE term when this is set.
    wants_err_objective = True
    # Both modules are used exclusively by err_logits().  The online host must freeze the complete
    # set when the optional detector is neither trained nor served; freezing err_head alone leaves
    # err_seed_in trainable-but-unused and makes DDP fail on its second iteration.
    err_only_module_names = ("err_seed_in", "err_head")
    # ``OnlineDFlashModel`` lays a training block out as ``[anchor, proposal_1..proposal_H]``.  The
    # anchor is not a proposal slot: it has no candidate list and its hidden state is conditioned on
    # a different set.  The host slices it off; this module re-introduces it as cloze position 0.
    selector_excludes_anchor = True
    supports_full_vocab = True

    def __init__(
        self,
        *,
        hidden_size: int,
        vocab_size: int,
        top_k: int = 16,
        d: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        state_dim: int = 512,
        delta_hidden: int = 2048,
        max_slots: int = 32,
        dropout: float = 0.1,
        n_scalars: int = 3,
        direct_hidden: bool = True,
        use_state: bool = True,
        bidirectional: bool = True,
        err_use_state: bool = False,
    ) -> None:
        super().__init__()
        if top_k < 2:
            raise ValueError("PSPR-cloze needs top_k >= 2 so a non-top-1 candidate can win")
        self.top_k = int(top_k)
        self.d = int(d)
        self.hidden_size = int(hidden_size)
        self.state_dim = int(state_dim)
        self.n_scalars = int(n_scalars)
        # Ablation switches.  Each removes one stream from `delta` and nothing else.
        self.direct_hidden = bool(direct_hidden)
        self.use_state = bool(use_state)
        if err_use_state and not self.use_state:
            raise ValueError("err_use_state=True requires use_state=True")
        self.err_use_state = bool(err_use_state)
        # Identical parameters and FLOPs, only the future half of the attention removed: the control
        # that attributes any gain to bidirectionality rather than to capacity.
        self.bidirectional = bool(bidirectional)
        self.max_slots = int(max_slots)

        # ---- cloze position inputs.  Each stream is normalised in its own space before projection;
        # `h` is a bf16 backbone activation and `E(token)` is a row of the embedding table, and their
        # scales differ enough that summing raw projections lets one of them set the cell.
        self.h_ln = nn.LayerNorm(hidden_size)
        self.h_in = nn.Linear(hidden_size, d, bias=False)
        self.tok_ln = nn.LayerNorm(hidden_size)
        self.tok_in = nn.Linear(hidden_size, d, bias=False)
        # top-1 log prob, top1-top2 margin, position entropy, top-k mass: how far the seed token at
        # this position can be trusted as context.
        self.conf_ln = nn.LayerNorm(4)
        self.conf_in = nn.Linear(4, d)
        self.mask_emb = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.mask_emb, std=0.02)
        self.role_emb = nn.Embedding(_N_ROLES, d)
        nn.init.normal_(self.role_emb.weight, std=0.02)
        # +1 for the anchor position that sits ahead of every proposal slot.
        self.pos_emb = nn.Parameter(torch.zeros(max_slots + 1, d))
        nn.init.normal_(self.pos_emb, std=0.02)
        # The anchor has no draft hidden state (the host sliced it off) and no candidate list, so its
        # non-token features are one learned vector.  It still carries its committed token, which is
        # what gives slot 0 any pre-block context inside the rows.
        self.anchor_feat = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.anchor_feat, std=0.02)
        self.cell_ln = nn.LayerNorm(d)

        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=n_heads,
            dim_feedforward=4 * d,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)

        # ---- causal stream.  The cloze rows see the SEED at ``j < i``; the walk actually knows what
        # it committed there.  That difference is real information and this is where it enters.  One
        # GRU step per slot keeps serving's per-slot cost O(1) while the encoder runs once per block.
        if self.use_state:
            self.gru = nn.GRU(hidden_size, self.state_dim, batch_first=True)

        # ---- late fusion.  Three separately-normalised streams, concatenated.
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
        # delta == 0 at step 0 => bit-identical to plain DFlash => any gain is attributable.  The
        # output layer keeps its own gradient, so all parameters train from step 1.
        self.delta_w2._pspr_zero_init = True
        nn.init.zeros_(self.delta_w2.weight)

        # ---- keep/repair gate.  The query position is masked, so ``z_i`` never sees WHICH token is
        # about to be kept -- only how confident the draft was.  The seed token's embedding is fed in
        # explicitly.  Its exact label is selected by the host objective: historical BCE asks whether
        # top-1 is wrong, while profitable_repair asks whether a correct alternative is available.
        err_in = (
            d + d + 4 + (hidden_size if self.direct_hidden else 0)
        )
        self.err_seed_in = nn.Linear(hidden_size, d, bias=False)
        self.err_head = nn.Sequential(
            nn.LayerNorm(err_in),
            nn.Linear(err_in, d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d, 1),
        )
        err_only_names = ["err_seed_in", "err_head"]
        if self.err_use_state:
            # Keep the mature detector exactly loadable and add the committed-prefix feature as a
            # zero-initialised residual.  Concatenating S into err_head would change its first-layer
            # shape and throw away the useful Stage-1 detector on every warm start.  The residual
            # also includes projected h/z/seed/confidence so it can learn interactions such as
            # "this seed is plausible under the all-MASK h but inconsistent with what was actually
            # committed", rather than reducing the GRU to a context-independent scalar bias.
            self.err_state_ln = nn.LayerNorm(self.state_dim)
            err_state_in = self.state_dim + 3 * d + 4
            self.err_state_head = nn.Sequential(
                nn.LayerNorm(err_state_in),
                nn.Linear(err_state_in, d),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d, 1),
            )
            self.err_state_head[-1]._pspr_zero_init = True
            nn.init.zeros_(self.err_state_head[-1].weight)
            nn.init.zeros_(self.err_state_head[-1].bias)
            err_only_names.extend(("err_state_ln", "err_state_head"))
        self.err_only_module_names = tuple(err_only_names)

        self.register_buffer(
            "target_embedding", torch.empty(0, dtype=torch.float32), persistent=False
        )
        self._config = {
            "model_type": "ClozeCorrector",
            "hidden_dim": int(hidden_size),
            "vocab_size": int(vocab_size),
            "K": self.top_k,
            "d": self.d,
            "n_layers": int(n_layers),
            "n_heads": int(n_heads),
            "ds": self.state_dim,
            "delta_h": int(delta_hidden),
            "max_slots": self.max_slots,
            "n_scal": self.n_scalars,
            "dropout": float(dropout),
            "direct_hidden": self.direct_hidden,
            "use_state": self.use_state,
            "bidirectional": self.bidirectional,
            "err_use_state": self.err_use_state,
        }

    def reference_config(self) -> dict:
        return dict(self._config)

    @property
    def K(self) -> int:
        """Alias the reference selector uses; the external decoder reads it to size the lattice."""
        return self.top_k

    # ---------------------------------------------------------------- binding

    def bind_target_embedding(self, embed_tokens: nn.Module) -> None:
        """Attach the target's tied input embedding; it is both the candidate table and the readout."""
        weight = getattr(embed_tokens, "weight", embed_tokens)
        self.target_embedding = weight.detach()

    def _embedding(self) -> torch.Tensor:
        weight = self.target_embedding
        if weight.numel() == 0:
            raise RuntimeError(
                "ClozeCorrector needs bind_target_embedding() before the first forward"
            )
        return weight.to(dtype=self.h_in.weight.dtype)

    # ---------------------------------------------------------------- lattice

    # Byte-identical lattice extraction and decision rules, so a cloze-vs-v2 comparison isolates the
    # head and nothing else.
    extract_lattice = LatticePathSelector.extract_lattice
    keep_repair_log_probs = staticmethod(LatticePathSelector.keep_repair_log_probs)
    select_margin_gate = staticmethod(LatticePathSelector.select_margin_gate)
    select_keep_repair = classmethod(LatticePathSelector.select_keep_repair.__func__)
    restore_zero_init_contract = None  # replaced below

    # ------------------------------------------------------------------ cloze

    def confidence(
        self, log_probs: torch.Tensor, scalars: torch.Tensor
    ) -> torch.Tensor:
        """``[..., H, 4]`` how far the seed token at each slot can be trusted as context."""
        param_dtype = self.h_in.weight.dtype
        log_probs = log_probs.to(param_dtype)
        margin = (log_probs[..., 0] - log_probs[..., 1]).clamp(-20.0, 20.0)
        scalars = scalars.to(param_dtype)
        return torch.stack(
            [log_probs[..., 0], margin, scalars[..., 0], scalars[..., 2]], dim=-1
        )

    def cloze_states(
        self,
        hidden_states: torch.Tensor,
        candidate_ids: torch.Tensor,
        anchor_ids: torch.Tensor,
        log_probs: torch.Tensor,
        scalars: torch.Tensor,
    ) -> torch.Tensor:
        """``[..., H, d]`` one batched masked-LM pass over the block. ``z_i`` per query slot.

        Every argument comes from the single all-MASK draft forward plus the block's own anchor
        token, so this is called identically during training and during the serving walk -- there is
        no teacher-forced variant to keep in sync.
        """
        param_dtype = self.h_in.weight.dtype
        embedding = self._embedding()
        H = candidate_ids.shape[-2]
        d = self.d
        lead = candidate_ids.shape[:-2]
        device = candidate_ids.device

        seed_ids = candidate_ids[..., 0]
        # Kept as two separate sums rather than "context minus seed": floating-point ``(a+b)-a`` is
        # not ``b``, so subtracting the seed back out of the query cell would leave a ~1e-7 residue
        # of the very token the row must not see.  The query cell never touches ``seed_tok``.
        base = self.h_in(self.h_ln(hidden_states.to(param_dtype))) + self.conf_in(
            self.conf_ln(self.confidence(log_probs, scalars))
        )  # [..., H, d]
        seed_tok = self.tok_in(self.tok_ln(F.embedding(seed_ids, embedding)))
        context_cell = base + seed_tok
        query_cell = base + self.mask_emb.to(base.dtype)

        i = torch.arange(H, device=device).view(H, 1)
        j = torch.arange(H, device=device).view(1, H)
        role = torch.where(
            j.eq(i),
            torch.full_like(j, _ROLE_QUERY),
            torch.where(
                j.gt(i),
                torch.full_like(j, _ROLE_RIGHT_SEED),
                torch.full_like(j, _ROLE_LEFT_SEED),
            ),
        )  # [H, H]

        # The query position keeps its own hidden state and confidence but not its seed token: the
        # row must not be able to read the token it is being asked to predict.
        is_query = j.eq(i).unsqueeze(-1)
        rows = torch.where(
            is_query, query_cell.unsqueeze(-3), context_cell.unsqueeze(-3)
        )  # [..., H, H, d]
        rows = rows + self.role_emb(role).to(rows.dtype) + self.pos_emb[1 : H + 1].to(rows.dtype)

        anchor = (
            self.tok_in(self.tok_ln(F.embedding(anchor_ids, embedding)))
            + self.anchor_feat.to(rows.dtype)
            + self.role_emb.weight[_ROLE_ANCHOR].to(rows.dtype)
            + self.pos_emb[0].to(rows.dtype)
        ).unsqueeze(-2).unsqueeze(-3).expand(*lead, H, 1, d)

        seq = self.cell_ln(torch.cat([anchor, rows], dim=-2))  # [..., H, H+1, d]
        mask = None
        attention_pattern = getattr(
            self, "attention_pattern", "bidirectional" if self.bidirectional else "causal"
        )
        if attention_pattern == "causal":
            mask = torch.full(
                (H + 1, H + 1), float("-inf"), device=device, dtype=seq.dtype
            ).triu(1)
        elif attention_pattern == "anchor_self":
            # Diagnostic control: each proposal cell may read the block anchor and itself, but no
            # other proposal slot.  Keep the full dense Transformer call so parameters/kernel FLOPs
            # are unchanged; only cross-slot information is removed.  The anchor itself is also
            # self-only, preventing a multi-layer anchor relay from smuggling slot information back
            # into the query.
            mask = torch.full(
                (H + 1, H + 1), float("-inf"), device=device, dtype=seq.dtype
            )
            mask.fill_diagonal_(0.0)
            mask[1:, 0] = 0.0
        elif attention_pattern != "bidirectional":
            raise ValueError(f"unknown cloze attention_pattern={attention_pattern!r}")
        out = self.encoder(seq.reshape(-1, H + 1, d), mask=mask).reshape(*lead, H, H + 1, d)
        # Row ``i``'s masked position is proposal slot ``i``, i.e. sequence index ``i + 1``.
        take = (
            (torch.arange(H, device=device) + 1)
            .view(*((1,) * len(lead)), H, 1, 1)
            .expand(*lead, H, 1, d)
        )
        return out.gather(-2, take).squeeze(-2)

    def causal_states(self, prefix_embeddings: torch.Tensor) -> torch.Tensor:
        """GRU over the committed prefix. ``[..., H, E]`` -> ``[..., H, state_dim]``.

        The cloze rows only ever see the seed at ``j < i``.  This is where what the walk *actually*
        committed enters, and it is the one input the backbone provably lacks: its block forward sees
        MASK at every proposal slot.
        """
        if not self.use_state:
            raise RuntimeError("use_state=False has no causal stream")
        horizon, width = prefix_embeddings.shape[-2], prefix_embeddings.shape[-1]
        flat = prefix_embeddings.reshape(-1, horizon, width).to(self.gru.weight_ih_l0.dtype)
        states, _ = self.gru(flat)
        return states.reshape(*prefix_embeddings.shape[:-1], self.state_dim)

    # -------------------------------------------------------------- corrector

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

        Equal to :meth:`full_vocab_logits` gathered at the candidate ids minus the base normaliser,
        so it is anchored on ``log_probs`` and reduces to plain DFlash when ``delta == 0``.
        """
        param_dtype = self.h_in.weight.dtype
        delta = self.hidden_delta(z, hidden_states, state)
        correction = (candidate_embeddings.to(param_dtype) * delta.unsqueeze(-2)).sum(-1)
        return log_probs.float() + correction.float()

    def err_logits(
        self,
        z: torch.Tensor,
        seed_embeddings: torch.Tensor,
        log_probs: torch.Tensor,
        scalars: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Keep/repair gate logit, with label semantics supplied by the host objective. ``[..., H]``.

        ``seed_embeddings`` is ``E(d0_i)``, the very token the gate decides whether to keep.  It is a
        separate argument because ``z_i`` cannot contain it: slot ``i`` is masked in row ``i``.
        """
        if self.direct_hidden and hidden_states is None:
            raise RuntimeError("direct_hidden=True requires hidden_states in err_logits")
        param_dtype = self.h_in.weight.dtype
        seed_feature = self.err_seed_in(self.tok_ln(seed_embeddings.to(param_dtype)))
        parts = [z.to(param_dtype), seed_feature, self.confidence(log_probs, scalars)]
        if self.direct_hidden:
            parts.insert(0, hidden_states.to(param_dtype))
        logits = self.err_head(torch.cat(parts, dim=-1)).squeeze(-1)
        if self.err_use_state:
            if state is None or hidden_states is None:
                raise RuntimeError(
                    "err_use_state=True requires committed-prefix state and hidden_states"
                )
            residual_parts = [
                self.err_state_ln(state.to(param_dtype)),
                self.h_in(self.h_ln(hidden_states.to(param_dtype))),
                z.to(param_dtype),
                seed_feature,
                self.confidence(log_probs, scalars),
            ]
            logits = logits + self.err_state_head(torch.cat(residual_parts, dim=-1)).squeeze(-1)
        return logits.float()

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
        """Signature-compatible with the DFlash family's selector contract.

        ``predecessor_ids[0]`` is the block anchor's committed token and feeds cloze position 0;
        ``predecessor_ids`` as a whole feeds the causal GRU.  The cloze rows themselves read only the
        seed path, so nothing here is a teacher-forced input the serving walk cannot reproduce.
        """
        embedding = self._embedding()
        candidate_embeddings = F.embedding(candidate_ids, embedding)
        unary_logits = unary_logits.float()
        if lattice_scalars is None:
            margin = (unary_logits[..., 0] - unary_logits[..., 1]).clamp(-20.0, 20.0)
            mass = unary_logits.exp().sum(dim=-1).clamp(max=1.0)
            lattice_scalars = torch.stack(
                [torch.zeros_like(margin), margin, mass], dim=-1
            )
        z = self.cloze_states(
            hidden_states,
            candidate_ids,
            predecessor_ids[..., 0],
            unary_logits,
            lattice_scalars,
        )
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


def _restore_zero_init_contract(self: ClozeCorrector) -> None:
    """Re-assert the zero-init the module's init-time no-op depends on.

    ``DFlashDraftModel.__init__`` runs ``post_init()`` after ``_init_draft_head``, and ``post_init``
    re-applies ``_init_weights`` to every submodule, so zeroing inside ``__init__`` alone does not
    survive.  The write goes through ``torch.nn.init`` on purpose: while HF runs ``_init_weights`` it
    swaps those primitives for guarded versions that skip tensors the loader already filled from a
    checkpoint, whereas a raw in-place ``zero_()`` is invisible to the guard and would silently
    destroy a warm start after ``from_pretrained`` re-runs init.
    """
    nn.init.zeros_(self.delta_w2.weight)
    if self.err_use_state:
        nn.init.zeros_(self.err_state_head[-1].weight)
        nn.init.zeros_(self.err_state_head[-1].bias)


ClozeCorrector.restore_zero_init_contract = _restore_zero_init_contract


@register_draft
class PSPRClozeDraftModel(DFlashDraftModel):
    """DFlash backbone with a one-shot masked-LM seed refiner."""

    warm_start_optional_prefixes = ("candidate_selector.",)
    # A mature cloze head predates the optional state-conditioned gate residual.  It is safe to
    # initialise this exact all-or-nothing group as a no-op while loading every existing selector
    # tensor; a partially present group remains a hard loader error.
    warm_start_optional_key_groups = (
        (
            "candidate_selector.err_state_ln.weight",
            "candidate_selector.err_state_ln.bias",
            "candidate_selector.err_state_head.0.weight",
            "candidate_selector.err_state_head.0.bias",
            "candidate_selector.err_state_head.1.weight",
            "candidate_selector.err_state_head.1.bias",
            "candidate_selector.err_state_head.4.weight",
            "candidate_selector.err_state_head.4.bias",
        ),
    )

    def _init_draft_head(self, config, dflash_config: dict) -> None:
        self.candidate_selector = ClozeCorrector(
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            top_k=int(dflash_config.get("selector_top_k", 16)),
            d=int(dflash_config.get("selector_dim", 512)),
            n_layers=int(dflash_config.get("selector_layers", 6)),
            n_heads=int(dflash_config.get("selector_heads", 8)),
            state_dim=int(dflash_config.get("selector_state_dim", 512)),
            delta_hidden=int(dflash_config.get("selector_delta_hidden", 2048)),
            max_slots=int(dflash_config.get("selector_max_slots", 32)),
            dropout=float(dflash_config.get("selector_dropout", 0.1)),
            direct_hidden=bool(dflash_config.get("selector_direct_hidden", True)),
            use_state=bool(dflash_config.get("selector_use_state", True)),
            bidirectional=bool(dflash_config.get("selector_bidirectional", True)),
            err_use_state=bool(dflash_config.get("selector_err_use_state", False)),
        )
        self.freeze_backbone = bool(dflash_config.get("freeze_backbone", False))
        # A warm gate-calibration run must not destroy an already-useful alternative scorer.
        # ``err_only`` freezes every selector parameter except the modules used exclusively by
        # err_logits(); the default remains ``all`` so existing recipes retain their behaviour.
        self.selector_train_scope = str(
            dflash_config.get("selector_train_scope", "all")
        )
        if self.selector_train_scope not in {"all", "err_only", "err_state_only"}:
            raise ValueError(
                "selector_train_scope must be 'all', 'err_only' or 'err_state_only', got "
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
        self.selector_decision_mode = str(
            dflash_config.get("selector_decision_mode", "margin_gate")
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
        if isinstance(module, ClozeCorrector):
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
        """Apply the configured backbone and selector training scope.

        ``err_only`` is the clean warm-start experiment: candidate ordering, ``z``, ``delta`` and
        the GRU remain bit-identical while the old detector plus its optional state residual learn
        KEEP-vs-REPAIR. ``err_state_only`` is the stricter residual-only diagnostic.
        """
        frozen = 0
        for name, parameter in self.named_parameters():
            is_selector = name.startswith("candidate_selector.")
            should_freeze = self.freeze_backbone and not is_selector
            if is_selector and self.selector_train_scope != "all":
                relative = name.removeprefix("candidate_selector.")
                if self.selector_train_scope == "err_only":
                    train_prefixes = tuple(
                        f"{module_name}."
                        for module_name in self.candidate_selector.err_only_module_names
                    )
                else:
                    if not self.candidate_selector.err_use_state:
                        raise ValueError(
                            "selector_train_scope='err_state_only' requires "
                            "selector_err_use_state=true"
                        )
                    train_prefixes = ("err_state_ln.", "err_state_head.")
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

        The expensive bidirectional pass runs ONCE for the block -- the cloze rows depend only on the
        seed path, which is fixed the moment the lattice is read -- and the walk then costs one GRU
        step plus one MLP per slot.
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

        z = selector.cloze_states(
            hidden, candidate_ids, block_output_ids[:, 0], unary_logits, lattice_scalars
        )

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


__all__ = ["ClozeCorrector", "PSPRClozeDraftModel"]
