# coding=utf-8
"""PSPR-v2: a lattice-conditioned hidden-state corrector with a frozen tied readout.

Why a second module instead of an edit to ``pspr.py``
-----------------------------------------------------
``pspr.py`` is bit-locked against the external reference selector that produced every stage-1
baseline (``scripts/gate_pspr_reference_equivalence.py`` asserts ``max|delta| = 0``).  v2 changes the
scorer's input signature, so it gets its own class and its own registry entry; every S1 artifact
stays loadable and every existing gate stays green.

What v1 actually computed
-------------------------
Qwen3-4B has ``tie_word_embeddings=True``, so ``lm_head.weight`` IS the input embedding ``E``.
Expanding v1's ``score()`` under that identity::

    score_k = lp_k + gamma * ( mlp([r_t; S_t; W_c E(c_k); lp_k]) + <E(c_k), delta_mlp([r_t; S_t])> )
            = <E(c_k), h_t + gamma * delta_t> - logZ_t + gamma * mlp(...)

so the ``delta`` branch is exactly "correct the hidden state, then re-read it through the frozen tied
lm_head".  The deep audit's counterfactuals measured ``delta-only`` (5.9329) at or above ``full``
(5.9043) and ``MLP-only`` (5.3988) barely above base (5.3402): the hidden-space correction is the
whole head, and the pointwise MLP is noise.  v2 keeps the correction and drops the MLP branch.

The bottleneck v2 removes
-------------------------
In v1 ``score()`` the ``hidden_states`` argument is consumed **only** by ``transition_term``, which is
disabled at ``trans_rank=0`` -- the S1 configuration.  So ``h_t`` (2560-d) reached the correction
solely through ``r_t = Encoder(dh_in(h_t) + ...)``, a LayerNorm'd, dropout'd **2560 -> 512** squeeze
that is simultaneously asked to carry the cross-slot signal.  Domino's competing head feeds
``[h_t ; S_t]`` at full width.  That single missing wire explains the audit's counterfactual table
better than the "candidate identity is lost in the centroid" reading does: removing cross-slot
*attention* barely moves the score (5.9414 -> 5.9150) because the attention was never the value --
the 512-d projection was, and it survives that ablation.

v2 therefore wires ``h_t`` straight into the corrector and lets ``r_t`` be a pure *additive*
cross-slot term.  This is what makes the bidirectional encoder falsifiable: with ``cross_slot="none"``
the module keeps identical parameters and compute and only loses the off-diagonal attention, so the
difference is attributable to cross-slot information alone.

Full-vocabulary supervision at zero architectural cost
------------------------------------------------------
``E (h_t + delta_t)`` is defined for every token, not just the K candidates.  Measured on the stage-1
trace, 20.93% of slots have their target outside top-16 and currently receive **no** selector
gradient at all, and the surviving 79.07% get a 16-way signal where Domino's head gets a 151936-way
one.  ``full_vocab_logits`` exposes the whole readout so training can use dense CE; serving may still
restrict the argmax to the lattice (``score_candidates``) or take the full argmax, and the two are
reported separately because the restriction costs a measured 8.4% of first errors.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .dflash import DFlashDraftModel
from .pspr import LatticePathSelector
from .registry import register_draft

_VALID_CROSS_SLOT = {"full", "none"}


class LatticeCorrector(nn.Module):
    """Corrects a draft hidden state from the block's lattice plus the committed prefix."""

    # The host adds the frontier detector's BCE term when this is set.
    wants_err_objective = True
    # ``OnlineDFlashModel`` lays a training block out as ``[anchor, proposal_1..proposal_H]``.  The
    # encoder has learned positional embeddings and (optionally) cross-slot attention, so leaving the
    # anchor in shifts every proposal's position and lets an inference-absent pseudo-slot alter all
    # proposal representations.  The host must slice it off before any lattice extraction.
    selector_excludes_anchor = True
    # Tells the objective this selector can emit a full-vocabulary distribution.
    supports_full_vocab = True

    def __init__(
        self,
        *,
        hidden_size: int,
        vocab_size: int,
        top_k: int = 16,
        d: int = 512,
        n_layers: int = 3,
        n_heads: int = 8,
        state_dim: int = 512,
        delta_hidden: int = 2048,
        max_slots: int = 32,
        dropout: float = 0.1,
        n_scalars: int = 3,
        cross_slot: str = "full",
        direct_hidden: bool = True,
        cand_slots: int = 4,
        prefix_rank: bool = False,
    ) -> None:
        super().__init__()
        if top_k < 2:
            raise ValueError("PSPR-v2 needs top_k >= 2 so a non-top-1 candidate can win")
        if cross_slot not in _VALID_CROSS_SLOT:
            raise ValueError(f"cross_slot must be one of {sorted(_VALID_CROSS_SLOT)}")
        self.top_k = int(top_k)
        self.d = int(d)
        self.hidden_size = int(hidden_size)
        self.state_dim = int(state_dim)
        self.n_scalars = int(n_scalars)
        self.cross_slot = str(cross_slot)
        self.direct_hidden = bool(direct_hidden)
        # How many candidate embeddings enter the encoder with their identity intact.  v1 fed a
        # single probability-weighted centroid, which collapses to noise exactly at the high-entropy
        # slots that carry the cross-slot signal.  0 restores v1's centroid-only behaviour.
        self.cand_slots = max(0, min(int(cand_slots), self.top_k))

        self.dh_in = nn.Linear(hidden_size, d)
        self.candset_in = nn.Linear(hidden_size, d)
        if self.cand_slots > 0:
            self.candtop_in = nn.Linear(hidden_size * self.cand_slots, d)
            self.candlp_in = nn.Linear(self.cand_slots, d)
        self.scal_in = nn.Linear(self.n_scalars, d)
        self.slot_ln = nn.LayerNorm(d)
        self.pos_emb = nn.Parameter(torch.zeros(max_slots, d))
        nn.init.normal_(self.pos_emb, std=0.02)
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

        self.gru = nn.GRU(hidden_size, self.state_dim, batch_first=True)

        # Where the committed path sits *in the lattice*, not just which token it is.
        #
        # In an all-MASK block forward every slot's hidden state is conditioned on exactly the same
        # information (the anchor prefix), so cross-slot attention only remixes readouts of one
        # conditioning set and adds no new evidence -- which is why it measures ~+0.02 here and
        # ~+0.03 in the deep audit, and why an H-by-K cell encoder would not change that either.
        # The one genuinely new signal available to the head is the committed prefix, and the GRU
        # currently sees only its token embeddings.  The rank of the committed token within the
        # *previous slot's own candidate list* tells the head something the embedding cannot: rank 0
        # means the draft's greedy path is still intact, rank>0 means this block has already been
        # repaired and the backbone's remaining slots were drafted under a prefix that no longer
        # matches.  It is causal, identical in training and serving, and it is the actual "path"
        # information the lattice framing promises.
        self.prefix_rank = bool(prefix_rank)
        if self.prefix_rank:
            # top_k buckets for in-lattice ranks, one for "off-lattice", one for the anchor slot.
            self.rank_emb = nn.Embedding(self.top_k + 2, hidden_size)
            nn.init.normal_(self.rank_emb.weight, std=0.02)

        # The corrector.  ``h_t`` enters at full width here; that wire is the point of v2.
        delta_in = d + self.state_dim + (hidden_size if self.direct_hidden else 0)
        self.delta_mlp = nn.Sequential(
            nn.LayerNorm(delta_in),
            nn.Linear(delta_in, delta_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(delta_hidden, hidden_size, bias=False),
        )
        # Zero output => delta == 0 => v2 is EXACTLY the DFlash backbone at step 0, so any measured
        # gain is attributable.  Unlike v1's ``gamma=0`` gate this leaves the output layer's own
        # gradient intact, so all ~13M corrector parameters train from step 1 instead of waiting for
        # one scalar to grow (v1's gamma crawled 0.32 -> 0.55 -> 0.98 across three epochs).
        self.delta_mlp[-1]._pspr_zero_init = True
        nn.init.zeros_(self.delta_mlp[-1].weight)

        # Frontier detector, kept from v1 so the calibrated decode gate keeps working.  Acceptance is
        # decided at the first wrong slot and the two error directions are not symmetric: recovering
        # the frontier extends the run, breaking a correct slot truncates it, and correct slots
        # outnumber wrong ones ~4.5:1.
        err_in = d + self.state_dim + 4 + (hidden_size if self.direct_hidden else 0)
        self.err_head = nn.Sequential(
            nn.LayerNorm(err_in),
            nn.Linear(err_in, d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d, 1),
        )

        self.register_buffer(
            "target_embedding", torch.empty(0, dtype=torch.float32), persistent=False
        )
        self._config = {
            "model_type": "LatticeCorrector",
            "hidden_dim": int(hidden_size),
            "vocab_size": int(vocab_size),
            "K": self.top_k,
            "d": self.d,
            "n_layers": int(n_layers),
            "n_heads": int(n_heads),
            "ds": self.state_dim,
            "delta_h": int(delta_hidden),
            "max_slots": int(max_slots),
            "n_scal": self.n_scalars,
            "dropout": float(dropout),
            "cross_slot": self.cross_slot,
            "direct_hidden": self.direct_hidden,
            "cand_slots": self.cand_slots,
            "prefix_rank": self.prefix_rank,
        }

    def reference_config(self) -> dict:
        return dict(self._config)

    @property
    def K(self) -> int:
        """Alias the reference selector uses; the external decoder reads it to size the lattice."""
        return self.top_k

    # ---------------------------------------------------------------- binding

    def bind_target_embedding(self, embed_tokens: nn.Module) -> None:
        """Attach the target's tied input embedding; it is both the candidate table and the readout.

        Non-persistent: a ``[vocab, hidden]`` matrix must not enter the draft checkpoint, and it is
        available wherever the DFlash family runs.
        """
        weight = getattr(embed_tokens, "weight", embed_tokens)
        self.target_embedding = weight.detach()

    def _embedding(self) -> torch.Tensor:
        weight = self.target_embedding
        if weight.numel() == 0:
            raise RuntimeError(
                "LatticeCorrector needs bind_target_embedding() before the first forward"
            )
        # The readout is part of the selector decision, so it follows the selector's configured
        # precision rather than a caller-supplied (possibly bf16 backbone) dtype.
        return weight.to(dtype=self.dh_in.weight.dtype)

    # ---------------------------------------------------------------- lattice

    # Tie-handling and the uncertainty channels are identical to v1; reuse them verbatim so the two
    # modules see byte-identical lattices and any v1/v2 comparison isolates the head.
    extract_lattice = LatticePathSelector.extract_lattice
    restore_zero_init_contract = None  # replaced below

    # ---------------------------------------------------------------- encoder

    def encode(
        self,
        hidden_states: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        log_probs: torch.Tensor,
        scalars: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-slot pass over one block. ``[..., H, *]`` -> ``[..., H, d]``.

        ``cross_slot="none"`` keeps every parameter and every FLOP and only removes the off-diagonal
        attention, so the ablation attributes its delta to cross-slot information rather than to the
        encoder's capacity.
        """
        param_dtype = self.dh_in.weight.dtype
        weights = torch.softmax(log_probs.float(), dim=-1).to(candidate_embeddings.dtype)
        centroid = (weights.unsqueeze(-1) * candidate_embeddings).sum(dim=-2)
        slot = (
            self.dh_in(hidden_states.to(param_dtype))
            + self.candset_in(centroid.to(param_dtype))
            + self.scal_in(scalars.to(param_dtype))
        )
        if self.cand_slots > 0:
            m = self.cand_slots
            head = candidate_embeddings[..., :m, :].to(param_dtype)
            slot = slot + self.candtop_in(head.flatten(-2, -1))
            slot = slot + self.candlp_in(log_probs[..., :m].to(param_dtype))
        horizon = slot.shape[-2]
        slot = self.slot_ln(slot) + self.pos_emb[:horizon]
        flat = slot.reshape(-1, horizon, self.d)
        mask = None
        if self.cross_slot == "none":
            mask = torch.full(
                (horizon, horizon), float("-inf"), device=flat.device, dtype=flat.dtype
            )
            mask.fill_diagonal_(0.0)
        return self.encoder(flat, mask=mask).reshape(slot.shape)

    def prefix_ranks(
        self, candidate_ids: torch.Tensor, predecessor_ids: torch.Tensor
    ) -> torch.Tensor:
        """``[..., H]`` rank of each committed token inside the previous slot's candidate list.

        ``top_k`` marks "committed token was not in that slot's lattice" (only reachable when the
        corpus label is off-lattice during teacher forcing) and ``top_k + 1`` marks slot 0, whose
        predecessor is the anchor and therefore has no candidate list.
        """
        prev_cands = candidate_ids[..., :-1, :]
        prev_token = predecessor_ids[..., 1:]
        matches = prev_cands.eq(prev_token.unsqueeze(-1))
        rank = torch.where(
            matches.any(dim=-1),
            matches.long().argmax(dim=-1),
            torch.full_like(prev_token, self.top_k),
        )
        anchor = torch.full_like(predecessor_ids[..., :1], self.top_k + 1)
        return torch.cat([anchor, rank], dim=-1)

    def causal_states(
        self,
        prefix_embeddings: torch.Tensor,
        prefix_rank_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """GRU over the committed prefix. ``[..., H, E]`` -> ``[..., H, state_dim]``.

        This is the one input the backbone provably lacks: its block forward sees MASK at every
        proposal slot, so the identity of the tokens actually committed before slot ``t`` is new
        information, not a re-reading of what ``h_t`` already encodes.
        """
        horizon, width = prefix_embeddings.shape[-2], prefix_embeddings.shape[-1]
        inputs = prefix_embeddings
        if self.prefix_rank:
            if prefix_rank_ids is None:
                raise RuntimeError("prefix_rank=True requires prefix_rank_ids")
            inputs = inputs + self.rank_emb(prefix_rank_ids).to(inputs.dtype)
        flat = inputs.reshape(-1, horizon, width).to(self.gru.weight_ih_l0.dtype)
        states, _ = self.gru(flat)
        return states.reshape(*prefix_embeddings.shape[:-1], self.state_dim)

    # -------------------------------------------------------------- corrector

    def hidden_delta(
        self,
        hidden_states: torch.Tensor,
        context: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """``[..., hidden]`` additive correction in the target's hidden space."""
        param_dtype = self.dh_in.weight.dtype
        parts = [context.to(param_dtype), state.to(param_dtype)]
        if self.direct_hidden:
            parts.insert(0, hidden_states.to(param_dtype))
        return self.delta_mlp(torch.cat(parts, dim=-1))

    def corrected_hidden(
        self,
        hidden_states: torch.Tensor,
        context: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        param_dtype = self.dh_in.weight.dtype
        return hidden_states.to(param_dtype) + self.hidden_delta(
            hidden_states, context, state
        )

    def full_vocab_logits(
        self,
        hidden_states: torch.Tensor,
        context: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """``[..., vocab]`` corrected logits. Training-only: dense CE needs every class."""
        return F.linear(
            self.corrected_hidden(hidden_states, context, state), self._embedding()
        )

    def score(
        self,
        hidden_states: torch.Tensor,
        context: torch.Tensor,
        state: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """``[..., K]`` corrected scores restricted to the lattice.

        Equal to ``full_vocab_logits`` gathered at the candidate ids, minus the base normaliser, so
        the value is anchored on ``log_probs`` and reduces to plain DFlash when ``delta == 0``.
        Computed by contraction against K vectors rather than the full readout, which is what keeps
        serving cheap.
        """
        param_dtype = self.dh_in.weight.dtype
        delta = self.hidden_delta(hidden_states, context, state)
        correction = (candidate_embeddings.to(param_dtype) * delta.unsqueeze(-2)).sum(-1)
        return log_probs.float() + correction.float()

    def err_logits(
        self,
        context: torch.Tensor,
        state: torch.Tensor,
        log_probs: torch.Tensor,
        scalars: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Logit of ``P(base top-1 is wrong at this slot)``. ``[..., H]``.

        Argument order matches the reference selector's ``err_logits(r, S, lp, scal)`` so the
        external six-domain decoder can call it positionally without a shim; ``hidden_states`` is
        only read when ``direct_hidden`` is on.
        """
        if self.direct_hidden and hidden_states is None:
            raise RuntimeError("direct_hidden=True requires hidden_states in err_logits")
        param_dtype = self.dh_in.weight.dtype
        log_probs = log_probs.to(param_dtype)
        margin = (log_probs[..., 0] - log_probs[..., 1]).unsqueeze(-1)
        parts = [
            context.to(param_dtype),
            state.to(param_dtype),
            log_probs[..., :1],
            margin,
            scalars[..., :1].to(param_dtype),
            scalars[..., 2:3].to(param_dtype),
        ]
        if self.direct_hidden:
            parts.insert(0, hidden_states.to(param_dtype))
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
        """Training-host entry point, matching the DFlash2 selector contract.

        ``return_full`` additionally returns the ``[..., vocab]`` corrected readout so the objective
        can apply dense cross-entropy; it is never requested at serving time.
        """
        embedding = self._embedding()
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
        state = self.causal_states(
            prefix_embeddings,
            self.prefix_ranks(candidate_ids, predecessor_ids) if self.prefix_rank else None,
        )
        scores = self.score(
            hidden_states, context, state, candidate_embeddings, unary_logits
        )
        if not (return_err or return_full):
            return scores
        out = [scores]
        if return_err:
            out.append(
                self.err_logits(
                    context, state, unary_logits, lattice_scalars, hidden_states
                )
            )
        if return_full:
            out.append(self.full_vocab_logits(hidden_states, context, state))
        return tuple(out)

    def forward(self, **kwargs):
        """Alias for :meth:`score_candidates` so DDP registers every parameter."""
        return self.score_candidates(**kwargs)

    # ------------------------------------------------------------- decisions

    keep_repair_log_probs = staticmethod(LatticePathSelector.keep_repair_log_probs)
    select_margin_gate = staticmethod(LatticePathSelector.select_margin_gate)
    select_keep_repair = classmethod(LatticePathSelector.select_keep_repair.__func__)


def _restore_zero_init_contract(self: LatticeCorrector) -> None:
    """Re-assert the zero-init the module's init-time no-op depends on.

    ``DFlashDraftModel.__init__`` runs ``post_init()`` after ``_init_draft_head``, and ``post_init``
    re-applies ``_init_weights`` to every submodule, so zeroing inside ``__init__`` alone does not
    survive.  The write goes through ``torch.nn.init`` on purpose: while HF runs ``_init_weights`` it
    swaps those primitives for guarded versions that skip tensors the loader already filled from a
    checkpoint, whereas a raw in-place ``zero_()`` is invisible to the guard and would silently
    destroy a warm start after ``from_pretrained`` re-runs init.
    """
    nn.init.zeros_(self.delta_mlp[-1].weight)


LatticeCorrector.restore_zero_init_contract = _restore_zero_init_contract


@register_draft
class PSPRv2DraftModel(DFlashDraftModel):
    """DFlash backbone with a lattice-conditioned hidden-state corrector."""

    warm_start_optional_prefixes = ("candidate_selector.",)

    def _init_draft_head(self, config, dflash_config: dict) -> None:
        self.candidate_selector = LatticeCorrector(
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            top_k=int(dflash_config.get("selector_top_k", 16)),
            d=int(dflash_config.get("selector_dim", 512)),
            n_layers=int(dflash_config.get("selector_layers", 3)),
            n_heads=int(dflash_config.get("selector_heads", 8)),
            state_dim=int(dflash_config.get("selector_state_dim", 512)),
            delta_hidden=int(dflash_config.get("selector_delta_hidden", 2048)),
            max_slots=int(dflash_config.get("selector_max_slots", 32)),
            dropout=float(dflash_config.get("selector_dropout", 0.1)),
            cross_slot=str(dflash_config.get("selector_cross_slot", "full")),
            direct_hidden=bool(dflash_config.get("selector_direct_hidden", True)),
            cand_slots=int(dflash_config.get("selector_cand_slots", 4)),
            prefix_rank=bool(dflash_config.get("selector_prefix_rank", False)),
        )
        self.freeze_backbone = bool(dflash_config.get("freeze_backbone", False))
        self.selector_compute_dtype = str(
            dflash_config.get("selector_compute_dtype", "model")
        )
        if self.selector_compute_dtype not in {"model", "float32"}:
            raise ValueError(
                "selector_compute_dtype must be 'model' or 'float32', got "
                f"{self.selector_compute_dtype!r}"
            )
        # Serving decision policy.  ``full_vocab`` takes the argmax of the corrected readout over the
        # whole vocabulary; ``margin_gate``/``keep_repair`` keep the lattice restriction.  The three
        # are reported separately because the restriction is worth a measured 8.4% of first errors.
        self.selector_decision_mode = str(
            dflash_config.get("selector_decision_mode", "full_vocab")
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
        if isinstance(module, LatticeCorrector):
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
        """Freeze everything except the selector; returns the number of frozen parameters."""
        if not self.freeze_backbone:
            return 0
        frozen = 0
        for name, parameter in self.named_parameters():
            if not name.startswith("candidate_selector."):
                parameter.requires_grad_(False)
                frozen += parameter.numel()
        return frozen

    def _sample_draft_tokens(
        self,
        target: nn.Module,
        draft_hidden: torch.Tensor,
        block_output_ids: torch.LongTensor,
    ) -> torch.LongTensor:
        """Native fixed-lattice, left-to-right walk, identical in semantics to the external decoder.

        The encoder pass is done once for the block (it is decode-legal: every input comes from the
        single all-MASK forward), then the GRU is stepped over the tokens this selector itself
        commits -- no target information beyond the already committed anchor.
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
        context = selector.encode(
            hidden, candidate_embeddings, unary_logits, lattice_scalars
        )

        predecessor_ids = block_output_ids[:, 0]
        gru_hidden = None
        path = []
        # top_k + 1 is the anchor bucket: slot 0's predecessor has no candidate list.  Afterwards the
        # rank is the index this walk actually committed at the previous slot, which is exactly the
        # teacher-forced quantity because teacher forcing commits the traced token.
        rank_ids = torch.full_like(predecessor_ids, selector.top_k + 1)
        for position in range(candidate_ids.shape[1]):
            prefix_embeddings = F.embedding(predecessor_ids, embedding)
            if selector.prefix_rank:
                prefix_embeddings = prefix_embeddings + selector.rank_emb(rank_ids).to(
                    prefix_embeddings.dtype
                )
            state, gru_hidden = selector.gru(
                prefix_embeddings.unsqueeze(1).to(selector.gru.weight_ih_l0.dtype),
                gru_hidden,
            )
            if self.selector_decision_mode == "full_vocab":
                logits = selector.full_vocab_logits(
                    hidden[:, position], context[:, position], state[:, 0]
                )
                predecessor_ids = logits.argmax(dim=-1)
                hit = candidate_ids[:, position].eq(predecessor_ids.unsqueeze(-1))
                rank_ids = torch.where(
                    hit.any(-1), hit.long().argmax(-1), torch.full_like(rank_ids, selector.top_k)
                )
                path.append(predecessor_ids)
                continue
            scores = selector.score(
                hidden[:, position],
                context[:, position],
                state[:, 0],
                candidate_embeddings[:, position],
                unary_logits[:, position],
            )
            err_logits = None
            if self.selector_decision_mode == "keep_repair" or self.selector_gate_theta > 0:
                err_logits = selector.err_logits(
                    context[:, position : position + 1],
                    state,
                    unary_logits[:, position : position + 1],
                    lattice_scalars[:, position : position + 1],
                    hidden[:, position : position + 1],
                )[:, 0]
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
            rank_ids = selected_index[:, 0]
            predecessor_ids = candidate_ids[:, position].gather(1, selected_index)[:, 0]
            path.append(predecessor_ids)
        return torch.stack(path, dim=1)


__all__ = ["LatticeCorrector", "PSPRv2DraftModel"]
