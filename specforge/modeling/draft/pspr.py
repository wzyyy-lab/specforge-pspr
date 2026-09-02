# coding=utf-8
"""PSPR: a bidirectional path selector over the draft block's top-K candidate lattice.

Positioning against what already exists in this codebase, since both neighbours are close:

``DominoDraftModel``
    Corrects each slot's full-vocab logits from ``[z_n ; causal_GRU(committed prefix)_n]``. The head
    at slot n never sees any other slot's candidate set. Decision rule is per-slot argmax.

``DFlash2DraftModel.CandidateSelector``
    Already does "top-K lattice + path selection", but ``score_candidates`` is a *strictly local*
    low-rank bigram, ``unary + <predecessor . W h, successor>``, with no cross-slot term, and
    ``greedy_path`` is a plain left-to-right walk.

This module keeps the top-K formulation and the causal committed-prefix state, and adds the piece
neither has: a **bidirectional transformer over the whole block's lattice**. Every slot's summary
attends to every other slot's candidate set, log-probs and uncertainty statistics before any token
is committed. That is decode-legal because the lattice comes from one parallel all-MASK pass and is
fixed for the whole block, unlike committed tokens.

Why the lattice is not redundant with the backbone's own bidirectionality: the backbone does mix
slots (its within-block attention is unrestricted), but it mixes *hidden states*. The lattice is the
``lm_head`` readout - a ``[vocab, hidden]`` projection followed by a top-K argsort - which the
5-layer backbone never computes. And the consumer here is the head, which in Domino sees only its
own slot.

Faithfully ported from the frozen-DFlash configuration that measured macro 5.705 (vs 5.594 for the
official Domino head on the same frozen backbone and 5.284 for plain one-shot). All 27.06M
parameters of that reference module are reproduced component for component, including the ``err_head``
frontier detector and its auxiliary BCE objective.

Three optional branches of the reference module are *not* ported, each because it was separately
measured as a loser against this configuration on the same frozen backbone and data: ``tok_rank``
(a learned per-token table, +0.315 vs +0.440), ``pair_dim`` (LiLiCorr-style bigram compatibility,
+0.388) and ``thidden_dim`` (feeding the target's anchor context to the encoder).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .dflash import DFlashDraftModel
from .registry import register_draft


class LatticePathSelector(nn.Module):
    """Scores a slot's top-K candidates using the whole block's lattice plus the committed prefix."""

    # The host reads this to decide whether to add the frontier detector's BCE term. It does NOT
    # read a `wants_full_logits` flag -- full logits reach this selector because it implements
    # `extract_lattice`, which the host prefers over a bare `topk` when present.
    wants_err_objective = True

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
        output_zero_init: bool = False,
        trans_rank: int = 0,
    ) -> None:
        super().__init__()
        if top_k < 2:
            raise ValueError("PSPR needs top_k >= 2 so a non-top-1 candidate can win")
        self.top_k = int(top_k)
        self.d = int(d)
        self.state_dim = int(state_dim)
        self.n_scalars = int(n_scalars)

        self.dh_in = nn.Linear(hidden_size, d)
        self.candset_in = nn.Linear(hidden_size, d)
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
        self.s_proj = nn.Linear(self.state_dim, d)

        self.cand_proj = nn.Linear(hidden_size, d)
        self.mlp = nn.Sequential(
            nn.LayerNorm(3 * d + 1),
            nn.Linear(3 * d + 1, d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d, 1),
        )
        # 1D, not scalar: FSDP rejects zero-dim parameters. Still a single learnable number.
        self.gamma = nn.Parameter(torch.zeros(1))

        # Dedicated binary "is the base top-1 wrong at this slot?" detector.
        #
        # Acceptance is decided at the frontier, and the two error directions are not symmetric:
        # recovering the first wrong slot extends the accepted run, while breaking an already-correct
        # slot truncates it, and correct slots outnumber wrong ones about 4.5:1. A margin read off
        # the K-way softmax is a poor confidence estimate for that decision, so detection gets its
        # own head, supervised only on frontier-reachable slots. The decode gate can threshold this
        # probability instead of the softmax margin.
        #
        # Declared BEFORE ``delta_mlp`` to match the reference ``LatticeSelector.__init__`` order:
        # module construction draws from the global RNG, so the declaration order fixes the initial
        # weights. ``scripts/gate_accept_selector_init.py`` checks the two are bit-identical.
        err_in = d + self.state_dim + 4
        self.err_head = nn.Sequential(
            nn.LayerNorm(err_in),
            nn.Linear(err_in, d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d, 1),
        )

        # Built in the reference's submodule order (input Linear before output Linear): construction
        # draws from the global RNG, so any reordering changes the initial weights.
        self.delta_mlp = nn.Sequential(
            nn.LayerNorm(d + self.state_dim),
            nn.Linear(d + self.state_dim, delta_hidden),
            nn.GELU(),
            nn.Linear(delta_hidden, hidden_size, bias=False),
        )
        # Zero-init output => exact no-op at initialisation (base-anchored, like the reference).
        self.delta_mlp[-1]._pspr_zero_init = True
        nn.init.zeros_(self.delta_mlp[-1].weight)

        # ``output_zero_init`` moves the no-op guarantee from the gamma SCALAR to the correction
        # heads' OUTPUT layers, which is how official DFlash2's CandidateSelector does it.
        #
        # Default (reference dh2048) behaviour: gamma = 0 and ``mlp[-1]`` randomly initialised.
        # ``score = lp + gamma * correction``, so at init every one of the ~27M correction
        # parameters has gradient ``gamma * (...) = 0``.  Only gamma itself can move, and the whole
        # network's learning rate is gated by how fast that ONE scalar grows -- which is why gamma
        # needs a 5x learning rate to be trainable at all, and why the observed trajectory crawls
        # (gamma 0.32 @ep1 -> 0.55 @ep2 -> 0.98 @ep3, where cal peaks).
        #
        # DFlash2 zero-inits only ONE SIDE of its bilinear transition instead:
        #   "The transition must start as a no-op ... The successor factor learns first; the other
        #    bilinear factors receive signal once it becomes non-zero."
        # so the zeroed tensor's own gradient (``pred * W h``) is non-zero and it trains immediately.
        #
        # Setting this flag reproduces that structure: gamma = 1 with zeroed output layers still
        # gives ``correction = 0`` at init (initial equivalence to DFlash top-1 is preserved, so any
        # gain remains attributable), but ``mlp[-1]`` and ``delta_mlp[-1]`` now receive gradient on
        # step 1 instead of waiting for a scalar to grow.  Forward algebra is untouched, so decode
        # and every exported checkpoint stay compatible.
        self.output_zero_init = bool(output_zero_init)
        if self.output_zero_init:
            self.mlp[-1]._pspr_zero_init = True
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)
            with torch.no_grad():
                self.gamma.fill_(1.0)

        # Official DFlash2's ENTIRE selector is this one term (CandidateSelector.score_candidates):
        #
        #     score_k = unary_k + < successor_codebook[cand_k],  predecessor_codebook[prev] * W h >
        #
        # Two properties distinguish it from anything this port or the reference already had:
        #
        #  1. BOTH sides are dedicated, learned, per-token tables, trained specifically to answer
        #     "does token k follow token prev here?".  This port reads candidate AND prefix vectors
        #     out of the FROZEN target embedding table, i.e. representations optimised for language
        #     modelling, never for transition scoring.
        #  2. The predecessor vector GATES the hidden state elementwise before the inner product, so
        #     the form is a genuine rank-r trilinear (CP) interaction between prefix identity,
        #     context, and candidate identity.
        #
        # The reference's two ablations are both strict degenerations of this and neither one tests
        # it: ``tok_rank`` has a successor table but no predecessor table (the prefix reaches it only
        # through the GRU's dense state, whose input is the frozen embedding), and ``pair_dim``
        # (LiLiCorr-style) builds both sides from the frozen embedding with no learned per-token
        # tables at all.  Their negative results (+0.315 / +0.388 vs +0.440) therefore say nothing
        # about the official form.
        #
        # Kept OUTSIDE the gamma product on purpose: gamma starts at zero in the reference recipe, and
        # anything multiplied by it is frozen at step 0.  Official DFlash2 gets its no-op from
        # zero-initialising the successor table alone, which leaves that table's own gradient
        # (``pred * W h``) intact -- "the successor factor learns first".  Reproducing that requires
        # the term to bypass gamma.
        self.trans_rank = int(trans_rank)
        if self.trans_rank > 0:
            self.predecessor_codebook = nn.Parameter(
                torch.empty(int(vocab_size), self.trans_rank)
            )
            self.successor_codebook = nn.Parameter(
                torch.empty(int(vocab_size), self.trans_rank)
            )
            self.trans_proj = nn.Linear(hidden_size, self.trans_rank, bias=False)
            nn.init.normal_(self.predecessor_codebook, std=0.02)
            self.successor_codebook._pspr_zero_init = True
            nn.init.zeros_(self.successor_codebook)
        self.register_buffer(
            "target_embedding", torch.empty(0, dtype=torch.float32), persistent=False
        )
        self._reference_config = {
            "hidden_dim": int(hidden_size),
            "vocab_embed_dim": int(hidden_size),
            "K": self.top_k,
            "d": self.d,
            "n_layers": int(n_layers),
            "n_heads": int(n_heads),
            "ds": self.state_dim,
            "max_slots": int(max_slots),
            "n_scal": self.n_scalars,
            "dropout": float(dropout),
            "tok_rank": 0,
            "vocab_size": int(vocab_size),
            "pair_dim": 0,
            "delta_h": int(delta_hidden),
            "thidden_dim": 0,
        }
        # Only recorded when enabled: dh2048's config has no such key, and the equivalence gate
        # asserts this port's config matches it key for key.
        if self.trans_rank > 0:
            self._reference_config["trans_rank"] = self.trans_rank

    def reference_config(self) -> dict:
        """This module's shape in the reference ``LatticeSelector``'s own config vocabulary.

        Checkpoints written with this config load into the reference class -- and therefore into the
        reference decoder that produced every baseline number -- without a conversion step. The three
        zeroed entries are reference-only options this port does not implement (``tok_rank``,
        ``pair_dim``, ``thidden_dim``); the reference constructor treats 0 as "absent", and the
        equivalence gate loads dh2048, whose config has exactly these zeros, with ``strict=True``.
        """
        return dict(self._reference_config)

    def bind_target_embedding(self, embed_tokens: nn.Module) -> None:
        """Attach the target's input embedding, used for candidate vectors and the delta decode.

        Non-persistent: a ``[vocab, hidden]`` matrix must not enter the draft checkpoint, and it is
        available wherever the DFlash family runs.
        """
        weight = getattr(embed_tokens, "weight", embed_tokens)
        self.target_embedding = weight.detach()

    def _embedding(self, dtype: torch.dtype) -> torch.Tensor:
        weight = self.target_embedding
        if weight.numel() == 0:
            raise RuntimeError(
                "LatticePathSelector needs bind_target_embedding() before the first forward"
            )
        return weight.to(dtype=dtype)

    def extract_lattice(
        self, logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(top_log_probs, candidate_ids, scalars)`` for ``logits[..., vocab]``.

        Candidate 0 is forced to be the ``argmax`` token. ``lm_head`` runs in bf16, so the maximal
        logit is exactly tied at a few percent of slots, and ``topk`` breaks such ties differently
        from ``argmax``. Greedy drafting uses ``argmax``, so without this swap candidate 0 is not the
        token the backbone would have drafted and the selector's "keep candidate 0" branch can no
        longer reproduce plain one-shot decoding, which is what makes acceptance structurally
        non-decreasing. Tied entries are numerically equal, so the swap changes no statistic.
        """
        float_logits = logits.float()
        top_logits, candidate_ids = torch.topk(float_logits, self.top_k, dim=-1)

        argmax_ids = float_logits.argmax(dim=-1, keepdim=True)
        matches = candidate_ids.eq(argmax_ids)
        found = matches.any(dim=-1, keepdim=True)
        position = torch.where(
            found, matches.long().argmax(dim=-1, keepdim=True), torch.zeros_like(argmax_ids)
        )
        if bool((position != 0).any()):
            gathered_ids = candidate_ids.gather(-1, position)
            gathered_logits = top_logits.gather(-1, position)
            first_ids = candidate_ids[..., :1]
            first_logits = top_logits[..., :1]
            candidate_ids = candidate_ids.scatter(-1, position, first_ids)
            top_logits = top_logits.scatter(-1, position, first_logits)
            candidate_ids = torch.cat([gathered_ids, candidate_ids[..., 1:]], dim=-1)
            top_logits = torch.cat([gathered_logits, top_logits[..., 1:]], dim=-1)

        log_z = torch.logsumexp(float_logits, dim=-1, keepdim=True)
        top_log_probs = top_logits - log_z
        entropy = -(
            torch.softmax(float_logits, dim=-1) * (float_logits - log_z)
        ).sum(dim=-1)
        margin = (top_log_probs[..., 0] - top_log_probs[..., 1]).clamp(-20.0, 20.0)
        mass = top_log_probs.exp().sum(dim=-1).clamp(max=1.0)
        scalars = torch.stack([entropy, margin, mass], dim=-1)
        # Deliberately float32, not the model dtype. The scores are a decision over 16 candidates
        # whose base log-probs sit around -10; in bf16 the spacing at that magnitude is ~0.05, so a
        # correction of order 1e-2 would round away entirely and the selector could never overrule
        # candidate 0. The frozen-DFlash reference implementation ran this whole head in float32.
        return top_log_probs, candidate_ids, scalars

    def encode(
        self,
        hidden_states: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        log_probs: torch.Tensor,
        scalars: torch.Tensor,
    ) -> torch.Tensor:
        """Bidirectional pass over one block's slots. Shapes ``[..., H, *]`` -> ``[..., H, d]``."""
        param_dtype = self.dh_in.weight.dtype
        weights = torch.softmax(log_probs.float(), dim=-1).to(candidate_embeddings.dtype)
        centroid = (weights.unsqueeze(-1) * candidate_embeddings).sum(dim=-2)
        slot = (
            self.dh_in(hidden_states.to(param_dtype))
            + self.candset_in(centroid.to(param_dtype))
            + self.scal_in(scalars.to(param_dtype))
        )
        horizon = slot.shape[-2]
        slot = self.slot_ln(slot) + self.pos_emb[:horizon]
        flat = slot.reshape(-1, horizon, self.d)
        return self.encoder(flat).reshape(slot.shape)

    def causal_states(self, prefix_embeddings: torch.Tensor) -> torch.Tensor:
        """GRU over the committed prefix. ``[..., H, E]`` -> ``[..., H, state_dim]``."""
        horizon, width = prefix_embeddings.shape[-2], prefix_embeddings.shape[-1]
        flat = prefix_embeddings.reshape(-1, horizon, width).to(self.s_proj.weight.dtype)
        states, _ = self.gru(flat)
        return states.reshape(*prefix_embeddings.shape[:-1], self.state_dim)

    def delta_term(
        self,
        context: torch.Tensor,
        state: torch.Tensor,
        candidate_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Correction expressed in the target's hidden space, decoded by its own embedding."""
        delta = self.delta_mlp(torch.cat([context, state], dim=-1))
        return (candidate_embeddings * delta.unsqueeze(-2)).sum(-1)

    def restore_zero_init_contract(self) -> None:
        """Re-assert every zero-init the module's init-time no-op depends on.

        ``DFlashDraftModel.__init__`` runs ``post_init()`` after ``_init_draft_head``, and
        ``post_init`` re-applies ``_init_weights`` to every submodule.  Tagging a Linear with
        ``_pspr_zero_init`` covers its ``weight`` but not its ``bias``, and bare ``nn.Parameter``
        tensors (``gamma``, the codebooks) are not submodules at all, so nothing guarantees they
        survive.  Everything the "identical to DFlash top-1 at step 0" property rests on is
        re-established here instead of being set once in ``__init__``.

        Every write goes through ``torch.nn.init``, never a raw in-place tensor op.  While HF runs
        ``_init_weights`` it swaps the ``nn.init`` primitives for guarded versions (see
        ``transformers.initialization.guard_torch_init_functions``) that skip any tensor the loader
        already filled from a checkpoint and tagged ``_is_hf_initialized``.  A raw
        ``self.gamma.zero_()`` is invisible to that guard, and ``from_pretrained`` re-runs
        ``_init_weights`` AFTER loading the state dict -- so it silently destroys a warm start while
        the loader still reports ``missing=0``.
        """
        nn.init.zeros_(self.delta_mlp[-1].weight)
        if self.output_zero_init:
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)
            nn.init.ones_(self.gamma)
        else:
            nn.init.zeros_(self.gamma)
        if self.trans_rank > 0:
            nn.init.zeros_(self.successor_codebook)

    def transition_term(
        self,
        hidden_states: torch.Tensor,
        predecessor_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        """``[..., K]`` transition scores, transcribed from DFlash2 ``CandidateSelector``.

        ``hidden_states[..., E]``, ``predecessor_ids[...]``, ``candidate_ids[..., K]``.
        """
        predecessor = self.predecessor_codebook[predecessor_ids]
        successor = self.successor_codebook[candidate_ids]
        context = predecessor * self.trans_proj(
            hidden_states.to(self.trans_proj.weight.dtype)
        )
        return torch.einsum("...r,...kr->...k", context, successor)

    def score(
        self,
        context: torch.Tensor,
        state: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        log_probs: torch.Tensor,
        *,
        hidden_states: Optional[torch.Tensor] = None,
        predecessor_ids: Optional[torch.Tensor] = None,
        candidate_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``[..., K]`` candidate scores, anchored on the base log-probs (gamma is zero-init)."""
        param_dtype = self.cand_proj.weight.dtype
        k = candidate_embeddings.shape[-2]
        context_k = context.unsqueeze(-2).expand(*context.shape[:-1], k, self.d)
        state_k = self.s_proj(state).unsqueeze(-2).expand(*state.shape[:-1], k, self.d)
        features = torch.cat(
            [
                context_k,
                state_k,
                self.cand_proj(candidate_embeddings.to(param_dtype)),
                log_probs.unsqueeze(-1).to(param_dtype),
            ],
            dim=-1,
        )
        correction = self.mlp(features).squeeze(-1) + self.delta_term(
            context, state, candidate_embeddings.to(param_dtype)
        )
        # float32 accumulation: see the note in extract_lattice.
        scores = log_probs.float() + self.gamma.float() * correction.float()
        if self.trans_rank > 0:
            scores = scores + self.transition_term(
                hidden_states, predecessor_ids, candidate_ids
            ).float()
        return scores

    def err_logits(
        self,
        context: torch.Tensor,
        state: torch.Tensor,
        log_probs: torch.Tensor,
        scalars: torch.Tensor,
    ) -> torch.Tensor:
        """Logit of ``P(base top-1 is wrong)`` per slot. ``[..., H]``.

        Reads the same lattice context the scorer uses, plus the four scalar confidence channels the
        reference implementation selected: the top-1 log-prob, the top1-top2 margin, the full-vocab
        entropy and the top-K mass.
        """
        param_dtype = self.s_proj.weight.dtype
        log_probs = log_probs.to(param_dtype)
        margin = (log_probs[..., 0] - log_probs[..., 1]).unsqueeze(-1)
        features = torch.cat(
            [
                context.to(param_dtype),
                state.to(param_dtype),
                log_probs[..., :1],
                margin,
                scalars[..., :1].to(param_dtype),
                scalars[..., 2:3].to(param_dtype),
            ],
            dim=-1,
        )
        return self.err_head(features).squeeze(-1).float()

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
        """Training-host entry point, matching DFlash2's selector contract.

        ``lattice_scalars`` is supplied by the objective when it routes lattice extraction through
        ``extract_lattice``; without it the uncertainty features are recovered from the top-K
        log-probs alone and the full-vocab entropy channel is zero.

        With ``return_err`` the frontier error detector is returned alongside the candidate scores,
        computed from the same encoder and GRU pass so the auxiliary objective costs no extra
        forward work.
        """
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

    def forward(self, **kwargs):
        """Alias for :meth:`score_candidates`.

        Selector-only training wraps this module in DDP directly, and DDP only hooks gradients for
        parameters touched inside ``forward``.
        """
        return self.score_candidates(**kwargs)


@register_draft
class PSPRDraftModel(DFlashDraftModel):
    """DFlash backbone with a bidirectional lattice path selector."""

    warm_start_optional_prefixes = ("candidate_selector.",)

    def _init_draft_head(self, config, dflash_config: dict) -> None:
        self.candidate_selector = LatticePathSelector(
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
            output_zero_init=bool(dflash_config.get("selector_output_zero_init", False)),
            trans_rank=int(dflash_config.get("selector_trans_rank", 0)),
        )
        self.freeze_backbone = bool(dflash_config.get("freeze_backbone", False))

    def _init_weights(self, module) -> None:
        """Keep the delta output at zero through HF's own init path.

        ``DFlashDraftModel.__init__`` runs ``post_init()`` after ``_init_draft_head``, and
        ``post_init`` re-applies ``_init_weights`` to every submodule, so zeroing a tensor inside
        ``_init_draft_head`` is silently overwritten.
        """
        super()._init_weights(module)
        if getattr(module, "_pspr_zero_init", False):
            nn.init.zeros_(module.weight)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, LatticePathSelector):
            module.restore_zero_init_contract()

    def transform_unary_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Identity: the lattice is read off the raw lm_head logits.

        The DFlash objective applies this whenever a candidate selector is present, because DFlash2
        rescales and softcaps its unary logits before serving. PSPR does neither, and the
        frozen-DFlash reference this port reproduces took the lattice straight from ``lm_head(h)``.
        Returning float32 matches the selector's float32 decision path.
        """
        return logits.float()

    def bind_target_decoder(self, embed_tokens: nn.Module) -> None:
        self.candidate_selector.bind_target_embedding(embed_tokens)

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


__all__ = ["LatticePathSelector", "PSPRDraftModel"]
