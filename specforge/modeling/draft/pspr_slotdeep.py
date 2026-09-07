"""SlotDeep: per-slot residual MLP, full-width hidden and causal-prefix correction.

Updated in place at the user's request (2026-09-05). The trained pspr/v2/cloze
implementations are untouched. The default constructor can still construct the
original untrained SlotDeep control; the training JSON explicitly enables:

* concat anchor/query fusion, rather than an irreversible sum before the tower;
* a small candidate-conditioned nonlinear residual, jointly trained with the
  WHOLE selector. It reads full-width h, z, committed-prefix S, candidate
  embedding and decode-available score features. No vocabulary matrix is learned.

The original tied-embedding delta readout is retained. Both correction outputs
start at zero, so step-zero selection is exactly the DFlash top-1. The added
candidate scorer is K-restricted; full-vocabulary serving is explicitly rejected
when it is enabled, rather than silently dropping its contribution.

Inference-time attention ablations motivate testing per-slot computation, but
do NOT establish a 91%/8.4% causal contribution decomposition or prove that
bidirectional information is useless. Accept-length gains remain experimental.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pspr_cloze import _ROLE_ANCHOR, _ROLE_QUERY, ClozeCorrector, PSPRClozeDraftModel
from .registry import register_draft


class _SlotBlock(nn.Module):
    """Pre-norm residual FFN.  Named sublayers, not ``nn.Sequential``, on purpose.

    ``Optimizer._parameter_groups`` selects weight decay with
    ``name.startswith(prefix) or f".{prefix}" in name`` (``specforge/optimizer.py:94``), and the
    cloze recipe's rule list is written in exactly that vocabulary (``linear1.weight``,
    ``err_head.1.weight``, ...).  An ``nn.Sequential`` here would produce
    ``slot_blocks.0.1.weight``, which can only be reached by a rule like ``1.weight`` -- and that
    rule also matches ``err_head.1.weight`` and ``err_state_head.1.weight``.  Distinct names make
    the rule list two lines long, independent of depth, and impossible to alias.
    """

    def __init__(self, d: int, ff: int, dropout: float) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, ff)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(ff, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(self.act(self.fc1(self.ln(x)))))


class SlotDeepCorrector(ClozeCorrector):
    """``ClozeCorrector`` whose ``z`` comes from a deep per-slot readout, not a dense encoder."""

    def __init__(self, *, slot_layers: int = 9, slot_hidden: int = 0,
                 anchor_fusion: str = "sum", candidate_rank_dim: int = 0,
                 candidate_query_hidden: int = 1024, **kwargs) -> None:
        if not kwargs.get("bidirectional", True):
            raise ValueError("SlotDeep has no attention; bidirectional=False is not an ablation")
        if anchor_fusion not in {"sum", "concat"}:
            raise ValueError("anchor_fusion must be sum or concat")
        if candidate_rank_dim < 0 or candidate_query_hidden < 1:
            raise ValueError("invalid candidate ranker dimensions")
        dropout = float(kwargs.get("dropout", 0.1))
        requested = int(kwargs.get("n_layers", 6))
        # The parent unconditionally builds the dense encoder.  Build the cheapest possible one and
        # drop it: constructing 6 layers only to delete them wastes ~19M of initialisation, and
        # leaving it attached would put 19M dead parameters into the optimizer state, the checkpoint
        # and every parameter-count gate.  ``n_layers`` is still recorded in ``reference_config``
        # below so the arm remains identifiable from a saved config.
        kwargs = dict(kwargs, n_layers=1)
        super().__init__(**kwargs)
        del self.encoder

        if slot_layers < 1:
            raise ValueError(f"selector_slot_layers must be >= 1, got {slot_layers}")
        d = self.d
        ff = int(slot_hidden) if int(slot_hidden) > 0 else 4 * d
        self.slot_layers = int(slot_layers)
        self.slot_hidden = ff
        # Pre-norm residual FFN stack.  Pre-norm because a 9-deep stack at initialiser_range 0.02
        # is otherwise poorly conditioned, and because it matches the encoder it replaces
        # (``norm_first=True``).  No zero-init on the block outputs: ``delta_w2`` already guarantees
        # the step-0 no-op, and zeroing every block's second matrix would leave the first matrices
        # with no gradient on step 0 for no benefit.
        self.slot_blocks = nn.ModuleList(
            _SlotBlock(d, ff, dropout) for _ in range(self.slot_layers)
        )
        # The parent's ``nn.TransformerEncoder`` carries no final norm, but it also never had to hand
        # a residual stream of this depth to three separately-normalised consumers.  One output norm
        # keeps ``z``'s scale independent of ``slot_layers`` so the depth sweep is not also a
        # learning-rate sweep.
        self.slot_out_ln = nn.LayerNorm(d)
        self.anchor_fusion = str(anchor_fusion)
        self.candidate_rank_dim = int(candidate_rank_dim)
        self.candidate_query_hidden = int(candidate_query_hidden)
        if self.anchor_fusion == "concat":
            self.anchor_fuse = nn.Linear(2 * d, d)
        if self.candidate_rank_dim:
            if not self.direct_hidden or not self.use_state:
                raise ValueError("candidate ranker requires full-width h and causal-prefix GRU")
            rank = self.candidate_rank_dim
            self.rank_query_in = nn.Linear(self.hidden_size + d + self.state_dim,
                                           self.candidate_query_hidden)
            self.rank_query_out = nn.Linear(self.candidate_query_hidden, rank)
            self.rank_candidate_ln = nn.LayerNorm(self.hidden_size)
            self.rank_candidate_in = nn.Linear(self.hidden_size, rank, bias=False)
            self.rank_features_in = nn.Linear(4, rank)
            self.rank_out = nn.Linear(rank, 1, bias=False)
            self.rank_out._pspr_zero_init = True
            nn.init.zeros_(self.rank_out.weight)
        self.supports_full_vocab = not bool(self.candidate_rank_dim)
        self._config = dict(
            self._config,
            model_type="SlotDeepCorrector",
            n_layers=requested,
            slot_layers=self.slot_layers,
            slot_hidden=self.slot_hidden,
            anchor_fusion=self.anchor_fusion,
            candidate_rank_dim=self.candidate_rank_dim,
            candidate_query_hidden=self.candidate_query_hidden,
        )

    @classmethod
    def from_reference_config(cls, c):
        # These knobs alter semantics even when shapes happen to match. Current
        # artifacts must record them; do not guess architecture at load time.
        return cls(
            hidden_size=c["hidden_dim"], vocab_size=c["vocab_size"], top_k=c["K"],
            d=c["d"], n_layers=c["n_layers"], n_heads=c["n_heads"], state_dim=c["ds"],
            delta_hidden=c["delta_h"], max_slots=c["max_slots"], dropout=c["dropout"],
            direct_hidden=c["direct_hidden"], use_state=c["use_state"],
            bidirectional=c["bidirectional"], err_use_state=c["err_use_state"],
            slot_layers=c["slot_layers"], slot_hidden=c["slot_hidden"],
            anchor_fusion=c["anchor_fusion"], candidate_rank_dim=c["candidate_rank_dim"],
            candidate_query_hidden=c["candidate_query_hidden"],
        )

    def restore_zero_init_contract(self):
        super().restore_zero_init_contract()
        if self.candidate_rank_dim:
            nn.init.zeros_(self.rank_out.weight)

    def cloze_states(
        self,
        hidden_states: torch.Tensor,
        candidate_ids: torch.Tensor,
        anchor_ids: torch.Tensor,
        log_probs: torch.Tensor,
        scalars: torch.Tensor,
    ) -> torch.Tensor:
        """``[..., H, d]`` one deep readout per proposal slot.

        Signature, output shape and call sites are the parent's, so training and the serving walk
        stay on one code path and ``candidate_ids`` keeps its role as the lattice: only column 0 is
        read, and only to be *excluded*.
        """
        if not self.bidirectional:
            raise ValueError("SlotDeep has no attention; a causal attention ablation is invalid")
        pattern = getattr(self, "attention_pattern", "bidirectional")
        if pattern != "bidirectional":
            # There is no attention here to mask.  A diagnostic that silently no-ops is worse than
            # one that fails: the causal/anchor_self arms would report "no effect" for a model that
            # never had cross-slot attention in the first place.
            raise ValueError(
                "SlotDeepCorrector has no cross-slot attention, so attention_pattern="
                f"{pattern!r} cannot be applied; it removes information that is already absent"
            )

        param_dtype = self.h_in.weight.dtype
        embedding = self._embedding()
        H = candidate_ids.shape[-2]
        if not 1 <= H <= self.max_slots:
            raise ValueError(f"proposal slots {H} outside [1, {self.max_slots}]")

        # Exactly the parent's query cell: the draft's own hidden state and its confidence at this
        # slot, plus the MASK marker that stands in for the withheld seed token.
        base = self.h_in(self.h_ln(hidden_states.to(param_dtype))) + self.conf_in(
            self.conf_ln(self.confidence(log_probs, scalars))
        )  # [..., H, d]
        # The anchor is the one committed token the block knows before any slot is decided.  In the
        # parent it was a separate sequence cell every row attended to; with no attention left it is
        # added into every slot instead, which is the same information at 1/16 of the positions.
        anchor = (
            self.tok_in(self.tok_ln(F.embedding(anchor_ids, embedding)))
            + self.anchor_feat.to(param_dtype)
            + self.role_emb.weight[_ROLE_ANCHOR].to(param_dtype)
            + self.pos_emb[0].to(param_dtype)
        ).unsqueeze(-2)  # [..., 1, d]

        if self.anchor_fusion == "concat":
            base = self.anchor_fuse(torch.cat([base, anchor.expand_as(base)], dim=-1))
        else:
            base = base + anchor
        cell = (
            base
            + self.mask_emb.to(param_dtype)
            + self.role_emb.weight[_ROLE_QUERY].to(param_dtype)
            + self.pos_emb[1 : H + 1].to(param_dtype)
        )
        z = self.cell_ln(cell)
        for block in self.slot_blocks:
            z = z + block(z)
        return self.slot_out_ln(z)

    def score(self, z, hidden_states, candidate_embeddings, log_probs, state=None):
        scores = super().score(z, hidden_states, candidate_embeddings, log_probs, state)
        if not self.candidate_rank_dim:
            return scores
        if state is None:
            raise ValueError("candidate ranker needs the committed-prefix state")
        dtype = self.rank_query_in.weight.dtype
        context = torch.cat([
            self.fuse_h_ln(hidden_states.to(dtype)), self.fuse_z_ln(z.to(dtype)),
            self.fuse_s_ln(state.to(dtype)),
        ], dim=-1)
        query = self.rank_query_out(F.silu(self.rank_query_in(context)))
        candidates = self.rank_candidate_in(self.rank_candidate_ln(candidate_embeddings.to(dtype)))
        # Detached score features avoid an auxiliary route for manipulating the
        # original readout merely to change its own input. The main scores and
        # h/z/S still receive joint gradients from the final multiclass CE.
        lp = log_probs.detach().to(dtype)
        gap = (scores.detach() - scores.detach()[..., :1]).to(dtype)
        is_base = torch.zeros_like(lp)
        is_base[..., 0] = 1
        features = torch.stack([
            lp.clamp(-40, 0) / 10,
            (lp - lp[..., :1]).clamp(-30, 0) / 10,
            gap.clamp(-30, 30) / 10,
            is_base,
        ], dim=-1)
        joint = query.unsqueeze(-2) + candidates + self.rank_features_in(features)
        residual = self.rank_out(F.gelu(joint)).squeeze(-1).float()
        # Centering does not change a softmax or score difference; it makes the
        # reference score's numerical contract explicit. Alternatives remain
        # permutation-equivariant, with candidate 0 the distinguished KEEP.
        return scores + (residual - residual[..., :1])

    def full_vocab_logits(self, z, hidden_states, state=None):
        if self.candidate_rank_dim:
            raise ValueError("SlotDeep candidate residual is top-K only; full_vocab would drop it")
        return super().full_vocab_logits(z, hidden_states, state)


@register_draft
class PSPRSlotDeepDraftModel(PSPRClozeDraftModel):
    """DFlash backbone with a per-slot deep seed refiner in place of the cloze encoder."""

    def _init_draft_head(self, config, dflash_config: dict) -> None:
        super()._init_draft_head(config, dflash_config)
        # Rebuild the selector rather than duplicating the parent's 30-line argument list: the parent
        # is the single place the knob names are defined, so a knob added there cannot be silently
        # dropped here.  The throwaway ClozeCorrector is discarded before ``post_init`` runs.
        parent = self.candidate_selector
        reference = parent.reference_config()
        self.candidate_selector = SlotDeepCorrector(
            hidden_size=int(reference["hidden_dim"]),
            vocab_size=int(reference["vocab_size"]),
            top_k=int(reference["K"]),
            d=int(reference["d"]),
            n_layers=int(reference["n_layers"]),
            n_heads=int(reference["n_heads"]),
            state_dim=int(reference["ds"]),
            delta_hidden=int(reference["delta_h"]),
            max_slots=int(reference["max_slots"]),
            dropout=float(reference["dropout"]),
            direct_hidden=bool(reference["direct_hidden"]),
            use_state=bool(reference["use_state"]),
            bidirectional=bool(reference["bidirectional"]),
            err_use_state=bool(reference["err_use_state"]),
            slot_layers=int(dflash_config.get("selector_slot_layers", 9)),
            slot_hidden=int(dflash_config.get("selector_slot_hidden", 0)),
            anchor_fusion=str(dflash_config.get("selector_anchor_fusion", "sum")),
            candidate_rank_dim=int(dflash_config.get("selector_candidate_rank_dim", 0)),
            candidate_query_hidden=int(dflash_config.get("selector_candidate_query_hidden", 1024)),
        )
        if self.candidate_selector.candidate_rank_dim and self.selector_decision_mode != "margin_gate":
            raise ValueError("candidate-conditioned SlotDeep validates margin_gate serving only")
        del parent
