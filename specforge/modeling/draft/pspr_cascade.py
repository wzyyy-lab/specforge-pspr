"""PSPR-cascade: the KEEP-vs-REPAIR gate is allowed to read the ranker's own corrected scores.

Why this exists
---------------
The champion (PSPR-cloze stage-1, macro 6.0498) decides KEEP-vs-REPAIR with a hand-set scalar rule at
serving: ``--gate-tau 0 --gate-rho 3 --gate-theta 0``, i.e. repair iff ``max_alt_prob > 3 * p0``.  The
decision-chain accounting on the held-out run says that rule, not the ranker, is where the recoveries
are lost.  Of 5789 slots whose target is a non-base candidate:

    alternative_rank_0   3870 = 66.86%   the target IS the ranker's best alternative
    selector_rank_0      2550 = 44.05%   the target also outranks the base score
    gate blocked          425 =  7.34%
    recovered            2125 = 36.71%   (2550 - 425 = 2125, exact)

The 22.81pp between 66.86% and 44.05% is entirely "the base score ranks ahead of the target".

What is and is not claimed (D1 was FLAWED as originally written)
---------------------------------------------------------------
* CORRECT: ``keep_repair_log_probs`` never compares against ``s0``.  It builds REPAIR from
  ``candidate_scores[..., 1:]`` alone (pspr.py:485-505), so repair happens iff ``e + log r_* > 0``.
  The 44.05% restriction is genuinely absent from the factorized rule.
* WRONG, RETRACTED: "the old rule is capped at 44.05% for any rho".  The old rule is
  ``p_alt > tau + rho * p0`` (pspr.py:535).  With ``tau=0`` and ``rho < 1`` an alternative can win
  while scoring BELOW candidate 0, and as ``rho -> 0`` it approaches always-take-best-alternative.
  The 44.05% cap holds only for ``rho >= 1``.  It applies to the DEPLOYED rho=3, not to the family.
* WRONG, RETRACTED: "66.86% is the cascade's ceiling".  66.86% is a conditional recovery rate measured
  ON the champion's trajectory.  Repairing a first error lengthens the block and exposes a different
  first error, which moves block boundaries and reachable GRU states, so no offline number bounds the
  deployed result.  This is why the 500-step run is a measurement and not a confirmation.

The honest argument for funding this arm: the cascade can move the recovery-vs-destruction Pareto
frontier without globally lowering rho, because it prices each slot individually instead of charging
every slot the same scalar margin.  Whether it does is exactly what the run measures.

Why the champion's detector cannot do this today
------------------------------------------------
``ClozeCorrector.err_logits`` receives ``z``, the seed embedding, ``confidence(log_probs, scalars)``
built from the BASE unary logits, optionally ``hidden_states`` and the GRU state -- but never the
corrected candidate scores that serving actually compares.  It is structurally blind to the quantity
its own decision is about.  This module feeds those scores in.

Structure of the gate logit
---------------------------
    e = anchor(scores) + base_gain * e_parent + score_head(features(scores))

* ``anchor(scores) = logsumexp(s_1:) - s_0 - log(rho_anchor)`` has NO parameters.  Substituting it
  into the factorized rule gives ``e + log r_* = s_max_alt - s_0 - log(rho)``, so repair happens iff
  ``s_max_alt - s_0 > log(rho)``, which is ``p_alt > rho * p0`` -- the champion rule at ``tau=0``,
  exactly, ties included (both sides are strict ``>``, and an exact tie leaves KEEP as ``argmax``'s
  first-index winner just as the champion's ``ok=False`` does).
* ``base_gain`` is a learnable scalar initialised to 0, and the readout of ``score_head`` is
  zero-initialised.  So at step 0 the gate IS the anchor and ``latrepair --repair-margin 0`` reproduces
  the champion's ``latgate --gate-rho 3`` decisions slot for slot.
* Initialising ``base_gain`` to 0 discards nothing: the champion deploys ``--gate-theta 0``, and
  ``decode_lattice.py`` only consults the detector under ``if ok and gate_theta > 0``, so the mature
  detector is not part of the champion's deployed policy at all.  The cascade starts strictly no worse
  and can learn to bring the detector back in through ``base_gain``.

Without the anchor, ``selector_decision_mode: keep_repair`` at step 0 would reproduce the historical
``latrepair m=0`` policy (macro 6.371 on the v2 lineage) rather than the champion -- that was the D4
FATAL.

Design constraints honoured here
--------------------------------
* ``pspr_cloze.py`` is not modified.  ``CascadeCorrector`` subclasses ``ClozeCorrector`` and
  ``PSPRCascadeDraftModel`` subclasses ``PSPRClozeDraftModel``.
* Features are DETACHED by default.  Under ``selector_train_scope: err_only`` the ranker is frozen
  anyway, but detaching makes it structural rather than dependent on a ``requires_grad`` flag.  Note
  the honest limit (D3): the cascade still calls the parent detector on ``z``, and ``z`` comes from the
  shared cloze encoder, so detaching the SCORES does not prove the gate can never perturb the ranker
  once the encoder is unfrozen.  For this arm the encoder is frozen, so the guarantee holds; joint
  training is a separate ablation with its own contract, not a flag flip.
* ``err_only_module_names`` is extended so the existing ``err_only`` freeze path trains the new head
  with no host change.
* Serving must supply ``scores``.  Both serving paths originally did not (D4 FATAL).  Native serving
  lives in ``PSPRClozeDraftModel._sample_draft_tokens``, which this module may not edit, so ``score``
  publishes a one-shot, identity-checked handoff that ``err_logits`` consumes; the external decoder
  passes ``scores`` explicitly.  A missing or mismatched handoff RAISES.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pspr_cloze import ClozeCorrector, PSPRClozeDraftModel
from .registry import register_draft

# Feature-set names.  D2 asked for the rank-aligned and full-score representations to be measured
# against the order-statistic summary on held-out data BEFORE spending GPU time, so the choice is a
# config knob and ``scripts/probe_cascade_features.py`` decides it offline.
#
# Every channel is either a difference against s0 or a shift-invariant functional of softmax(s).  That
# is deliberate: `s` is `log_probs + correction`, so it carries the base normaliser as an additive
# per-slot constant, and a head that could see it would be reading the base model's confidence scale
# through a side channel.  The first version included raw ``s0`` and so did exactly that.
#
# ``p0`` is NOT a channel anywhere: p0 = 1 / (1 + exp(LSE(s_1:) - s0)) is an exact function of the
# ``lse`` channel, and with a LayerNorm downstream an exact duplicate is not free -- it changes the
# normalisation of every other channel.
_FEATURE_SETS = ("margins", "margins_summary", "summary")


def n_score_features(top_k: int, feature_set: str) -> int:
    if feature_set == "margins":
        return top_k - 1
    if feature_set == "margins_summary":
        return top_k + 1
    if feature_set == "summary":
        return 5
    raise ValueError(
        f"unknown selector_err_score_features {feature_set!r}; expected one of {_FEATURE_SETS}"
    )


class CascadeCorrector(ClozeCorrector):
    """``ClozeCorrector`` whose keep/repair gate additionally reads the corrected scores."""

    def __init__(
        self,
        *,
        err_score_detach: bool = True,
        err_score_features: str = "margins_summary",
        err_score_enabled: bool = True,
        err_score_anchor_rho: float = 3.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if err_score_features not in _FEATURE_SETS:
            raise ValueError(
                f"selector_err_score_features must be one of {_FEATURE_SETS}, "
                f"got {err_score_features!r}"
            )
        if err_score_anchor_rho < 0.0:
            raise ValueError(
                f"selector_err_score_anchor_rho must be >= 0 (0 disables the anchor), "
                f"got {err_score_anchor_rho}"
            )
        self.err_score_detach = bool(err_score_detach)
        self.err_score_features = str(err_score_features)
        # The blind control arm (D5 in the change list) is this same class with the score head switched
        # off.  Keeping it in one class makes the control byte-identical everywhere else: same anchor,
        # same base_gain schedule, same objective, same freeze set, same data order.
        self.err_score_enabled = bool(err_score_enabled)
        self.err_score_anchor_rho = float(err_score_anchor_rho)
        self.n_score_features = n_score_features(self.top_k, self.err_score_features)

        d = self.d
        if self.err_score_enabled:
            self.err_score_ln = nn.LayerNorm(self.n_score_features)
            self.err_score_head = nn.Sequential(
                nn.Linear(self.n_score_features, d, bias=True),
                nn.SiLU(),
                nn.Linear(d, 1, bias=True),
            )
            self.err_score_head[-1]._pspr_zero_init = True
        # Multiplicative gain on the mature blind detector: 0 when the anchor supplies the step-0
        # policy, 1 when it does not (so the anchor-off configuration equals the plain parent gate plus
        # a zeroed readout, and the ablation is a clean one-variable change in both directions).
        #
        # A zero gain also zeroes ``d loss / d err_head``, so the mature detector would sit in a dead
        # region -- except that ``d loss / d gain = (d loss / d e) * e_parent`` is nonzero, so the gain
        # moves on the first update and the detector is live from step 2 onward.  This is the standard
        # zero-init residual gate, not a permanent deletion.
        #
        # It is a ``ParameterDict`` rather than a bare ``nn.Parameter`` because ``apply_backbone_freeze``
        # builds its train set from ``f"{module_name}."`` prefixes (pspr_cloze.py:727) and matches with
        # ``str.startswith``.  A bare parameter is named ``err_base_gain`` with no dot, would never
        # match ``err_base_gain.``, and would be silently FROZEN at zero -- which would delete the
        # mature detector from the model for good while every gate still passed.
        self.err_base_gain = nn.ParameterDict(
            {
                "gain": nn.Parameter(
                    torch.zeros(()) if self.err_score_anchor_rho > 0.0 else torch.ones(())
                )
            }
        )
        self.restore_zero_init_contract()

        names = list(self.err_only_module_names) + ["err_base_gain"]
        if self.err_score_enabled:
            names.extend(("err_score_ln", "err_score_head"))
        self.err_only_module_names = tuple(names)

        self._config["model_type"] = "CascadeCorrector"
        self._config.update(
            {
                "err_score_detach": self.err_score_detach,
                "err_score_features": self.err_score_features,
                "err_score_enabled": self.err_score_enabled,
                "err_score_anchor_rho": self.err_score_anchor_rho,
            }
        )
        self._published_scores: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    # --------------------------------------------------------------- contracts

    def restore_zero_init_contract(self) -> None:  # type: ignore[override]
        """Re-assert every init-time no-op, the parent's and the cascade's.

        ``DFlashDraftModel.__init__`` calls ``post_init()`` AFTER ``_init_draft_head``, and
        ``post_init`` re-applies ``_init_weights`` to every submodule, so zeroing in ``__init__`` alone
        does not survive.  The writes go through ``torch.nn.init`` for the reason the parent documents:
        while HF runs ``_init_weights`` it swaps those primitives for guarded versions that skip
        tensors the loader already filled, whereas a raw ``zero_()`` would silently destroy a warm
        start or a resume.
        """
        super().restore_zero_init_contract()
        if getattr(self, "err_score_enabled", False) and hasattr(self, "err_score_head"):
            nn.init.zeros_(self.err_score_head[-1].weight)
            nn.init.zeros_(self.err_score_head[-1].bias)
        if hasattr(self, "err_base_gain"):
            nn.init.constant_(
                self.err_base_gain["gain"],
                0.0 if self.err_score_anchor_rho > 0.0 else 1.0,
            )

    # ---------------------------------------------------------------- features

    def score_anchor(self, scores: torch.Tensor) -> Optional[torch.Tensor]:
        """``LSE(s_1:) - s_0 - log(rho)``: the parameter-free champion-equivalent gate logit.

        Returns ``None`` when the anchor is disabled.  Always computed from detached scores: it is a
        fixed policy statement about the ranker, never a gradient path into it.
        """
        if self.err_score_anchor_rho <= 0.0:
            return None
        s = scores.float().detach()
        return (
            torch.logsumexp(s[..., 1:], dim=-1)
            - s[..., 0]
            - math.log(self.err_score_anchor_rho)
        )

    def score_features(self, scores: torch.Tensor) -> torch.Tensor:
        """``[..., H, n_score_features]`` shift-invariant statistics of the corrected scores."""
        s = scores.float()
        if self.err_score_detach:
            s = s.detach()
        s0 = s[..., :1]
        alt = s[..., 1:]
        margins = alt - s0
        lse = torch.logsumexp(alt, dim=-1, keepdim=True) - s0
        probability = torch.softmax(s, dim=-1)
        entropy = -(probability * probability.clamp_min(1e-9).log()).sum(
            -1, keepdim=True
        )
        if self.err_score_features == "margins":
            # Complete and non-redundant: softmax(s) is invariant to a shift, so the K-1 margins are a
            # sufficient statistic for every decision quantity, `lse` and `entropy` included.
            return margins
        if self.err_score_features == "margins_summary":
            # Redundant BY CONSTRUCTION, and the redundancy is the point: `lse` is the exact
            # sufficient statistic of the keep/repair comparison and a one-hidden-layer MLP has to
            # spend capacity approximating log-sum-exp to recover it.  Supplying it is a basis change,
            # not new information.  Whether the basis pays for the LayerNorm perturbation it causes is
            # measured offline, not assumed.
            return torch.cat([margins, lse, entropy], dim=-1)
        # "summary": permutation-invariant order statistics only, for the cheapest hypothesis.
        k_alt = alt.shape[-1]
        top = alt.topk(min(3, k_alt), dim=-1).values
        if k_alt < 3:
            top = torch.cat(
                [top, top[..., -1:].expand(*top.shape[:-1], 3 - k_alt)], dim=-1
            )
        return torch.cat(
            [
                lse,
                entropy,
                top[..., 0:1] - s0,
                top[..., 0:1] - top[..., 1:2],
                top[..., 1:2] - top[..., 2:3],
            ],
            dim=-1,
        )

    # ------------------------------------------------------- serving handoff

    def score(  # type: ignore[override]
        self,
        z: torch.Tensor,
        hidden_states: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        log_probs: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Parent scoring, plus a one-shot publication of the result for the gate to consume.

        ``PSPRClozeDraftModel._sample_draft_tokens`` computes ``scores`` and then calls ``err_logits``
        with the same ``z``, and this module may not edit ``pspr_cloze.py``.  Re-implementing those 82
        lines here would create a second copy of the serving walk that silently drifts from the one the
        trainer validates, which is a worse failure mode than a handoff.  So the handoff is made
        auditable instead of implicit: it is keyed on tensor IDENTITY of ``z`` (not shape, not a hash),
        it is consumed exactly once, and every way it can be wrong raises.
        """
        scores = super().score(z, hidden_states, candidate_embeddings, log_probs, state)
        # ``detach()`` so a stale entry can never retain an autograd graph across steps.  Legitimate
        # consumers of the handoff are inference-only, and ``err_logits`` refuses to use it whenever
        # grad is enabled, so this detach is never on a training path.
        self._published_scores = (z, scores.detach())
        return scores

    def _consume_published_scores(
        self, z: torch.Tensor, published: Optional[Tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        if published is None:
            raise RuntimeError(
                "CascadeCorrector.err_logits got scores=None and no published scores. The gate is "
                "defined on the corrected scores; running it without them would silently degrade "
                "this arm to the champion's blind gate and report the result as a cascade number. "
                "Call score() on the same z first, or pass scores= explicitly."
            )
        published_z, scores = published
        if published_z is not z:
            raise RuntimeError(
                "CascadeCorrector.err_logits found published scores from a DIFFERENT z "
                f"(published id={id(published_z)}, requested id={id(z)}). Refusing to gate slot A "
                "with slot B's scores."
            )
        return scores

    def err_logits(  # type: ignore[override]
        self,
        z: torch.Tensor,
        seed_embeddings: torch.Tensor,
        log_probs: torch.Tensor,
        scalars: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
        scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pop the serving handoff unconditionally, even when ``scores`` is supplied: leaving an entry
        # behind would let a later call consume scores computed for a different slot.
        published = self._published_scores
        self._published_scores = None
        base = super().err_logits(
            z, seed_embeddings, log_probs, scalars, hidden_states, state
        )
        if scores is None:
            if torch.is_grad_enabled():
                raise RuntimeError(
                    "CascadeCorrector.err_logits requires scores= on any grad-enabled call. The "
                    "published-score handoff exists only for the inference walk; using it under "
                    "autograd would train the gate on detached scores without saying so."
                )
            scores = self._consume_published_scores(z, published)
        if scores.shape[:-1] != base.shape or scores.shape[-1] != self.top_k:
            raise RuntimeError(
                f"CascadeCorrector.err_logits scores shape {tuple(scores.shape)} is inconsistent "
                f"with the gate logits shape {tuple(base.shape)} and top_k={self.top_k}."
            )
        logits = self.err_base_gain["gain"].float() * base
        anchor = self.score_anchor(scores)
        if anchor is not None:
            logits = logits + anchor
        if self.err_score_enabled:
            param_dtype = self.h_in.weight.dtype
            features = self.err_score_ln(self.score_features(scores).to(param_dtype))
            logits = logits + self.err_score_head(features).squeeze(-1).float()
        return logits

    # ------------------------------------------------------------ host entry

    def score_candidates(  # type: ignore[override]
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
        """Identical to the parent up to passing ``scores`` into ``err_logits``.

        The body is duplicated rather than delegated because the parent computes ``scores`` and then
        immediately consumes ``z``/``state`` for the gate, with no seam to hook.  Duplication risks
        drifting from the parent, so ``scripts/gate_pspr_cascade.py`` asserts bit-identical ``scores``
        against ``ClozeCorrector.score_candidates`` on shared weights.
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
        scores = self.score(z, hidden_states, candidate_embeddings, unary_logits, state)
        # The training path always passes ``scores`` explicitly, so drop the serving handoff rather
        # than leave a stale entry that a later inference call could consume.
        self._published_scores = None
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
                    scores,
                )
            )
        if return_full:
            out.append(self.full_vocab_logits(z, hidden_states, state))
        return tuple(out)


@register_draft
class PSPRCascadeDraftModel(PSPRClozeDraftModel):
    """PSPR-cloze with the score-aware keep/repair gate."""

    @property
    def warm_start_optional_key_groups(self):
        """Parent groups UNION the cascade-only tensors, as one extra all-or-nothing group.

        Without this the warm start fails outright (D5 FATAL).
        ``warm_start_optional_prefixes`` already declares ``candidate_selector.`` optional, but the
        loader applies that exemption all-or-nothing per prefix and SKIPS it whenever the source holds
        any key under the prefix (training/model_loading.py:461).  The champion checkpoint does carry
        ``candidate_selector.*``, so the prefix exemption is disabled by design -- a checkpoint holding
        part of a head is a partially-applied warm start, not a fresh head -- and the cascade tensors
        would be reported as required-but-missing.

        ``warm_start_optional_key_groups`` is the intended escape: an exact key set exempt only when
        EVERY member is absent from the source, so a half-written head is still a hard error.  This is
        a PROPERTY over a parent CLASS ATTRIBUTE, so it must re-emit the parent's ``err_state_*`` group
        or a checkpoint that predates the state-conditioned residual would stop loading -- shadowing it
        was a regression introduced by the first fix and caught here.

        The cascade group is enumerated from the live modules rather than hardcoded, so renaming or
        resizing the head cannot desynchronise the declaration from the parameters it covers.
        """
        inherited = tuple(
            tuple(group)
            for group in PSPRClozeDraftModel.warm_start_optional_key_groups
        )
        selector = getattr(self, "candidate_selector", None)
        if selector is None:
            return inherited
        keys = ["candidate_selector.err_base_gain.gain"]
        for module_name in ("err_score_ln", "err_score_head"):
            module = getattr(selector, module_name, None)
            if module is None:
                continue
            keys.extend(
                f"candidate_selector.{module_name}.{name}"
                for name, _ in module.named_parameters()
            )
        return inherited + (tuple(keys),)

    def _init_draft_head(self, config, dflash_config: dict) -> None:
        super()._init_draft_head(config, dflash_config)
        # Rebuild the selector as the cascade variant.  Running the parent first keeps every default
        # and validation in one place (pspr_cloze.py stays untouched) at the cost of one discarded
        # module construction, which is negligible against a multi-hour run.
        reference = self.candidate_selector
        self.candidate_selector = CascadeCorrector(
            err_score_detach=bool(dflash_config.get("selector_err_score_detach", True)),
            err_score_features=str(
                dflash_config.get("selector_err_score_features", "margins_summary")
            ),
            err_score_enabled=bool(
                dflash_config.get("selector_err_score_enabled", True)
            ),
            err_score_anchor_rho=float(
                dflash_config.get("selector_err_score_anchor_rho", 3.0)
            ),
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            top_k=reference.top_k,
            d=reference.d,
            n_layers=int(dflash_config.get("selector_layers", 6)),
            n_heads=int(dflash_config.get("selector_heads", 8)),
            state_dim=reference.state_dim,
            delta_hidden=int(dflash_config.get("selector_delta_hidden", 2048)),
            max_slots=int(dflash_config.get("selector_max_slots", 32)),
            dropout=float(dflash_config.get("selector_dropout", 0.1)),
            direct_hidden=reference.direct_hidden,
            use_state=reference.use_state,
            bidirectional=bool(dflash_config.get("selector_bidirectional", True)),
            err_use_state=reference.err_use_state,
        )
