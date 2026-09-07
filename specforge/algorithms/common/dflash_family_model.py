# coding=utf-8
"""DFlash-family training models and shared masking helpers."""

import math
import os
from typing import Dict, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from specforge.core.chunking import checkpointed_chunk_reduce
from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.draft.flex_attention_backend import flex_attention_backend

try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask

    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    BlockMask = None
    create_block_mask = None

# NPU workaround: flex_attention is not available on Ascend NPU.
if hasattr(torch, "npu") and torch.npu.is_available():
    FLEX_ATTENTION_AVAILABLE = False

_VALID_LOSS_TYPES = {
    "dflash",
    "dpace",
    "dpace-cumulative-confidence-only",
    "dpace-continuation-value-only",
}
_DPACE_LOSS_TYPES = _VALID_LOSS_TYPES - {"dflash"}
_VALID_LK_LOSS_TYPES = {None, "alpha", "lambda", "tv"}
# How the selector's per-slot CE is weighted. See `selector_weight_mode` in the model's __init__.
_VALID_SELECTOR_WEIGHT_MODES = {
    "base_dpace",
    "uniform",
    "uniform_frontier_boost",
    "reachable_frontier_boost",
    "teacher_forced_frontier",
    "expected_accept",
    "smoothed_accept",
}
_VALID_SELECTOR_OBJECTIVES = {
    "multiclass",
    "candidate_distill",
    "keep_repair",
    "profitable_repair",
    "accept_repair",
    "profitable_action",
    "full_vocab",
    "covered_conditional",
}
_FACTORIZED_SELECTOR_OBJECTIVES = {
    "keep_repair",
    "profitable_repair",
    "accept_repair",
}
# Objectives whose err-head output is a term of the MAIN selector likelihood rather than an optional
# post-hoc detector.  For these, (a) the head must be requested from score_candidates, (b) it must
# stay trainable even though selector_err_loss_alpha == 0, and (c) a standalone err BCE would be a
# second, conflicting label on the same logits.  Keying these three behaviours off the factorised set
# alone would freeze covered_conditional's coverage head and then fail on a missing err_logits.
_MAIN_LIKELIHOOD_ERR_OBJECTIVES = _FACTORIZED_SELECTOR_OBJECTIVES | {"covered_conditional"}
_VALID_SELECTOR_ERR_WEIGHT_MODES = {"base_frontier", "all_valid"}


@torch.no_grad()
def redistribute_reachable_weights(
    original: torch.Tensor,
    reachable: torch.Tensor,
    covered: torch.Tensor,
    floor: float,
) -> torch.Tensor:
    """Shift covered CE mass toward the teacher-prefix policy's reachable slots.

    Conserves each block's *covered* weight sum, not a batch/local-rank mean.
    A nonzero floor retains later-slot supervision. This is a detached training
    curriculum, NOT native on-policy anchor sampling or an exact AL gradient.
    No target outside the candidate set is relabelled as KEEP. The same weights
    feed the existing alternative CE; its class-specific mass need not match.
    """
    if not math.isfinite(floor) or not 0 < floor <= 1:
        raise ValueError("reachable weight floor must be finite and in (0, 1]")
    if original.shape != reachable.shape or original.shape != covered.shape:
        raise ValueError("reachable weight tensors must have identical shapes")
    if floor == 1:
        return original.detach()
    mixed = original * (floor + (1 - floor) * reachable.to(original.dtype))
    mass = (original * covered).sum(-1, keepdim=True)
    mixed_mass = (mixed * covered).sum(-1, keepdim=True)
    scale = torch.where(mixed_mass > 0, mass / mixed_mass.clamp_min(1e-12), 1.)
    return mixed * scale


class SelectorAuxTerms(NamedTuple):
    """Training-only additive statistics; no probabilities or learned parameters."""

    alt_ce_num: torch.Tensor
    alt_weight_den: torch.Tensor
    alt_correct_num: torch.Tensor
    safe_num: torch.Tensor
    safe_den: torch.Tensor
    safe_violation_num: torch.Tensor
    repair_num: torch.Tensor
    repair_den: torch.Tensor
    repair_violation_num: torch.Tensor

    @classmethod
    def zeros(cls, reference: torch.Tensor) -> "SelectorAuxTerms":
        return cls(*(reference.new_zeros(()) for _ in cls._fields))


class SelectorTerms(NamedTuple):
    """Additive selector objective and metric terms for one objective chunk.

    The three ``err_*`` fields carry the optional frontier error-detector objective. They stay zero
    for selectors that do not declare ``wants_err_objective``, so the DFlash2 selector path reduces
    to exactly its previous behaviour.
    """

    ce_num: torch.Tensor
    objective_probability_num: torch.Tensor
    probability_num: torch.Tensor
    correct_num: torch.Tensor
    all_valid_correct_num: torch.Tensor
    weight_den: torch.Tensor
    covered_num: torch.Tensor
    err_ce_num: torch.Tensor
    err_den: torch.Tensor
    err_correct_num: torch.Tensor
    # Diagnostics that must coexist with the err objective, so they cannot borrow err_* fields:
    #   base_right_num  weighted count of valid slots whose target IS the draft top-1
    #   slot_weight_num sum of the per-slot CE weights actually applied
    # Denominator for both is `weight_den_all` = every valid slot, which differs from `weight_den`
    # (the supervised, weighted denominator) and is the population the decode-side 75.4% is over.
    base_right_num: torch.Tensor
    slot_weight_num: torch.Tensor
    weight_den_all: torch.Tensor
    alt_ce_num: torch.Tensor
    alt_weight_den: torch.Tensor
    alt_correct_num: torch.Tensor
    safe_num: torch.Tensor
    safe_den: torch.Tensor
    safe_violation_num: torch.Tensor
    repair_num: torch.Tensor
    repair_den: torch.Tensor
    repair_violation_num: torch.Tensor
    distill_kl_num: torch.Tensor
    distill_weight_den: torch.Tensor

    @classmethod
    def zeros(cls, reference: torch.Tensor) -> "SelectorTerms":
        return cls(*(reference.new_zeros(()) for _ in cls._fields))


def accept_survival_weights(
    valid: torch.Tensor,
    correct_action_probability: torch.Tensor,
    excludes_anchor: bool,
    survival_floor: float,
) -> torch.Tensor:
    """Suffix-value weights w_i = sum_{t>=i} prod_{j<=t} survival_j on the slot axis.

    With ``survival_floor == 0`` these are the EXACT gradient weights of expected accepted-prefix
    length: for CE_i = -log q_i and E[A] = sum_t prod_{j<=t} q_j,

        dE[A]/dCE_i = sum_{t>=i} (prod_{j<=t} q_j / q_i) * dq_i/dCE_i = -sum_{t>=i} prod_{j<=t} q_j

    because dq_i/dCE_i = -q_i.  So a CE weighted by these detached w_i has exactly the parameter
    gradient of -E[A].  A positive floor mixes survival to floor+(1-floor)*q, which is the D-PACE
    surrogate that keeps later covered slots from being starved behind an early q=0.

    Pulled out of the loss body purely so scripts/gate_smoothed_accept.py can check the identity
    against autograd on the real code rather than on a copy of it.
    """

    if excludes_anchor:
        contiguous_valid = valid.long().cumprod(dim=-1).bool()
    else:
        proposal_contiguous = valid[..., 1:].long().cumprod(dim=-1).bool()
        contiguous_valid = torch.cat(
            [torch.zeros_like(proposal_contiguous[..., :1]), proposal_contiguous],
            dim=-1,
        )
    survival_probability = (
        survival_floor + (1.0 - survival_floor) * correct_action_probability
    )
    prefix_probability = torch.cumprod(
        torch.where(
            contiguous_valid,
            survival_probability,
            torch.ones_like(survival_probability),
        ),
        dim=-1,
    )
    return torch.flip(
        torch.cumsum(
            torch.flip(prefix_probability * contiguous_valid.float(), dims=[-1]),
            dim=-1,
        ),
        dims=[-1],
    )


def frontier_mask(valid: torch.Tensor, top_ok: torch.Tensor) -> torch.Tensor:
    """Slots up to AND INCLUDING the first position where the base top-1 is wrong.

    A drafted block's accepted run ends at its first mismatch, so these are the only slots a decode
    actually reaches, and therefore the only slots at which a repair decision is ever taken.
    Operates on the last dimension, which is the within-block slot axis.
    """

    reached = torch.cumsum(((~valid) | (~top_ok)).long(), dim=-1) == 0
    run_length = reached.sum(dim=-1, keepdim=True)
    horizon = valid.shape[-1]
    first_error = torch.zeros_like(reached).scatter(
        -1, run_length.clamp(max=horizon - 1), run_length < horizon
    )
    return (reached | first_error) & valid


def candidate_distillation_kl(student_scores, teacher_logits, weight_mask, temperature=1.0):
    """Additive KL on the SAME draft candidates, with detached teacher targets.

    The teacher distribution is conditional on this candidate set, not a claim
    that it contains the target. On a miss it supervises relative rankings only;
    it never invents a KEEP label or inserts the target into the draft lattice.
    No local mean: the caller retains the global DDP/accumulation denominator.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("candidate distillation temperature must be finite and positive")
    if student_scores.shape != teacher_logits.shape or student_scores.shape[:-1] != weight_mask.shape:
        raise ValueError("candidate distillation score/mask shapes disagree")
    log_q = F.log_softmax(student_scores.float() / temperature, dim=-1)
    log_p = F.log_softmax(teacher_logits.detach().float() / temperature, dim=-1)
    per_slot = F.kl_div(log_q, log_p, reduction="none", log_target=True).sum(-1)
    return (per_slot * weight_mask * temperature**2).sum(), weight_mask.sum()


def gather_selector_teacher_hidden(target_last_hidden_states, safe_label_indices):
    """Post-norm target state at p-1 supervises token p; never feed it to the head."""
    if target_last_hidden_states.ndim != 3 or safe_label_indices.ndim != 3:
        raise ValueError("teacher states require [B,L,D] and label positions [B,A,H]")
    batch, _, width = target_last_hidden_states.shape
    if safe_label_indices.shape[0] != batch:
        raise ValueError("teacher state/label batch sizes disagree")
    shifted = (safe_label_indices - 1).clamp_min(0)
    gathered = torch.gather(target_last_hidden_states.detach(), 1,
                            shifted.reshape(batch, -1, 1).expand(-1, -1, width))
    return gathered.reshape(*safe_label_indices.shape, width)


def selector_training_auxiliaries(
    scores: torch.Tensor,
    target_index: torch.Tensor,
    covered: torch.Tensor,
    weight_mask: torch.Tensor,
    ranking_weights: torch.Tensor,
    selected_index: torch.Tensor,
    *,
    excludes_anchor: bool,
    rho: float,
    safe_margin: float,
    repair_margin: float,
) -> SelectorAuxTerms:
    """Alternative CE plus existing margin-gate boundary losses.

    This is a teacher-prefix surrogate, NOT on-policy rollout or an exact
    greedy acceptance objective. Misses receive no invented KEEP label.
    All returned terms are sums; normalize once across chunks/microbatches/ranks.
    The caller applies coefficients only after the additive chunk reduction.
    """
    if scores.shape[-1] < 2 or scores.shape[:-1] != target_index.shape:
        raise ValueError("selector auxiliary scores/targets have incompatible shapes")
    if not math.isfinite(rho) or rho <= 0:
        raise ValueError("selector auxiliary rho must be finite and positive")
    scores = scores.float()
    repairable = covered & target_index.ne(0)
    base_correct = covered & target_index.eq(0)
    with torch.no_grad():
        policy_ok = covered & selected_index.squeeze(-1).eq(target_index)
        if excludes_anchor:
            reachable = frontier_mask(weight_mask > 0, policy_ok)
        else:
            proposal_reachable = frontier_mask(weight_mask[..., 1:] > 0, policy_ok[..., 1:])
            reachable = torch.cat([torch.zeros_like(weight_mask[..., :1], dtype=torch.bool),
                                   proposal_reachable], dim=-1)
    alt_log_probs = F.log_softmax(scores[..., 1:], dim=-1)
    # Uncovered targets have only a placeholder index. Their weights are zero.
    alt_index = (target_index - 1).clamp(min=0, max=scores.shape[-1] - 2)
    alt_ce = -alt_log_probs.gather(-1, alt_index.unsqueeze(-1)).squeeze(-1)
    alt_weights = ranking_weights * repairable.float()
    safe_weights = weight_mask * (reachable & base_correct).float()
    repair_weights = weight_mask * (reachable & repairable).float()
    base = scores[..., 0]
    best_alt = scores[..., 1:].max(dim=-1).values
    truth = scores.gather(-1, target_index.unsqueeze(-1)).squeeze(-1)
    safe_hinge = F.relu(best_alt - base - math.log(rho) + safe_margin)
    repair_hinge = F.relu(base + math.log(rho) + repair_margin - truth)
    return SelectorAuxTerms(
        alt_ce_num=(alt_ce * alt_weights).sum(),
        alt_weight_den=alt_weights.sum(),
        alt_correct_num=((alt_log_probs.detach().argmax(-1) + 1).eq(target_index).float()
                         * alt_weights).sum(),
        safe_num=(safe_hinge * safe_weights).sum(),
        safe_den=safe_weights.sum(),
        safe_violation_num=((safe_hinge.detach() > 0).float() * safe_weights).sum(),
        repair_num=(repair_hinge * repair_weights).sum(),
        repair_den=repair_weights.sum(),
        repair_violation_num=((repair_hinge.detach() > 0).float() * repair_weights).sum(),
    )


class DFlashObjectiveTerms(NamedTuple):
    """Additive terms for ``OnlineDFlashModel``'s block objective.

    This tuple covers the standard DFlash and D-PACE loss variants, including
    the optional DFlash2 selector terms. It is not shared with Domino or DSpark;
    those models define separate objective reductions. The fields stay flat and
    tensor-only to satisfy ``checkpointed_chunk_reduce``'s chunk-function contract.
    """

    ce_loss_num: torch.Tensor
    tv_loss_num: torch.Tensor
    loss_den: torch.Tensor
    target_probability_num: torch.Tensor
    correct_num: torch.Tensor
    accuracy_den: torch.Tensor
    selector_ce_num: torch.Tensor
    selector_objective_probability_num: torch.Tensor
    selector_probability_num: torch.Tensor
    selector_correct_num: torch.Tensor
    selector_all_valid_correct_num: torch.Tensor
    selector_weight_den: torch.Tensor
    selector_covered_num: torch.Tensor
    selector_err_ce_num: torch.Tensor
    selector_err_den: torch.Tensor
    selector_err_correct_num: torch.Tensor
    selector_base_right_num: torch.Tensor
    selector_slot_weight_num: torch.Tensor
    selector_weight_den_all: torch.Tensor
    selector_alt_ce_num: torch.Tensor
    selector_alt_weight_den: torch.Tensor
    selector_alt_correct_num: torch.Tensor
    selector_safe_num: torch.Tensor
    selector_safe_den: torch.Tensor
    selector_safe_violation_num: torch.Tensor
    selector_repair_num: torch.Tensor
    selector_repair_den: torch.Tensor
    selector_repair_violation_num: torch.Tensor
    selector_distill_kl_num: torch.Tensor
    selector_distill_weight_den: torch.Tensor


def compute_accept_len(
    pred_ids_4d: torch.Tensor,
    target_ids_4d: torch.Tensor,
    valid_mask_4d: torch.Tensor,
) -> torch.Tensor:
    """Compute per-block acceptance length."""
    correct = (pred_ids_4d == target_ids_4d) | (~valid_mask_4d)
    accept_prefix = correct.long().cumprod(dim=2) * valid_mask_4d.long()
    return accept_prefix.sum(dim=2).float()


def create_dflash_sdpa_mask(
    anchor_positions,
    block_keep_mask,
    S,
    block_size,
    device,
    sliding_window: Optional[int] = None,
):
    """Construct a full or sliding dense boolean DFlash mask."""

    if sliding_window is not None and sliding_window <= 0:
        raise ValueError("sliding_window must be > 0")
    B, N = anchor_positions.shape
    Q_LEN = N * block_size
    KV_LEN = S + N * block_size

    q_indices = torch.arange(Q_LEN, device=device).view(1, 1, -1, 1)  # (1, 1, Q_LEN, 1)
    kv_indices = torch.arange(KV_LEN, device=device).view(
        1, 1, 1, -1
    )  # (1, 1, 1, KV_LEN)

    q_block_ids = q_indices // block_size
    q_block_offsets = q_indices % block_size

    anchor_expanded = anchor_positions.view(B, 1, N, 1).repeat_interleave(
        block_size, dim=2
    )

    mask_context = (kv_indices < S) & (kv_indices < anchor_expanded)
    if sliding_window is not None:
        # The current draft token occupies one slot in the window.
        context_lower_bound = anchor_expanded + q_block_offsets - (sliding_window - 1)
        mask_context = mask_context & (kv_indices >= context_lower_bound)

    is_draft = kv_indices >= S
    kv_block_ids = (kv_indices - S) // block_size
    mask_draft = is_draft & (q_block_ids == kv_block_ids)
    if sliding_window is not None:
        kv_block_offsets = (kv_indices - S) % block_size
        mask_draft = mask_draft & (kv_block_offsets <= q_block_offsets)

    valid_block = block_keep_mask.view(B, 1, N, 1).repeat_interleave(block_size, dim=2)

    final_mask = (mask_context | mask_draft) & valid_block
    return final_mask


def create_dflash_block_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    S: int,
    block_size: int,
    device: torch.device,
    flex_block_size=None,
    sliding_window: Optional[int] = None,
):
    """Construct a full or sliding Flex Attention mask for DFlash training."""

    if sliding_window is not None and sliding_window <= 0:
        raise ValueError("sliding_window must be > 0")

    def dflash_mask_mod(b, h, q_idx, kv_idx):
        q_block_id = q_idx // block_size
        q_block_offset = q_idx % block_size
        safe_q_block_id = q_block_id.clamp(max=N - 1)
        anchor_pos = anchor_positions[b, safe_q_block_id]

        is_context = kv_idx < S
        # Strictly less than: matches inference where target_hidden[anchor_pos]
        # is not available as context.
        mask_context = is_context & (kv_idx < anchor_pos)
        if sliding_window is not None:
            # The current draft token occupies one slot in the window.
            context_lower_bound = anchor_pos + q_block_offset - (sliding_window - 1)
            mask_context = mask_context & (kv_idx >= context_lower_bound)

        is_draft = kv_idx >= S
        kv_block_id = (kv_idx - S) // block_size
        mask_draft = is_draft & (q_block_id == kv_block_id)
        if sliding_window is not None:
            kv_block_offset = (kv_idx - S) % block_size
            mask_draft = mask_draft & (kv_block_offset <= q_block_offset)

        is_valid_block = block_keep_mask[b, safe_q_block_id]
        in_bounds = q_block_id < N
        return (mask_context | mask_draft) & is_valid_block & in_bounds

    B, N = anchor_positions.shape
    Q_LEN = N * block_size
    KV_LEN = S + N * block_size

    kwargs = {}
    if flex_block_size is not None:
        kwargs["BLOCK_SIZE"] = flex_block_size
    return create_block_mask(
        dflash_mask_mod,
        B=B,
        H=None,
        Q_LEN=Q_LEN,
        KV_LEN=KV_LEN,
        device=device,
        **kwargs,
    )


class OnlineDFlashModel(nn.Module):
    """DFlash online training wrapper with DFlash and D-PACE losses."""

    def __init__(
        self,
        draft_model: DFlashDraftModel,
        target_lm_head: nn.Module,
        target_embed_tokens: nn.Module,
        mask_token_id: int,
        block_size: int = 16,
        attention_backend: str = "flex_attention",
        num_anchors: int = 512,
        loss_decay_gamma: Optional[float] = None,
        objective_chunk_blocks: int = 128,
        loss_type: str = "dflash",
        dpace_alpha: float = 0.5,
        selector_loss_alpha: float = 1.0,
        selector_err_loss_alpha: float = 0.0,
        selector_own_denominator: bool = False,
        selector_weight_mode: str = "base_dpace",
        selector_frontier_boost: float = 0.0,
        selector_survival_floor: float = 0.5,
        selector_objective: str = "multiclass",
        selector_err_weight_mode: str = "base_frontier",
        selector_warmup_ratio: float = 0.0,
        selector_ramp_ratio: float = 0.0,
        selector_stop_gradient: bool = False,
        selector_target_greedy_labels: bool = False,
        lk_loss_type: Optional[str] = None,
        kl_scale: float = 1.0,
        kl_decay: float = 1.0,
        selector_alt_loss_alpha: float = 0.0,
        selector_safe_loss_alpha: float = 0.0,
        selector_repair_loss_alpha: float = 0.0,
        selector_safe_margin: float = 0.1,
        selector_repair_margin: float = 0.1,
        selector_preserve_fp32: bool = False,
        selector_distill_alpha: float = 0.5,
        selector_distill_temperature: float = 1.0,
    ):
        super().__init__()
        if loss_type not in _VALID_LOSS_TYPES:
            raise ValueError(
                f"loss_type={loss_type!r}; must be one of {sorted(_VALID_LOSS_TYPES)}"
            )
        if not 0.0 <= dpace_alpha <= 1.0:
            raise ValueError(f"dpace_alpha must be in [0, 1], got {dpace_alpha}")
        if objective_chunk_blocks < 0:
            raise ValueError("objective_chunk_blocks must be >= 0")
        if selector_loss_alpha < 0:
            raise ValueError("selector_loss_alpha must be >= 0")
        if selector_err_loss_alpha < 0:
            raise ValueError("selector_err_loss_alpha must be >= 0")
        if selector_own_denominator:
            raise ValueError(
                "selector_own_denominator=True is not partition-invariant under DDP/gradient "
                "accumulation: local selector means are mixed before the trainer applies its "
                "single global base denominator. Set dflash2_selector_own_denominator=false; "
                "the selector numerator will then share the globally reduced base denominator, "
                "and tune dflash2_selector_loss_alpha for its relative scale."
            )
        if selector_weight_mode not in _VALID_SELECTOR_WEIGHT_MODES:
            raise ValueError(
                f"selector_weight_mode={selector_weight_mode!r}; must be one of "
                f"{sorted(_VALID_SELECTOR_WEIGHT_MODES)}"
            )
        if selector_objective not in _VALID_SELECTOR_OBJECTIVES:
            raise ValueError(
                f"selector_objective={selector_objective!r}; must be one of "
                f"{sorted(_VALID_SELECTOR_OBJECTIVES)}"
            )
        if (
            selector_objective in _FACTORIZED_SELECTOR_OBJECTIVES
            and selector_err_loss_alpha > 0
        ):
            raise ValueError(
                "selector_err_loss_alpha must be 0 for a factorised selector objective: "
                "the main likelihood already contains its binary gate BCE"
            )
        if selector_objective == "covered_conditional" and selector_err_loss_alpha > 0:
            raise ValueError(
                "selector_err_loss_alpha must be 0 for covered_conditional: the coverage BCE is "
                "already a term of the main likelihood, and a second BCE on the same logits with a "
                "different label (err_target=~base_top1_ok) would fight it"
            )
        candidate_selector = getattr(draft_model, "candidate_selector", None)
        if selector_weight_mode == "reachable_frontier_boost" and not (
            type(candidate_selector).__name__ == "SlotDeepCorrector"
            and selector_objective == "multiclass" and selector_loss_alpha > 0
        ):
            raise ValueError("reachable_frontier_boost requires active SlotDeep multiclass CE")
        for name, value in (("selector_distill_alpha", selector_distill_alpha),
                            ("selector_distill_temperature", selector_distill_temperature)):
            if not math.isfinite(value) or value < 0 or (name.endswith("temperature") and value == 0):
                raise ValueError(f"{name} must be finite and {'positive' if name.endswith('temperature') else 'nonnegative'}")
        if selector_objective == "candidate_distill":
            if (type(candidate_selector).__name__ != "SlotDeepCorrector"
                    or selector_loss_alpha <= 0 or selector_distill_alpha <= 0
                    or selector_weight_mode not in {"uniform", "uniform_frontier_boost"}):
                raise ValueError("candidate_distill requires active SlotDeep CE/KL with uniform or uniform_frontier_boost weights")
            if not isinstance(getattr(target_lm_head, "weight", None), torch.Tensor):
                raise ValueError("candidate_distill requires a linear target lm_head weight")
        aux_options = {
            "selector_alt_loss_alpha": selector_alt_loss_alpha,
            "selector_safe_loss_alpha": selector_safe_loss_alpha,
            "selector_repair_loss_alpha": selector_repair_loss_alpha,
            "selector_safe_margin": selector_safe_margin,
            "selector_repair_margin": selector_repair_margin,
        }
        for name, value in aux_options.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        auxiliary_enabled = any(x > 0 for x in (
            selector_alt_loss_alpha, selector_safe_loss_alpha, selector_repair_loss_alpha))
        if auxiliary_enabled or selector_preserve_fp32:
            if type(candidate_selector).__name__ != "SlotDeepCorrector":
                raise ValueError("training-only selector extensions currently require SlotDeepCorrector")
        if selector_preserve_fp32 and getattr(draft_model, "selector_compute_dtype", None) != "float32":
            raise ValueError("selector_preserve_fp32 requires selector_compute_dtype='float32'")
        if auxiliary_enabled:
            rho = float(getattr(draft_model, "selector_gate_rho", 1.0))
            if not (
                selector_objective == "multiclass" and selector_loss_alpha > 0
                and selector_weight_mode in {"uniform", "uniform_frontier_boost", "reachable_frontier_boost"}
                and getattr(draft_model, "selector_decision_mode", None) == "margin_gate"
                and math.isfinite(rho) and rho > 0
                and float(getattr(draft_model, "selector_gate_tau", 0.0)) == 0
                and float(getattr(draft_model, "selector_gate_theta", 0.0)) == 0
                and not bool(getattr(draft_model, "selector_gate_skip_first", False))
            ):
                raise ValueError("SlotDeep auxiliary losses require active multiclass CE, uniform or "
                                 "uniform_frontier_boost/reachable_frontier_boost weights, margin_gate, finite rho>0, "
                                 "tau=theta=0 and no gate_skip_first")
        if selector_objective == "full_vocab" and not getattr(
            candidate_selector, "supports_full_vocab", False
        ):
            raise ValueError(
                "selector_objective='full_vocab' requires a selector exposing a full-vocabulary "
                "readout (supports_full_vocab=True); the K-way heads cannot score the classes "
                "outside their own candidate set"
            )
        serving_decision_mode = getattr(draft_model, "selector_decision_mode", None)
        if candidate_selector is not None and serving_decision_mode is not None:
            expected_objectives = {
                # profitable_action is a K-way CE over the *deployable* actions: class 0 means
                # KEEP (both when top-1 is right and when the target is outside the lattice), and
                # classes 1..K-1 are concrete repairs.  With rho=1/tau=theta=0 the historical
                # margin gate is exactly argmax over those same K scores.
                # covered_conditional decodes argmax over the same K conditional scores, which is
                # exactly margin_gate at rho=1/tau=0/theta=0.  Keeping it on this decision mode (as
                # opposed to inventing a new one) means the native MAP policy and every conservative
                # rho are the SAME rule with one scalar changed, so the calibration question
                # "is rho=1 optimal?" is directly measurable rather than confounded by a rule change.
                "margin_gate": {"multiclass", "candidate_distill", "profitable_action", "covered_conditional"},
                # All three objectives decode the same normalized abstain/repair distribution.
                # They differ only on top-k misses: historical keep_repair labels them REPAIR,
                # profitable_repair labels them ABSTAIN, and accept_repair masks them because every
                # fixed-lattice action has identical zero immediate acceptance utility.
                "keep_repair": _FACTORIZED_SELECTOR_OBJECTIVES,
                "full_vocab": {"full_vocab"},
            }.get(str(serving_decision_mode))
            if expected_objectives is None:
                raise ValueError(
                    f"unsupported selector_decision_mode={serving_decision_mode!r}"
                )
            if selector_objective not in expected_objectives:
                raise ValueError(
                    "selector train/serve policy mismatch: "
                    f"training objective is {selector_objective!r}, but the draft config serves "
                    f"with selector_decision_mode={serving_decision_mode!r} "
                    f"(allowed objectives={sorted(expected_objectives)})"
                )
        if selector_objective == "profitable_action" and serving_decision_mode is not None:
            rho = float(getattr(draft_model, "selector_gate_rho", 1.0))
            tau = float(getattr(draft_model, "selector_gate_tau", 0.0))
            theta = float(getattr(draft_model, "selector_gate_theta", 0.0))
            skip_first = bool(getattr(draft_model, "selector_gate_skip_first", False))
            if str(serving_decision_mode) != "margin_gate" or (
                rho != 1.0 or tau != 0.0 or theta != 0.0 or skip_first
            ):
                raise ValueError(
                    "profitable_action requires exact K-action argmax at serving: "
                    "selector_decision_mode='margin_gate', selector_gate_rho=1, "
                    "selector_gate_tau=0, selector_gate_theta=0, and "
                    "selector_gate_skip_first=false"
                )
        if not 0.0 < selector_survival_floor <= 1.0:
            raise ValueError(
                "selector_survival_floor must be in (0, 1], got "
                f"{selector_survival_floor}"
            )
        if selector_weight_mode in {"expected_accept", "smoothed_accept"}:
            # The survival weights need one thing only: a per-slot probability that the SERVING
            # rule emits the target, so that q_i is the acceptance-survival factor of slot i.
            # ``keep_repair`` qualifies via its normalized action distribution, and so does plain
            # ``multiclass``: its `selector_probability = exp(-CE)` is exactly P(argmax over the K
            # deployable candidates hits the target), and its serving rule at
            # rho=1/tau=0/theta=0 is that same argmax.  Restricting this to keep_repair was an
            # accident of the order the objectives were added, not a mathematical requirement --
            # the weight code below only ever touches weight_mask, target_is_candidate,
            # selector_probability and selector_excludes_anchor, all of which multiclass defines.
            #
            # covered_conditional is deliberately NOT allowed: its `selector_probability` is the
            # JOINT P(covered) * P(target | covered) over K+1 classes, but NONE is not emittable,
            # so using it as a survival factor would systematically understate q_i.  Enabling it
            # requires exposing the renormalized conditional first.
            if selector_objective not in {"keep_repair", "multiclass"}:
                raise ValueError(
                    f"selector_weight_mode={selector_weight_mode!r} requires a selector "
                    "probability defined over exactly the deployable actions "
                    f"(keep_repair or multiclass), got selector_objective={selector_objective!r}"
                )
            if selector_objective == "keep_repair":
                repair_margin = float(
                    getattr(draft_model, "selector_keep_repair_margin", 0.0)
                )
                if repair_margin != 0.0:
                    raise ValueError(
                        f"selector_weight_mode={selector_weight_mode!r} requires "
                        "selector_keep_repair_margin=0: a post-hoc repair penalty changes the "
                        "serving policy relative to the trained action distribution"
                    )
            elif selector_weight_mode == "expected_accept":
                # `expected_accept` claims to be the EXACT gradient of E[accepted prefix], which
                # only holds if the survival factor q_i equals the probability that the deployed
                # rule emits the target.  For multiclass q_i is the K-way softmax entry of the
                # target, i.e. the probability of the ARGMAX action, so a conservative gate would
                # silently break the exactness claim.  Fail closed rather than mislabel it.
                #
                # `smoothed_accept` is explicitly documented as a surrogate (floor-mixed survival,
                # not the exact gradient), so it is intentionally NOT subject to this check: its
                # ramp only needs to express "an early slot's error costs more", which is true for
                # every rho.  That is what makes it a genuine one-variable swap against the
                # uniform_frontier_boost champion, which is served at rho=3.
                rho = float(getattr(draft_model, "selector_gate_rho", 1.0))
                tau = float(getattr(draft_model, "selector_gate_tau", 0.0))
                theta = float(getattr(draft_model, "selector_gate_theta", 0.0))
                skip_first = bool(
                    getattr(draft_model, "selector_gate_skip_first", False)
                )
                if rho != 1.0 or tau != 0.0 or theta != 0.0 or skip_first:
                    raise ValueError(
                        "selector_weight_mode='expected_accept' with multiclass requires the "
                        "serving rule to be exact K-way argmax so that the trained survival "
                        "factor equals the deployed one: selector_gate_rho=1, "
                        "selector_gate_tau=0, selector_gate_theta=0, "
                        "selector_gate_skip_first=false; use 'smoothed_accept' for a conservative "
                        "gate, which is documented as a surrogate rather than an exact gradient"
                    )
        if selector_err_weight_mode not in _VALID_SELECTOR_ERR_WEIGHT_MODES:
            raise ValueError(
                f"selector_err_weight_mode={selector_err_weight_mode!r}; must be one of "
                f"{sorted(_VALID_SELECTOR_ERR_WEIGHT_MODES)}"
            )
        if not 0.0 <= selector_warmup_ratio <= 1.0:
            raise ValueError("selector_warmup_ratio must be in [0, 1]")
        if not 0.0 <= selector_ramp_ratio <= 1.0:
            raise ValueError("selector_ramp_ratio must be in [0, 1]")
        if lk_loss_type not in _VALID_LK_LOSS_TYPES:
            raise ValueError(
                "lk_loss_type must be one of None, 'alpha', 'lambda', or 'tv'"
            )

        self.draft_model = draft_model
        self.lm_head = target_lm_head
        self.embed_tokens = target_embed_tokens
        self.block_size = block_size
        self.mask_token_id = mask_token_id
        self.attention_backend = attention_backend
        self.num_anchors = num_anchors
        self.loss_decay_gamma = loss_decay_gamma
        self.objective_chunk_blocks = int(objective_chunk_blocks)
        self.loss_type = loss_type
        self.dpace_alpha = dpace_alpha
        self.selector_loss_alpha = float(selector_loss_alpha)
        self.selector_err_loss_alpha = float(selector_err_loss_alpha)
        self.selector_own_denominator = bool(selector_own_denominator)
        self.selector_weight_mode = str(selector_weight_mode)
        self.selector_frontier_boost = float(selector_frontier_boost)
        self.selector_survival_floor = float(selector_survival_floor)
        self.selector_objective = str(selector_objective)
        self.selector_err_weight_mode = str(selector_err_weight_mode)
        self.selector_warmup_ratio = float(selector_warmup_ratio)
        self.selector_ramp_ratio = float(selector_ramp_ratio)
        self.selector_stop_gradient = bool(selector_stop_gradient)
        self.selector_target_greedy_labels = bool(selector_target_greedy_labels)
        self.selector_alt_loss_alpha = float(selector_alt_loss_alpha)
        self.selector_safe_loss_alpha = float(selector_safe_loss_alpha)
        self.selector_repair_loss_alpha = float(selector_repair_loss_alpha)
        self.selector_safe_margin = float(selector_safe_margin)
        self.selector_repair_margin = float(selector_repair_margin)
        self.selector_preserve_fp32 = bool(selector_preserve_fp32)
        self.selector_distill_alpha = float(selector_distill_alpha)
        self.selector_distill_temperature = float(selector_distill_temperature)
        self._selector_auxiliary_enabled = auxiliary_enabled
        self.lk_loss_type = lk_loss_type
        self.kl_scale = float(kl_scale)
        self.kl_decay = float(kl_decay)

        self._selector_objective_enabled = (
            candidate_selector is not None and self.selector_loss_alpha > 0
        )
        if self.selector_objective in _MAIN_LIKELIHOOD_ERR_OBJECTIVES and not (
            self._selector_objective_enabled
            and bool(getattr(candidate_selector, "wants_err_objective", False))
        ):
            raise ValueError(
                f"selector_objective={self.selector_objective!r} puts the err/coverage logits in "
                "the main likelihood, so it requires a selector with an err head and "
                "selector_loss_alpha > 0"
            )
        # Static, like the CE gate above: the detector's parameters must either be trained for the
        # whole run or frozen for the whole run, never toggled per step, or DDP with
        # ``find_unused_parameters=False`` would stall on the steps that skip them.
        self._selector_err_objective_enabled = (
            self._selector_objective_enabled
            and bool(getattr(candidate_selector, "wants_err_objective", False))
            and self.selector_err_loss_alpha > 0
        )
        if (
            candidate_selector is not None
            and not self._selector_objective_enabled
            and isinstance(candidate_selector, nn.Module)
        ):
            # A zero configured weight statically disables selector training.
            # Freezing keeps those parameters out of BF16Optimizer and prevents
            # DDP(find_unused_parameters=False) from waiting for their gradients.
            candidate_selector.requires_grad_(False)
        if (
            candidate_selector is not None
            and bool(getattr(candidate_selector, "wants_err_objective", False))
            and not self._selector_err_objective_enabled
            and self.selector_objective not in _MAIN_LIKELIHOOD_ERR_OBJECTIVES
        ):
            # Same reason, one level down: a selector may be trained while its optional detector is
            # not, which would otherwise leave a gradient-less subtree in the DDP bucket.
            err_only_names = tuple(
                getattr(candidate_selector, "err_only_module_names", ("err_head",))
            )
            if not err_only_names:
                raise ValueError(
                    "a selector with wants_err_objective=True must declare at least one "
                    "err-only module"
                )
            for module_name in err_only_names:
                err_only_module = getattr(candidate_selector, module_name, None)
                if not isinstance(err_only_module, nn.Module):
                    raise ValueError(
                        "selector err_only_module_names contains a missing/non-module entry: "
                        f"{module_name!r}"
                    )
                err_only_module.requires_grad_(False)

        self._cached_block_mask: Optional[BlockMask] = None
        self._cached_seq_len: Optional[int] = None
        self._cached_bsz: Optional[int] = None

    def _sample_anchor_positions(
        self,
        seq_len: int,
        loss_mask: torch.Tensor,
        device: torch.device,
        max_valid_anchors: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample anchors whose clean token and first target are supervised."""

        num_candidates = max(seq_len - 1, 0)
        valid = (loss_mask[:, :num_candidates] > 0.5) & (
            loss_mask[:, 1 : num_candidates + 1] > 0.5
        )
        valid_counts = valid.sum(dim=1)
        if max_valid_anchors is None:
            # Direct model callers may supply an already-device-resident mask.
            # Training strategies pass the CPU-computed value and avoid this
            # synchronizing fallback on CUDA.
            max_valid_anchors = int(valid_counts.max().item())
        width = min(self.num_anchors, max(0, int(max_valid_anchors)))
        if width == 0:
            raise ValueError(
                "DFlash-family training requires two consecutive supervised tokens"
            )

        random_values = torch.rand(valid.shape, device=device)
        random_values.masked_fill_(~valid, 2.0)
        candidates = random_values.argsort(dim=1)[:, :width]
        keep_mask = torch.arange(width, device=device).unsqueeze(
            0
        ) < valid_counts.clamp(max=width).unsqueeze(1)

        sentinel = valid.shape[1]
        anchors = torch.where(
            keep_mask,
            candidates,
            torch.full_like(candidates, sentinel),
        )
        anchors = anchors.sort(dim=1).values
        keep_mask = anchors < sentinel
        return torch.where(keep_mask, anchors, 0), keep_mask

    def _create_position_ids(self, anchor_positions: torch.Tensor) -> torch.Tensor:
        """Create absolute position IDs for parallel draft blocks."""
        bsz, n_blocks = anchor_positions.shape
        device = anchor_positions.device
        offsets = torch.arange(self.block_size, device=device).view(1, 1, -1)
        pos_ids = anchor_positions.unsqueeze(-1) + offsets
        return pos_ids.view(bsz, -1)

    def _create_noise_embed(self, input_ids, anchor_positions, block_keep_mask):
        bsz, seq_len = input_ids.shape
        n = anchor_positions.shape[1]
        bs = self.block_size
        device = input_ids.device

        noise_ids = torch.full(
            (bsz, n * bs), self.mask_token_id, dtype=torch.long, device=device
        )

        block_starts = torch.arange(n, device=device) * bs
        block_starts = block_starts.unsqueeze(0).expand(bsz, -1)

        valid_anchor_positions = anchor_positions.clamp(0, seq_len - 1)
        anchor_tokens = torch.gather(input_ids, 1, valid_anchor_positions)

        flat_batch_idx = torch.arange(bsz, device=device).unsqueeze(1).expand(bsz, n)
        noise_ids[flat_batch_idx, block_starts] = torch.where(
            block_keep_mask,
            anchor_tokens,
            torch.tensor(self.mask_token_id, dtype=torch.long, device=device),
        )

        return self.embed_tokens(noise_ids)

    def _dpace_weight(
        self,
        prob: torch.Tensor,
        binary_mask: torch.Tensor,
        binary_mask_b: torch.Tensor,
        loss_type: str,
    ) -> torch.Tensor:
        """Compute detached D-PACE position weights.

        ``prob`` is the draft probability on the target token at each draft
        position. Invalid positions are treated as multiplicative no-ops inside
        prefix products and excluded from suffix sums; the caller still
        multiplies the returned weights by ``binary_mask`` before reduction.
        """
        smooth = (1.0 - self.dpace_alpha) * prob + self.dpace_alpha
        smooth = torch.where(binary_mask_b, smooth, torch.ones_like(smooth))
        prefix = torch.cumprod(smooth, dim=-1)

        if loss_type == "dpace-cumulative-confidence-only":
            return prefix

        suffix = torch.flip(
            torch.cumsum(torch.flip(prefix * binary_mask, dims=[-1]), dim=-1),
            dims=[-1],
        )

        if loss_type == "dpace":
            return suffix
        if loss_type == "dpace-continuation-value-only":
            return suffix / prefix.clamp_min(torch.finfo(prefix.dtype).tiny)
        raise ValueError(f"unknown D-PACE loss_type {loss_type!r}")

    def _forward_draft_blocks(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        max_valid_anchors: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len,
            loss_mask,
            device,
            max_valid_anchors=max_valid_anchors,
        )

        noise_embedding = self._create_noise_embed(
            input_ids, anchor_positions, block_keep_mask
        )

        context_position_ids = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        )
        draft_position_ids = self._create_position_ids(anchor_positions)
        full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)

        mask_builder = (
            create_dflash_block_mask
            if self.attention_backend == "flex_attention"
            else create_dflash_sdpa_mask
        )
        mask_args = {
            "anchor_positions": anchor_positions,
            "block_keep_mask": block_keep_mask,
            "S": seq_len,
            "block_size": self.block_size,
            "device": device,
        }
        if (
            self.attention_backend == "flex_attention"
            and flex_attention_backend() == "FLASH"
        ):
            # FLASH requires a minimum of this block size.
            mask_args["flex_block_size"] = (256, 128)
        full_attn_mask = mask_builder(**mask_args)
        sliding_window = self.draft_model.sliding_window
        dflash_attn_mask = full_attn_mask
        if sliding_window is not None:
            dflash_attn_mask = {
                "full_attention": full_attn_mask,
                "sliding_attention": mask_builder(
                    **mask_args,
                    sliding_window=sliding_window,
                ),
            }

        draft_kwargs = {}
        if self.attention_backend == "flex_attention":
            # DFlash's dynamic short-query batches are training/prefill shaped,
            # not autoregressive decoding.  AUTO may route q_len < 128 to the
            # more restrictive flex-decoding kernel, whose config set can be
            # empty for DFlash's sparse BlockMask.  Force the general Triton
            # Flex Attention kernel for every DFlash-family batch.
            #
            # The "BACKEND" kernel_option only exists on torch >= 2.11, where
            # the inductor lowering sanitizes it out of the generated Triton
            # constexprs.  On older builds (including current torch ROCm wheels)
            # the string leaks into the kernel as a bare identifier and fails to
            # compile (NameError: 'TRITON' is not defined), so we fall back to
            # FORCE_USE_FLEX_ATTENTION, which selects the same kernel and has
            # been supported since torch 2.5.
            if torch.__version__ >= "2.11":
                draft_kwargs["kernel_options"] = {"BACKEND": "TRITON"}
            else:
                draft_kwargs["kernel_options"] = {"FORCE_USE_FLEX_ATTENTION": True}
        output_hidden = self.draft_model(
            position_ids=full_position_ids,
            noise_embedding=noise_embedding,
            target_hidden=hidden_states,
            attention_mask=dflash_attn_mask,
            **draft_kwargs,
        )
        return anchor_positions, block_keep_mask, output_hidden

    def _selector_chunk_terms(
        self,
        candidate_selector: nn.Module,
        objective_logits: torch.Tensor,
        hidden: torch.Tensor,
        target_ids: torch.Tensor,
        predecessor_ids: torch.Tensor,
        loss_weights: torch.Tensor,
        weight_mask: torch.Tensor,
        selector_target_ids: Optional[torch.Tensor] = None,
        selector_memory: Optional[torch.Tensor] = None,
        selector_memory_mask: Optional[torch.Tensor] = None,
        selector_teacher_hidden: Optional[torch.Tensor] = None,
    ) -> SelectorTerms:
        """Return additive selector terms for one enabled objective chunk.

        The caller gates this helper only on the model-static selector
        configuration, never on the per-step effective selector alpha. During
        warmup the zero-scaled CE term must remain in the autograd graph so DDP
        with ``find_unused_parameters=False`` observes every selector parameter.
        The caller flattens these fields into ``DFlashObjectiveTerms`` to preserve
        ``checkpointed_chunk_reduce``'s flat tuple contract.

        ``selector_target_ids`` optionally replaces the corpus labels with the
        target model's own greedy token for the *same* conditioning prefix. Only
        the labels move: ``predecessor_ids`` stays on the corpus because that is
        what the target was conditioned on when the greedy token was computed, so
        the (prefix, label) pair remains self-consistent and matches the decode
        acceptance test ``pick == argmax P(. | committed prefix)``.
        """

        if selector_target_ids is None:
            selector_target_ids = target_ids

        if self.selector_stop_gradient:
            # Isolate only the selector objective. The caller still uses the
            # original tensors for the primary DFlash/D-PACE/LK objective.
            objective_logits = objective_logits.detach()
            hidden = hidden.detach()

        # The online DFlash objective carries an anchor at slot 0, whereas some
        # sequence-level selectors (PSPR) are defined over proposal slots only.
        # Masking the anchor's CE is not equivalent for such selectors: it still
        # shifts learned positions and participates in cross-slot attention.
        # Slice every selector-side tensor together so training has exactly the
        # same H slots and predecessor alignment as stage-1 and serving.  Local
        # selectors such as DFlash2 keep the historical block-sized path.
        selector_excludes_anchor = bool(
            getattr(candidate_selector, "selector_excludes_anchor", False)
        )
        if selector_excludes_anchor:
            objective_logits = objective_logits[..., 1:, :]
            hidden = hidden[..., 1:, :]
            target_ids = target_ids[..., 1:]
            predecessor_ids = predecessor_ids[..., 1:]
            loss_weights = loss_weights[..., 1:]
            weight_mask = weight_mask[..., 1:]
            selector_target_ids = selector_target_ids[..., 1:]
            if selector_teacher_hidden is not None:
                selector_teacher_hidden = selector_teacher_hidden[..., 1:, :]

        # Match serving exactly: construct actions from the strict unary top-k.
        # A multiclass selector has no label on a candidate miss.  The
        # factorised objective may still use its exact binary "top-1 is wrong"
        # label, while expected_accept correctly gives an unrepairable miss no
        # selector credit.  In every case the full-vocabulary backbone loss is
        # what can move the missing target into the candidate set.
        if hasattr(candidate_selector, "extract_lattice"):
            # Selectors that consume full-vocab uncertainty statistics, or that need candidate 0 to
            # be the argmax token under bf16 logit ties, own their own lattice extraction.
            unary_logits, candidate_ids, lattice_scalars = (
                candidate_selector.extract_lattice(objective_logits)
            )
            selector_extras = {"lattice_scalars": lattice_scalars}
        else:
            unary_logits, candidate_ids = objective_logits.topk(
                candidate_selector.top_k,
                dim=-1,
            )
            selector_extras = {}
        target_matches = candidate_ids.eq(selector_target_ids.unsqueeze(-1))
        target_is_candidate = target_matches.any(dim=-1)
        target_candidate_index = target_matches.long().argmax(dim=-1)
        # ``keep_repair`` factorises the decision into
        #
        #   P(keep top-1)              = sigmoid(-e)
        #   P(repair with candidate j) = sigmoid(e) * softmax(alt_scores)[j]
        #
        # so the error logit is part of the main selector likelihood, not merely
        # an optional post-hoc gate.  This directly matches the asymmetric job:
        # preserve a correct top-1, otherwise choose the right replacement.
        serving_uses_err_gate = (
            getattr(self.draft_model, "selector_decision_mode", None) == "margin_gate"
            and float(getattr(self.draft_model, "selector_gate_theta", 0.0)) > 0
        )
        wants_err = (
            self._selector_err_objective_enabled
            or self.selector_objective in _MAIN_LIKELIHOOD_ERR_OBJECTIVES
            or serving_uses_err_gate
        )
        if wants_err:
            selector_extras["return_err"] = True
        wants_full = self.selector_objective == "full_vocab"
        if wants_full:
            selector_extras["return_full"] = True
        if getattr(candidate_selector, "requires_verified_memory", False):
            selector_extras["target_memory"] = selector_memory
            selector_extras["memory_mask"] = selector_memory_mask
        selector_output = candidate_selector.score_candidates(
            candidate_ids=candidate_ids,
            unary_logits=unary_logits,
            hidden_states=hidden,
            predecessor_ids=predecessor_ids,
            **selector_extras,
        )
        full_logits = None
        if wants_err or wants_full:
            selector_logits = selector_output[0]
            cursor = 1
            if wants_err:
                err_logits = selector_output[cursor]
                cursor += 1
            else:
                err_logits = None
            if wants_full:
                full_logits = selector_output[cursor]
        else:
            selector_logits, err_logits = selector_output, None
        if self.selector_objective == "full_vocab":
            # Dense cross-entropy over the corrected readout.  Qwen3-style tied embeddings make
            # ``E (h + delta)`` a full-vocabulary distribution at no architectural cost, which fixes
            # two defects of the K-way objectives at once: the ~21% of slots whose target falls
            # outside top-K currently receive *no* selector gradient, and the slots that do are
            # trained on a 16-way decision while the competing Domino head trains on a 151936-way
            # one.  The candidate set stays the serving restriction, not the training support.
            if full_logits is None:
                raise RuntimeError("full_vocab objective requires the selector's full readout")
            selector_ce = F.cross_entropy(
                full_logits.float().reshape(-1, full_logits.shape[-1]),
                selector_target_ids.reshape(-1),
                reduction="none",
            ).reshape_as(selector_target_ids)
            selector_probability = torch.exp(-selector_ce)
            with torch.no_grad():
                # Reported through the candidate-index channel so the existing accuracy telemetry
                # keeps its meaning: the full argmax is mapped back onto the lattice when it lands
                # there, and onto a sentinel outside it otherwise.
                full_pick = full_logits.detach().argmax(dim=-1)
                pick_matches = candidate_ids.eq(full_pick.unsqueeze(-1))
                selected_index = torch.where(
                    pick_matches.any(dim=-1, keepdim=True),
                    pick_matches.long().argmax(dim=-1, keepdim=True),
                    torch.full_like(target_candidate_index.unsqueeze(-1), -1),
                )
        elif self.selector_objective == "profitable_action":
            # Train the action the fixed lattice can actually execute.  Candidate 0 is KEEP.
            # When the target is outside top-K no repair can extend acceptance, so KEEP is the
            # unique non-destructive selector action; mapping a miss to class 0 also gives every
            # valid slot a scorer/delta gradient.  This is deliberately different from historical
            # multiclass CE, which drops misses, and from the factorised objective, whose
            # alternative scorer sees only positive repair examples.
            action_target = torch.where(
                target_is_candidate,
                target_candidate_index,
                torch.zeros_like(target_candidate_index),
            )
            selector_ce = F.cross_entropy(
                selector_logits.float().reshape(-1, selector_logits.shape[-1]),
                action_target.reshape(-1),
                reduction="none",
            ).reshape_as(selector_target_ids)
            selector_probability = torch.exp(-selector_ce)
            selected_index = selector_logits.argmax(dim=-1, keepdim=True)
        elif self.selector_objective == "covered_conditional":
            # Flat (K+1)-way softmax with an explicit NONE class: z = [s_0 .. s_{K-1}, s_none],
            # s_none := the existing err-head channel.  NONE means "the target is not in this
            # candidate set at all".
            #
            #     L = -log softmax(z)[observed]      observed = target index, or NONE on a top-k miss
            #
            # WHY FLAT AND NOT A q/r FACTORISATION.  The factorised form
            # (q = P(covered), r = P(c_j | covered), loss = BCE(q) + covered*CE(r)) looks cleaner and
            # is free at serving, but its supervision of the ranker is *bit-identical to multiclass*:
            # a miss contributes only BCE on a separate head, so `r` gets gradient on exactly the
            # same slots with exactly the same labels.  It cannot move the statistic this project is
            # bottlenecked on.
            #
            # That bottleneck, measured on cloze@7115 (latgate rho=3, 5789 fixable slots):
            #     truth ranked first AMONG ALTERNATIVES   66.86%
            #     truth ranked first once candidate 0 is included   44.05%
            # The 22.8 pp gap is entirely "candidate 0 outranks the truth".  A top-k miss is a hard
            # negative for precisely that: if the target is outside C then c_0 is definitely wrong.
            # multiclass drops those slots; the factorised form routes them to a head that is
            # discarded at decode; the flat softmax spends them on s_0, where
            #     miss slot:            dL/ds_0 = P(c_0) > 0   -> s_0 is pushed DOWN
            #     covered, target=c_0:  dL/ds_0 = P(c_0) - 1   -> s_0 is pushed UP  (unchanged)
            # All 16 candidates are pushed down on a miss, but P(c_0) is the largest of them (c_0 is
            # the draft's own argmax), so the net effect is to lower s_0 RELATIVE to the alternatives.
            #
            # Why that is the right axis, provably.  At tau=0, rho=1 the serving rule
            # `pr[1:].max() > tau + rho*pr[0]` reduces to `s_alt_max > s_0`, i.e. plain argmax, which
            # is invariant to any monotone rescaling.  So the measured advantage of rho=3 over rho=1
            # (-0.0936 +- 0.0549 new card, -0.1110 +- 0.0436 old card, same sign and magnitude)
            # CANNOT be reproduced by any global temperature -- a temperature sweep at rho=1 is
            # mathematically a no-op, and measuring it confirmed five identical results.  The only
            # thing that changes an argmax decision is moving s_0 relative to the alternatives, which
            # is exactly what rho does by hand and what the miss gradient above does by training.
            #
            # Serving cost is unchanged from multiclass, and identical to the factorised form's: the
            # decoder excludes NONE and takes argmax over j<K, in which both the partition function
            # and s_none cancel.  s_none is therefore never evaluated at decode.
            if err_logits is None:
                raise RuntimeError(
                    "covered_conditional needs the NONE-class logit (err_logits channel)"
                )
            extended_logits = torch.cat(
                [selector_logits.float(), err_logits.float().unsqueeze(-1)], dim=-1
            )
            none_index = selector_logits.shape[-1]
            flat_target = torch.where(
                target_is_candidate,
                target_candidate_index,
                torch.full_like(target_candidate_index, none_index),
            )
            selector_ce = F.cross_entropy(
                extended_logits.reshape(-1, extended_logits.shape[-1]),
                flat_target.reshape(-1),
                reduction="none",
            ).reshape_as(selector_target_ids)
            # Probability of the true outcome under the (K+1)-way model.  On a miss this is P(NONE),
            # which is a real predicted quantity here rather than leftover mass.
            selector_probability = torch.exp(-selector_ce)
            # Serving picks among candidates only; NONE is not an emittable action.  Reporting the
            # restricted argmax keeps `selector_accuracy` comparable to multiclass.
            selected_index = selector_logits.argmax(dim=-1, keepdim=True)
        elif self.selector_objective in _FACTORIZED_SELECTOR_OBJECTIVES:
            if err_logits is None:
                raise RuntimeError("factorised selector objective requires selector err logits")
            if selector_logits.shape[-1] < 2:
                raise ValueError("factorised selector objective requires at least two candidates")
            alt_log_probs = F.log_softmax(selector_logits.float()[..., 1:], dim=-1)
            repair_is_covered = target_is_candidate & target_candidate_index.ne(0)
            if self.selector_objective in {"profitable_repair", "accept_repair"}:
                # The deployable question is not merely "is top-1 wrong?".  On a top-k miss that
                # answer is yes, but every available repair action is still wrong.  Train the gate
                # on the action it can profitably take: override iff the target is an alternative.
                err_target = repair_is_covered
            else:
                # Historical semantics retained for reproducibility.
                err_target = ~(
                    target_is_candidate & target_candidate_index.eq(0)
                )
            binary_ce = F.binary_cross_entropy_with_logits(
                err_logits.float(), err_target.float(), reduction="none"
            )
            alt_target = (target_candidate_index - 1).clamp_min(0)
            alt_ce = -alt_log_probs.gather(
                -1, alt_target.unsqueeze(-1)
            ).squeeze(-1)
            # The unweighted factorised likelihood has an exact keep-vs-repair
            # label even on a top-k miss; conditional alternative CE exists
            # only when the target is one of candidates 1..K-1.  Whether the
            # miss BCE is actually optimized is deliberately decided below by
            # selector_weight_mode: uniform/frontier can retain it, whereas
            # expected_accept gives an unrepairable miss zero selector credit.
            selector_ce = binary_ce + torch.where(
                repair_is_covered, alt_ce, torch.zeros_like(alt_ce)
            )
            keep_probability = torch.sigmoid(-err_logits.float())
            repair_probability = torch.sigmoid(err_logits.float()) * torch.exp(-alt_ce)
            if self.selector_objective in {"profitable_repair", "accept_repair"}:
                # Probability of the *correct deployable action*.  A miss cannot emit the target,
                # but ABSTAIN is still the only non-destructive selector action and has a valid
                # likelihood.  Expected-accept weighting separately masks target misses to q=0.
                selector_probability = torch.where(
                    repair_is_covered, repair_probability, keep_probability
                )
            else:
                selector_probability = torch.where(
                    target_is_candidate & target_candidate_index.eq(0),
                    keep_probability,
                    torch.where(
                        repair_is_covered,
                        repair_probability,
                        torch.zeros_like(repair_probability),
                    ),
                )
            repair_margin = float(
                getattr(self.draft_model, "selector_keep_repair_margin", 0.0)
            )
            if hasattr(candidate_selector, "select_keep_repair"):
                selected_index = candidate_selector.select_keep_repair(
                    selector_logits,
                    err_logits,
                    repair_margin=repair_margin,
                )
            else:
                decision_log_probs = torch.cat(
                    [
                        F.logsigmoid(-err_logits.float()).unsqueeze(-1),
                        F.logsigmoid(err_logits.float()).unsqueeze(-1) + alt_log_probs,
                    ],
                    dim=-1,
                )
                if repair_margin:
                    decision_log_probs = decision_log_probs.clone()
                    decision_log_probs[..., 1:] -= repair_margin
                selected_index = decision_log_probs.argmax(dim=-1, keepdim=True)
        else:
            selector_ce = F.cross_entropy(
                selector_logits.float().reshape(-1, selector_logits.shape[-1]),
                target_candidate_index.reshape(-1),
                reduction="none",
            ).reshape_as(selector_target_ids)
            selector_probability = torch.exp(-selector_ce)
            if (
                getattr(self.draft_model, "selector_decision_mode", None) == "margin_gate"
                and hasattr(candidate_selector, "select_margin_gate")
            ):
                force_keep = None
                if bool(getattr(self.draft_model, "selector_gate_skip_first", False)):
                    force_keep = torch.zeros_like(
                        selector_logits[..., 0], dtype=torch.bool
                    )
                    force_keep[..., 0] = True
                selected_index = candidate_selector.select_margin_gate(
                    selector_logits,
                    err_logits=err_logits,
                    rho=float(getattr(self.draft_model, "selector_gate_rho", 1.0)),
                    tau=float(getattr(self.draft_model, "selector_gate_tau", 0.0)),
                    theta=float(getattr(self.draft_model, "selector_gate_theta", 0.0)),
                    force_keep_mask=force_keep,
                )
            else:
                selected_index = selector_logits.argmax(dim=-1, keepdim=True)
        # Which slot weighting the selector's CE inherits.
        #
        # `loss_weights` is the BASE objective's weighting, and under any D-PACE loss_type it is
        # `weight_mask * dpace_weights` -- a prefix-survival reweighting recomputed every step. A
        # selector warm-started from stage-1 was trained under the uniform covered-slot CE that
        # `scripts/train_pspr_accept_selector.py` applies by default, so inheriting the D-PACE
        # weights silently changes the objective at the moment the warm start is loaded, which is
        # the one thing a warm start exists to avoid. `uniform` keeps the stage-1 objective and
        # makes the selector's weighting identical to `selector_metric_mask` below, so the logged
        # accuracy and the trained loss finally measure the same thing.
        #
        # `base_dpace` reproduces the previous behaviour exactly and stays the default so existing
        # runs and the joint-equivalence gate are unaffected.
        if self.selector_weight_mode == "uniform":
            selector_base_weights = weight_mask
        elif self.selector_weight_mode in {"uniform_frontier_boost", "reachable_frontier_boost"}:
            # `uniform` trains every covered slot equally, but accept length is decided at ONE slot
            # per block: the first the selector gets wrong.  `teacher_forced_frontier` goes to the
            # other extreme and zeroes every slot past it, which starves the representation of the
            # non-frontier slots -- and those slots are exactly the context a cloze selector reads
            # when it repairs the frontier.  This keeps the full uniform signal and adds
            # `selector_frontier_boost` extra weight on the current first error, so capacity is
            # steered at the metric without any slot losing supervision.  boost=0 is bit-identical
            # to `uniform`.
            with torch.no_grad():
                policy_ok = target_is_candidate & selected_index.squeeze(-1).eq(
                    target_candidate_index
                )
                if selector_excludes_anchor:
                    reachable = frontier_mask(weight_mask > 0, policy_ok)
                else:
                    proposal_reachable = frontier_mask(
                        weight_mask[..., 1:] > 0, policy_ok[..., 1:]
                    )
                    reachable = torch.cat(
                        [
                            torch.zeros_like(proposal_reachable[..., :1]),
                            proposal_reachable,
                        ],
                        dim=-1,
                    )
                first_error = reachable & (~policy_ok)
                if self.selector_objective in {
                    "profitable_repair",
                    "accept_repair",
                    "profitable_action",
                }:
                    # A top-k miss ends acceptance regardless of which selector action is chosen.
                    # Keep it as an ordinary negative gate example, but do not spend the 3x
                    # actionable-frontier boost on an impossible repair.
                    first_error = first_error & target_is_candidate
            selector_base_weights = weight_mask * (
                1.0 + self.selector_frontier_boost * first_error.float()
            )
            if self.selector_weight_mode == "reachable_frontier_boost":
                selector_base_weights = redistribute_reachable_weights(
                    selector_base_weights, reachable, target_is_candidate,
                    self.selector_survival_floor,
                )
        elif self.selector_weight_mode == "teacher_forced_frontier":
            # Detached curriculum over the current selector evaluated on the
            # teacher-forced predecessor sequence: its leading correct run plus
            # its first error.  On target-regenerated data that predecessor
            # sequence is the target-greedy correct prefix and therefore is the
            # serving state up to this first error.  On arbitrary corpus labels
            # this must not be described as on-policy rollout occupancy.
            with torch.no_grad():
                policy_ok = target_is_candidate & selected_index.squeeze(-1).eq(
                    target_candidate_index
                )
                if selector_excludes_anchor:
                    policy_reachable = frontier_mask(weight_mask > 0, policy_ok)
                else:
                    # Block-sized DFlash2 tensors retain a masked anchor at
                    # position zero.  Scanning from it would terminate every
                    # row and silently make the selector objective identically
                    # zero, so scan proposal positions and pad the anchor back.
                    proposal_reachable = frontier_mask(
                        weight_mask[..., 1:] > 0, policy_ok[..., 1:]
                    )
                    policy_reachable = torch.cat(
                        [
                            torch.zeros_like(proposal_reachable[..., :1]),
                            proposal_reachable,
                        ],
                        dim=-1,
                    )
                # Only a covered first failure has a repair action that can extend acceptance.
                # Reuse the same explicit frontier boost as uniform_frontier_boost, but keep the
                # strict occupancy mask: easy/off-policy slots after this failure receive no local
                # CE.  Context cells can still receive gradient through a reachable query's
                # attention, so masking their own token losses does not remove them as inputs.
                first_actionable_error = (
                    policy_reachable & (~policy_ok) & target_is_candidate
                )
            selector_base_weights = weight_mask * policy_reachable.float() * (
                1.0
                + self.selector_frontier_boost
                * first_actionable_error.float()
            )
        elif self.selector_weight_mode in {"expected_accept", "smoothed_accept"}:
            # Exact gradient of expected accepted-prefix length *on the supplied
            # teacher-forced states*.  Let
            # q_i=P(the selector chooses the target at slot i | target prefix)
            # and E[A]=sum_t prod_{j<=t}q_j.  For CE_i=-log(q_i),
            #
            #   d(-E[A]) / d CE_i = sum_{t>=i} prod_{j<=t} q_j.
            #
            # A CE with these detached weights therefore has the same parameter
            # gradient as -E[A], up to the shared positive global base-loss
            # normalizer. Unlike the hard frontier above, early uncertainty
            # reduces later credit smoothly rather than at an argmax
            # discontinuity. Calling this the decode-time objective
            # additionally requires the predecessor tokens and labels to form
            # one self-consistent target continuation.  Merely swapping labels
            # from a target_greedy sidecar while retaining different corpus
            # predecessors does not satisfy that condition.
            #
            # In exact mode a top-k miss has q_i=0 and therefore kills selector
            # credit at and after that slot: no current selector action can
            # extend the accepted prefix there.  This objective is mathematically
            # faithful but has a serious optimization pathology: q_i=0 is also a
            # saturated stationary boundary, so the selector can collapse to
            # keeping top-1 and receive essentially no repair gradient.
            #
            # ``smoothed_accept`` is the D-PACE-style training surrogate used to
            # avoid that gradient starvation.  It replaces each survival factor
            # by floor + (1-floor)*q before constructing detached suffix-value
            # weights.  It is deliberately *not* described as the exact gradient
            # of expected acceptance.  The positive floor gives later covered
            # repair decisions a curriculum signal even when an earlier action is
            # currently bad or outside top-k.
            with torch.no_grad():
                # A block cannot resume acceptance after an invalid label
                # (EOS, a user turn, or padding).  Real conversation masks can
                # very rarely contain 1...0...1 within one 15-slot window, so
                # treating an invalid slot as q=1 would incorrectly splice two
                # assistant spans into one accepted prefix.  Retained-anchor
                # selectors have one intentional leading zero; skip that anchor
                # before constructing the contiguous proposal prefix.  All of
                # that lives in ``accept_survival_weights`` so the gate can
                # check the exact-gradient identity against the real code.
                #
                # A miss has no emittable target, so q=0: under `expected_accept`
                # (floor 0) that correctly zeroes its own and every later slot's
                # credit, and under `smoothed_accept` the floor keeps them alive.
                correct_action_probability = torch.where(
                    target_is_candidate,
                    selector_probability.detach(),
                    torch.zeros_like(selector_probability),
                )
                marginal_value = accept_survival_weights(
                    valid=weight_mask > 0,
                    correct_action_probability=correct_action_probability,
                    excludes_anchor=selector_excludes_anchor,
                    survival_floor=(
                        self.selector_survival_floor
                        if self.selector_weight_mode == "smoothed_accept"
                        else 0.0
                    ),
                )
            selector_base_weights = weight_mask * marginal_value
        else:
            selector_base_weights = loss_weights
        # Multiclass CE has no valid class on a top-k miss.  Factorised objectives have a binary
        # gate label there: historical keep_repair says "top-1 is wrong", while
        # profitable_repair says "ABSTAIN because no profitable alternative exists";
        # accept_repair masks a miss because KEEP and every repair have equal zero acceptance value.
        selector_supervised = target_is_candidate.float()
        if (
            self.selector_objective in _FACTORIZED_SELECTOR_OBJECTIVES
            and self.selector_objective != "accept_repair"
            and self.selector_weight_mode != "smoothed_accept"
        ):
            selector_supervised = torch.ones_like(
                target_is_candidate, dtype=torch.float
            )
        if self.selector_objective == "full_vocab":
            # Every valid slot has a well-defined label over the whole vocabulary, so nothing is
            # dropped here.  ``weight_mask`` (folded into selector_base_weights) still restricts to
            # real assistant positions.
            selector_supervised = torch.ones_like(
                target_is_candidate, dtype=torch.float
            )
        if self.selector_objective == "profitable_action":
            # A top-k miss has no token-label inside the lattice, but it does have the exact
            # serving-action label KEEP, so unlike multiclass it remains supervised.
            selector_supervised = torch.ones_like(
                target_is_candidate, dtype=torch.float
            )
        if self.selector_objective == "covered_conditional":
            # A miss is a first-class outcome (NONE) with an exact coverage label, so it stays
            # supervised.  The conditional CE is masked internally rather than here, which is what
            # keeps the coverage head trained on 100% of valid slots while the ranker still sees
            # only slots where "which candidate" is a well-posed question.
            selector_supervised = torch.ones_like(
                target_is_candidate, dtype=torch.float
            )
        selector_loss_weights = selector_base_weights * selector_supervised
        selector_metric_mask = weight_mask * target_is_candidate.float()
        ce_num = (selector_ce * selector_loss_weights).sum()
        # Two probability views are intentionally separate. The first follows
        # the actual objective support/weights; the second is the historical
        # all-valid statistic conditioned on target-in-top-k.
        objective_probability_num = (
            selector_probability.detach() * selector_loss_weights
        ).sum()
        probability_num = (selector_probability.detach() * selector_metric_mask).sum()
        weight_den = selector_loss_weights.sum()
        covered_num = selector_metric_mask.sum()
        with torch.no_grad():
            # ``full_vocab`` marks "argmax landed outside the lattice" with -1; that pick is by
            # construction not the traced candidate, so map it to a sentinel id that never matches
            # rather than gathering at a negative index.
            in_lattice = selected_index.ge(0)
            selected_ids = candidate_ids.gather(
                -1, selected_index.clamp_min(0),
            ).squeeze(-1)
            if not bool(in_lattice.all()):
                selected_ids = torch.where(
                    in_lattice.squeeze(-1),
                    selected_ids,
                    torch.full_like(selected_ids, -1),
                )
            objective_correct = selected_ids == selector_target_ids
            if self.selector_objective in {
                "profitable_repair",
                "accept_repair",
                "profitable_action",
            }:
                # On a miss, candidate 0 is not the target, but choosing index 0 is the correct
                # ABSTAIN action for the selector's own objective.
                objective_correct = torch.where(
                    target_is_candidate,
                    objective_correct,
                    selected_index.squeeze(-1).eq(0),
                )
            correct_num = (
                objective_correct.float() * selector_loss_weights
            ).sum()
            all_valid_correct_num = (
                (selected_ids == selector_target_ids).float() * selector_metric_mask
            ).sum()
        err_ce_num = ce_num.new_zeros(())
        err_den = ce_num.new_zeros(())
        err_correct_num = ce_num.new_zeros(())
        # Measured for EVERY objective: these describe the data and the loss weights, not the
        # likelihood.  Purpose is to explain why a conservative serving rho beats the trained MAP.
        # A global temperature cannot be the cause, because rho=1/tau=0/theta=0 IS argmax and argmax
        # is scale-invariant, so the defect is candidate-0 specific.
        #
        # SIGN, stated carefully because I got it backwards once: the gate repairs iff
        # `p_alt > tau + rho*p0`, so a LARGER rho makes repair harder and therefore FAVOURS
        # candidate 0.  Hence if class 0 is UNDER-scored the compensating direction is rho>1.
        # Measured: p_train (all valid slots) = 0.4749 against a decode base_right of 0.7541 on
        # policy-reachable slots -- class 0 under-scored, which is consistent with the tuned rho=3.
        #
        # These are NOT matched populations and must not be compared directly: this counter runs over
        # every valid corpus slot, whereas the decode figure is conditioned on having survived the
        # prefix (slot k is only reached when 0..k-1 were accepted), which selects easy regions.
        # Conditioning on coverage narrows it to 0.4749/0.8086 = 0.5874 versus 19872/25661 = 0.7744.
        # The residual is an occupancy/reachability effect, not a label-noise effect: the dataset was
        # regenerated at temperature 0 (PSPR_README.md:186) and audited at 98.76% greedy agreement
        # (PSPR_STAGE2_PLAN.md:35).
        #
        # A miss slot has all-False `target_matches`, so `argmax` returns a PLACEHOLDER 0 that would
        # be silently counted as base_right; AND with target_is_candidate.
        base_right_num = (
            (target_candidate_index.eq(0) & target_is_candidate).float() * weight_mask
        ).sum()
        # Mean per-slot loss weight: exactly 1.0 under `uniform`, and >1 once an expected-accept
        # survival ramp is active, so this doubles as proof the ramp is actually on.
        slot_weight_num = selector_base_weights.sum()
        weight_den_all = weight_mask.sum()
        if self.selector_objective == "covered_conditional":
            # The NONE class is otherwise completely unobservable: its contribution is summed into
            # selector_loss with the 16 candidate terms, so a model that never predicts NONE would
            # look identical to one that predicts it well.  Reuse the three dead err_* channels for
            # NONE statistics (this objective sets err_loss_alpha=0, so they carry nothing else).
            #
            # `none_accuracy` is agreement between "NONE is the (K+1)-way argmax" and "the slot is
            # actually a miss".  A head that has learned nothing sits at max(miss, 1-miss) ~= 0.85;
            # anything at or below that means the NONE class is inert and the whole point of this
            # objective -- spending miss slots as negatives on s_0 -- is not happening.
            with torch.no_grad():
                none_weights = selector_base_weights * weight_mask
                is_miss = ~target_is_candidate
                none_log_prob = F.log_softmax(extended_logits.detach(), dim=-1)[..., none_index]
                err_ce_num = (-none_log_prob * none_weights).sum()
                err_den = none_weights.sum()
                none_is_argmax = extended_logits.detach().argmax(dim=-1).eq(none_index)
                err_correct_num = ((none_is_argmax == is_miss).float() * none_weights).sum()
        if self._selector_err_objective_enabled:
            if err_logits is None:
                raise RuntimeError("selector err objective requires selector err logits")
            base_top1_ok = target_is_candidate & target_candidate_index.eq(0)
            if self.selector_err_weight_mode == "all_valid":
                # Teacher-forced prefixes are the exact states reached after every successful
                # repair, so later slots are valid supervision.  Restricting the detector to the
                # *uncorrected base* frontier leaves it untrained precisely where PSPR continues
                # after repairing that frontier.
                reachable = weight_mask > 0
            elif selector_excludes_anchor:
                # Slot 0 is now proposal_1, exactly as in the stage-1 trace and decoder.
                reachable = frontier_mask(weight_mask > 0, base_top1_ok)
            else:
                # Historical block-sized selector path: the leading slot is the
                # masked anchor and must be skipped or it terminates the scan.
                reachable = frontier_mask(weight_mask[..., 1:] > 0, base_top1_ok[..., 1:])
                reachable = torch.cat(
                    [torch.zeros_like(reachable[..., :1]), reachable], dim=-1
                )
            # The label is "the base top-1 is wrong here", which deliberately includes slots whose
            # target falls outside the top-k. Those are unrepairable by the selector, so the CE
            # above drops them, but their binary label is perfectly well defined and the detector
            # must flag them: at decode they end the accepted run just the same.
            err_target = (~base_top1_ok).float()
            err_weights = reachable.float()
            err_per_slot = F.binary_cross_entropy_with_logits(
                err_logits.float(), err_target, reduction="none"
            )
            err_ce_num = (err_per_slot * err_weights).sum()
            err_den = err_weights.sum()
            with torch.no_grad():
                err_correct_num = (
                    ((err_logits.float() > 0).float() == err_target).float()
                    * err_weights
                ).sum()
        aux = SelectorAuxTerms.zeros(ce_num)
        if self._selector_auxiliary_enabled:
            aux = selector_training_auxiliaries(
                selector_logits, target_candidate_index, target_is_candidate,
                weight_mask, selector_base_weights, selected_index,
                excludes_anchor=selector_excludes_anchor,
                rho=float(self.draft_model.selector_gate_rho),
                safe_margin=self.selector_safe_margin,
                repair_margin=self.selector_repair_margin,
            )
        distill_num, distill_den = ce_num.new_zeros(()), ce_num.new_zeros(())
        if self.selector_objective == "candidate_distill":
            if selector_teacher_hidden is None or selector_teacher_hidden.shape != hidden.shape:
                raise ValueError("candidate_distill requires aligned online target final hidden states")
            # Privileged target states are LABELS ONLY. They never enter
            # score_candidates, the backbone, the GRU, or the exported model.
            with torch.no_grad():
                teacher_weights = F.embedding(candidate_ids, self.lm_head.weight.detach()).float()
                teacher_logits = torch.einsum("...d,...kd->...k",
                                               selector_teacher_hidden.detach().float(), teacher_weights)
                bias = getattr(self.lm_head, "bias", None)
                if bias is not None:
                    teacher_logits = teacher_logits + bias.detach()[candidate_ids].float()
            distill_num, distill_den = candidate_distillation_kl(
                selector_logits, teacher_logits, weight_mask, self.selector_distill_temperature)
        return SelectorTerms(
            ce_num=ce_num,
            objective_probability_num=objective_probability_num,
            probability_num=probability_num,
            correct_num=correct_num,
            all_valid_correct_num=all_valid_correct_num,
            weight_den=weight_den,
            covered_num=covered_num,
            err_ce_num=err_ce_num,
            err_den=err_den,
            err_correct_num=err_correct_num,
            base_right_num=base_right_num,
            slot_weight_num=slot_weight_num,
            weight_den_all=weight_den_all,
            **aux._asdict(),
            distill_kl_num=distill_num,
            distill_weight_den=distill_den,
        )

    def _dflash_objective_chunk_terms(
        self,
        hidden: torch.Tensor,
        target_ids: torch.Tensor,
        weight_mask: torch.Tensor,
        predecessor_ids: torch.Tensor,
        selector_target_ids: Optional[torch.Tensor] = None,
        selector_memory: Optional[torch.Tensor] = None,
        selector_memory_mask: Optional[torch.Tensor] = None,
        selector_teacher_hidden: Optional[torch.Tensor] = None,
    ) -> DFlashObjectiveTerms:
        """Return a flat tuple of additive objective and metric tensors."""

        batch_size, num_blocks, block_size, hidden_size = hidden.shape
        logits = self.lm_head(
            hidden.reshape(batch_size, num_blocks * block_size, hidden_size)
        ).reshape(batch_size, num_blocks, block_size, -1)
        candidate_selector = getattr(self.draft_model, "candidate_selector", None)
        objective_logits = (
            self.draft_model.transform_unary_logits(logits)
            if candidate_selector is not None
            else logits
        )
        neg_log_q = F.cross_entropy(
            objective_logits.reshape(-1, objective_logits.shape[-1]),
            target_ids.reshape(-1),
            reduction="none",
        ).reshape_as(target_ids)

        target_probability = torch.exp(-neg_log_q)
        loss_weights = weight_mask
        if self.loss_type == "dflash":
            if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
                positions = torch.arange(
                    self.block_size,
                    device=hidden.device,
                ).view(1, 1, -1)
                decay_weights = torch.exp(
                    -(positions - 1).clamp(min=0).float() / self.loss_decay_gamma
                )
                loss_weights = loss_weights * decay_weights
            loss_den = loss_weights.sum()
        elif self.loss_type in _DPACE_LOSS_TYPES:
            with torch.no_grad():
                dpace_weights = self._dpace_weight(
                    target_probability.detach(),
                    weight_mask,
                    weight_mask > 0,
                    self.loss_type,
                )
            loss_weights = weight_mask * dpace_weights
            loss_den = loss_weights.sum()
        else:  # defensive: __init__ validates the configured loss type.
            raise ValueError(f"unknown loss_type {self.loss_type!r}")

        ce_loss_num = (neg_log_q * loss_weights).sum()
        if self.lk_loss_type in {"lambda", "tv"}:
            tv_loss_num = ((1.0 - target_probability) * loss_weights).sum()
        else:
            tv_loss_num = ce_loss_num.new_zeros(())
        target_probability_num = (target_probability.detach() * weight_mask).sum()

        selector_terms = SelectorTerms.zeros(ce_loss_num)
        if self._selector_objective_enabled:
            selector_terms = self._selector_chunk_terms(
                candidate_selector=candidate_selector,
                objective_logits=objective_logits,
                hidden=hidden,
                target_ids=target_ids,
                predecessor_ids=predecessor_ids,
                loss_weights=loss_weights,
                weight_mask=weight_mask,
                selector_target_ids=selector_target_ids,
                selector_memory=selector_memory,
                selector_memory_mask=selector_memory_mask,
                selector_teacher_hidden=selector_teacher_hidden,
            )

        with torch.no_grad():
            predicted_ids = objective_logits.argmax(dim=-1)
            correct_num = (
                ((predicted_ids == target_ids) & (weight_mask > 0.5)).sum().float()
            )
            accuracy_den = weight_mask.sum()
        return DFlashObjectiveTerms(
            ce_loss_num=ce_loss_num,
            tv_loss_num=tv_loss_num,
            loss_den=loss_den,
            target_probability_num=target_probability_num,
            correct_num=correct_num,
            accuracy_den=accuracy_den,
            selector_ce_num=selector_terms.ce_num,
            selector_objective_probability_num=(
                selector_terms.objective_probability_num
            ),
            selector_probability_num=selector_terms.probability_num,
            selector_correct_num=selector_terms.correct_num,
            selector_all_valid_correct_num=selector_terms.all_valid_correct_num,
            selector_weight_den=selector_terms.weight_den,
            selector_covered_num=selector_terms.covered_num,
            selector_err_ce_num=selector_terms.err_ce_num,
            selector_err_den=selector_terms.err_den,
            selector_err_correct_num=selector_terms.err_correct_num,
            selector_base_right_num=selector_terms.base_right_num,
            selector_slot_weight_num=selector_terms.slot_weight_num,
            selector_weight_den_all=selector_terms.weight_den_all,
            selector_alt_ce_num=selector_terms.alt_ce_num,
            selector_alt_weight_den=selector_terms.alt_weight_den,
            selector_alt_correct_num=selector_terms.alt_correct_num,
            selector_safe_num=selector_terms.safe_num,
            selector_safe_den=selector_terms.safe_den,
            selector_safe_violation_num=selector_terms.safe_violation_num,
            selector_repair_num=selector_terms.repair_num,
            selector_repair_den=selector_terms.repair_den,
            selector_repair_violation_num=selector_terms.repair_violation_num,
            selector_distill_kl_num=selector_terms.distill_kl_num,
            selector_distill_weight_den=selector_terms.distill_weight_den,
        )

    def _compose_token_objective(
        self,
        ce_num: torch.Tensor,
        tv_num: torch.Tensor,
        probability_num: torch.Tensor,
        probability_den: torch.Tensor,
    ) -> torch.Tensor:
        """Compose a hard-target CE/TV/LK numerator after chunk reduction."""

        if self.lk_loss_type is None or self.lk_loss_type == "alpha":
            # With a one-hot target distribution, LK-alpha is exactly NLL/CE.
            return ce_num
        if self.lk_loss_type == "tv":
            return tv_num
        if self.lk_loss_type == "lambda":
            acceptance = probability_num / probability_den.clamp_min(1.0)
            kl_weight = self.kl_scale * torch.exp(-self.kl_decay * acceptance.detach())
            return kl_weight * ce_num + (1.0 - kl_weight) * tv_num
        raise ValueError(f"unknown lk_loss_type {self.lk_loss_type!r}")

    def _selector_target_ids(
        self,
        *,
        target_greedy: Optional[torch.Tensor],
        safe_label_indices: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Gather the target model's own greedy token for each supervised slot.

        Alignment. The dump stores ``target_greedy[i] = argmax P(. | tokens[:i+1])``,
        i.e. the target's prediction *for* position ``i + 1``, so the greedy token
        at absolute position ``p`` is ``target_greedy[p - 1]``. Slot ``k`` of a
        block supervises absolute position ``anchor + k``, hence the shift by one
        against ``safe_label_indices``. Slot 0 is the anchor itself and would read
        index ``-1``; the objective already zeroes ``weight_mask`` at
        ``pos_in_block == 0`` unconditionally, so the clamp below only keeps the
        gather in bounds and never contributes a term.

        Returns ``None`` when the feature is off, which makes the selector fall
        back to the corpus labels on exactly the same code path as before.
        """

        if not self.selector_target_greedy_labels:
            return None
        if target_greedy is None:
            raise ValueError(
                "selector_target_greedy_labels is enabled but the batch carries no "
                "'target_greedy' tensor. Build the sidecar with "
                "scripts/dump_target_greedy.py and place it at "
                "'<hidden_states_dir>.target_greedy'."
            )
        greedy = target_greedy.to(torch.long)
        shifted = (safe_label_indices - 1).clamp_min(0)
        return torch.gather(
            greedy.unsqueeze(1).expand(-1, shifted.size(1), -1),
            2,
            shifted,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        max_valid_anchors: Optional[int] = None,
        selector_loss_alpha: Optional[float] = None,
        selector_err_loss_alpha: Optional[float] = None,
        target_greedy: Optional[torch.Tensor] = None,
        target_last_hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
        """Parallel block-wise training forward pass; returns
        (loss, accuracy, metrics) — same shape as Domino's forward."""
        if self.attention_backend == "flex_attention" and not FLEX_ATTENTION_AVAILABLE:
            raise ValueError(
                "flex_attention is not available on this device; use sdpa/eager."
            )
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        anchor_positions, block_keep_mask, output_hidden = self._forward_draft_blocks(
            input_ids=input_ids,
            hidden_states=hidden_states,
            loss_mask=loss_mask,
            max_valid_anchors=max_valid_anchors,
        )

        # --- Labels: same-position prediction (position k predicts token anchor+k) ---
        label_offsets = torch.arange(0, self.block_size, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        valid_label_mask = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)

        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )
        predecessor_ids = torch.cat(
            [target_ids[:, :, :1], target_ids[:, :, :-1]],
            dim=-1,
        )
        selector_target_ids = self._selector_target_ids(
            target_greedy=target_greedy,
            safe_label_indices=safe_label_indices,
        )
        teacher_args = ()
        if self.selector_objective == "candidate_distill":
            if (target_last_hidden_states is None
                    or target_last_hidden_states.ndim != 3
                    or target_last_hidden_states.shape[:2] != input_ids.shape
                    or target_last_hidden_states.shape[-1] != output_hidden.shape[-1]):
                raise ValueError("candidate_distill requires online target_last_hidden_states [B,L,D] matching the input")
            teacher_args = (gather_selector_teacher_hidden(target_last_hidden_states, safe_label_indices),)

        # --- Weight mask: block validity * bounds * exclude anchor (pos 0) * loss_mask ---
        weight_mask = (
            block_keep_mask.unsqueeze(-1).expand(-1, -1, self.block_size).float()
        )
        weight_mask = weight_mask * valid_label_mask.float()

        pos_in_block = torch.arange(self.block_size, device=device).view(1, 1, -1)
        weight_mask = weight_mask * (pos_in_block > 0).float()

        original_loss_mask_gathered = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )
        weight_mask = weight_mask * original_loss_mask_gathered

        hidden_4d = output_hidden.reshape(
            bsz,
            anchor_positions.shape[1],
            self.block_size,
            -1,
        )
        # Opt-in context plumbing; old models retain their exact argument list.
        memory_args = ()
        candidate_selector = getattr(self.draft_model, "candidate_selector", None)
        if getattr(candidate_selector, "requires_verified_memory", False):
            from specforge.modeling.draft.pspr_memory import gather_verified_memory
            memory_args = gather_verified_memory(
                hidden_states, anchor_positions, candidate_selector.memory_length,
                candidate_selector.hidden_size)
        if teacher_args and not memory_args:
            memory_args = (None, None)
        (
            ce_loss_num,
            tv_loss_num,
            loss_den,
            target_probability_num,
            correct_num,
            accuracy_denom,
            selector_ce_num,
            selector_objective_probability_num,
            selector_probability_num,
            selector_correct_num,
            selector_all_valid_correct_num,
            selector_weight_den,
            selector_covered_num,
            selector_err_ce_num,
            selector_err_den,
            selector_err_correct_num,
            selector_base_right_num,
            selector_slot_weight_num,
            selector_weight_den_all,
            selector_alt_ce_num,
            selector_alt_weight_den,
            selector_alt_correct_num,
            selector_safe_num,
            selector_safe_den,
            selector_safe_violation_num,
            selector_repair_num,
            selector_repair_den,
            selector_repair_violation_num,
            selector_distill_kl_num,
            selector_distill_weight_den,
        ) = checkpointed_chunk_reduce(
            self._dflash_objective_chunk_terms,
            hidden_4d,
            target_ids,
            weight_mask,
            predecessor_ids,
            selector_target_ids,
            *memory_args,
            *teacher_args,
            chunk_size=self.objective_chunk_blocks,
            dim=1,
        )
        loss_num = self._compose_token_objective(
            ce_loss_num,
            tv_loss_num,
            target_probability_num,
            accuracy_denom,
        )
        effective_selector_alpha = (
            self.selector_loss_alpha
            if selector_loss_alpha is None
            else float(selector_loss_alpha)
        )
        if effective_selector_alpha < 0:
            raise ValueError("selector_loss_alpha must be >= 0")
        selector_loss_num = loss_num.new_zeros(())
        has_selector_objective = self._selector_objective_enabled
        loss_denominator = loss_den
        if has_selector_objective:
            # Keep the selector's serving-action likelihood independent of the
            # base model's optional LK/TV composition.  This is categorical CE
            # for the historical objective and binary+conditional CE for
            # keep_repair/profitable_repair.
            selector_loss_num = selector_ce_num
            # Keep this as an additive numerator. TrainerCore sums it over all
            # microbatches/ranks and applies the same globally reduced base
            # denominator once, so the gradient is invariant to data
            # partitioning. A local selector mean here cannot be repaired after
            # its gradient has mixed with the base gradient.
            loss_num = loss_num + effective_selector_alpha * selector_loss_num
        if self.selector_objective == "candidate_distill":
            loss_num = loss_num + effective_selector_alpha * self.selector_distill_alpha * selector_distill_kl_num

        selector_aux_loss_num = loss_num.new_zeros(())
        if self._selector_auxiliary_enabled:
            selector_aux_loss_num = (
                self.selector_alt_loss_alpha * selector_alt_ce_num
                + self.selector_safe_loss_alpha * selector_safe_num
                + self.selector_repair_loss_alpha * selector_repair_num
            )
            loss_num = loss_num + effective_selector_alpha * selector_aux_loss_num

        selector_err_loss_num = loss_num.new_zeros(())
        effective_err_alpha = (
            self.selector_err_loss_alpha
            if selector_err_loss_alpha is None
            else float(selector_err_loss_alpha)
        )
        if effective_err_alpha < 0:
            raise ValueError("selector_err_loss_alpha must be >= 0")
        if self._selector_err_objective_enabled:
            # This is also an additive numerator. A per-rank/per-microbatch
            # frontier mean would make the objective depend on how examples are
            # partitioned; the shared globally reduced base denominator keeps
            # the gradient well defined. Its relative scale is controlled by
            # selector_err_loss_alpha.
            selector_err_loss_num = selector_err_ce_num
            loss_num = (
                loss_num
                + effective_err_alpha * selector_err_loss_num
            )
        ratio_metrics = {
            "acc": (correct_num.detach(), accuracy_denom.detach()),
            "target_probability": (
                target_probability_num.detach(),
                accuracy_denom.detach(),
            ),
        }
        if has_selector_objective:
            ratio_metrics.update(
                {
                    "selector_loss": (
                        selector_loss_num.detach(),
                        selector_weight_den.detach(),
                    ),
                    "selector_accuracy": (
                        selector_correct_num.detach(),
                        selector_weight_den.detach(),
                    ),
                    "selector_accuracy_given_topk_all_valid": (
                        selector_all_valid_correct_num.detach(),
                        selector_covered_num.detach(),
                    ),
                    "selector_accuracy_all_valid": (
                        selector_all_valid_correct_num.detach(),
                        accuracy_denom.detach(),
                    ),
                    "selector_coverage": (
                        selector_covered_num.detach(),
                        accuracy_denom.detach(),
                    ),
                    "selector_target_probability": (
                        selector_probability_num.detach(),
                        selector_covered_num.detach(),
                    ),
                    "selector_target_probability_given_topk_all_valid": (
                        selector_probability_num.detach(),
                        selector_covered_num.detach(),
                    ),
                    "selector_target_probability_all_valid": (
                        selector_probability_num.detach(),
                        accuracy_denom.detach(),
                    ),
                    "selector_target_probability_on_objective": (
                        selector_objective_probability_num.detach(),
                        selector_weight_den.detach(),
                    ),
                    "selector_loss_per_base_weight": (
                        selector_loss_num.detach(),
                        loss_denominator.detach(),
                    ),
                    "selector_weight_per_base_weight": (
                        selector_weight_den.detach(),
                        loss_denominator.detach(),
                    ),
                }
            )
        if has_selector_objective:
            #   selector_base_right       fraction of valid slots whose target IS the draft top-1.
            #                             Compare against the decoder's 75.4%.
            #   selector_mean_slot_weight 1.0 under `uniform`; >1 proves the survival ramp is on.
            # Guarded: a model with no selector leaves both numerator and denominator at zero, and
            # registering 0/0 would publish a NaN for every plain DFlash run.
            ratio_metrics.update(
                {
                    "selector_base_right": (
                        selector_base_right_num.detach(),
                        selector_weight_den_all.detach(),
                    ),
                    "selector_mean_slot_weight": (
                        selector_slot_weight_num.detach(),
                        selector_weight_den_all.detach(),
                    ),
                }
            )
        if self.selector_objective == "candidate_distill":
            ratio_metrics.update({
                "selector_distill_kl": (selector_distill_kl_num.detach(), selector_distill_weight_den.detach()),
                "selector_distill_loss_per_base_weight": (
                    (self.selector_distill_alpha * selector_distill_kl_num).detach(), loss_denominator.detach()),
                "selector_distill_weight_per_valid": (selector_distill_weight_den.detach(), accuracy_denom.detach()),
            })
        if self._selector_auxiliary_enabled:
            ratio_metrics.update({
                "selector_aux_loss_per_base_weight": (selector_aux_loss_num.detach(), loss_denominator.detach()),
                "selector_alt_ce": (selector_alt_ce_num.detach(), selector_alt_weight_den.detach()),
                "selector_alt_accuracy_weighted": (selector_alt_correct_num.detach(), selector_alt_weight_den.detach()),
                "selector_safe_hinge": (selector_safe_num.detach(), selector_safe_den.detach()),
                "selector_safe_violation": (selector_safe_violation_num.detach(), selector_safe_den.detach()),
                "selector_repair_hinge": (selector_repair_num.detach(), selector_repair_den.detach()),
                "selector_repair_violation": (selector_repair_violation_num.detach(), selector_repair_den.detach()),
                "selector_alt_weight_per_valid": (selector_alt_weight_den.detach(), selector_weight_den_all.detach()),
                "selector_safe_weight_per_valid": (selector_safe_den.detach(), selector_weight_den_all.detach()),
                "selector_repair_weight_per_valid": (selector_repair_den.detach(), selector_weight_den_all.detach()),
            })
        if self.selector_objective == "covered_conditional":
            # Distinct names from `selector_err_*`: these describe the explicit NONE class of the
            # (K+1)-way softmax, not the historical "is top-1 wrong" detector.
            #   selector_none_nll      mean -log P(NONE); should FALL below -log(miss_rate) ~= 1.9
            #   selector_none_accuracy agreement of "NONE is argmax" with "slot is a miss"; a head
            #                          that learned nothing sits at max(miss, 1-miss) ~= 0.85
            # `selector_coverage` (already above) remains the EMPIRICAL top-k hit rate and is
            # unchanged in meaning.
            ratio_metrics.update(
                {
                    "selector_none_nll": (
                        selector_err_ce_num.detach(),
                        selector_err_den.detach(),
                    ),
                    "selector_none_accuracy": (
                        selector_err_correct_num.detach(),
                        selector_err_den.detach(),
                    ),
                }
            )
        if self._selector_err_objective_enabled:
            ratio_metrics.update(
                {
                    "selector_err_loss": (
                        selector_err_ce_num.detach(),
                        selector_err_den.detach(),
                    ),
                    "selector_err_accuracy": (
                        selector_err_correct_num.detach(),
                        selector_err_den.detach(),
                    ),
                    "selector_err_frontier_rate": (
                        selector_err_den.detach(),
                        accuracy_denom.detach(),
                    ),
                }
            )
        metrics: Dict[str, object] = {
            "accuracy_denom": accuracy_denom.detach(),
            "ratio_metrics": ratio_metrics,
        }
        if has_selector_objective:
            metrics["selector_loss_alpha"] = effective_selector_alpha
        if self._selector_err_objective_enabled:
            metrics["selector_err_loss_alpha"] = effective_err_alpha
        # Reduce all chunks before flooring the denominator so the result does
        # not depend on how many chunks happen to contain no effective weight.
        denominator_floor = torch.finfo(loss_denominator.dtype).tiny
        loss = loss_num / loss_denominator.clamp_min(denominator_floor)
        metrics["loss_terms"] = (loss_num, loss_denominator.detach())
        accuracy = correct_num / accuracy_denom
        return loss, accuracy, metrics


class OnlineDominoModel(OnlineDFlashModel):
    """Domino online training wrapper over DFlash block-parallel components."""

    def __init__(
        self,
        draft_model: DFlashDraftModel,
        target_lm_head: nn.Module,
        target_embed_tokens: nn.Module,
        mask_token_id: int,
        block_size: int = 16,
        attention_backend: str = "flex_attention",
        num_anchors: int = 512,
        loss_decay_gamma: Optional[float] = None,
        objective_chunk_blocks: int = 128,
        shift_label: bool = False,
    ):
        super().__init__(
            draft_model=draft_model,
            target_lm_head=target_lm_head,
            target_embed_tokens=target_embed_tokens,
            mask_token_id=mask_token_id,
            block_size=block_size,
            attention_backend=attention_backend,
            num_anchors=num_anchors,
            loss_decay_gamma=loss_decay_gamma,
            objective_chunk_blocks=objective_chunk_blocks,
            loss_type="dflash",
        )
        self.shift_label = shift_label
        self._use_fused_domino_ce = (
            os.environ.get("SPECFORGE_DOMINO_TRITON_CE", "1") == "1"
        )

    def _build_domino_head_inputs(
        self,
        input_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        target_ids: torch.Tensor,
        output_hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, n, bs = target_ids.shape
        hidden4d = output_hidden.reshape(bsz, n, bs, output_hidden.shape[-1])

        prev_ids = target_ids
        if self.shift_label:
            prev_offsets = torch.arange(
                0, self.block_size, device=input_ids.device
            ).view(1, 1, -1)
            prev_indices = (anchor_positions.unsqueeze(-1) + prev_offsets).clamp(
                max=input_ids.size(1) - 1
            )
            prev_ids = torch.gather(
                input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
                2,
                prev_indices,
            )

        return hidden4d, prev_ids

    def _domino_objective_chunk_terms(
        self,
        hidden: torch.Tensor,
        prev_ids: torch.Tensor,
        target_ids: torch.Tensor,
        weight_mask: torch.Tensor,
        eval_weight_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """Return additive Domino loss and telemetry terms for one block slice."""
        from specforge.core.domino_loss import domino_weighted_cross_entropy

        batch_size, num_blocks, block_size, hidden_size = hidden.shape
        base_logits = self.lm_head(
            hidden.reshape(batch_size, num_blocks * block_size, hidden_size)
        ).reshape(batch_size, num_blocks, block_size, -1)
        head_token_ids = prev_ids if self.shift_label else target_ids
        head_token_embeddings = self.embed_tokens(head_token_ids)
        correction_logits = self.draft_model.compute_correction_logits(
            hidden_states=hidden,
            prev_token_embeddings=head_token_embeddings,
        )
        final_num, base_num, predicted_ids, base_predicted_ids = (
            domino_weighted_cross_entropy(
                base_logits.reshape(-1, base_logits.shape[-1]),
                correction_logits.reshape(-1, correction_logits.shape[-1]),
                target_ids.reshape(-1),
                weight_mask.reshape(-1),
                block_size=block_size,
                suffix_start=self.draft_model.suffix_start,
                use_fused=self._use_fused_domino_ce and base_logits.is_cuda,
            )
        )
        loss_den = weight_mask.sum()

        with torch.no_grad():
            predicted_ids = predicted_ids.reshape_as(target_ids)
            base_predicted_ids = base_predicted_ids.reshape_as(target_ids)
            binary_accuracy_mask = eval_weight_mask > 0.5
            correct_num = (
                ((predicted_ids == target_ids) & binary_accuracy_mask).sum().float()
            )
            base_correct_num = (
                ((base_predicted_ids == target_ids) & binary_accuracy_mask)
                .sum()
                .float()
            )
            accuracy_den = eval_weight_mask.sum()

            valid_mask = eval_weight_mask > 0
            accepted = compute_accept_len(predicted_ids, target_ids, valid_mask)
            base_accepted = compute_accept_len(
                base_predicted_ids,
                target_ids,
                valid_mask,
            )
            valid_blocks = valid_mask.any(dim=-1).float()
            accept_num = ((accepted + 1.0) * valid_blocks).sum()
            base_accept_num = ((base_accepted + 1.0) * valid_blocks).sum()
            accept_den = valid_blocks.sum()

        return (
            final_num,
            base_num,
            loss_den,
            correct_num,
            base_correct_num,
            accuracy_den,
            accept_num,
            base_accept_num,
            accept_den,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        lambda_base: float = 0.0,
        max_valid_anchors: Optional[int] = None,
    ):
        """Parallel Domino training forward pass."""
        if self.attention_backend == "flex_attention" and not FLEX_ATTENTION_AVAILABLE:
            raise ValueError(
                "flex_attention is not available on this device; use sdpa/eager."
            )
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        anchor_positions, block_keep_mask, output_hidden = self._forward_draft_blocks(
            input_ids=input_ids,
            hidden_states=hidden_states,
            loss_mask=loss_mask,
            max_valid_anchors=max_valid_anchors,
        )

        label_start = 1 if self.shift_label else 0
        label_offsets = torch.arange(
            label_start, label_start + self.block_size, device=device
        ).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        valid_label_mask = label_indices < seq_len
        safe_target_indices = label_indices.clamp(max=seq_len - 1)

        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_target_indices,
        )

        bsz, n, bs = target_ids.shape
        hidden4d, prev_ids = self._build_domino_head_inputs(
            input_ids=input_ids,
            anchor_positions=anchor_positions,
            target_ids=target_ids,
            output_hidden=output_hidden,
        )
        weight_mask = (
            block_keep_mask.unsqueeze(-1).expand(-1, -1, self.block_size).float()
        )
        weight_mask = weight_mask * valid_label_mask.float()

        if not self.shift_label:
            pos_in_block = torch.arange(self.block_size, device=device).view(1, 1, -1)
            weight_mask = weight_mask * (pos_in_block > 0).float()

        original_loss_mask_gathered = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_target_indices,
        )
        weight_mask = weight_mask * original_loss_mask_gathered

        eval_weight_mask = weight_mask.clone()

        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            k = torch.arange(self.block_size, device=device).view(1, 1, -1)
            offset = 0 if self.shift_label else 1
            decay_weights = torch.exp(
                -(k - offset).clamp(min=0).float() / self.loss_decay_gamma
            )
            weight_mask = weight_mask * decay_weights

        (
            final_num,
            base_num,
            loss_den,
            correct_num,
            base_correct_num,
            accuracy_denom,
            accept_num,
            base_accept_num,
            accept_den,
        ) = checkpointed_chunk_reduce(
            self._domino_objective_chunk_terms,
            hidden4d,
            prev_ids,
            target_ids,
            weight_mask,
            eval_weight_mask,
            chunk_size=self.objective_chunk_blocks,
            dim=1,
        )

        valid_token_count = loss_den + 1e-6
        final_loss = final_num / valid_token_count
        base_loss = base_num / valid_token_count
        loss = (1.0 - lambda_base) * final_loss + lambda_base * base_loss
        accuracy = correct_num / (accuracy_denom + 1e-6)
        metrics = {
            "final_loss": final_loss.detach(),
            "base_loss": base_loss.detach(),
            "base_accuracy": (base_correct_num / (accuracy_denom + 1e-6)).detach(),
            "accept_len": (accept_num / (accept_den + 1e-6)).detach(),
            "base_accept_len": (base_accept_num / (accept_den + 1e-6)).detach(),
            # Telemetry does not participate in the objective. Keeping this as
            # a host scalar avoids a tiny H2D copy that otherwise synchronizes
            # the whole forward stream once per micro-step.
            "lambda_base": float(lambda_base),
            "accuracy_denom": accuracy_denom.detach(),
        }

        return loss, accuracy, metrics


class OnlineDSparkModel(OnlineDFlashModel):
    """DSpark online training wrapper over DFlash block-parallel components."""

    def __init__(
        self,
        draft_model: DFlashDraftModel,
        target_lm_head: nn.Module,
        target_embed_tokens: nn.Module,
        mask_token_id: int,
        block_size: int = 16,
        attention_backend: str = "flex_attention",
        num_anchors: int = 512,
        loss_decay_gamma: Optional[float] = None,
        dspark_ce_loss_alpha: float = 0.1,
        dspark_l1_loss_alpha: float = 0.9,
        dspark_confidence_head_alpha: float = 1.0,
        objective_chunk_blocks: int = 128,
    ):
        super().__init__(
            draft_model=draft_model,
            target_lm_head=target_lm_head,
            target_embed_tokens=target_embed_tokens,
            mask_token_id=mask_token_id,
            block_size=block_size,
            attention_backend=attention_backend,
            num_anchors=num_anchors,
            loss_decay_gamma=loss_decay_gamma,
            objective_chunk_blocks=objective_chunk_blocks,
            loss_type="dflash",
        )
        if dspark_ce_loss_alpha < 0:
            raise ValueError("dspark_ce_loss_alpha must be >= 0")
        if dspark_l1_loss_alpha < 0:
            raise ValueError("dspark_l1_loss_alpha must be >= 0")
        if dspark_confidence_head_alpha < 0:
            raise ValueError("dspark_confidence_head_alpha must be >= 0")
        self.loss_type = "dspark"
        self.dspark_ce_loss_alpha = float(dspark_ce_loss_alpha)
        self.dspark_l1_loss_alpha = float(dspark_l1_loss_alpha)
        self.dspark_confidence_head_alpha = float(dspark_confidence_head_alpha)

    def _build_dspark_labels_and_mask(
        self,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_len = input_ids.shape[1]
        device = input_ids.device
        label_offsets = torch.arange(1, self.block_size + 1, device=device).view(
            1, 1, -1
        )
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        safe_label_indices = label_indices.clamp(max=seq_len - 1)
        safe_label_indices = torch.where(
            block_keep_mask.unsqueeze(-1),
            safe_label_indices,
            torch.zeros_like(safe_label_indices),
        )
        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )

        target_valid = label_indices < seq_len
        target_loss_mask = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, label_indices.size(1), -1),
            2,
            safe_label_indices,
        )
        eval_mask = target_valid & (target_loss_mask > 0.5)
        eval_mask = eval_mask & block_keep_mask.unsqueeze(-1)
        eval_mask = eval_mask.to(torch.int32).cumprod(dim=-1).bool()
        return target_ids, eval_mask, safe_label_indices

    def _dspark_loss_weight_mask(
        self,
        eval_mask: torch.Tensor,
    ) -> torch.Tensor:
        loss_weight_mask = eval_mask.to(torch.float32)
        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            positions = torch.arange(self.block_size, device=eval_mask.device).view(
                1, 1, -1
            )
            decay_weights = torch.exp(-positions.float() / float(self.loss_decay_gamma))
            loss_weight_mask = loss_weight_mask * decay_weights
        return loss_weight_mask

    def _aligned_target_hidden(
        self,
        target_last_hidden_states: torch.Tensor,
        safe_label_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Gather the target state that predicts each DSpark label token."""

        target_pred_indices = (safe_label_indices - 1).clamp(min=0)
        batch_size = target_last_hidden_states.shape[0]
        hidden_size = target_last_hidden_states.shape[-1]
        gather_indices = target_pred_indices.reshape(batch_size, -1, 1).expand(
            -1, -1, hidden_size
        )
        return torch.gather(
            target_last_hidden_states,
            1,
            gather_indices,
        ).reshape(*safe_label_indices.shape, hidden_size)

    def _dspark_objective_chunk_terms(
        self,
        hidden: torch.Tensor,
        prev_token_ids: torch.Tensor,
        target_ids: torch.Tensor,
        loss_weights: torch.Tensor,
        eval_mask: torch.Tensor,
        aligned_target_hidden: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, ...]:
        """Return additive loss and telemetry numerators for one block slice."""

        batch_size, num_blocks, block_size, hidden_size = hidden.shape
        base_logits = self.lm_head(
            hidden.reshape(batch_size, num_blocks * block_size, hidden_size)
        ).reshape(batch_size, num_blocks, block_size, -1)
        draft_logits = self.draft_model.apply_logits_head(
            base_logits,
            prev_token_ids=prev_token_ids,
            hidden_states=hidden,
        )
        vocab_size = draft_logits.shape[-1]
        cross_entropy = F.cross_entropy(
            draft_logits.reshape(-1, vocab_size),
            target_ids.reshape(-1),
            reduction="none",
        ).reshape_as(target_ids)
        ce_num = (cross_entropy * loss_weights).sum()

        zero = ce_num.new_zeros(())
        l1_num = zero
        confidence_num = zero
        confidence_error_num = zero
        teacher_agreement_num = zero
        teacher_top1_num = zero
        draft_top1_num = zero
        tau_num = zero
        tau_den = zero
        accept_probability = None

        draft_probabilities = None
        teacher_ids = None
        if aligned_target_hidden is not None:
            with torch.no_grad():
                target_logits = self.lm_head(
                    aligned_target_hidden.reshape(
                        batch_size,
                        num_blocks * block_size,
                        hidden_size,
                    )
                ).reshape_as(draft_logits)
                target_probabilities = torch.softmax(target_logits.float(), dim=-1)
                teacher_ids = target_logits.argmax(dim=-1)
            draft_probabilities = torch.softmax(draft_logits.float(), dim=-1)
            l1_per_token = (
                (draft_probabilities - target_probabilities).abs().sum(dim=-1)
            )
            accept_probability = (1.0 - 0.5 * l1_per_token).clamp(0.0, 1.0)
            if self.dspark_l1_loss_alpha > 0:
                l1_num = (l1_per_token * loss_weights).sum()

        confidence_pred = self.draft_model.predict_confidence(
            hidden,
            prev_token_ids=prev_token_ids,
        )
        if confidence_pred is not None and self.dspark_confidence_head_alpha > 0:
            if accept_probability is None:
                raise ValueError(
                    "DSpark confidence loss requires target_last_hidden_states"
                )
            confidence_per_token = F.binary_cross_entropy_with_logits(
                confidence_pred.float(),
                accept_probability.detach(),
                reduction="none",
            )
            confidence_num = (confidence_per_token * loss_weights).sum()
            confidence_error_num = (
                (confidence_pred.float().sigmoid() - accept_probability).abs()
                * loss_weights
            ).sum()

        with torch.no_grad():
            predicted_ids = draft_logits.argmax(dim=-1)
            correct = ((predicted_ids == target_ids) & eval_mask).float()
            correct_num = correct.sum()
            eval_den = eval_mask.float().sum()
            ce_position_num = (cross_entropy.detach() * eval_mask).sum(dim=(0, 1))
            correct_position_num = correct.sum(dim=(0, 1))
            position_den = eval_mask.float().sum(dim=(0, 1))
            if aligned_target_hidden is not None:
                assert draft_probabilities is not None and teacher_ids is not None
                teacher_agreement_num = (
                    (predicted_ids == teacher_ids).float() * eval_mask
                ).sum()
                teacher_top1_num = (
                    target_probabilities.max(dim=-1).values * eval_mask
                ).sum()
                draft_top1_num = (
                    draft_probabilities.max(dim=-1).values * eval_mask
                ).sum()
                valid_blocks = eval_mask.any(dim=-1).float()
                accepted_expectation = (
                    accept_probability.detach() * eval_mask
                ).cumprod(dim=-1).sum(dim=-1) + 1.0
                tau_num = (accepted_expectation * valid_blocks).sum()
                tau_den = valid_blocks.sum()

        return (
            ce_num,
            l1_num,
            confidence_num,
            confidence_error_num,
            correct_num,
            eval_den,
            ce_position_num,
            correct_position_num,
            position_den,
            teacher_agreement_num,
            teacher_top1_num,
            draft_top1_num,
            tau_num,
            tau_den,
        )

    def _compute_dspark_loss(
        self,
        *,
        output_hidden: torch.Tensor,
        target_ids: torch.Tensor,
        eval_mask: torch.Tensor,
        prev_token_ids: torch.Tensor,
        safe_label_indices: torch.Tensor,
        target_last_hidden_states: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        """Token-pooled DSpark objective with bounded vocab-logit memory."""

        batch_size, num_blocks, block_size = target_ids.shape
        hidden_4d = output_hidden.reshape(
            batch_size,
            num_blocks,
            block_size,
            -1,
        )
        loss_weights = self._dspark_loss_weight_mask(eval_mask)
        local_loss_den = loss_weights.sum()
        need_target = self.dspark_l1_loss_alpha > 0 or (
            self.dspark_confidence_head_alpha > 0
            and getattr(self.draft_model, "confidence_head", None) is not None
        )
        aligned_target_hidden = None
        if need_target:
            if target_last_hidden_states is None:
                raise ValueError(
                    "DSpark L1/confidence loss requires target_last_hidden_states"
                )
            aligned_target_hidden = self._aligned_target_hidden(
                target_last_hidden_states,
                safe_label_indices,
            )

        totals = checkpointed_chunk_reduce(
            self._dspark_objective_chunk_terms,
            hidden_4d,
            prev_token_ids,
            target_ids,
            loss_weights,
            eval_mask,
            aligned_target_hidden,
            chunk_size=self.objective_chunk_blocks,
            dim=1,
        )

        (
            ce_num,
            l1_num,
            confidence_num,
            confidence_error_num,
            correct_num,
            eval_den,
            ce_position_num,
            correct_position_num,
            position_den,
            teacher_agreement_num,
            teacher_top1_num,
            draft_top1_num,
            tau_num,
            tau_den,
        ) = totals

        global_loss_den = local_loss_den.detach().clone()
        world_size = 1
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            if world_size > 1:
                dist.all_reduce(global_loss_den, op=dist.ReduceOp.SUM)
        if float(global_loss_den) <= 0:
            raise ValueError("DSpark objective has no supervised target tokens")
        loss = (
            world_size
            * (
                self.dspark_ce_loss_alpha * ce_num
                + self.dspark_l1_loss_alpha * l1_num
                + self.dspark_confidence_head_alpha * confidence_num
            )
            / global_loss_den
        )

        ratio_metrics = {
            "acc": (correct_num, eval_den),
            "ce_loss": (ce_num.detach(), local_loss_den.detach()),
            "l1_loss": (l1_num.detach(), local_loss_den.detach()),
            "confidence_loss": (
                confidence_num.detach(),
                local_loss_den.detach(),
            ),
            "confidence_abs_error": (
                confidence_error_num.detach(),
                local_loss_den.detach(),
            ),
            "ce_position": (ce_position_num, position_den),
            "accuracy_position": (correct_position_num, position_den),
        }
        if aligned_target_hidden is not None:
            ratio_metrics.update(
                {
                    "teacher_agreement": (teacher_agreement_num, eval_den),
                    "teacher_top1_prob": (teacher_top1_num, eval_den),
                    "draft_top1_prob": (draft_top1_num, eval_den),
                    "tau_probabilistic": (tau_num, tau_den),
                }
            )
        metrics: Dict[str, object] = {
            "ratio_metrics": {
                name: (numerator.detach(), denominator.detach())
                for name, (numerator, denominator) in ratio_metrics.items()
            },
            "accuracy_denom": eval_den.detach(),
        }
        accuracy = correct_num / eval_den.clamp_min(1.0)
        return loss, {"accuracy": accuracy.detach(), **metrics}

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        target_last_hidden_states: Optional[torch.Tensor] = None,
        max_valid_anchors: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
        """Parallel DSpark training forward pass."""
        if self.attention_backend == "flex_attention" and not FLEX_ATTENTION_AVAILABLE:
            raise ValueError(
                "flex_attention is not available on this device; use sdpa/eager."
            )
        anchor_positions, block_keep_mask, output_hidden = self._forward_draft_blocks(
            input_ids=input_ids,
            hidden_states=hidden_states,
            loss_mask=loss_mask,
            max_valid_anchors=max_valid_anchors,
        )

        (
            target_ids,
            eval_mask,
            safe_label_indices,
        ) = self._build_dspark_labels_and_mask(
            input_ids=input_ids,
            loss_mask=loss_mask,
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
        )
        anchor_token_ids = torch.gather(input_ids, 1, anchor_positions)
        prev_token_ids = torch.cat(
            [anchor_token_ids.unsqueeze(-1), target_ids[:, :, :-1]],
            dim=-1,
        )
        loss, metrics = self._compute_dspark_loss(
            output_hidden=output_hidden,
            target_ids=target_ids,
            eval_mask=eval_mask,
            prev_token_ids=prev_token_ids,
            safe_label_indices=safe_label_indices,
            target_last_hidden_states=target_last_hidden_states,
        )
        accuracy = metrics.pop("accuracy")
        return loss, accuracy, metrics
