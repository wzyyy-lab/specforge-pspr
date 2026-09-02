"""Precomputed lattice-trace features: the data path the reference selector actually trained on.

The DFlash-family offline path stores whole sequences (``input_ids``/``loss_mask``/``hidden_states``)
and re-runs the draft backbone every step to obtain per-slot logits. The reference selector does not
work that way: its traces are collected by *running speculative decoding*, and each record is one
anchor's already-extracted lattice --

    top_token_ids [H, K]     the unary top-K candidate ids at each in-block slot
    top_log_probs [H, K]     their log-probs under the base draft
    draft_hidden  [H, Dh]    the draft's own hidden state at each slot
    g_block       [H]        ground truth, read off the COMMITTED sequence (-1 where unavailable)
    anchor_tok               the last committed token, i.e. the conditioning anchor
    position_entropy / top1_top2_margin / topk_mass  [H]   the lattice scalars

Two properties follow from *how* those records are produced, and neither is reachable from a
teacher-forced corpus dump:

* **Anchors are on-policy.** Collection advances ``start += accept + 1``, so the anchor distribution
  is the one decoding actually visits, not a uniform sample over token positions.
* **Labels are the target's own committed continuation.** Greedy speculative decoding is lossless, so
  ``g_block`` is exactly what the target would emit -- which is exactly the acceptance test at decode
  time. A corpus next-token label disagrees with it on ~22% of positions (measured).

So this module exists to train the selector on its own data instead of an approximation of it.
"""

from __future__ import annotations

from functools import partial

NORMALIZER_ID = "lattice_trace_v1"

# Keys the reference records always carry. ``anchor_thidden`` and the ``ctx_*`` group are optional in
# the reference loader too ("trace dirs may be mixed"), and the dh2048 selector config has
# ``thidden_dim: 0`` / ``pair_dim: 0``, i.e. it consumes neither -- so they are deliberately not read.
TRACE_KEYS = (
    "top_token_ids",
    "top_log_probs",
    "draft_hidden",
    "g_block",
    "position_entropy",
    "top1_top2_margin",
    "topk_mass",
    "drafted_top1",
    "anchor_tok",
)

# Transcribed from the reference collate: the margin feature is clamped before it is stacked.
MARGIN_CLAMP = 20.0


def align_top1(top_token_ids, top_log_probs, drafted_top1):
    """Swap candidate column 0 to the token DFlash actually drafts.

    Transcribed from ``TAPS-SP/scripts/train_pairwise.py::align_top1``, including its reasoning:
    ``lm_head`` runs in bf16, so the maximal logit is *exactly* tied across tokens at 2.92% of slots,
    and ``torch.topk`` breaks such ties differently from ``torch.argmax``. Column 0 therefore
    disagreed with ``drafted_top1`` at 2.92% of slots, which puts that much label noise on the single
    most important signal the selector has ("is the base top-1 already right?") and breaks the
    ``tau=inf == plain DFlash`` guarantee at decode. The tied entries have identical log-probs, so the
    swap leaves ``top_log_probs`` (and entropy/margin/mass) unchanged in value.

    Returns new tensors; the inputs are not modified.
    """
    import torch

    d1 = drafted_top1.long()
    top = top_token_ids.long()
    eq = top == d1.unsqueeze(1)
    j = torch.where(eq.any(dim=1), eq.long().argmax(dim=1), torch.zeros_like(d1))
    if not bool((j != 0).any()):
        return top, top_log_probs
    rows = torch.arange(top.shape[0])
    out_top = top.clone()
    out_lp = top_log_probs.clone()
    for tensor in (out_top, out_lp):
        column0 = tensor[:, 0].clone()
        tensor[:, 0] = tensor[rows, j]
        tensor[rows, j] = column0
    return out_top, out_lp


def target_index(top_token_ids, g_block):
    """Index of the ground-truth token inside the candidate pool, or -1 if it is not there.

    Transcribed from ``train_pairwise.py::precompute``. ``g_block < 0`` marks a slot with no label
    (past the end of the committed sequence); those must not become class 0 by accident, hence the
    explicit ``valid`` guard before the argmax.
    """
    import torch

    top = top_token_ids.long()
    g = g_block.long()
    horizon = top.shape[0]
    valid = g >= 0
    matches = (top == g.unsqueeze(1)) & valid.unsqueeze(1)
    has = matches.any(dim=1)
    return torch.where(
        has,
        matches.float().argmax(dim=1),
        torch.full((horizon,), -1, dtype=torch.long),
    )


def normalize_trace_record(raw, max_len: int = 0):
    """One reference trace record -> the tensors the selector objective consumes.

    ``max_len`` is accepted and ignored: a record is one fixed-width block, not a sequence, so there
    is nothing to truncate. Keeping the parameter lets this reuse the offline normalizer factory
    signature.
    """
    import torch

    top, log_probs = align_top1(
        raw["top_token_ids"], raw["top_log_probs"], raw["drafted_top1"]
    )
    g_block = raw["g_block"].long()
    scalars = torch.stack(
        [
            raw["position_entropy"].float(),
            raw["top1_top2_margin"].float().clamp(-MARGIN_CLAMP, MARGIN_CLAMP),
            raw["topk_mass"].float(),
        ],
        dim=-1,
    )
    anchor = raw["anchor_tok"]
    anchor = int(anchor) if not hasattr(anchor, "item") else int(anchor.item())
    # prefix[0] is the anchor (a committed token); prefix[t] for t>0 is the previous slot's ground
    # truth, clamped so an unlabelled tail cannot index the embedding with -1. Matches the reference
    # exactly: torch.cat([anchor, g[:, :-1].clamp(min=0)]).
    predecessor = torch.cat(
        [torch.tensor([anchor], dtype=torch.long), g_block[:-1].clamp(min=0)]
    )
    return {
        "candidate_ids": top.unsqueeze(0),
        "unary_logits": log_probs.float().unsqueeze(0),
        "draft_hidden": raw["draft_hidden"].float().unsqueeze(0),
        "lattice_scalars": scalars.unsqueeze(0),
        "predecessor_ids": predecessor.unsqueeze(0),
        "target_ids": g_block.unsqueeze(0),
        "target_index": target_index(top, g_block).unsqueeze(0),
    }


def build_trace_normalizer(max_len=0, **_topology):
    return partial(normalize_trace_record, max_len=max_len)


def build_trace_collator():
    """Stack fixed-width records. No padding: every record has the same [H, K] block shape."""

    def collate(features):
        import torch

        if not features:
            raise ValueError("cannot collate an empty trace batch")
        keys = tuple(features[0])
        for index, feature in enumerate(features[1:], start=1):
            if tuple(feature) != keys:
                raise ValueError(
                    f"trace record {index} exposes {sorted(feature)}, expected {sorted(keys)}"
                )
        shapes = {key: {tuple(f[key].shape[1:]) for f in features} for key in keys}
        ragged = {key: sorted(v) for key, v in shapes.items() if len(v) != 1}
        if ragged:
            raise ValueError(
                "trace records must share one block shape; mismatched: " f"{ragged}"
            )
        return {key: torch.cat([f[key] for f in features], dim=0) for key in keys}

    return collate


__all__ = [
    "MARGIN_CLAMP",
    "NORMALIZER_ID",
    "TRACE_KEYS",
    "align_top1",
    "build_trace_collator",
    "build_trace_normalizer",
    "normalize_trace_record",
    "target_index",
]
