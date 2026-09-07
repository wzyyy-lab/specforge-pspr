#!/usr/bin/env python3
"""Gate for PSPR-cascade: the keep/repair gate reads the ranker's corrected scores.

The claim being funded, stated so it cannot be re-justified after the fact
--------------------------------------------------------------------------
The champion deploys ``--gate-tau 0 --gate-rho 3 --gate-theta 0``: repair iff
``max_alt_prob > 3 * p0``.  On the held-out accounting, of 5789 slots whose target is a non-base
candidate, the target is the ranker's BEST alternative 66.86% of the time but also outranks the base
score only 44.05% of the time, and 36.71% survive the gate.  The 22.81pp difference is entirely "the
base score ranks ahead of the target".

What this gate does NOT assert, because both statements were wrong:
  * NOT "capped at 44.05% for any rho".  ``p_alt > tau + rho * p0`` with ``tau=0`` and ``rho < 1``
    admits alternatives that score BELOW candidate 0, approaching always-take-best-alternative as
    ``rho -> 0``.  The cap belongs to ``rho >= 1``, which is where the deployed rho=3 sits.  Check 12
    states exactly that and no more.
  * NOT "66.86% is the cascade's ceiling".  That rate is conditional on the champion's trajectory;
    repairing a first error moves every later block boundary and GRU state.  No offline number bounds
    the deployed result, which is why the 500-step run is a measurement.

What it does assert: that the cascade is a strict superset of the champion's deployed policy at step 0,
that all three code paths that can produce a gate logit produce the SAME logit, and that no path can
silently run the gate blind.

Checks
------
  1.  reference_config announces CascadeCorrector, not the parent, and carries the cascade knobs
  2.  only the cascade tensors are new; no champion tensor is dropped
  3.  warm_start_optional_key_groups is parent groups UNION the cascade group, enumerated from live
      modules, and the cascade group is exactly the cascade-only tensor set
  4.  the REAL warm-start loader loads a real ClozeCorrector-shaped state into a real cascade model
  5.  the duplicated score_candidates body has not drifted: scores bit-identical to the parent
  6.  the anchor is exactly LSE(alt) - s0 - log(rho), parameter-free, and is the whole gate at step 0
  7.  step 0 reproduces the champion's rho decisions slot-for-slot, on random and on near-tie scores,
      for several rho -- including ties
  8.  that zero state survives a post_init-style re-application of _init_weights
  9.  err_only trains exactly the gate tensors (base_gain included) and freezes the ranker, using the
      host's real freeze walk
  10. training / native-serving / external-serving gate logits agree with a NONZERO score head
  11. every way of running the gate without scores raises; the serving handoff is identity-checked
  12. feature sets are shift-invariant, carry no exact duplicate channel, and the summary set is
      permutation-invariant
  13. detach cuts the gradient path from the gate into the ranker's delta
  14. the blind control arm differs from the cascade ONLY by the score head
  15. profitable_repair already labels REPAIR iff the target is a non-base candidate
"""

from __future__ import annotations

import importlib.util
import math
import pathlib
import sys
import tempfile

import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from transformers import Qwen3Config  # noqa: E402

from specforge.modeling.draft.pspr import LatticePathSelector  # noqa: E402
from specforge.modeling.draft.pspr_cascade import (  # noqa: E402
    CascadeCorrector,
    PSPRCascadeDraftModel,
    n_score_features,
)
from specforge.modeling.draft.pspr_cloze import (  # noqa: E402
    ClozeCorrector,
    PSPRClozeDraftModel,
)

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


KW = dict(
    hidden_size=64,
    vocab_size=97,
    top_k=16,
    d=32,
    n_layers=2,
    n_heads=4,
    state_dim=32,
    delta_hidden=64,
    max_slots=8,
    dropout=0.0,
)
ANCHOR_RHO = 3.0


def build_pair(**overrides):
    torch.manual_seed(0)
    parent = ClozeCorrector(**KW)
    torch.manual_seed(0)
    child = CascadeCorrector(**KW, **overrides)
    embed = nn.Embedding(KW["vocab_size"], KW["hidden_size"])
    torch.manual_seed(7)
    nn.init.normal_(embed.weight)
    parent.bind_target_embedding(embed)
    child.bind_target_embedding(embed)
    # Copy every shared tensor so the two rankers are byte-identical; the child then differs only by
    # the cascade tensors.
    missing = child.load_state_dict(parent.state_dict(), strict=False)
    parent.eval()
    child.eval()
    return parent, child, missing


def sample_batch(seed: int = 3, B: int = 2, H: int = 5):
    torch.manual_seed(seed)
    return dict(
        candidate_ids=torch.randint(0, KW["vocab_size"], (B, H, KW["top_k"])),
        unary_logits=torch.randn(B, H, KW["top_k"]).log_softmax(-1),
        hidden_states=torch.randn(B, H, KW["hidden_size"]),
        predecessor_ids=torch.randint(0, KW["vocab_size"], (B, H)),
    )


def tiny_config(architecture: str, **dflash_overrides) -> Qwen3Config:
    config = Qwen3Config(
        architectures=[architecture],
        block_size=4,
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=1,
        num_target_layers=4,
        head_dim=4,
        max_position_embeddings=64,
        vocab_size=32,
    )
    config._attn_implementation = "eager"
    dflash = dict(
        mask_token_id=5,
        target_layer_ids=[0],
        selector_top_k=4,
        selector_dim=16,
        selector_layers=1,
        selector_heads=2,
        selector_delta_hidden=16,
        selector_max_slots=8,
        selector_dropout=0.0,
        selector_direct_hidden=True,
        selector_decision_mode="keep_repair",
        selector_state_dim=16,
        selector_use_state=True,
        selector_train_scope="err_only",
        selector_err_use_state=False,
        freeze_backbone=True,
        selector_bidirectional=True,
    )
    dflash.update(dflash_overrides)
    config.dflash_config = dflash
    return config


def main() -> None:  # noqa: C901
    parent, child, missing = build_pair(err_score_anchor_rho=ANCHOR_RHO)
    batch = sample_batch()
    cascade_only = set(child.state_dict()) - set(parent.state_dict())

    # ---- 1: provenance ---------------------------------------------------------------------------
    reference = child.reference_config()
    check(
        "reference_config announces CascadeCorrector and carries the cascade knobs",
        reference.get("model_type") == "CascadeCorrector"
        and reference.get("err_score_features") == child.err_score_features
        and reference.get("err_score_anchor_rho") == ANCHOR_RHO
        and reference.get("err_score_enabled") is True,
        f"model_type={reference.get('model_type')!r}, "
        f"features={reference.get('err_score_features')!r}, "
        f"anchor_rho={reference.get('err_score_anchor_rho')}",
    )

    # ---- 2: tensor delta -------------------------------------------------------------------------
    check(
        "only the cascade tensors are new, and no champion tensor is dropped",
        sorted({n.split(".")[0] for n in cascade_only})
        == ["err_base_gain", "err_score_head", "err_score_ln"]
        and not (set(parent.state_dict()) - set(child.state_dict()))
        and not missing.unexpected_keys,
        f"new roots={sorted({n.split('.')[0] for n in cascade_only})}, "
        f"dropped={sorted(set(parent.state_dict()) - set(child.state_dict()))}",
    )

    # ---- 3/4: the warm start must be declared AND actually work ---------------------------------
    # model_loading.py applies `warm_start_optional_prefixes` all-or-nothing per prefix and SKIPS it
    # whenever the source carries any key under that prefix.  A champion checkpoint does carry
    # candidate_selector.*, so the prefix exemption never fires and the cascade tensors would be
    # reported as required-but-missing: the launch would die after loading, not after training, but it
    # would die.  `warm_start_optional_key_groups` is the intended escape.
    torch.manual_seed(0)
    real_cascade = PSPRCascadeDraftModel(
        tiny_config("PSPRCascadeDraftModel", selector_err_score_anchor_rho=ANCHOR_RHO)
    )
    torch.manual_seed(0)
    real_cloze = PSPRClozeDraftModel(tiny_config("PSPRClozeDraftModel"))
    groups = tuple(tuple(g) for g in real_cascade.warm_start_optional_key_groups)
    inherited = tuple(tuple(g) for g in PSPRClozeDraftModel.warm_start_optional_key_groups)
    real_new = set(real_cascade.state_dict()) - set(real_cloze.state_dict())
    check(
        "warm_start_optional_key_groups re-emits the parent's groups (a property must not shadow it)",
        all(g in groups for g in inherited) and len(groups) == len(inherited) + 1,
        f"{len(groups)} groups, parent contributed {len(inherited)}",
    )
    check(
        "the cascade group is EXACTLY the cascade-only tensor set, enumerated from live modules",
        set(groups[-1]) == real_new,
        f"declared={len(groups[-1])}, actual={len(real_new)}, "
        f"symmetric difference={sorted(set(groups[-1]) ^ real_new)}",
    )

    from specforge.training.model_loading import warm_start_draft_model  # noqa: E402

    with tempfile.TemporaryDirectory() as tmp:
        state_path = pathlib.Path(tmp) / "step1" / "training_state.pt"
        state_path.parent.mkdir(parents=True)
        torch.save(
            {
                "draft_state_dict": real_cloze.state_dict(),
                "strategy": "online",
            },
            state_path,
        )
        loaded_error = None
        try:
            report = warm_start_draft_model(
                real_cascade,
                str(state_path),
                draft_config=real_cascade.config,
                strategy="online",
            )
        except Exception as exc:  # noqa: BLE001
            report = None
            loaded_error = repr(exc)
    check(
        "the REAL warm-start loader accepts a real ClozeCorrector state into a cascade model",
        report is not None
        and report.loaded_keys == len(real_cloze.state_dict())
        and set(report.missing_keys) == real_new,
        loaded_error
        or f"loaded={report.loaded_keys}/{len(real_cloze.state_dict())}, "
        f"missing={len(report.missing_keys)} (expected {len(real_new)})",
    )

    # A half-written head must still be a hard error, which is the whole point of the group being
    # all-or-nothing.  Drop one member from the source and re-run.
    with tempfile.TemporaryDirectory() as tmp:
        partial = dict(real_cloze.state_dict())
        partial["candidate_selector.err_base_gain.gain"] = torch.zeros(())
        state_path = pathlib.Path(tmp) / "step1" / "training_state.pt"
        state_path.parent.mkdir(parents=True)
        torch.save({"draft_state_dict": partial, "strategy": "online"}, state_path)
        torch.manual_seed(0)
        fresh = PSPRCascadeDraftModel(
            tiny_config("PSPRCascadeDraftModel", selector_err_score_anchor_rho=ANCHOR_RHO)
        )
        partial_rejected = False
        try:
            warm_start_draft_model(
                fresh, str(state_path), draft_config=fresh.config, strategy="online"
            )
        except ValueError:
            partial_rejected = True
    check(
        "a PARTIALLY present cascade head is still rejected (the group is all-or-nothing)",
        partial_rejected,
    )

    # ---- 5: no drift in the duplicated body -----------------------------------------------------
    with torch.no_grad():
        s_parent = parent.score_candidates(**batch)
        s_child = child.score_candidates(**batch)
        e_parent = parent.score_candidates(**batch, return_err=True)[1]
        s_child_err, e_child = child.score_candidates(**batch, return_err=True)
    check(
        "duplicated score_candidates has not drifted: scores bit-identical to the parent",
        torch.equal(s_parent, s_child) and torch.equal(s_child, s_child_err),
        f"max|d| = {(s_parent - s_child).abs().max().item():.3e}",
    )

    # ---- 6: the anchor is exactly the analytic champion logit -----------------------------------
    with torch.no_grad():
        expected_anchor = (
            torch.logsumexp(s_child[..., 1:].float(), dim=-1)
            - s_child[..., 0].float()
            - math.log(ANCHOR_RHO)
        )
    check(
        "at step 0 the gate logit IS the analytic anchor LSE(alt) - s0 - log(rho), exactly",
        torch.allclose(e_child, expected_anchor, atol=0.0, rtol=0.0),
        f"max|d| = {(e_child - expected_anchor).abs().max().item():.3e}",
    )
    check(
        "and it is NOT the parent's blind logit (that would be the historical latrepair m=0 policy)",
        not torch.allclose(e_child, e_parent, atol=1e-3),
        f"max|d| vs parent = {(e_child - e_parent).abs().max().item():.3e}",
    )
    check(
        "err_base_gain starts at zero, so the blind detector contributes nothing at step 0",
        float(child.err_base_gain["gain"].detach()) == 0.0,
        f"gain = {float(child.err_base_gain['gain'].detach())}",
    )

    # ---- 7: step 0 == champion rho decisions, slot for slot -------------------------------------
    # keep_repair repairs iff `e + log r_* > 0`; substituting the anchor gives `s_max - s0 > log rho`,
    # which is `p_alt > rho * p0` at tau=0.  Ties: the champion's `>` yields KEEP, and argmax over
    # [KEEP, REPAIR...] also yields KEEP because KEEP is index 0.  Both are tested.
    for rho in (1.0, 3.0, 10.0):
        _, gated, _ = build_pair(err_score_anchor_rho=rho)
        mismatch = 0
        total = 0
        for scores in (
            s_child,
            # near-ties: put the best alternative exactly at the rho boundary in probability space,
            # then perturb by +-0 to hit the equality case as well.
            torch.stack(
                [
                    torch.tensor(
                        [0.0, -math.log(rho)] + [-30.0] * (KW["top_k"] - 2)
                    ).expand(s_child.shape[0], s_child.shape[1], KW["top_k"])[i]
                    for i in range(s_child.shape[0])
                ]
            ),
        ):
            with torch.no_grad():
                err = (
                    torch.logsumexp(scores[..., 1:].float(), -1)
                    - scores[..., 0].float()
                    - math.log(rho)
                )
                action = LatticePathSelector.keep_repair_log_probs(scores, err).argmax(-1)
                champion = LatticePathSelector.select_margin_gate(
                    scores, rho=rho, tau=0.0
                ).squeeze(-1)
            mismatch += int((action != champion).sum())
            total += action.numel()
        check(
            f"step-0 factorized MAP == champion margin_gate(rho={rho:g}, tau=0) on every slot",
            mismatch == 0,
            f"{mismatch}/{total} mismatched",
        )
        del gated

    # ---- 8: the zero state must survive post_init re-running _init_weights ---------------------
    class Host(torch.nn.Module):
        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.normal_(module.bias, std=0.02)
            if getattr(module, "_pspr_zero_init", False):
                nn.init.zeros_(module.weight)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
            if isinstance(module, ClozeCorrector):
                module.restore_zero_init_contract()

    host = Host()
    with torch.no_grad():
        child.err_base_gain["gain"].fill_(0.77)
        child.err_score_head[-1].weight.normal_()
    for module in child.modules():
        host._init_weights(module)
    host._init_weights(child)
    with torch.no_grad():
        readout = float(child.err_score_head[-1].weight.abs().sum()) + float(
            child.err_score_head[-1].bias.abs().sum()
        )
        gain = float(child.err_base_gain["gain"].detach())
    check(
        "score read-out and base gain are restored to zero after a post_init-style re-init",
        readout == 0.0 and gain == 0.0,
        f"|readout|={readout:.3e}, gain={gain:.3e}",
    )

    # ---- 9: the host's REAL freeze walk ---------------------------------------------------------
    frozen_numel = real_cascade.apply_backbone_freeze()
    trainable = sorted(
        n for n, p in real_cascade.named_parameters() if p.requires_grad
    )
    trainable_roots = sorted({n.split(".")[1] for n in trainable})
    check(
        "err_only trains exactly the gate tensors, base gain included",
        trainable_roots == ["err_base_gain", "err_head", "err_score_head", "err_score_ln", "err_seed_in"],
        f"trainable roots = {trainable_roots}",
    )
    check(
        "err_base_gain is TRAINABLE (a bare nn.Parameter would be silently frozen at zero)",
        "candidate_selector.err_base_gain.gain" in trainable,
        f"{len(trainable)} trainable tensors, {frozen_numel} frozen elements",
    )
    check(
        "and the ranker's delta path plus the whole backbone stay frozen",
        not any(
            n.startswith("candidate_selector.delta_")
            or n.startswith("candidate_selector.blocks")
            or n.startswith("candidate_selector.gru")
            or n.startswith("model.")
            for n in trainable
        ),
        f"trainable = {[n.removeprefix('candidate_selector.') for n in trainable]}",
    )

    # ---- 10: training / native / external must agree with a NONZERO head -----------------------
    torch.manual_seed(21)
    _, live, _ = build_pair(err_score_anchor_rho=ANCHOR_RHO)
    with torch.no_grad():
        live.err_score_head[0].weight.normal_(std=0.5)
        live.err_score_head[0].bias.normal_(std=0.5)
        live.err_score_head[-1].weight.normal_(std=0.5)
        live.err_score_head[-1].bias.normal_(std=0.5)
        live.err_base_gain["gain"].fill_(0.6)
    live.eval()

    one = sample_batch(seed=5, B=1, H=3)
    with torch.no_grad():
        scores_train, err_train = live.score_candidates(**one, return_err=True)

    # (a) native serving: pspr_cloze.py:786-800 computes `scores = selector.score(zi, ...)` and then
    #     calls `selector.err_logits(zi, ...)` with the SAME zi and no scores argument.  Replay that
    #     exact call order here so the one-shot handoff is exercised the way serving uses it.
    with torch.no_grad():
        embedding = live._embedding()
        cand_emb = torch.nn.functional.embedding(one["candidate_ids"], embedding)
        unary = one["unary_logits"].float()
        margin = (unary[..., 0] - unary[..., 1]).clamp(-20.0, 20.0)
        mass = unary.exp().sum(-1).clamp(max=1.0)
        scalars = torch.stack([torch.zeros_like(margin), margin, mass], dim=-1)
        # Argument shapes exactly as score_candidates builds them: cloze_states takes the block
        # ANCHOR (predecessor_ids[..., 0]) while causal_states walks the whole seed path.
        z = live.cloze_states(
            one["hidden_states"],
            one["candidate_ids"],
            one["predecessor_ids"][..., 0],
            unary,
            scalars,
        )
        state = live.causal_states(
            torch.nn.functional.embedding(one["predecessor_ids"], embedding)
        )
        native_err = []
        native_scores = []
        for position in range(one["hidden_states"].shape[1]):
            zi = z[:, position]
            hi = one["hidden_states"][:, position]
            si = None if state is None else state[:, position]
            sc = live.score(zi, hi, cand_emb[:, position], unary[:, position], si)
            native_scores.append(sc)
            native_err.append(
                live.err_logits(
                    zi,
                    cand_emb[:, position, 0],
                    unary[:, position],
                    scalars[:, position],
                    hi,
                    si,
                )
            )
        native_err = torch.stack(native_err, dim=1)
        native_scores = torch.stack(native_scores, dim=1)
    check(
        "native serving's score/err_logits call order reproduces the training gate logit",
        torch.allclose(native_err, err_train, atol=1e-5)
        and torch.allclose(native_scores, scores_train, atol=1e-5),
        f"max|d err| = {(native_err - err_train).abs().max().item():.3e}, "
        f"max|d scores| = {(native_scores - scores_train).abs().max().item():.3e}",
    )
    # Prove the replay above matches the real serving source rather than a convenient rewrite.
    native_src = (
        pathlib.Path(__file__).resolve().parents[1]
        / "specforge/modeling/draft/pspr_cloze.py"
    ).read_text()
    check(
        "native serving really does pass the same zi to score() and err_logits()",
        "scores = selector.score(\n" in native_src
        and "zi, hi, candidate_embeddings[:, position], unary_logits[:, position], state"
        in native_src
        and "err_logits = selector.err_logits(\n                    zi," in native_src,
        "so the one-shot handoff keyed on z identity is the real call pattern",
    )

    # (b) external serving: the actual decode_lattice function.
    spec = importlib.util.spec_from_file_location(
        "decode_lattice",
        pathlib.Path(__file__).resolve().parents[2]
        / "TAPS-SP/scripts/decode_lattice.py",
    )
    decode = importlib.util.module_from_spec(spec)
    sys.path.insert(
        0, str(pathlib.Path(__file__).resolve().parents[2] / "TAPS-SP/scripts")
    )
    spec.loader.exec_module(decode)
    check(
        "decode_lattice recognises the cascade head by capability",
        decode.is_cascade(live) and not decode.is_cascade(parent),
    )
    with torch.no_grad():
        external_err = []
        for position in range(one["hidden_states"].shape[1]):
            external_err.append(
                decode.slot_err_logit(
                    live,
                    z[0],
                    state[:, position].unsqueeze(0),
                    unary[0],
                    scalars[0],
                    cand_emb[0],
                    one["hidden_states"],
                    position,
                    scores=scores_train[0, position],
                ).reshape(())
            )
        external_err = torch.stack(external_err).unsqueeze(0)
    check(
        "external serving (decode_lattice.slot_err_logit) reproduces the training gate logit",
        torch.allclose(external_err, err_train, atol=1e-5),
        f"max|d| = {(external_err - err_train).abs().max().item():.3e}",
    )
    external_src = (
        pathlib.Path(__file__).resolve().parents[2] / "TAPS-SP/scripts/decode_lattice.py"
    ).read_text()
    check(
        "every slot_err_logit call site in decode_lattice passes scores",
        external_src.count("slot_err_logit(") == 1 + external_src.count("scores=sc"),
        f"{external_src.count('slot_err_logit(') - 1} call sites, "
        f"{external_src.count('scores=sc')} carry scores",
    )

    # ---- 11: fail closed --------------------------------------------------------------------- --
    blind_args = (
        torch.zeros(2, 5, KW["d"]),
        torch.zeros(2, 5, KW["hidden_size"]),
        torch.zeros(2, 5, KW["top_k"]),
        torch.zeros(2, 5, 3),
        torch.zeros(2, 5, KW["hidden_size"]),
        None,
    )
    raised_no_scores = False
    try:
        with torch.no_grad():
            live.err_logits(*blind_args)
    except RuntimeError:
        raised_no_scores = True
    check(
        "err_logits with no scores and no handoff raises instead of running the blind gate",
        raised_no_scores,
    )
    raised_grad = False
    try:
        live.err_logits(*blind_args)
    except RuntimeError:
        raised_grad = True
    check(
        "a grad-enabled call refuses the handoff entirely (training must pass scores explicitly)",
        raised_grad,
    )
    raised_wrong_z = False
    with torch.no_grad():
        live.score(z[:, 0], one["hidden_states"][:, 0], cand_emb[:, 0], unary[:, 0], state[:, 0])
        try:
            live.err_logits(
                z[:, 1],
                cand_emb[:, 1, 0],
                unary[:, 1],
                scalars[:, 1],
                one["hidden_states"][:, 1],
                state[:, 1],
            )
        except RuntimeError:
            raised_wrong_z = True
    check(
        "the handoff is identity-checked: slot A's scores cannot gate slot B",
        raised_wrong_z,
    )
    raised_shape = False
    try:
        with torch.no_grad():
            live.err_logits(*blind_args[:-1], None, scores=torch.zeros(2, 5, KW["top_k"] + 1))
    except RuntimeError:
        raised_shape = True
    check("a wrong-width scores tensor raises", raised_shape)
    raised_external = False
    try:
        with torch.no_grad():
            decode.slot_err_logit(
                live, z[0], state[:, 0].unsqueeze(0), unary[0], scalars[0],
                cand_emb[0], one["hidden_states"], 0,
            )
    except ValueError:
        raised_external = True
    check(
        "decode_lattice refuses to decode a cascade checkpoint without scores",
        raised_external,
    )

    # ---- 12: feature-set hygiene ----------------------------------------------------------------
    torch.manual_seed(11)
    probe_scores = torch.randn(2, 5, KW["top_k"])
    for feature_set in ("margins", "margins_summary", "summary"):
        _, variant, _ = build_pair(err_score_features=feature_set)
        feats = variant.score_features(probe_scores)
        shifted = variant.score_features(probe_scores + 3.0)
        width_ok = feats.shape[-1] == n_score_features(KW["top_k"], feature_set)
        shift_ok = torch.allclose(feats, shifted, atol=1e-5)
        # No channel may be an exact duplicate or an exact affine copy of another: with a LayerNorm
        # downstream a duplicate silently rescales every other channel.
        flat = feats.reshape(-1, feats.shape[-1])
        duplicate = None
        for a in range(flat.shape[-1]):
            for b in range(a + 1, flat.shape[-1]):
                if torch.allclose(flat[:, a], flat[:, b], atol=1e-6):
                    duplicate = (a, b)
        # p0 is an exact function of the lse channel, so its ABSENCE is what is asserted.
        p0 = torch.softmax(probe_scores, dim=-1)[..., 0]
        carries_p0 = any(
            torch.allclose(feats[..., c], p0, atol=1e-6) for c in range(feats.shape[-1])
        )
        check(
            f"features[{feature_set}]: declared width, shift-invariant, no duplicate, no p0 channel",
            width_ok and shift_ok and duplicate is None and not carries_p0,
            f"width={feats.shape[-1]} (want {n_score_features(KW['top_k'], feature_set)}), "
            f"shift_ok={shift_ok}, duplicate={duplicate}, carries_p0={carries_p0}",
        )
        if feature_set == "summary":
            permuted = probe_scores.clone()
            idx = torch.randperm(KW["top_k"] - 1) + 1
            permuted[..., 1:] = probe_scores[..., idx]
            check(
                "features[summary] is permutation-invariant in the alternatives",
                torch.allclose(feats, variant.score_features(permuted), atol=1e-6),
            )
        if feature_set == "margins":
            check(
                "features[margins] is NOT permutation-invariant: base rank is retained on purpose",
                not torch.allclose(
                    feats,
                    variant.score_features(
                        torch.cat(
                            [probe_scores[..., :1], probe_scores[..., 1:].flip(-1)], dim=-1
                        )
                    ),
                    atol=1e-6,
                ),
            )

    # ---- 13: detach cuts the path into the ranker ----------------------------------------------
    live.train()
    err = live.score_candidates(**batch, return_err=True)[1]
    err.sum().backward()
    ranker_grad = sum(
        float(p.grad.abs().sum())
        for n, p in live.named_parameters()
        if n.startswith("delta_") and p.grad is not None
    )
    gate_grad = sum(
        float(p.grad.abs().sum())
        for n, p in live.named_parameters()
        if n.startswith("err_score_") and p.grad is not None
    )
    gain_grad = float(live.err_base_gain["gain"].grad.abs().sum())
    check(
        "the gate loss reaches the score head and the base gain",
        gate_grad > 0.0 and gain_grad > 0.0,
        f"score head grad = {gate_grad:.3e}, base gain grad = {gain_grad:.3e}",
    )
    check(
        "and NOT the ranker's delta path (features and anchor are detached)",
        ranker_grad == 0.0,
        f"delta grad = {ranker_grad:.3e}",
    )
    live.zero_grad(set_to_none=True)
    torch.manual_seed(0)
    attached = CascadeCorrector(**KW, err_score_detach=False)
    check(
        "err_score_detach is a real switch, not a hardcoded constant",
        attached.err_score_detach is False and live.err_score_detach is True,
    )

    # ---- 14: the blind control arm --------------------------------------------------------------
    _, control, _ = build_pair(err_score_enabled=False, err_score_anchor_rho=ANCHOR_RHO)
    control_only = set(control.state_dict()) - set(parent.state_dict())
    with torch.no_grad():
        control_err = control.score_candidates(**batch, return_err=True)[1]
    check(
        "the blind control differs from the cascade ONLY by the score head",
        control_only == {"err_base_gain.gain"}
        and cascade_only - control_only == {
            k for k in cascade_only if k.startswith("err_score_")
        },
        f"control-only tensors = {sorted(control_only)}",
    )
    check(
        "the control keeps the same anchor, so the arms share an identical step-0 policy",
        torch.allclose(control_err, e_child, atol=0.0, rtol=0.0),
        f"max|d| = {(control_err - e_child).abs().max().item():.3e}",
    )

    # ---- 15: the objective already has the right labels ----------------------------------------
    host_src = (
        pathlib.Path(__file__).resolve().parents[1]
        / "specforge/algorithms/common/dflash_family_model.py"
    ).read_text()
    check(
        "profitable_repair labels REPAIR iff the target is a non-base candidate",
        "repair_is_covered = target_is_candidate & target_candidate_index.ne(0)" in host_src
        and "err_target = repair_is_covered" in host_src,
        "so base-right and top-k misses are both KEEP; no new objective is needed",
    )

    # ---- 12b: the narrow, TRUE version of the ceiling claim -------------------------------------
    # Only rho >= 1 is capped by the s0 comparison.  rho < 1 is not, and the cascade's advantage must
    # not be argued from the false general statement.
    s = torch.zeros(1, 1, KW["top_k"])
    s[0, 0, 0] = 5.0
    s[0, 0, 1] = 1.0
    repairs = {
        rho: bool(
            LatticePathSelector.select_margin_gate(s, rho=rho, tau=0.0).squeeze().item()
        )
        for rho in (0.001, 0.01, 1.0, 3.0, 10.0)
    }
    check(
        "on a base-dominated slot every rho >= 1 keeps, while small rho repairs (the honest claim)",
        not repairs[1.0] and not repairs[3.0] and not repairs[10.0] and repairs[0.001],
        f"repairs by rho = {repairs}",
    )

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
