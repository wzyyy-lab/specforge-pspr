import unittest

import torch
from torch import nn
from transformers import Qwen3Config

from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
from specforge.modeling.draft.pspr import LatticePathSelector
from specforge.modeling.draft.pspr_cloze import ClozeCorrector, PSPRClozeDraftModel


class _RecordingProposalSelector(nn.Module):
    """Minimal selector that exposes exactly what the online host passed to it."""

    selector_excludes_anchor = True
    top_k = 2

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.seen = {}

    def extract_lattice(self, logits):
        self.seen["objective_logits"] = logits.detach().clone()
        values, indices = logits.topk(self.top_k, dim=-1)
        scalars = torch.zeros(*logits.shape[:-1], 3, device=logits.device)
        return values, indices, scalars

    def score_candidates(
        self,
        *,
        candidate_ids,
        unary_logits,
        hidden_states,
        predecessor_ids,
        lattice_scalars,
    ):
        self.seen.update(
            candidate_ids=candidate_ids.detach().clone(),
            hidden_states=hidden_states.detach().clone(),
            predecessor_ids=predecessor_ids.detach().clone(),
            lattice_scalars=lattice_scalars.detach().clone(),
        )
        return unary_logits * self.scale


class _Draft(nn.Module):
    def __init__(self, selector):
        super().__init__()
        self.candidate_selector = selector


class _KeepRepairSelector(_RecordingProposalSelector):
    wants_err_objective = True

    def score_candidates(self, *, return_err=False, **kwargs):
        scores = super().score_candidates(**kwargs)
        # Proposal 0: keep; proposal 1: repair; proposal 2: keep. K=2 makes the
        # conditional replacement distribution a single action and isolates the
        # keep-vs-repair arithmetic in this test.
        err = torch.tensor([[[-2.0, 2.0, -2.0]]], dtype=scores.dtype)
        return (scores, err) if return_err else scores


class _LearnableKeepRepairSelector(_RecordingProposalSelector):
    wants_err_objective = True

    def __init__(self):
        super().__init__()
        self.err_bias = nn.Parameter(torch.tensor(0.0))

    def score_candidates(self, *, return_err=False, **kwargs):
        scores = super().score_candidates(**kwargs)
        err = self.err_bias.expand(scores.shape[:-1])
        return (scores, err) if return_err else scores


class _PerSlotKeepRepairSelector(_RecordingProposalSelector):
    """Independent error logits make per-slot supervision observable in tests."""

    wants_err_objective = True

    def __init__(self):
        super().__init__()
        self.err_bias = nn.Parameter(torch.zeros(3))

    def score_candidates(self, *, return_err=False, **kwargs):
        scores = super().score_candidates(**kwargs)
        if scores.shape[-2] != self.err_bias.numel():
            raise ValueError("test selector expects exactly three proposal slots")
        err = self.err_bias.view(1, 1, -1).expand(scores.shape[:-1])
        return (scores, err) if return_err else scores


class _SelectorWithTwoErrOnlyModules(_RecordingProposalSelector):
    wants_err_objective = True
    err_only_module_names = ("err_seed_in", "err_head")

    def __init__(self):
        super().__init__()
        self.err_seed_in = nn.Linear(3, 3, bias=False)
        self.err_head = nn.Linear(3, 1)


class _BlockSelector(_RecordingProposalSelector):
    """DFlash2-shaped local selector that retains the masked anchor slot."""

    selector_excludes_anchor = False


class PSPROnlineAlignmentTest(unittest.TestCase):
    def test_cloze_err_only_scope_preserves_scorer_and_trains_complete_gate(self):
        config = Qwen3Config(
            architectures=["PSPRClozeDraftModel"],
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
        config.dflash_config = {
            "freeze_backbone": True,
            "selector_top_k": 2,
            "selector_dim": 4,
            "selector_layers": 1,
            "selector_heads": 1,
            "selector_state_dim": 4,
            "selector_delta_hidden": 8,
            "selector_max_slots": 4,
            "selector_err_use_state": True,
            "selector_train_scope": "err_only",
        }
        model = PSPRClozeDraftModel(config)
        model.apply_backbone_freeze()

        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        self.assertTrue(trainable)
        self.assertTrue(
            all(
                name.startswith(
                    (
                        "candidate_selector.err_seed_in.",
                        "candidate_selector.err_head.",
                        "candidate_selector.err_state_ln.",
                        "candidate_selector.err_state_head.",
                    )
                )
                for name in trainable
            )
        )
        self.assertTrue(any("err_state_head" in name for name in trainable))
        self.assertFalse(model.candidate_selector.delta_w2.weight.requires_grad)
        self.assertFalse(model.candidate_selector.gru.weight_ih_l0.requires_grad)

    def test_cloze_state_gate_is_a_warm_start_noop_then_reads_committed_prefix(self):
        common = dict(
            hidden_size=8,
            vocab_size=11,
            top_k=2,
            d=4,
            n_layers=1,
            n_heads=1,
            state_dim=4,
            delta_hidden=8,
            max_slots=3,
            dropout=0.0,
        )
        baseline = ClozeCorrector(**common, err_use_state=False).eval()
        stateful = ClozeCorrector(**common, err_use_state=True).eval()
        loaded = stateful.load_state_dict(baseline.state_dict(), strict=False)
        self.assertFalse(loaded.unexpected_keys)
        self.assertEqual(
            set(loaded.missing_keys),
            {
                "err_state_ln.weight",
                "err_state_ln.bias",
                "err_state_head.0.weight",
                "err_state_head.0.bias",
                "err_state_head.1.weight",
                "err_state_head.1.bias",
                "err_state_head.4.weight",
                "err_state_head.4.bias",
            },
        )

        z = torch.randn(2, 3, 4)
        seed = torch.randn(2, 3, 8)
        log_probs = torch.log_softmax(torch.randn(2, 3, 2), dim=-1)
        scalars = torch.randn(2, 3, 3)
        hidden = torch.randn(2, 3, 8)
        state_a = torch.randn(2, 3, 4)
        state_b = torch.randn(2, 3, 4)
        expected = baseline.err_logits(z, seed, log_probs, scalars, hidden)
        torch.testing.assert_close(
            stateful.err_logits(z, seed, log_probs, scalars, hidden, state_a),
            expected,
        )
        torch.testing.assert_close(
            stateful.err_logits(z, seed, log_probs, scalars, hidden, state_b),
            expected,
        )

        with torch.no_grad():
            stateful.err_state_head[-1].weight.fill_(0.1)
        self.assertFalse(
            torch.equal(
                stateful.err_logits(z, seed, log_probs, scalars, hidden, state_a),
                stateful.err_logits(z, seed, log_probs, scalars, hidden, state_b),
            )
        )

    def test_selector_local_denominator_is_rejected_as_partition_dependent(self):
        selector = _RecordingProposalSelector()
        with self.assertRaisesRegex(ValueError, "not partition-invariant"):
            OnlineDFlashModel(
                draft_model=_Draft(selector),
                target_lm_head=nn.Identity(),
                target_embed_tokens=nn.Embedding(5, 3),
                mask_token_id=4,
                block_size=4,
                attention_backend="eager",
                selector_loss_alpha=1.0,
                selector_own_denominator=True,
            )

    def test_expected_accept_rejects_margin_shifted_policy(self):
        selector = _LearnableKeepRepairSelector()
        draft = _Draft(selector)
        draft.selector_decision_mode = "margin_gate"
        # Multiclass with the unshifted argmax policy is now supported. This
        # gate should exercise the incompatible margin, not an obsolete ban
        # on multiclass or a later missing err-head check in the tiny fixture.
        draft.selector_gate_rho = 3.0
        with self.assertRaisesRegex(ValueError, "requires the serving rule to be exact K-way argmax"):
            OnlineDFlashModel(
                draft_model=draft,
                target_lm_head=nn.Identity(),
                target_embed_tokens=nn.Embedding(5, 3),
                mask_token_id=4,
                selector_weight_mode="expected_accept",
                selector_objective="multiclass",
            )

        draft.selector_decision_mode = "keep_repair"
        draft.selector_keep_repair_margin = 0.25
        with self.assertRaisesRegex(ValueError, "requires selector_keep_repair_margin=0"):
            OnlineDFlashModel(
                draft_model=draft,
                target_lm_head=nn.Identity(),
                target_embed_tokens=nn.Embedding(5, 3),
                mask_token_id=4,
                selector_weight_mode="expected_accept",
                selector_objective="keep_repair",
            )

    def test_lattice_column_zero_is_argmax_even_when_topk_omits_a_large_tie(self):
        selector = LatticePathSelector(
            hidden_size=4,
            vocab_size=6,
            top_k=2,
            d=4,
            n_layers=1,
            n_heads=1,
            state_dim=4,
            delta_hidden=4,
        )
        # All six tokens share the maximum while K=2.  torch.topk is free to
        # return any two of them, but greedy drafting deterministically uses
        # torch.argmax's lowest index (token 0).
        logits = torch.ones(1, 1, 6)
        top_log_probs, candidate_ids, _ = selector.extract_lattice(logits)

        self.assertEqual(candidate_ids[..., 0].item(), logits.argmax(dim=-1).item())
        torch.testing.assert_close(top_log_probs[..., 0], top_log_probs[..., 1])

    def test_selector_embedding_precision_is_owned_by_selector_not_backbone(self):
        selector = LatticePathSelector(
            hidden_size=4,
            vocab_size=6,
            top_k=2,
            d=4,
            n_layers=1,
            n_heads=1,
            state_dim=4,
            delta_hidden=4,
            dropout=0.0,
        ).float().eval()
        target_embedding = nn.Embedding(6, 4, dtype=torch.bfloat16)
        selector.bind_target_embedding(target_embedding)

        hidden = torch.randn(1, 2, 4, dtype=torch.bfloat16)
        candidate_ids = torch.tensor([[[0, 1], [2, 3]]])
        predecessors = torch.tensor([[4, 0]])
        unary = torch.log_softmax(torch.randn(1, 2, 2), dim=-1)
        scalars = torch.zeros(1, 2, 3)
        scores = selector.score_candidates(
            candidate_ids=candidate_ids,
            unary_logits=unary,
            hidden_states=hidden,
            predecessor_ids=predecessors,
            lattice_scalars=scalars,
        )

        self.assertEqual(selector._embedding().dtype, torch.float32)
        self.assertEqual(scores.dtype, torch.float32)

    def test_online_host_removes_anchor_before_lattice_and_cross_slot_selector(self):
        selector = _RecordingProposalSelector()
        model = OnlineDFlashModel(
            draft_model=_Draft(selector),
            target_lm_head=nn.Identity(),
            target_embed_tokens=nn.Embedding(5, 3),
            mask_token_id=4,
            block_size=4,
            attention_backend="eager",
            selector_loss_alpha=1.0,
            selector_weight_mode="uniform",
        )

        logits = torch.tensor(
            [[[[9.0, 1.0, 0.0, -1.0, -2.0],
               [0.0, 9.0, 1.0, -1.0, -2.0],
               [0.0, 1.0, 9.0, -1.0, -2.0],
               [0.0, 1.0, 2.0, 9.0, -2.0]]]]
        )
        hidden = torch.arange(12.0).reshape(1, 1, 4, 3)
        targets = torch.tensor([[[0, 1, 2, 3]]])
        predecessors = torch.tensor([[[4, 0, 1, 2]]])
        weights = torch.tensor([[[0.0, 0.25, 0.5, 0.75]]])
        valid = torch.tensor([[[0.0, 1.0, 1.0, 1.0]]])

        terms = model._selector_chunk_terms(
            selector,
            logits,
            hidden,
            targets,
            predecessors,
            weights,
            valid,
        )

        # Proposal_1 is selector position 0.  The anchor is absent not only
        # from CE, but from lattice extraction and cross-slot attention input.
        torch.testing.assert_close(selector.seen["objective_logits"], logits[..., 1:, :])
        torch.testing.assert_close(selector.seen["hidden_states"], hidden[..., 1:, :])
        torch.testing.assert_close(selector.seen["predecessor_ids"], predecessors[..., 1:])
        self.assertEqual(selector.seen["candidate_ids"].shape[-2], 3)
        self.assertEqual(selector.seen["lattice_scalars"].shape[-2], 3)
        self.assertEqual(terms.weight_den.item(), 3.0)
        self.assertEqual(terms.covered_num.item(), 3.0)

    def test_keep_repair_distribution_is_normalized_and_decodes_joint_map(self):
        scores = torch.tensor([[8.0, 4.0, 3.0], [8.0, 4.0, 3.0]])
        err = torch.tensor([-2.0, 2.0])
        log_probs = LatticePathSelector.keep_repair_log_probs(scores, err)

        torch.testing.assert_close(log_probs.exp().sum(dim=-1), torch.ones(2))
        self.assertEqual(log_probs.argmax(dim=-1).tolist(), [0, 1])

    def test_keep_repair_main_likelihood_trains_slots_after_base_frontier(self):
        selector = _KeepRepairSelector()
        model = OnlineDFlashModel(
            draft_model=_Draft(selector),
            target_lm_head=nn.Identity(),
            target_embed_tokens=nn.Embedding(5, 3),
            mask_token_id=4,
            block_size=4,
            attention_backend="eager",
            selector_loss_alpha=1.0,
            selector_err_loss_alpha=0.0,
            selector_weight_mode="uniform",
            selector_objective="keep_repair",
        )
        logits = torch.tensor(
            [[[[9.0, 1.0, 0.0, -1.0, -2.0],
               [9.0, 8.0, 1.0, -1.0, -2.0],
               [8.0, 9.0, 1.0, -1.0, -2.0],
               [0.0, 1.0, 9.0, -1.0, -2.0]]]]
        )
        hidden = torch.arange(12.0).reshape(1, 1, 4, 3)
        # After anchor removal: keep token 0, repair token 1 -> 0, then keep token 2.
        targets = torch.tensor([[[0, 0, 0, 2]]])
        predecessors = torch.tensor([[[4, 0, 0, 0]]])
        valid = torch.tensor([[[0.0, 1.0, 1.0, 1.0]]])

        terms = model._selector_chunk_terms(
            selector, logits, hidden, targets, predecessors, valid, valid
        )

        self.assertEqual(terms.weight_den.item(), 3.0)
        self.assertEqual(terms.err_den.item(), 0.0)
        self.assertEqual(terms.correct_num.item(), 3.0)

    def test_keep_repair_topk_miss_still_trains_binary_error_detector(self):
        selector = _PerSlotKeepRepairSelector()
        model = OnlineDFlashModel(
            draft_model=_Draft(selector),
            target_lm_head=nn.Identity(),
            target_embed_tokens=nn.Embedding(5, 3),
            mask_token_id=4,
            block_size=4,
            attention_backend="eager",
            selector_loss_alpha=1.0,
            selector_err_loss_alpha=0.0,
            selector_weight_mode="uniform",
            selector_objective="keep_repair",
        )
        logits = torch.tensor(
            [[[[9.0, 1.0, 0.0, -1.0, -2.0],
               [9.0, 8.0, 1.0, -1.0, -2.0],
               [9.0, 8.0, 1.0, -1.0, -2.0],
               [9.0, 8.0, 1.0, -1.0, -2.0]]]],
            requires_grad=True,
        )
        hidden = torch.arange(12.0).reshape(1, 1, 4, 3)
        # Only proposal slot 1 is valid and targets token 4, outside strict
        # top-2. It has no alternative-ranking label, but under the uniform
        # diagnostic objective it exactly says top-1 is wrong.
        targets = torch.tensor([[[0, 0, 4, 0]]])
        predecessors = torch.tensor([[[4, 0, 0, 4]]])
        valid = torch.tensor([[[0.0, 0.0, 1.0, 0.0]]])

        terms = model._selector_chunk_terms(
            selector, logits, hidden, targets, predecessors, valid, valid
        )
        terms.ce_num.backward()

        self.assertEqual(terms.weight_den.item(), 1.0)
        self.assertEqual(terms.covered_num.item(), 0.0)
        self.assertIsNotNone(selector.err_bias.grad)
        torch.testing.assert_close(
            selector.err_bias.grad, torch.tensor([0.0, -0.5, 0.0])
        )

    def test_profitable_repair_topk_miss_trains_abstain_not_impossible_override(self):
        selector = _PerSlotKeepRepairSelector()
        model = OnlineDFlashModel(
            draft_model=_Draft(selector),
            target_lm_head=nn.Identity(),
            target_embed_tokens=nn.Embedding(5, 3),
            mask_token_id=4,
            block_size=4,
            attention_backend="eager",
            selector_loss_alpha=1.0,
            selector_err_loss_alpha=0.0,
            selector_weight_mode="uniform",
            selector_objective="profitable_repair",
        )
        logits = torch.tensor(
            [[[[9.0, 1.0, 0.0, -1.0, -2.0],
               [9.0, 8.0, 1.0, -1.0, -2.0],
               [9.0, 8.0, 1.0, -1.0, -2.0],
               [9.0, 8.0, 1.0, -1.0, -2.0]]]],
            requires_grad=True,
        )
        hidden = torch.arange(12.0).reshape(1, 1, 4, 3)
        targets = torch.tensor([[[0, 0, 4, 0]]])
        predecessors = torch.tensor([[[4, 0, 0, 4]]])
        valid = torch.tensor([[[0.0, 0.0, 1.0, 0.0]]])

        terms = model._selector_chunk_terms(
            selector, logits, hidden, targets, predecessors, valid, valid
        )
        terms.ce_num.backward()

        self.assertEqual(terms.weight_den.item(), 1.0)
        self.assertEqual(terms.covered_num.item(), 0.0)
        # At logit zero, an ABSTAIN label has d(BCE)/d(logit)=+0.5.  The historical
        # keep_repair objective above produces -0.5 and therefore learns the opposite action.
        torch.testing.assert_close(
            selector.err_bias.grad, torch.tensor([0.0, 0.5, 0.0])
        )
        self.assertEqual(terms.correct_num.item(), 1.0)

    def test_accept_repair_topk_miss_has_no_arbitrary_gate_gradient(self):
        selector = _PerSlotKeepRepairSelector()
        draft = _Draft(selector)
        draft.selector_decision_mode = "keep_repair"
        draft.selector_keep_repair_margin = 0.0
        model = OnlineDFlashModel(
            draft_model=draft,
            target_lm_head=nn.Identity(),
            target_embed_tokens=nn.Embedding(5, 3),
            mask_token_id=4,
            block_size=4,
            attention_backend="eager",
            selector_loss_alpha=1.0,
            selector_err_loss_alpha=0.0,
            selector_weight_mode="uniform",
            selector_objective="accept_repair",
        )
        logits = torch.tensor(
            [[[[9.0, 1.0, 0.0, -1.0, -2.0],
               [9.0, 8.0, 1.0, -1.0, -2.0],
               [9.0, 8.0, 1.0, -1.0, -2.0],
               [9.0, 8.0, 1.0, -1.0, -2.0]]]],
            requires_grad=True,
        )
        hidden = torch.arange(12.0).reshape(1, 1, 4, 3)
        targets = torch.tensor([[[0, 0, 4, 0]]])
        predecessors = torch.tensor([[[4, 0, 0, 4]]])
        valid = torch.tensor([[[0.0, 0.0, 1.0, 0.0]]])

        terms = model._selector_chunk_terms(
            selector, logits, hidden, targets, predecessors, valid, valid
        )
        terms.ce_num.backward()

        self.assertEqual(terms.weight_den.item(), 0.0)
        self.assertEqual(terms.covered_num.item(), 0.0)
        self.assertIsNotNone(selector.err_bias.grad)
        torch.testing.assert_close(selector.err_bias.grad, torch.zeros(3))

    def test_profitable_action_topk_miss_trains_stateful_scorer_to_keep(self):
        selector = _RecordingProposalSelector()
        draft = _Draft(selector)
        draft.selector_decision_mode = "margin_gate"
        draft.selector_gate_rho = 1.0
        draft.selector_gate_tau = 0.0
        draft.selector_gate_theta = 0.0
        draft.selector_gate_skip_first = False
        model = OnlineDFlashModel(
            draft_model=draft,
            target_lm_head=nn.Identity(),
            target_embed_tokens=nn.Embedding(5, 3),
            mask_token_id=4,
            block_size=4,
            attention_backend="eager",
            selector_loss_alpha=1.0,
            selector_err_loss_alpha=0.0,
            selector_weight_mode="uniform",
            selector_objective="profitable_action",
        )
        logits = torch.tensor(
            [[[[9.0, 1.0, 0.0, -1.0, -2.0],
               [9.0, 8.0, 1.0, -1.0, -2.0],
               [9.0, 8.0, 1.0, -1.0, -2.0],
               [9.0, 8.0, 1.0, -1.0, -2.0]]]],
            requires_grad=True,
        )
        hidden = torch.arange(12.0).reshape(1, 1, 4, 3)
        # The only valid proposal targets token 4, outside strict top-2.  Unlike historical
        # multiclass this has the exact deployable label class 0 (KEEP), and the K-way scorer—not
        # a disconnected binary head—must receive its gradient.
        targets = torch.tensor([[[0, 0, 4, 0]]])
        predecessors = torch.tensor([[[4, 0, 0, 4]]])
        valid = torch.tensor([[[0.0, 0.0, 1.0, 0.0]]])

        terms = model._selector_chunk_terms(
            selector, logits, hidden, targets, predecessors, valid, valid
        )
        terms.ce_num.backward()

        self.assertEqual(terms.weight_den.item(), 1.0)
        self.assertEqual(terms.covered_num.item(), 0.0)
        self.assertEqual(terms.correct_num.item(), 1.0)
        self.assertLess(logits.grad[0, 0, 2, 0].item(), 0.0)
        self.assertGreater(logits.grad[0, 0, 2, 1].item(), 0.0)

    def test_profitable_action_rejects_non_argmax_serving_gate(self):
        selector = _RecordingProposalSelector()
        draft = _Draft(selector)
        draft.selector_decision_mode = "margin_gate"
        draft.selector_gate_rho = 3.0
        draft.selector_gate_tau = 0.0
        draft.selector_gate_theta = 0.0
        draft.selector_gate_skip_first = False
        with self.assertRaisesRegex(ValueError, "requires exact K-action argmax"):
            OnlineDFlashModel(
                draft_model=draft,
                target_lm_head=nn.Identity(),
                target_embed_tokens=nn.Embedding(5, 3),
                mask_token_id=4,
                selector_objective="profitable_action",
            )

    def test_disabling_detector_freezes_every_declared_err_only_module(self):
        selector = _SelectorWithTwoErrOnlyModules()
        draft = _Draft(selector)
        draft.selector_decision_mode = "margin_gate"
        draft.selector_gate_rho = 1.0
        draft.selector_gate_tau = 0.0
        draft.selector_gate_theta = 0.0
        draft.selector_gate_skip_first = False
        OnlineDFlashModel(
            draft_model=draft,
            target_lm_head=nn.Identity(),
            target_embed_tokens=nn.Embedding(5, 3),
            mask_token_id=4,
            selector_loss_alpha=1.0,
            selector_err_loss_alpha=0.0,
            selector_objective="profitable_action",
        )

        self.assertFalse(selector.err_seed_in.weight.requires_grad)
        self.assertFalse(selector.err_head.weight.requires_grad)
        self.assertFalse(selector.err_head.bias.requires_grad)
        self.assertTrue(selector.scale.requires_grad)

    def test_teacher_forced_frontier_stops_after_current_selectors_first_error(self):
        selector = _KeepRepairSelector()
        model = OnlineDFlashModel(
            draft_model=_Draft(selector),
            target_lm_head=nn.Identity(),
            target_embed_tokens=nn.Embedding(5, 3),
            mask_token_id=4,
            block_size=4,
            attention_backend="eager",
            selector_loss_alpha=1.0,
            selector_err_loss_alpha=0.0,
            selector_weight_mode="teacher_forced_frontier",
            selector_objective="keep_repair",
            selector_frontier_boost=3.0,
        )
        logits = torch.tensor(
            [[[[9.0, 1.0, 0.0, -1.0, -2.0],
               [9.0, 8.0, 1.0, -1.0, -2.0],
               [8.0, 9.0, 1.0, -1.0, -2.0],
               [0.0, 1.0, 9.0, -1.0, -2.0]]]]
        )
        hidden = torch.arange(12.0).reshape(1, 1, 4, 3)
        # After anchor removal the selector keeps slot 0 correctly.  Its fixed
        # positive error logit then overrides slot 1 even though candidate 0 is
        # the target, so slot 2 must not consume objective weight yet.
        targets = torch.tensor([[[0, 0, 1, 2]]])
        predecessors = torch.tensor([[[4, 0, 0, 1]]])
        valid = torch.tensor([[[0.0, 1.0, 1.0, 1.0]]])

        terms = model._selector_chunk_terms(
            selector, logits, hidden, targets, predecessors, valid, valid
        )

        # One correctly-kept prefix slot has weight 1; the first covered policy error has
        # weight 1+boost=4.  Every later slot is unreachable and carries no local CE.
        self.assertEqual(terms.weight_den.item(), 5.0)
        # Coverage telemetry remains an all-valid diagnostic even though the
        # optimized action loss follows current-policy occupancy.
        self.assertEqual(terms.covered_num.item(), 3.0)

    def test_teacher_forced_frontier_skips_retained_dflash2_anchor(self):
        selector = _BlockSelector()
        model = OnlineDFlashModel(
            draft_model=_Draft(selector),
            target_lm_head=nn.Identity(),
            target_embed_tokens=nn.Embedding(5, 3),
            mask_token_id=4,
            block_size=4,
            attention_backend="eager",
            selector_loss_alpha=1.0,
            selector_weight_mode="teacher_forced_frontier",
        )
        logits = torch.tensor(
            [[[[9.0, 1.0, 0.0, -1.0, -2.0],
               [0.0, 9.0, 1.0, -1.0, -2.0],
               [0.0, 1.0, 9.0, -1.0, -2.0],
               [0.0, 1.0, 2.0, 9.0, -2.0]]]]
        )
        hidden = torch.arange(12.0).reshape(1, 1, 4, 3)
        targets = torch.tensor([[[0, 1, 2, 3]]])
        predecessors = torch.tensor([[[4, 0, 1, 2]]])
        valid = torch.tensor([[[0.0, 1.0, 1.0, 1.0]]])

        terms = model._selector_chunk_terms(
            selector, logits, hidden, targets, predecessors, valid, valid
        )

        self.assertEqual(terms.weight_den.item(), 3.0)

    def test_expected_accept_weights_have_exact_sequence_objective_gradient(self):
        selector = _LearnableKeepRepairSelector()
        model = OnlineDFlashModel(
            draft_model=_Draft(selector),
            target_lm_head=nn.Identity(),
            target_embed_tokens=nn.Embedding(5, 3),
            mask_token_id=4,
            block_size=4,
            attention_backend="eager",
            selector_loss_alpha=1.0,
            selector_err_loss_alpha=0.0,
            selector_weight_mode="expected_accept",
            selector_objective="keep_repair",
        )
        # Every proposal target is candidate 0.  With the shared err_bias at
        # zero, q_i=P(keep)=0.5 and E[A]=q+q^2+q^3.
        logits = torch.tensor(
            [[[[9.0, 1.0, 0.0, -1.0, -2.0],
               [9.0, 8.0, 1.0, -1.0, -2.0],
               [9.0, 8.0, 1.0, -1.0, -2.0],
               [9.0, 8.0, 1.0, -1.0, -2.0]]]]
        )
        hidden = torch.arange(12.0).reshape(1, 1, 4, 3)
        targets = torch.tensor([[[0, 0, 0, 0]]])
        predecessors = torch.tensor([[[4, 0, 0, 0]]])
        valid = torch.tensor([[[0.0, 1.0, 1.0, 1.0]]])

        terms = model._selector_chunk_terms(
            selector, logits, hidden, targets, predecessors, valid, valid
        )
        terms.ce_num.backward()
        surrogate_grad = selector.err_bias.grad.detach().clone()

        direct_bias = torch.tensor(0.0, requires_grad=True)
        q = torch.sigmoid(-direct_bias)
        direct_loss = -(q + q.square() + q.pow(3))
        direct_loss.backward()

        torch.testing.assert_close(surrogate_grad, direct_bias.grad)
        # Detached marginal weights are [q+q^2+q^3, q^2+q^3, q^3].
        self.assertAlmostEqual(terms.weight_den.item(), 1.375, places=6)

    def test_expected_accept_gives_no_selector_credit_at_or_after_topk_miss(self):
        selector = _PerSlotKeepRepairSelector()
        model = OnlineDFlashModel(
            draft_model=_Draft(selector),
            target_lm_head=nn.Identity(),
            target_embed_tokens=nn.Embedding(5, 3),
            mask_token_id=4,
            block_size=4,
            attention_backend="eager",
            selector_loss_alpha=1.0,
            selector_err_loss_alpha=0.0,
            selector_weight_mode="expected_accept",
            selector_objective="keep_repair",
        )
        logits = torch.tensor(
            [[[[9.0, 1.0, 0.0, -1.0, -2.0],
               [9.0, 8.0, 1.0, -1.0, -2.0],
               [9.0, 8.0, 1.0, -1.0, -2.0],
               [9.0, 8.0, 1.0, -1.0, -2.0]]]]
        )
        hidden = torch.arange(12.0).reshape(1, 1, 4, 3)
        # The second proposal target (token 4) is outside top-2.  The first
        # proposal has q=.5; no acceptance term reaches the miss or slot after.
        targets = torch.tensor([[[0, 0, 4, 0]]])
        predecessors = torch.tensor([[[4, 0, 0, 4]]])
        valid = torch.tensor([[[0.0, 1.0, 1.0, 1.0]]])

        terms = model._selector_chunk_terms(
            selector, logits, hidden, targets, predecessors, valid, valid
        )
        terms.ce_num.backward()

        self.assertAlmostEqual(terms.weight_den.item(), 0.5, places=6)
        # The first covered keep is trained; both the unrepairable miss and the
        # slot after it have exactly zero exact-acceptance gradient.
        torch.testing.assert_close(
            selector.err_bias.grad, torch.tensor([0.25, 0.0, 0.0])
        )

    def test_smoothed_accept_survives_miss_but_does_not_label_miss_as_repair(self):
        selector = _PerSlotKeepRepairSelector()
        model = OnlineDFlashModel(
            draft_model=_Draft(selector),
            target_lm_head=nn.Identity(),
            target_embed_tokens=nn.Embedding(5, 3),
            mask_token_id=4,
            block_size=4,
            attention_backend="eager",
            selector_loss_alpha=1.0,
            selector_err_loss_alpha=0.0,
            selector_weight_mode="smoothed_accept",
            selector_survival_floor=0.5,
            selector_objective="keep_repair",
        )
        logits = torch.tensor(
            [[[[9.0, 1.0, 0.0, -1.0, -2.0],
               [9.0, 8.0, 1.0, -1.0, -2.0],
               [9.0, 8.0, 1.0, -1.0, -2.0],
               [9.0, 8.0, 1.0, -1.0, -2.0]]]]
        )
        hidden = torch.arange(12.0).reshape(1, 1, 4, 3)
        # keep, top-k miss, keep. q=[.5, 0, .5], smoothed survival
        # [.75, .5, .75], prefix [.75, .375, .28125]. The miss has no
        # valid action label, but its floor leaves positive value on slot 2.
        targets = torch.tensor([[[0, 0, 4, 0]]])
        predecessors = torch.tensor([[[4, 0, 0, 4]]])
        valid = torch.tensor([[[0.0, 1.0, 1.0, 1.0]]])

        terms = model._selector_chunk_terms(
            selector, logits, hidden, targets, predecessors, valid, valid
        )
        terms.ce_num.backward()

        self.assertAlmostEqual(terms.weight_den.item(), 1.6875, places=6)
        torch.testing.assert_close(
            selector.err_bias.grad,
            torch.tensor([0.703125, 0.0, 0.140625]),
        )

    def test_expected_accept_does_not_splice_across_internal_invalid_gap(self):
        selector = _LearnableKeepRepairSelector()
        model = OnlineDFlashModel(
            draft_model=_Draft(selector),
            target_lm_head=nn.Identity(),
            target_embed_tokens=nn.Embedding(5, 3),
            mask_token_id=4,
            block_size=4,
            attention_backend="eager",
            selector_loss_alpha=1.0,
            selector_err_loss_alpha=0.0,
            selector_weight_mode="expected_accept",
            selector_objective="keep_repair",
        )
        logits = torch.tensor(
            [[[[9.0, 1.0, 0.0, -1.0, -2.0],
               [9.0, 8.0, 1.0, -1.0, -2.0],
               [9.0, 8.0, 1.0, -1.0, -2.0],
               [9.0, 8.0, 1.0, -1.0, -2.0]]]]
        )
        hidden = torch.arange(12.0).reshape(1, 1, 4, 3)
        targets = torch.tensor([[[0, 0, 0, 0]]])
        predecessors = torch.tensor([[[4, 0, 0, 0]]])
        # After removing the anchor this is [1, 0, 1].  The final supervised
        # token belongs to a later span and is unreachable in this block.
        valid = torch.tensor([[[0.0, 1.0, 0.0, 1.0]]])

        terms = model._selector_chunk_terms(
            selector, logits, hidden, targets, predecessors, valid, valid
        )

        self.assertAlmostEqual(terms.weight_den.item(), 0.5, places=6)

    def test_keep_repair_rejects_duplicate_auxiliary_binary_loss(self):
        selector = _LearnableKeepRepairSelector()
        with self.assertRaisesRegex(ValueError, "already contains"):
            OnlineDFlashModel(
                draft_model=_Draft(selector),
                target_lm_head=nn.Identity(),
                target_embed_tokens=nn.Embedding(5, 3),
                mask_token_id=4,
                block_size=4,
                attention_backend="eager",
                selector_loss_alpha=1.0,
                selector_err_loss_alpha=1.0,
                selector_objective="keep_repair",
            )


if __name__ == "__main__":
    unittest.main()
