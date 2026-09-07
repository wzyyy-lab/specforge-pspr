import importlib.util
import math
import pathlib
import sys
import tempfile
import unittest

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen3Config

from specforge.modeling.draft.pspr import LatticePathSelector
from specforge.modeling.draft.pspr_cascade import (
    CascadeCorrector,
    PSPRCascadeDraftModel,
)
from specforge.modeling.draft.pspr_cloze import ClozeCorrector, PSPRClozeDraftModel
from specforge.training.model_loading import warm_start_draft_model

ANCHOR_RHO = 3.0
SELECTOR_KW = dict(
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
_TAPS_DECODER = (
    pathlib.Path(__file__).resolve().parents[3] / "TAPS-SP/scripts/decode_lattice.py"
)


def _tiny_draft_config(architecture, **dflash_overrides):
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
        selector_err_score_anchor_rho=ANCHOR_RHO,
    )
    dflash.update(dflash_overrides)
    config.dflash_config = dflash
    return config


def _selector_pair(**overrides):
    torch.manual_seed(0)
    parent = ClozeCorrector(**SELECTOR_KW)
    torch.manual_seed(0)
    child = CascadeCorrector(
        **SELECTOR_KW, err_score_anchor_rho=ANCHOR_RHO, **overrides
    )
    embed = nn.Embedding(SELECTOR_KW["vocab_size"], SELECTOR_KW["hidden_size"])
    torch.manual_seed(7)
    nn.init.normal_(embed.weight)
    parent.bind_target_embedding(embed)
    child.bind_target_embedding(embed)
    child.load_state_dict(parent.state_dict(), strict=False)
    return parent.eval(), child.eval()


def _batch(seed=3, batch=2, slots=5):
    torch.manual_seed(seed)
    return dict(
        candidate_ids=torch.randint(
            0, SELECTOR_KW["vocab_size"], (batch, slots, SELECTOR_KW["top_k"])
        ),
        unary_logits=torch.randn(batch, slots, SELECTOR_KW["top_k"]).log_softmax(-1),
        hidden_states=torch.randn(batch, slots, SELECTOR_KW["hidden_size"]),
        predecessor_ids=torch.randint(0, SELECTOR_KW["vocab_size"], (batch, slots)),
    )


class CascadeWarmStartTest(unittest.TestCase):
    """The cascade adds tensors to a mature head, which the loader treats as fail-closed by design."""

    def test_optional_key_groups_extend_rather_than_shadow_the_parent(self):
        torch.manual_seed(0)
        cascade = PSPRCascadeDraftModel(_tiny_draft_config("PSPRCascadeDraftModel"))
        torch.manual_seed(0)
        cloze = PSPRClozeDraftModel(_tiny_draft_config("PSPRClozeDraftModel"))

        groups = [tuple(g) for g in cascade.warm_start_optional_key_groups]
        inherited = [tuple(g) for g in PSPRClozeDraftModel.warm_start_optional_key_groups]
        for group in inherited:
            self.assertIn(
                group,
                groups,
                "a property override must re-emit the parent's groups or a checkpoint that "
                "predates err_state_* stops loading",
            )
        self.assertEqual(len(groups), len(inherited) + 1)
        self.assertEqual(
            set(groups[-1]),
            set(cascade.state_dict()) - set(cloze.state_dict()),
            "the declared exemption must be exactly the cascade-only tensor set",
        )

    def test_real_loader_warm_starts_from_a_cloze_state(self):
        torch.manual_seed(0)
        cascade = PSPRCascadeDraftModel(_tiny_draft_config("PSPRCascadeDraftModel"))
        torch.manual_seed(0)
        cloze = PSPRClozeDraftModel(_tiny_draft_config("PSPRClozeDraftModel"))
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "run-step1" / "training_state.pt"
            path.parent.mkdir(parents=True)
            torch.save(
                {"draft_state_dict": cloze.state_dict(), "strategy": "online"}, path
            )
            report = warm_start_draft_model(
                cascade, str(path), draft_config=cascade.config, strategy="online"
            )
        self.assertEqual(report.loaded_keys, len(cloze.state_dict()))
        self.assertEqual(
            set(report.missing_keys),
            set(cascade.state_dict()) - set(cloze.state_dict()),
        )

    def test_partially_present_cascade_head_is_rejected(self):
        torch.manual_seed(0)
        cascade = PSPRCascadeDraftModel(_tiny_draft_config("PSPRCascadeDraftModel"))
        torch.manual_seed(0)
        cloze = PSPRClozeDraftModel(_tiny_draft_config("PSPRClozeDraftModel"))
        state = dict(cloze.state_dict())
        state["candidate_selector.err_base_gain.gain"] = torch.zeros(())
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "run-step1" / "training_state.pt"
            path.parent.mkdir(parents=True)
            torch.save({"draft_state_dict": state, "strategy": "online"}, path)
            with self.assertRaises(ValueError):
                warm_start_draft_model(
                    cascade, str(path), draft_config=cascade.config, strategy="online"
                )


class CascadeStepZeroPolicyTest(unittest.TestCase):
    """At step 0 the factorized rule must be the champion's margin gate, not latrepair m=0."""

    def test_gate_logit_is_the_analytic_anchor(self):
        _, child = _selector_pair()
        batch = _batch()
        with torch.no_grad():
            scores, err = child.score_candidates(**batch, return_err=True)
            expected = (
                torch.logsumexp(scores[..., 1:].float(), dim=-1)
                - scores[..., 0].float()
                - math.log(ANCHOR_RHO)
            )
        self.assertTrue(torch.equal(err, expected))

    def test_step_zero_matches_margin_gate_for_several_rho(self):
        for rho in (0.5, 1.0, 3.0, 10.0):
            _, child = _selector_pair()
            child.err_score_anchor_rho = rho
            batch = _batch()
            with torch.no_grad():
                scores, err = child.score_candidates(**batch, return_err=True)
                action = LatticePathSelector.keep_repair_log_probs(scores, err).argmax(-1)
                champion = LatticePathSelector.select_margin_gate(
                    scores, rho=rho, tau=0.0
                ).squeeze(-1)
            self.assertTrue(
                torch.equal(action, champion),
                f"rho={rho}: {int((action != champion).sum())} slots disagree",
            )

    def test_exact_tie_keeps_on_both_rules(self):
        _, child = _selector_pair()
        scores = torch.full((1, 1, SELECTOR_KW["top_k"]), -30.0)
        scores[0, 0, 0] = 0.0
        scores[0, 0, 1] = -math.log(ANCHOR_RHO)
        err = (
            torch.logsumexp(scores[..., 1:], dim=-1)
            - scores[..., 0]
            - math.log(ANCHOR_RHO)
        )
        action = LatticePathSelector.keep_repair_log_probs(scores, err).argmax(-1)
        champion = LatticePathSelector.select_margin_gate(
            scores, rho=ANCHOR_RHO, tau=0.0
        ).squeeze(-1)
        self.assertEqual(int(action), 0)
        self.assertTrue(torch.equal(action, champion))


class CascadeServingConsistencyTest(unittest.TestCase):
    """Training, native serving and external serving must agree with a NONZERO score head."""

    def _live_selector(self):
        torch.manual_seed(21)
        _, child = _selector_pair()
        with torch.no_grad():
            for layer in (child.err_score_head[0], child.err_score_head[-1]):
                layer.weight.normal_(std=0.5)
                layer.bias.normal_(std=0.5)
            child.err_base_gain["gain"].fill_(0.6)
        return child.eval()

    @staticmethod
    def _selector_inputs(child, batch):
        embedding = child._embedding()
        cand_emb = F.embedding(batch["candidate_ids"], embedding)
        unary = batch["unary_logits"].float()
        margin = (unary[..., 0] - unary[..., 1]).clamp(-20.0, 20.0)
        mass = unary.exp().sum(-1).clamp(max=1.0)
        scalars = torch.stack([torch.zeros_like(margin), margin, mass], dim=-1)
        z = child.cloze_states(
            batch["hidden_states"],
            batch["candidate_ids"],
            batch["predecessor_ids"][..., 0],
            unary,
            scalars,
        )
        state = child.causal_states(F.embedding(batch["predecessor_ids"], embedding))
        return cand_emb, unary, scalars, z, state

    def test_native_call_order_reproduces_the_training_logit(self):
        child = self._live_selector()
        batch = _batch(seed=5, batch=1, slots=3)
        with torch.no_grad():
            _, err_train = child.score_candidates(**batch, return_err=True)
            cand_emb, unary, scalars, z, state = self._selector_inputs(child, batch)
            native = []
            for position in range(batch["hidden_states"].shape[1]):
                zi = z[:, position]
                # Exactly pspr_cloze._sample_draft_tokens: score() first, then err_logits() with the
                # same zi and NO scores argument.
                child.score(
                    zi,
                    batch["hidden_states"][:, position],
                    cand_emb[:, position],
                    unary[:, position],
                    state[:, position],
                )
                native.append(
                    child.err_logits(
                        zi,
                        cand_emb[:, position, 0],
                        unary[:, position],
                        scalars[:, position],
                        batch["hidden_states"][:, position],
                        state[:, position],
                    )
                )
        self.assertTrue(
            torch.allclose(torch.stack(native, dim=1), err_train, atol=1e-5)
        )

    @unittest.skipUnless(_TAPS_DECODER.exists(), "external decoder not present")
    def test_external_decoder_reproduces_the_training_logit(self):
        spec = importlib.util.spec_from_file_location("decode_lattice", _TAPS_DECODER)
        decode = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(_TAPS_DECODER.parent))
        spec.loader.exec_module(decode)

        child = self._live_selector()
        batch = _batch(seed=5, batch=1, slots=3)
        with torch.no_grad():
            scores, err_train = child.score_candidates(**batch, return_err=True)
            cand_emb, unary, scalars, z, state = self._selector_inputs(child, batch)
            external = [
                decode.slot_err_logit(
                    child,
                    z[0],
                    state[:, position].unsqueeze(0),
                    unary[0],
                    scalars[0],
                    cand_emb[0],
                    batch["hidden_states"],
                    position,
                    scores=scores[0, position],
                ).reshape(())
                for position in range(batch["hidden_states"].shape[1])
            ]
        self.assertTrue(decode.is_cascade(child))
        self.assertTrue(
            torch.allclose(torch.stack(external).unsqueeze(0), err_train, atol=1e-5)
        )

    @unittest.skipUnless(_TAPS_DECODER.exists(), "external decoder not present")
    def test_external_decoder_refuses_to_run_the_gate_blind(self):
        spec = importlib.util.spec_from_file_location("decode_lattice", _TAPS_DECODER)
        decode = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(_TAPS_DECODER.parent))
        spec.loader.exec_module(decode)

        child = self._live_selector()
        batch = _batch(seed=5, batch=1, slots=3)
        with torch.no_grad():
            cand_emb, unary, scalars, z, state = self._selector_inputs(child, batch)
            with self.assertRaises(ValueError):
                decode.slot_err_logit(
                    child,
                    z[0],
                    state[:, 0].unsqueeze(0),
                    unary[0],
                    scalars[0],
                    cand_emb[0],
                    batch["hidden_states"],
                    0,
                )


class CascadeFailClosedTest(unittest.TestCase):
    def setUp(self):
        _, self.child = _selector_pair()

    def _blind_args(self):
        return (
            torch.zeros(2, 5, SELECTOR_KW["d"]),
            torch.zeros(2, 5, SELECTOR_KW["hidden_size"]),
            torch.zeros(2, 5, SELECTOR_KW["top_k"]),
            torch.zeros(2, 5, 3),
            torch.zeros(2, 5, SELECTOR_KW["hidden_size"]),
            None,
        )

    def test_no_scores_and_no_handoff_raises(self):
        with torch.no_grad(), self.assertRaises(RuntimeError):
            self.child.err_logits(*self._blind_args())

    def test_grad_enabled_call_never_uses_the_handoff(self):
        with self.assertRaises(RuntimeError):
            self.child.err_logits(*self._blind_args())

    def test_handoff_is_identity_checked(self):
        batch = _batch(seed=5, batch=1, slots=3)
        child = self.child
        with torch.no_grad():
            cand_emb, unary, scalars, z, state = (
                CascadeServingConsistencyTest._selector_inputs(child, batch)
            )
            child.score(
                z[:, 0],
                batch["hidden_states"][:, 0],
                cand_emb[:, 0],
                unary[:, 0],
                state[:, 0],
            )
            with self.assertRaises(RuntimeError):
                child.err_logits(
                    z[:, 1],
                    cand_emb[:, 1, 0],
                    unary[:, 1],
                    scalars[:, 1],
                    batch["hidden_states"][:, 1],
                    state[:, 1],
                )

    def test_wrong_width_scores_raises(self):
        with torch.no_grad(), self.assertRaises(RuntimeError):
            self.child.err_logits(
                *self._blind_args(),
                scores=torch.zeros(2, 5, SELECTOR_KW["top_k"] + 1),
            )


class CascadeTrainScopeTest(unittest.TestCase):
    def test_err_only_trains_the_gate_including_the_base_gain(self):
        torch.manual_seed(0)
        model = PSPRCascadeDraftModel(_tiny_draft_config("PSPRCascadeDraftModel"))
        model.apply_backbone_freeze()
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        self.assertIn("candidate_selector.err_base_gain.gain", trainable)
        self.assertEqual(
            sorted({n.split(".")[1] for n in trainable}),
            [
                "err_base_gain",
                "err_head",
                "err_score_head",
                "err_score_ln",
                "err_seed_in",
            ],
        )

    def test_detach_cuts_the_path_into_the_ranker(self):
        _, child = _selector_pair()
        with torch.no_grad():
            child.err_score_head[-1].weight.normal_(std=0.5)
        child.train()
        child.score_candidates(**_batch(), return_err=True)[1].sum().backward()
        delta_grad = sum(
            float(p.grad.abs().sum())
            for n, p in child.named_parameters()
            if n.startswith("delta_") and p.grad is not None
        )
        self.assertEqual(delta_grad, 0.0)
        self.assertGreater(float(child.err_score_head[-1].weight.grad.abs().sum()), 0.0)


class CascadeBlindControlTest(unittest.TestCase):
    def test_control_differs_only_by_the_score_head(self):
        parent, cascade = _selector_pair()
        _, control = _selector_pair(err_score_enabled=False)
        cascade_only = set(cascade.state_dict()) - set(parent.state_dict())
        control_only = set(control.state_dict()) - set(parent.state_dict())
        self.assertEqual(control_only, {"err_base_gain.gain"})
        self.assertEqual(
            cascade_only - control_only,
            {k for k in cascade_only if k.startswith("err_score_")},
        )
        batch = _batch()
        with torch.no_grad():
            self.assertTrue(
                torch.equal(
                    control.score_candidates(**batch, return_err=True)[1],
                    cascade.score_candidates(**batch, return_err=True)[1],
                ),
                "both arms must share an identical step-0 policy",
            )


if __name__ == "__main__":
    unittest.main()
