"""Built-in DFlash registration and executable providers."""

from __future__ import annotations

from functools import partial
from dataclasses import replace

from specforge.algorithms.common.defaults import (
    empty_options,
    no_missing_checkpoint_keys,
)
from specforge.algorithms.common.hidden_states_data import (
    NORMALIZER_ID,
    TARGET_GREEDY_KEY,
    build_collator,
    build_dspark_collator,
    build_offline_normalizer,
    build_offline_reader,
)
from specforge.algorithms.common.providers import (
    AlgorithmProviders,
    DraftConfigProvider,
    ModelProvider,
    OfflineCaptureLayout,
    OfflineDataProvider,
    ServerCaptureLayout,
    ServerStreamingProvider,
    StepProvider,
    TargetDerivedDraftDefaults,
    make_registration,
)
from specforge.algorithms.contracts import (
    AlgorithmCapabilities,
    AlgorithmSpec,
    DraftRequirement,
    FeatureContract,
    FeatureMode,
    OfflineStorageContract,
)
from specforge.data.loss_mask import has_consecutive_supervised_tokens

ALGORITHM_NAME = "dflash"
DRAFT_ARCHITECTURE = "DFlashDraftModel"
SLOTDEEP_TRAINING_EXTENSION_KEY = "pspr_slotdeep_training_extension_v1"
# Only this exact inactive/default contract may be supplied for a legacy checkpoint.
# New checkpoints always record it, including all-zero arms, so disabling an active
# loss on resume cannot silently evade the contract comparison.
SLOTDEEP_LEGACY_TRAINING_EXTENSION = {
    "alt_loss_alpha": 0.0, "safe_loss_alpha": 0.0, "repair_loss_alpha": 0.0,
    "safe_margin": 0.1, "repair_margin": 0.1, "preserve_fp32": False,
}
DFLASH2_DRAFT_ARCHITECTURE = "DFlash2DraftModel"
PSPR_DRAFT_ARCHITECTURE = "PSPRDraftModel"
# PSPR-v2 is the same DFlash backbone with a different selector head, so it trains under the same
# `dflash` strategy; only the candidate_selector module differs.
PSPR_V2_DRAFT_ARCHITECTURE = "PSPRv2DraftModel"
# Same story for PSPR-cloze: identical backbone and identical selector contract, only the
# candidate_selector's internals differ.
PSPR_CLOZE_DRAFT_ARCHITECTURE = "PSPRClozeDraftModel"
# PSPR-cascade is PSPR-cloze with one structural change: the keep/repair gate reads the ranker's
# corrected scores, which the cloze gate is blind to.  Same backbone, same selector contract.
PSPR_CASCADE_DRAFT_ARCHITECTURE = "PSPRCascadeDraftModel"
# And for PSPR-Decision: same frozen backbone, same selector contract; it replaces the
# cross-slot encoder with a per-slot residual tower and factorises the keep/repair decision.
PSPR_DECISION_DRAFT_ARCHITECTURE = "PSPRDecisionDraftModel"
# PSPR-slotdeep is PSPR-cloze with the dense bidirectional encoder replaced by a deep per-slot
# readout.  It subclasses PSPRClozeDraftModel and overrides one method (`cloze_states`), so the
# backbone, the selector contract and every objective term are inherited unchanged.
PSPR_SLOTDEEP_DRAFT_ARCHITECTURE = "PSPRSlotDeepDraftModel"
COMPATIBLE_DRAFT_ARCHITECTURES = frozenset(
    {
        DRAFT_ARCHITECTURE,
        DFLASH2_DRAFT_ARCHITECTURE,
        PSPR_DRAFT_ARCHITECTURE,
        PSPR_V2_DRAFT_ARCHITECTURE,
        PSPR_CLOZE_DRAFT_ARCHITECTURE,
        "PSPRMemoryDraftModel",
        PSPR_CASCADE_DRAFT_ARCHITECTURE,
        PSPR_DECISION_DRAFT_ARCHITECTURE,
        PSPR_SLOTDEEP_DRAFT_ARCHITECTURE,
    }
)


def build_step(wrapped_model, *, target_head=None, **_options):
    del target_head
    from specforge.training.strategies.base import DFlashTrainStrategy

    return DFlashTrainStrategy(wrapped_model)


def resume_contract(_config, draft_model, training_model):
    """Persist resolved DFlash architecture, sampling, and loss semantics."""

    contract = {
        "dflash_draft_num_hidden_layers": int(draft_model.config.num_hidden_layers),
        "dflash_target_layer_ids": tuple(
            int(layer_id) for layer_id in draft_model.target_layer_ids
        ),
        "dflash_block_size": int(training_model.block_size),
        "dflash_mask_token_id": int(training_model.mask_token_id),
        "dflash_attention_backend": str(training_model.attention_backend),
        "dflash_num_anchors": int(training_model.num_anchors),
        "dflash_loss_decay_gamma": training_model.loss_decay_gamma,
        "dflash_loss_type": str(training_model.loss_type),
        "dflash_dpace_alpha": float(training_model.dpace_alpha),
        "dflash_lk_loss_type": training_model.lk_loss_type,
        "dflash_kl_scale": float(training_model.kl_scale),
        "dflash_kl_decay": float(training_model.kl_decay),
    }
    if getattr(draft_model, "candidate_selector", None) is not None:
        method_config = dict(getattr(draft_model.config, "dflash_config", None) or {})
        # A candidate selector is an algorithm-level feature; the grouped convolution and the
        # low-rank transition are specific to DFlash2's architecture. Recording the latter
        # unconditionally would reject any other selector-bearing architecture.
        if "conv_kernel_size" in method_config:
            contract.update(
                {
                    "dflash2_conv_kernel_size": int(method_config["conv_kernel_size"]),
                    "dflash2_conv_group_size": int(method_config["conv_group_size"]),
                    "dflash2_selector_rank": int(method_config["selector_rank"]),
                }
            )
        contract.update(
            {
                "dflash2_selector_top_k": int(method_config["selector_top_k"]),
                "dflash2_selector_loss_alpha": float(
                    training_model.selector_loss_alpha
                ),
                "dflash2_selector_err_loss_alpha": float(
                    getattr(training_model, "selector_err_loss_alpha", 0.0)
                ),
                "dflash2_selector_own_denominator": bool(
                    getattr(training_model, "selector_own_denominator", False)
                ),
                "dflash2_selector_weight_mode": str(
                    getattr(training_model, "selector_weight_mode", "base_dpace")
                ),
                "dflash2_selector_frontier_boost": float(
                    getattr(training_model, "selector_frontier_boost", 0.0)
                ),
                "dflash2_selector_survival_floor": float(
                    getattr(training_model, "selector_survival_floor", 0.5)
                ),
                "dflash2_selector_objective": str(
                    getattr(training_model, "selector_objective", "multiclass")
                ),
                "dflash2_selector_err_weight_mode": str(
                    getattr(training_model, "selector_err_weight_mode", "base_frontier")
                ),
                "dflash2_selector_warmup_ratio": float(
                    training_model.selector_warmup_ratio
                ),
                "dflash2_selector_ramp_ratio": float(
                    training_model.selector_ramp_ratio
                ),
                "dflash2_selector_stop_gradient": bool(
                    training_model.selector_stop_gradient
                ),
                "dflash2_selector_target_greedy_labels": bool(
                    getattr(training_model, "selector_target_greedy_labels", False)
                ),
                # Serving-policy fields are semantic checkpoint state.  None
                # of them changes a selector tensor shape, so strict state_dict
                # loading cannot detect a changed gate or repair margin.
                "dflash2_selector_decision_mode": str(
                    getattr(draft_model, "selector_decision_mode", "margin_gate")
                ),
                "dflash2_selector_keep_repair_margin": float(
                    getattr(draft_model, "selector_keep_repair_margin", 0.0)
                ),
                "dflash2_selector_gate_rho": float(
                    getattr(draft_model, "selector_gate_rho", 1.0)
                ),
                "dflash2_selector_gate_tau": float(
                    getattr(draft_model, "selector_gate_tau", 0.0)
                ),
                "dflash2_selector_gate_theta": float(
                    getattr(draft_model, "selector_gate_theta", 0.0)
                ),
                "dflash2_selector_gate_skip_first": bool(
                    getattr(draft_model, "selector_gate_skip_first", False)
                ),
                "dflash2_selector_compute_dtype": str(
                    getattr(draft_model, "selector_compute_dtype", "model")
                ),
            }
        )
    selector = getattr(draft_model, "candidate_selector", None)
    if selector is not None and type(selector).__name__ == "SlotDeepCorrector":
        # Includes fusion/scoring semantics, not just tensor shapes. Old model
        # contracts are untouched; a SlotDeep resume must use its exact head.
        contract["pspr_slotdeep_reference_config"] = selector.reference_config()
        contract[SLOTDEEP_TRAINING_EXTENSION_KEY] = {
            name: getattr(training_model, "selector_" + name, default)
            for name, default in SLOTDEEP_LEGACY_TRAINING_EXTENSION.items()
        }
        if getattr(training_model, "selector_objective", None) == "candidate_distill":
            contract["pspr_slotdeep_candidate_distill_v1"] = {
                "alpha": training_model.selector_distill_alpha,
                "temperature": training_model.selector_distill_temperature,
                "teacher_alignment": "post_norm_target_hidden_at_label_position_minus_one",
                "support": "strict_draft_topk_conditional_distribution_all_valid_slots",
            }
    return contract


def resolve_dflash_kernels(config):
    if not config.model.use_liger_kernel:
        return None
    if config.training.strategy != "dflash":
        raise ValueError(
            "model.use_liger_kernel is currently supported only with "
            "training.strategy=dflash"
        )

    from specforge.modeling.draft.dflash_kernels import load_liger_dflash_kernels

    return load_liger_dflash_kernels()


def build_draft(config, draft_config):
    from specforge.algorithms.model_providers import build_dflash_draft

    return build_dflash_draft(
        config,
        draft_config,
        resolve_dflash_kernels(config),
    )


def build_training_model(config, draft_model, draft_config, target_config, tokenizer):
    from specforge.algorithms.model_providers import build_dflash_model

    return build_dflash_model(
        config,
        draft_model,
        draft_config,
        target_config,
        tokenizer,
    )


def resolve_capture_layers(config, draft_config, target_config):
    from specforge.algorithms.model_providers import resolve_dflash_capture_layers

    return resolve_dflash_capture_layers(config, draft_config, target_config)


def populate_target_defaults(payload, target_config, config):
    from specforge.algorithms.model_providers import populate_dflash_generated_config

    return populate_dflash_generated_config(payload, target_config, config)


def apply_draft_overrides(config, draft_config):
    from specforge.algorithms.model_providers import apply_dflash_overrides

    return apply_dflash_overrides(config, draft_config)


def minimum_loss_tokens(config, draft_config):
    from specforge.algorithms.model_providers import dflash_min_loss_tokens

    return dflash_min_loss_tokens(config, draft_config)


def needs_input_tools(config, draft_model):
    from specforge.algorithms.model_providers import dflash_needs_input_tools

    return dflash_needs_input_tools(config, draft_model)


def algorithm_spec() -> AlgorithmSpec:
    ready = {"input_ids", "loss_mask", "hidden_states"}
    # Selector-only supervision, served from a sidecar next to the dump. Optional
    # because every existing dump predates it and the backbone objective never
    # reads it.
    optional = {TARGET_GREEDY_KEY}
    return AlgorithmSpec(
        name=ALGORITHM_NAME,
        draft=DraftRequirement(
            compatible_architectures=COMPATIBLE_DRAFT_ARCHITECTURES,
            default_architecture=DRAFT_ARCHITECTURE,
            supported_overrides={"num_hidden_layers", "block_size"},
        ),
        feature_contracts=(
            FeatureContract(
                mode=FeatureMode.OFFLINE,
                modality="text",
                required_tensors=ready,
                optional_tensors=optional,
                storage=OfflineStorageContract(
                    format="specforge_hidden_states_v1",
                    required_tensors=ready,
                    optional_tensors=optional,
                    normalizer=NORMALIZER_ID,
                ),
            ),
            FeatureContract(
                mode=FeatureMode.STREAMING,
                modality="text",
                required_tensors=ready,
            ),
        ),
        capabilities=AlgorithmCapabilities(
            attention_backends={"eager", "sdpa", "flex_attention"},
        ),
    )


def algorithm_providers() -> AlgorithmProviders:
    collator = build_collator
    return AlgorithmProviders(
        algorithm_name=ALGORITHM_NAME,
        step=StepProvider(
            build=build_step,
            options=empty_options,
            resume_contract=resume_contract,
            allowed_missing_checkpoint_keys=no_missing_checkpoint_keys,
            uses_external_target_head=False,
        ),
        model=ModelProvider(
            draft_config=DraftConfigProvider(
                architecture=DRAFT_ARCHITECTURE,
                compatible_architectures=COMPATIBLE_DRAFT_ARCHITECTURES,
                expected_auto_map_model="dflash.DFlashDraftModel",
                target_defaults=TargetDerivedDraftDefaults(
                    model_type="qwen3",
                    num_hidden_layers=1,
                    populate=populate_target_defaults,
                ),
                apply_overrides=apply_draft_overrides,
            ),
            build_draft=build_draft,
            build_training_model=build_training_model,
            resolve_capture_layers=resolve_capture_layers,
            minimum_loss_tokens=minimum_loss_tokens,
            needs_input_tools=needs_input_tools,
            default_dataloader_num_workers=8,
            loss_mask_filter=has_consecutive_supervised_tokens,
        ),
        offline=(
            OfflineDataProvider(
                modality="text",
                normalizer_id=NORMALIZER_ID,
                capture_layout=OfflineCaptureLayout(
                    capture_method="dflash",
                    aux_feature="hidden_states",
                    last_hidden_feature=None,
                    passthrough=(
                        ("input_ids", "input_ids"),
                        ("loss_mask", "loss_mask"),
                    ),
                ),
                build_reader=partial(build_offline_reader, ALGORITHM_NAME),
                build_normalizer=build_offline_normalizer,
                build_collator=collator,
            ),
        ),
        server_streaming=(
            ServerStreamingProvider(
                modality="text",
                capture_method="dflash",
                target_representation=None,
                layout=ServerCaptureLayout(
                    aux_feature="hidden_states",
                    last_hidden_feature=None,
                    passthrough=(
                        ("input_ids", "input_ids", ()),
                        ("loss_mask", "loss_mask", ()),
                    ),
                ),
                build_collator=collator,
            ),
        ),
    )


def create_registration():
    return make_registration(algorithm_spec(), algorithm_providers())


def with_candidate_teacher(registration, config):
    """Run-local online feature extension; defaults and old dumps are untouched.

    SGLang's existing last_hidden artifact is post-final-norm. Reuse its tested
    transport and the four-feature collator, without adding a new algorithm or
    a new selector architecture. The extra tensor is strictly training labels.
    """
    if registration.name != ALGORITHM_NAME or config.mode != "online":
        raise ValueError("candidate_distill currently requires online training.strategy='dflash'")
    if config.model.input_modality != "text":
        raise ValueError("candidate_distill currently requires text input")
    contracts = tuple(
        replace(c, required_tensors=c.required_tensors | {"target_last_hidden_states"},
                allowed_target_representations={"hidden_state"},
                default_target_representation="hidden_state")
        if c.mode is FeatureMode.STREAMING else c
        for c in registration.spec.feature_contracts
    )
    streaming = tuple(
        replace(p, layout=replace(p.layout, last_hidden_feature="target_last_hidden_states"),
                target_representation="hidden_state", build_collator=build_dspark_collator)
        for p in registration.providers.server_streaming
    )
    return make_registration(replace(registration.spec, feature_contracts=contracts),
                             replace(registration.providers, server_streaming=streaming))


__all__ = ["algorithm_providers", "algorithm_spec", "create_registration"]
