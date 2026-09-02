"""Shared hidden-states normalization and padding adapters.

Used by the DFlash-family (DFlash/Domino/DSpark) and MTP algorithms.
"""

from __future__ import annotations

import os
from functools import partial

from specforge.algorithms.common.collation import pad_and_concatenate_features
from specforge.data.loss_mask import has_consecutive_supervised_tokens

NORMALIZER_ID = "dflash_family_offline_v1"
DSPARK_NORMALIZER_ID = "dspark_offline_v1"
MTP_NORMALIZER_ID = "mtp_offline_v1"

# Optional per-sample supervision kept beside a hidden-state dump rather than
# inside it: ``target_greedy[i]`` is the target model's own argmax for position
# ``i + 1``, so it is directly comparable to ``input_ids[1:]``.
#
# The corpus next token and the target's greedy token disagree on ~22% of
# supervised ShareGPT positions (measured). A backbone learning a distribution
# averages that noise out, but a candidate *selector* cannot: its entire job is
# deciding when to overrule the base top-1, which usually *is* the target's
# greedy token, so a corpus label at those positions is a direct instruction to
# make an override that decode can only reject. Hence a separate label set,
# opt-in per algorithm, with the backbone objective left on the corpus labels.
TARGET_GREEDY_KEY = "target_greedy"
TARGET_GREEDY_SIDECAR_SUFFIX = ".target_greedy"


def target_greedy_sidecar_dir(hidden_states_path):
    """Return the conventional sidecar directory iff it exists.

    Sibling of the dump (``<dump>.target_greedy``) so that the recursive
    ``*.ckpt`` walk in ``list_feature_files`` can never pick sidecars up as
    samples, and so that a dump and its labels can be moved together.
    """
    if not hidden_states_path:
        return None
    candidate = str(hidden_states_path).rstrip("/") + TARGET_GREEDY_SIDECAR_SUFFIX
    return candidate if os.path.isdir(candidate) else None


def _normalize_hidden_states(
    raw,
    key: str,
    max_len: int,
    *,
    description: str,
):
    hidden_states = raw[key]
    if hidden_states.dim() == 3:
        if hidden_states.shape[0] != 1:
            raise ValueError(
                f"offline {description} must have shape [seq, width] or "
                f"[1, seq, width], got {tuple(hidden_states.shape)}"
            )
        hidden_states = hidden_states.squeeze(0)
    if hidden_states.dim() != 2:
        raise ValueError(
            f"offline {description} must have shape [seq, width] or "
            f"[1, seq, width], got {tuple(hidden_states.shape)}"
        )
    return hidden_states[:max_len].unsqueeze(0)


def normalize_offline_sample(raw, max_len: int):
    """Normalize raw DFlash/Domino capture tensors without target projection."""

    input_ids = raw["input_ids"][:max_len].unsqueeze(0)
    loss_mask = raw["loss_mask"][:max_len].unsqueeze(0)
    hidden_states = _normalize_hidden_states(
        raw,
        "hidden_states",
        max_len,
        description="DFlash-family hidden_states",
    )
    lengths = {
        input_ids.shape[1],
        loss_mask.shape[1],
        hidden_states.shape[1],
    }
    if len(lengths) != 1:
        raise ValueError(
            "offline DFlash-family features have mismatched sequence lengths "
            f"after truncation: input_ids={input_ids.shape[1]}, "
            f"loss_mask={loss_mask.shape[1]}, "
            f"hidden_states={hidden_states.shape[1]}"
        )
    if not has_consecutive_supervised_tokens(loss_mask[0]):
        raise ValueError(
            "offline DFlash-family samples require two consecutive supervised tokens"
        )
    normalized = {
        "input_ids": input_ids,
        "loss_mask": loss_mask,
        "hidden_states": hidden_states,
    }
    if TARGET_GREEDY_KEY in raw:
        target_greedy = raw[TARGET_GREEDY_KEY][:max_len].unsqueeze(0)
        if target_greedy.shape[1] != input_ids.shape[1]:
            raise ValueError(
                "offline DFlash-family target_greedy has mismatched sequence "
                f"length after truncation: input_ids={input_ids.shape[1]}, "
                f"target_greedy={target_greedy.shape[1]}"
            )
        normalized[TARGET_GREEDY_KEY] = target_greedy
    return normalized


def normalize_dspark_offline_sample(raw, max_len: int):
    """Normalize DSpark capture tensors, including target final-layer states."""

    normalized = normalize_offline_sample(raw, max_len)
    target_last_hidden_states = _normalize_hidden_states(
        raw,
        "target_last_hidden_states",
        max_len,
        description="DSpark target_last_hidden_states",
    )
    expected_length = normalized["input_ids"].shape[1]
    if target_last_hidden_states.shape[1] != expected_length:
        raise ValueError(
            "offline DSpark features have mismatched sequence lengths after "
            f"truncation: input_ids={expected_length}, "
            "target_last_hidden_states="
            f"{target_last_hidden_states.shape[1]}"
        )
    return {
        **normalized,
        "target_last_hidden_states": target_last_hidden_states,
    }


def build_offline_reader(
    strategy,
    hidden_states_path,
    *,
    run_id,
    ttt_length,
    max_len,
):
    # Transitional runtime import; the composition root will inject this port.
    from specforge.runtime.data_plane.offline_reader import OfflineManifestReader

    feature_keys = ("input_ids", "loss_mask", "hidden_states")
    # Presence of the sidecar directory is itself the opt-in for *reading* the
    # extra labels; whether an objective *uses* them is a separate config flag.
    # Keying the read off the directory keeps the default path byte-identical:
    # with no sidecar the key never enters feature_keys, never reaches the
    # collator, and never reaches the model.
    sidecar_dir = target_greedy_sidecar_dir(hidden_states_path)
    if sidecar_dir is not None:
        feature_keys = feature_keys + (TARGET_GREEDY_KEY,)

    return OfflineManifestReader(
        hidden_states_path,
        run_id=run_id,
        strategy=strategy,
        feature_keys=feature_keys,
        target_repr=None,
        ttt_length=ttt_length,
        max_len=max_len,
        sidecar_dir=sidecar_dir,
    )


def build_dspark_offline_reader(
    strategy,
    hidden_states_path,
    *,
    run_id,
    ttt_length,
    max_len,
):
    # Transitional runtime import; the composition root will inject this port.
    from specforge.runtime.data_plane.offline_reader import OfflineManifestReader

    return OfflineManifestReader(
        hidden_states_path,
        run_id=run_id,
        strategy=strategy,
        feature_keys=(
            "input_ids",
            "loss_mask",
            "hidden_states",
            "target_last_hidden_states",
        ),
        target_repr="hidden_state",
        ttt_length=ttt_length,
        max_len=max_len,
    )


def build_offline_normalizer(max_len, **_topology):
    return partial(normalize_offline_sample, max_len=max_len)


def build_dspark_offline_normalizer(max_len, **_topology):
    return partial(normalize_dspark_offline_sample, max_len=max_len)


def build_collator():
    def collate(features):
        return pad_and_concatenate_features(
            features,
            sequence_axes={
                "input_ids": 1,
                "loss_mask": 1,
                "hidden_states": 1,
                TARGET_GREEDY_KEY: 1,
            },
            required_keys=("input_ids", "loss_mask", "hidden_states"),
            optional_keys=(TARGET_GREEDY_KEY,),
        )

    return collate


def build_dspark_collator():
    def collate(features):
        return pad_and_concatenate_features(
            features,
            sequence_axes={
                "input_ids": 1,
                "loss_mask": 1,
                "hidden_states": 1,
                "target_last_hidden_states": 1,
            },
            required_keys=(
                "input_ids",
                "loss_mask",
                "hidden_states",
                "target_last_hidden_states",
            ),
        )

    return collate


def normalize_mtp_offline_sample(raw, max_len: int):
    """Normalize MTP capture tensors (no aux-layer concat, final hidden only)."""

    input_ids = raw["input_ids"][:max_len].unsqueeze(0)
    loss_mask = raw["loss_mask"][:max_len].unsqueeze(0)
    target_last_hidden_states = _normalize_hidden_states(
        raw,
        "target_last_hidden_states",
        max_len,
        description="MTP target_last_hidden_states",
    )
    lengths = {
        input_ids.shape[1],
        loss_mask.shape[1],
        target_last_hidden_states.shape[1],
    }
    if len(lengths) != 1:
        raise ValueError(
            "offline MTP features have mismatched sequence lengths after "
            f"truncation: input_ids={input_ids.shape[1]}, "
            f"loss_mask={loss_mask.shape[1]}, "
            f"target_last_hidden_states={target_last_hidden_states.shape[1]}"
        )
    return {
        "input_ids": input_ids,
        "loss_mask": loss_mask,
        "target_last_hidden_states": target_last_hidden_states,
    }


def build_mtp_offline_reader(
    strategy,
    hidden_states_path,
    *,
    run_id,
    ttt_length,
    max_len,
):
    # Transitional runtime import; the composition root will inject this port.
    from specforge.runtime.data_plane.offline_reader import OfflineManifestReader

    return OfflineManifestReader(
        hidden_states_path,
        run_id=run_id,
        strategy=strategy,
        feature_keys=(
            "input_ids",
            "loss_mask",
            "target_last_hidden_states",
        ),
        target_repr="hidden_state",
        ttt_length=ttt_length,
        max_len=max_len,
    )


def build_mtp_offline_normalizer(max_len, **_topology):
    return partial(normalize_mtp_offline_sample, max_len=max_len)


def build_mtp_collator():
    def collate(features):
        return pad_and_concatenate_features(
            features,
            sequence_axes={
                "input_ids": 1,
                "loss_mask": 1,
                "target_last_hidden_states": 1,
            },
            required_keys=(
                "input_ids",
                "loss_mask",
                "target_last_hidden_states",
            ),
        )

    return collate


__all__ = [
    "DSPARK_NORMALIZER_ID",
    "MTP_NORMALIZER_ID",
    "NORMALIZER_ID",
    "TARGET_GREEDY_KEY",
    "TARGET_GREEDY_SIDECAR_SUFFIX",
    "build_collator",
    "build_dspark_collator",
    "build_dspark_offline_normalizer",
    "build_dspark_offline_reader",
    "build_mtp_collator",
    "build_mtp_offline_normalizer",
    "build_mtp_offline_reader",
    "build_offline_normalizer",
    "build_offline_reader",
    "normalize_dspark_offline_sample",
    "normalize_mtp_offline_sample",
    "normalize_offline_sample",
    "target_greedy_sidecar_dir",
]
