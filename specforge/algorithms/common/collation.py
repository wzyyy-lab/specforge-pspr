"""Small algorithm-owned batch collation helpers."""

from __future__ import annotations

from typing import Mapping, Sequence


def concatenate_features(features):
    """Concatenate equal-shaped per-sample feature dictionaries."""

    if not features:
        raise ValueError("cannot collate an empty feature batch")
    import torch

    keys = tuple(features[0])
    expected_keys = set(keys)
    if any(set(feature) != expected_keys for feature in features[1:]):
        raise ValueError("all samples must expose the same feature keys")
    return {
        key: torch.cat([feature[key] for feature in features], dim=0) for key in keys
    }


def pad_and_concatenate_features(
    features,
    *,
    sequence_axes: Mapping[str, int],
    required_keys: Sequence[str],
    optional_keys: Sequence[str] = (),
):
    """Zero-pad configured tensor axes to the longest input sequence.

    ``optional_keys`` are collated exactly like required ones, but only when the
    whole batch carries them; a key absent from every sample is simply not in the
    returned batch. Present on some samples and not others is a dataset bug, not
    a supported mode, so it raises: silently dropping the key there would make a
    partially prepared dataset look like an unprepared one and train on the wrong
    supervision without a single log line.
    """

    if not features:
        raise ValueError("cannot collate an empty feature batch")
    required = tuple(required_keys)
    missing = [
        (index, key)
        for index, feature in enumerate(features)
        for key in required
        if key not in feature
    ]
    if missing:
        raise KeyError(f"feature batch is missing required keys: {missing}")
    present_optional = []
    for key in tuple(optional_keys):
        count = sum(1 for feature in features if key in feature)
        if count == 0:
            continue
        if count != len(features):
            raise KeyError(
                f"optional feature {key!r} is present on {count} of "
                f"{len(features)} samples; it must be on all or none"
            )
        present_optional.append(key)
    max_length = max(int(feature["input_ids"].shape[-1]) for feature in features)

    import torch

    batch = {}
    for key in required + tuple(present_optional):
        axis = sequence_axes[key]
        padded = []
        for feature in features:
            tensor = feature[key]
            length = int(tensor.shape[axis])
            if length > max_length:
                raise ValueError(
                    f"feature {key!r} sequence length {length} exceeds "
                    f"input_ids length {max_length}"
                )
            if length < max_length:
                shape = list(tensor.shape)
                shape[axis] = max_length - length
                tensor = torch.cat(
                    [tensor, tensor.new_zeros(shape)],
                    dim=axis,
                )
            padded.append(tensor)
        batch[key] = torch.cat(padded, dim=0)
    return batch


__all__ = ["concatenate_features", "pad_and_concatenate_features"]
