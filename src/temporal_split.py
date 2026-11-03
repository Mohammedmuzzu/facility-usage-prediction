"""Global 70/15/15 target chronology, with purged crossing prediction cutoffs.

The extra purge ensures even infrequent residents cannot be scored by a model
selected using outcomes that occurred after their prediction cutoff.
"""

import numpy as np
import pandas as pd


def chronological_split(y: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict]:
    if len(y) < 30:
        raise ValueError("Need at least 30 sequential examples")
    times = y.target_booking_timestamp
    if not times.is_monotonic_increasing:
        raise ValueError("Targets must be globally chronological")
    first = times.iloc[int(len(y) * 0.7)]
    second = times.iloc[int(len(y) * 0.85)]
    masks = {
        "train": times < first,
        "validation": (times >= first)
        & (times < second)
        & (y.prediction_cutoff >= first),
        "test": (times >= second) & (y.prediction_cutoff >= second),
    }
    splits = {name: np.flatnonzero(mask.to_numpy()) for name, mask in masks.items()}
    if any(len(indices) < 10 for indices in splits.values()):
        raise ValueError("Too few examples after temporal purge")
    metadata = {
        "strategy": "70/15/15 target creation chronology; purge validation/test cutoffs before model-ready boundaries",
        "validation_start": str(first),
        "test_start": str(second),
        "counts": {k: len(v) for k, v in splits.items()},
        "purged_examples": int(len(y) - sum(len(v) for v in splits.values())),
        "ranges": {
            k: {
                "target_start": str(times.iloc[v].min()),
                "target_end": str(times.iloc[v].max()),
                "earliest_cutoff": str(y.prediction_cutoff.iloc[v].min()),
            }
            for k, v in splits.items()
        },
    }
    validate_splits(y, splits)
    return splits, metadata


def validate_splits(y: pd.DataFrame, splits: dict[str, np.ndarray]) -> None:
    if set(splits) != {"train", "validation", "test"}:
        raise ValueError("Require train, validation and test partitions")
    used: set[int] = set()
    for name, indices in splits.items():
        indices = np.asarray(indices)
        if indices.ndim != 1 or len(indices) == 0 or indices.dtype.kind not in "iu":
            raise ValueError(f"{name}: require nonempty integer row indices")
        if (
            (indices < 0).any()
            or (indices >= len(y)).any()
            or len(set(indices)) != len(indices)
            or used.intersection(indices)
        ):
            raise ValueError("Partition indices must be disjoint, unique and in bounds")
        used.update(indices.tolist())
    train, validation, test = (
        y.iloc[splits[name]] for name in ["train", "validation", "test"]
    )
    if not (y.target_booking_timestamp > y.prediction_cutoff).all():
        raise ValueError("Targets must strictly follow their cutoff")
    if not (
        train.target_booking_timestamp.max() < validation.prediction_cutoff.min()
        and validation.target_booking_timestamp.max() < test.prediction_cutoff.min()
    ):
        raise ValueError("Partition chronology/model-ready boundaries are violated")
