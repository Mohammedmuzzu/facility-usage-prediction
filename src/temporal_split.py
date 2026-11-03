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
    first = times.iloc[int(len(y) * .7)]
    second = times.iloc[int(len(y) * .85)]
    masks = {"train": times < first,
             "validation": (times >= first) & (times < second) & (y.prediction_cutoff >= first),
             "test": (times >= second) & (y.prediction_cutoff >= second)}
    splits = {name: np.flatnonzero(mask.to_numpy()) for name, mask in masks.items()}
    if any(len(indices) < 10 for indices in splits.values()):
        raise ValueError("Too few examples after temporal purge")
    metadata = {"strategy": "70/15/15 target creation chronology; purge validation/test cutoffs before model-ready boundaries",
                "validation_start": str(first), "test_start": str(second), "counts": {k: len(v) for k, v in splits.items()},
                "purged_examples": int(len(y) - sum(len(v) for v in splits.values())),
                "ranges": {k: {"target_start": str(times.iloc[v].min()), "target_end": str(times.iloc[v].max()),
                                "earliest_cutoff": str(y.prediction_cutoff.iloc[v].min())} for k, v in splits.items()}}
    return splits, metadata
