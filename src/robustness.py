"""Development-only temporal windows and explicit cold-start observations."""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from src.train import fit_models
from src.predict import predict_components
from src.feature_engineering import FeatureBuilder
from src.evaluate import review_predictions, metrics
from src.utils import timestamp


def development_windows(
    y: pd.DataFrame, final_test_start: str
) -> list[tuple[pd.DataFrame, dict, dict]]:
    development = y[
        y.target_booking_timestamp < timestamp(final_test_start)
    ].reset_index(drop=True)
    windows = []
    for fraction in [0.40, 0.55, 0.70]:
        first = development.target_booking_timestamp.iloc[
            int(len(development) * fraction)
        ]
        second = development.target_booking_timestamp.iloc[
            int(len(development) * (fraction + 0.10))
        ]
        end = development.target_booking_timestamp.iloc[
            min(len(development) - 1, int(len(development) * (fraction + 0.25)))
        ]
        masks = {
            "train": development.target_booking_timestamp < first,
            "validation": (development.target_booking_timestamp >= first)
            & (development.target_booking_timestamp < second)
            & (development.prediction_cutoff >= first),
            "test": (development.target_booking_timestamp >= second)
            & (development.target_booking_timestamp < end)
            & (development.prediction_cutoff >= second),
        }
        splits = {name: np.flatnonzero(mask.to_numpy()) for name, mask in masks.items()}
        if any(len(indices) < 10 for indices in splits.values()):
            continue
        windows.append(
            (
                development,
                splits,
                {
                    "validation_start": str(first),
                    "test_start": str(second),
                    "evaluation_end": str(end),
                },
            )
        )
    return windows


def temporal_backtests(X: pd.DataFrame, y: pd.DataFrame, final_test_start: str) -> dict:
    results = []
    for development, splits, split in development_windows(y, final_test_start):
        with tempfile.TemporaryDirectory(prefix="facility-backtest-") as directory:
            bundle = fit_models(
                X.iloc[: len(development)].reset_index(drop=True),
                development,
                splits,
                {"dataset_sha256": "development-only", "split": split},
                Path(directory),
            )
            indices = splits["test"]
            prediction = predict_components(
                bundle,
                X.iloc[indices],
                development.prediction_cutoff.iloc[indices],
                "champion",
            )
            review = review_predictions(
                development.iloc[indices], prediction, X.iloc[indices]
            )
            results.append(
                {
                    "split": split,
                    "sample_count": len(review),
                    "metrics": metrics(review),
                    "quarter_transition_samples": int(review.season_transition.sum()),
                    "component_routes": bundle["metadata"]["routes"],
                }
            )
    return {
        "windows": results,
        "original_final_test_accessed": False,
        "note": "Expanding train/validation/evaluation windows strictly inside development; diagnostic only, no test-driven retuning.",
    }


def cold_start_review(
    bookings: pd.DataFrame, bundle: dict, final_test_start: str
) -> dict:
    first = (
        bookings.sort_values(["booking_timestamp", "booking_id"])
        .groupby("resident_reference", sort=False)
        .head(1)
    )
    first = first[
        first.booking_timestamp > timestamp(final_test_start) + pd.Timedelta(seconds=1)
    ]
    if first.empty:
        return {
            "sample_count": 0,
            "status": "No genuine new-resident first observation in the retained period; no cold-start accuracy claim",
        }
    builder = FeatureBuilder(bookings)
    features, targets = [], []
    for row in first.itertuples(index=False):
        cutoff = row.booking_timestamp - pd.Timedelta(seconds=1)
        features.append(builder.build(row.resident_reference, cutoff))
        targets.append(
            {
                "booking_id": row.booking_id,
                "resident_reference": row.resident_reference,
                "prediction_cutoff": cutoff,
                "target_booking_timestamp": row.booking_timestamp,
                "target_usage_timestamp": row.usage_timestamp,
                "target_facility": row.facility_reference,
                "days_until_usage": (row.usage_timestamp - cutoff).total_seconds()
                / 86400,
                "lead_hours": (
                    row.usage_timestamp - row.booking_timestamp
                ).total_seconds()
                / 3600,
            }
        )
    X, y = pd.DataFrame(features), pd.DataFrame(targets)
    prediction = predict_components(bundle, X, y.prediction_cutoff, "champion")
    return {
        "sample_count": len(y),
        "metrics": metrics(review_predictions(y, prediction, X)),
        "note": "Separate first-observation cohort at one second before the observed booking; not the standard previous-event task or a booking-occurrence estimate.",
    }
