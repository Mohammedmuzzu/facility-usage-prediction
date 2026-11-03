from dataclasses import replace
import pandas as pd
from src.data_generator import generate_bookings
from src.data_validation import validate_bookings
from src.robustness import development_windows
from src.predict import assemble_predictions
import pytest


def test_backtests_never_enter_final_holdout(examples):
    _, y, _, split = examples
    windows = development_windows(y, split["test_start"])
    assert len(windows) == 3
    for development, indices, _ in windows:
        assert development.target_booking_timestamp.max() < pd.Timestamp(
            split["test_start"]
        )
        assert not set(indices["train"]) & set(indices["test"])


@pytest.mark.parametrize(
    "change",
    [
        {"drift_fraction": 1.0},
        {"exploration": 0.8},
        {"sparse_fraction": 0.8},
        {"start": "2024-02-01"},
    ],
)
def test_generator_extreme_regimes_keep_contract(config, change):
    configuration = replace(config, **change)
    assert validate_bookings(generate_bookings(configuration), configuration)[
        "validation_passed"
    ]


@pytest.mark.parametrize(
    "cutoff",
    [
        "2024-02-29T23:59:59+05:30",
        "2025-12-31T23:59:59+05:30",
        "2026-06-30T00:00:00+05:30",
        "2026-06-01T23:59:59Z",
    ],
)
def test_timestamp_calendar_edges(cutoff):
    cutoff = pd.Timestamp(cutoff)
    for days in [-1, 0, 1, 365, 10000]:
        result = assemble_predictions(
            pd.Series([cutoff]),
            {
                "facility": ["Gym"],
                "days": [days],
                "minutes": [1439.9],
                "lead_hours": [10000],
            },
        )
        row = result.iloc[0]
        assert (
            cutoff
            < row.predicted_notification_timestamp
            < row.predicted_usage_timestamp
            <= cutoff + pd.Timedelta(days=365)
        )
        assert 0.25 <= row.predicted_lead_hours <= 720
