import numpy as np
import pandas as pd
from src.feature_engineering import FeatureBuilder


def test_cutoff_counts_ordering_and_finite_features(bookings):
    builder = FeatureBuilder(bookings)
    resident = bookings.resident_reference.value_counts().index[0]
    history = builder.histories[resident]
    cutoff = history.booking_timestamp.iloc[4]
    features = builder.build(resident, cutoff)
    assert features["total_bookings"] == len(
        history[history.booking_timestamp <= cutoff]
    )
    assert features["most_recent_facility"] == history.facility_reference.iloc[4]
    assert sum(features[f"facility_{i}_preference"] for i in range(6)) == 1
    assert np.isfinite([v for v in features.values() if not isinstance(v, str)]).all()
    assert all(
        group.booking_timestamp.is_monotonic_increasing
        for group in builder.histories.values()
    )


def test_cold_start_and_first_history(bookings):
    builder = FeatureBuilder(bookings)
    cutoff = bookings.booking_timestamp.min() - pd.Timedelta(seconds=1)
    features = builder.build("new-resident", cutoff)
    assert features["total_bookings"] == 0
    assert features["cold_start"] == 1
    assert features["most_frequent_facility"] == "Unknown"
    first = bookings.iloc[0]
    assert (
        builder.build(first.resident_reference, first.booking_timestamp)[
            "total_bookings"
        ]
        >= 1
    )
