import numpy as np
import pandas as pd
from src.feature_engineering import FeatureBuilder
from src.train import fit_models


def test_appending_and_changing_future_cannot_change_history(bookings):
    resident = bookings.resident_reference.value_counts().index[0]
    history = bookings[bookings.resident_reference == resident]
    cutoff = history.booking_timestamp.iloc[5]
    expected = FeatureBuilder(bookings).build(resident, cutoff)
    future = history.iloc[[-1]].copy()
    future["booking_id"] = "future-record"
    future["booking_timestamp"] = cutoff + pd.Timedelta(days=500)
    future["usage_timestamp"] = cutoff + pd.Timedelta(days=501)
    future["facility_reference"] = "Multipurpose Hall"
    appended = pd.concat([bookings, future], ignore_index=True)
    assert FeatureBuilder(appended).build(resident, cutoff) == expected
    changed = bookings.copy()
    mask = changed.booking_timestamp > cutoff
    changed.loc[mask, "facility_reference"] = "Tennis Court"
    changed.loc[mask, "usage_timestamp"] += pd.Timedelta(days=7)
    assert FeatureBuilder(changed).build(resident, cutoff) == expected


def test_targets_strictly_future_and_no_target_columns(examples):
    X, y, _, _ = examples
    assert (y.target_booking_timestamp > y.prediction_cutoff).all()
    assert not set(X.columns) & set(y.columns)
    assert not any("target" in column or "persona" in column for column in X)


def test_final_test_perturbation_cannot_change_fitting_or_selection(examples, trained, tmp_path):
    X, y, splits, split = examples
    original, _ = trained
    changed_X, changed_y = X.copy(), y.copy()
    test = splits["test"]
    changed_X.loc[test, "total_bookings"] = 999999
    changed_X.loc[test, "most_recent_facility"] = "test-only-category"
    changed_y.loc[test, "target_facility"] = "TEST-POISON"
    changed_y.loc[test, "days_until_usage"] = 1000000
    changed_y.loc[test, "lead_hours"] = 1000000
    changed_y.loc[test, "target_usage_timestamp"] += pd.Timedelta(days=2000)
    refit = fit_models(changed_X, changed_y, splits, {"dataset_sha256": "test-fixture", "split": split}, tmp_path)
    assert original["metadata"]["validation_candidates"] == refit["metadata"]["validation_candidates"]
    assert original["metadata"]["selected_configurations"] == refit["metadata"]["selected_configurations"]
    for name in original["models"]:
        np.testing.assert_array_equal(original["models"][name].predict(X.iloc[test]), refit["models"][name].predict(X.iloc[test]))
    assert not set(test) & set(refit["metadata"]["fit_indices"])
    assert not set(test) & set(refit["metadata"]["selection_indices"])
