import numpy as np
import pandas as pd
import pytest
from src.predict import assemble_predictions, PredictionEngine, predict_components
from src.utils import circular_error


def test_circular_midnight_distance():
    np.testing.assert_allclose(circular_error([1435, 60, 0], [5, 1430, 720]), [10, 70, 720])


def test_notification_arithmetic_and_constraints():
    cutoff = pd.Series([pd.Timestamp("2026-06-01T10:00:00+05:30")])
    predicted = assemble_predictions(cutoff, {"facility": ["Gym"], "days": [2.], "minutes": [18 * 60], "lead_hours": [24.]})
    assert predicted.predicted_notification_timestamp.iloc[0] == pd.Timestamp("2026-06-02T18:00:00+05:30")
    extreme = assemble_predictions(cutoff, {"facility": ["Gym"], "days": [-3], "minutes": [0], "lead_hours": [100000]})
    assert cutoff.iloc[0] < extreme.predicted_notification_timestamp.iloc[0] < extreme.predicted_usage_timestamp.iloc[0]


def test_saved_inference_cold_start_and_model_availability(bookings, trained):
    _, root = trained
    engine = PredictionEngine(root / "model_bundle.joblib", bookings)
    cutoff = bookings.booking_timestamp.max() + pd.Timedelta(days=1)
    result = engine.predict("unseen-resident", cutoff)
    assert result["cold_start"] is True
    assert pd.Timestamp(result["recommended_notification_timestamp"]) > cutoff
    with pytest.raises(ValueError, match="predates"):
        engine.predict("R-0001", "2024-01-01T10:00:00+05:30")
    with pytest.raises(ValueError, match="timezone"):
        engine.predict("R-0001", "2026-06-01")


def test_all_prediction_outputs_are_valid(examples, trained):
    X, y, splits, _ = examples
    bundle, _ = trained
    test = splits["test"]
    prediction = predict_components(bundle, X.iloc[test], y.prediction_cutoff.iloc[test])
    cutoff = y.prediction_cutoff.iloc[test].reset_index(drop=True)
    assert (prediction.predicted_usage_timestamp > cutoff).all()
    assert (prediction.predicted_notification_timestamp > cutoff).all()
    assert (prediction.predicted_notification_timestamp < prediction.predicted_usage_timestamp).all()
    assert prediction.predicted_lead_hours.between(.25, 720).all()
