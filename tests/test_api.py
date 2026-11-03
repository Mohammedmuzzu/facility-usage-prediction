from fastapi.testclient import TestClient
from app import api
from src.predict import PredictionEngine
import pandas as pd


def test_api_saved_model_predictions_and_validation(bookings, trained, monkeypatch):
    _, root = trained
    monkeypatch.setattr(
        api,
        "get_engine",
        lambda: PredictionEngine(root / "model_bundle.joblib", bookings),
    )
    client = TestClient(api.app)
    cutoff = (bookings.booking_timestamp.max() + pd.Timedelta(days=1)).isoformat()
    response = client.post(
        "/predict", json={"resident_reference": "R-9999", "prediction_cutoff": cutoff}
    )
    assert response.status_code == 200
    assert response.json()["cold_start"] is True
    assert (
        client.post("/predict/R-0001", json={"prediction_cutoff": cutoff}).status_code
        == 200
    )
    assert (
        client.post(
            "/predict",
            json={"resident_reference": "R-0001", "prediction_cutoff": "bad"},
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/predict",
            json={"resident_reference": "R-0001", "prediction_cutoff": "2026-01-01"},
        ).status_code
        == 422
    )
