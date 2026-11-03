"""Frozen-artifact inference, shared temporal constraints, and cold-start fallback."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.feature_engineering import FeatureBuilder
from src.utils import timestamp


def assemble_predictions(cutoffs: pd.Series, raw: dict) -> pd.DataFrame:
    cutoff_series = cutoffs.reset_index(drop=True)
    days = np.clip(np.asarray(raw["days"], dtype=float), 0., 365.)
    minutes = np.asarray(raw["minutes"], dtype=float) % 1440
    leads = np.clip(np.asarray(raw["lead_hours"], dtype=float), .25, 720.)
    usage, notification = [], []
    for cutoff, day, minute, lead in zip(cutoff_series, days, minutes, leads, strict=True):
        cutoff = timestamp(cutoff)
        predicted = (cutoff + pd.Timedelta(days=float(day))).normalize() + pd.Timedelta(minutes=float(minute))
        # Keep inferred usage after the cutoff while preserving the predicted time of day.
        if predicted <= cutoff + pd.Timedelta(minutes=15):
            predicted += pd.Timedelta(days=1)
        nudge = max(cutoff + pd.Timedelta(seconds=1), predicted - pd.Timedelta(hours=float(lead)))
        nudge = min(nudge, predicted - pd.Timedelta(minutes=15))
        usage.append(predicted)
        notification.append(nudge)
    return pd.DataFrame({"predicted_facility": raw["facility"], "predicted_days_until_usage": days, "predicted_usage_timestamp": usage,
                         "predicted_notification_timestamp": notification,
                         "predicted_model_lead_hours": leads,
                         "predicted_lead_hours": [(u - n).total_seconds() / 3600 for u, n in zip(usage, notification)]})


class PredictionEngine:
    """Loads trusted local artifacts once; never fits or retrains at prediction time."""
    def __init__(self, model_path: Path, bookings: pd.DataFrame):
        self.bundle = joblib.load(model_path)
        self.builder = FeatureBuilder(bookings)

    def predict(self, resident: str, cutoff: str | pd.Timestamp) -> dict:
        cutoff = timestamp(cutoff)
        if cutoff < timestamp(self.bundle["metadata"]["ready_at"]):
            raise ValueError("Cutoff predates model selection; use a separately fitted historical model")
        X = pd.DataFrame([self.builder.build(resident, cutoff)])
        prediction = predict_components(self.bundle, X, pd.Series([cutoff])).iloc[0]
        return {"resident_reference": resident, "prediction_cutoff": cutoff.isoformat(),
                "predicted_facility": prediction.predicted_facility,
                "predicted_usage_date": str(prediction.predicted_usage_timestamp.date()),
                "predicted_usage_time": prediction.predicted_usage_timestamp.strftime("%H:%M:%S"),
                "predicted_usage_timestamp": prediction.predicted_usage_timestamp.isoformat(),
                "recommended_notification_timestamp": prediction.predicted_notification_timestamp.isoformat(),
                "cold_start": bool(X.cold_start.iloc[0]),
                "confidence_note": "Classifier scores are uncalibrated; no probability claim is made.",
                "notification_note": "Historical booking-time proxy; notification effectiveness is unvalidated.",
                "model_dataset_sha256": self.bundle["metadata"]["dataset_sha256"]}


def predict_components(bundle: dict, X: pd.DataFrame, cutoffs: pd.Series, kind: str = "ml") -> pd.DataFrame:
    baseline = bundle["baseline"].predict(X, "resident" if kind == "ml" else kind)
    if kind != "ml":
        return assemble_predictions(cutoffs, baseline)
    models = bundle["models"]
    sin = models["usage_sin"].predict(X)
    cos = models["usage_cos"].predict(X)
    minutes = np.mod(np.arctan2(sin, cos), 2 * np.pi) * 1440 / (2 * np.pi)
    # Near-zero resultant vectors are ambiguous; prefer the historical fallback.
    minutes = np.where(np.hypot(sin, cos) < .05, baseline["minutes"], minutes)
    raw = {"facility": models["facility"].predict(X), "days": models["usage_days"].predict(X),
           "minutes": minutes, "lead_hours": models["lead_hours"].predict(X)}
    cold = X.total_bookings.to_numpy() < 2
    for component in raw:
        raw[component] = np.where(cold, baseline[component], raw[component])
    return assemble_predictions(cutoffs, raw)
