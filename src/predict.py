"""Frozen-artifact inference, shared temporal constraints, and cold-start fallback."""

from pathlib import Path
import numpy as np
import pandas as pd
from src.feature_engineering import FeatureBuilder
from src.utils import timestamp
from src.artifacts import load_bundle, validate_history, ArtifactError, resolve_release
from src.config import FACILITIES
from src.data_validation import validate_bookings


def assemble_predictions(cutoffs: pd.Series, raw: dict) -> pd.DataFrame:
    cutoff_series = cutoffs.reset_index(drop=True)
    if set(raw) != {"facility", "days", "minutes", "lead_hours"}:
        raise ValueError("Require facility, days, minutes and lead_hours components")
    for component, values in raw.items():
        array = np.asarray(values)
        if array.ndim != 1 or len(array) != len(cutoff_series):
            raise ValueError("Prediction component shapes must match cutoffs")
        if component != "facility" and not np.isfinite(array.astype(float)).all():
            raise ValueError("Predictions must be finite")
    if not np.isin(raw["facility"], FACILITIES).all():
        raise ValueError("Unknown predicted facility")
    days = np.clip(np.asarray(raw["days"], dtype=float), 0.0, 365.0)
    minutes = np.asarray(raw["minutes"], dtype=float) % 1440
    leads = np.clip(np.asarray(raw["lead_hours"], dtype=float), 0.25, 720.0)
    usage, notification = [], []
    for cutoff, day, minute, lead in zip(
        cutoff_series, days, minutes, leads, strict=True
    ):
        cutoff = timestamp(cutoff)
        predicted = (cutoff + pd.Timedelta(days=float(day))).normalize() + pd.Timedelta(
            minutes=float(minute)
        )
        # Keep inferred usage after the cutoff while preserving the predicted time of day.
        if predicted <= cutoff + pd.Timedelta(minutes=15):
            predicted += pd.Timedelta(days=1)
        if predicted > cutoff + pd.Timedelta(days=365):
            predicted -= pd.Timedelta(days=1)
        nudge = max(
            cutoff + pd.Timedelta(seconds=1),
            predicted - pd.Timedelta(hours=float(lead)),
        )
        nudge = min(nudge, predicted - pd.Timedelta(minutes=15))
        usage.append(predicted)
        notification.append(nudge)
    return pd.DataFrame(
        {
            "predicted_facility": raw["facility"],
            "predicted_days_until_usage": days,
            "predicted_usage_timestamp": usage,
            "predicted_notification_timestamp": notification,
            "predicted_model_lead_hours": leads,
            "predicted_lead_hours": [
                (u - n).total_seconds() / 3600 for u, n in zip(usage, notification)
            ],
        }
    )


class PredictionEngine:
    """Loads trusted local artifacts once; never fits or retrains at prediction time."""

    def __init__(self, model_path: Path, bookings: pd.DataFrame):
        self.bundle = load_bundle(model_path)
        validate_bookings(bookings)
        validate_history(bookings, self.bundle["metadata"]["history_contract"])
        self.builder = FeatureBuilder(bookings)
        self.latest_booking = bookings.booking_timestamp.max()
        self.release_id = resolve_release(model_path).parent.name

    def predict(self, resident: str, cutoff: str | pd.Timestamp) -> dict:
        cutoff = timestamp(cutoff)
        if cutoff < timestamp(self.bundle["metadata"]["ready_at"]):
            raise ValueError(
                "Cutoff predates model selection; use a separately fitted historical model"
            )
        X = pd.DataFrame([self.builder.build(resident, cutoff)])
        if list(X.columns) != self.bundle["metadata"]["feature_names"]:
            raise ArtifactError("Inference feature schema mismatch")
        prediction = predict_components(
            self.bundle, X, pd.Series([cutoff]), "champion"
        ).iloc[0]
        return {
            "resident_reference": resident,
            "prediction_cutoff": cutoff.isoformat(),
            "predicted_facility": prediction.predicted_facility,
            "predicted_usage_date": str(prediction.predicted_usage_timestamp.date()),
            "predicted_usage_time": prediction.predicted_usage_timestamp.strftime(
                "%H:%M:%S"
            ),
            "predicted_usage_timestamp": prediction.predicted_usage_timestamp.isoformat(),
            "recommended_notification_timestamp": prediction.predicted_notification_timestamp.isoformat(),
            "cold_start": bool(X.cold_start.iloc[0]),
            "confidence_note": "Classifier scores are uncalibrated; no probability claim is made.",
            "notification_note": "Historical booking-time proxy; notification effectiveness is unvalidated.",
            "model_dataset_sha256": self.bundle["metadata"]["dataset_sha256"],
            "release_id": self.release_id,
            "history_count": int(X.total_bookings.iloc[0]),
            "prediction_strategy": "validation_selected",
            "component_routes": self.bundle["metadata"].get("routes", {}),
            "validation_error_guides": self.bundle["metadata"].get(
                "validation_error_guides", {}
            ),
            "prediction_policy": "next observed booking conditional on occurrence; trained at previous booking creation",
            "warnings": [
                message
                for condition, message in [
                    (
                        bool(X.cold_start.iloc[0]),
                        "Sparse or unseen history; timing uses global fallback when fewer than two records exist",
                    ),
                    (
                        float(X.days_since_last.iloc[0]) > 0,
                        "This cutoff is later than the latest resident booking; arbitrary-cutoff performance is not prospectively validated",
                    ),
                    (
                        (cutoff - self.latest_booking).total_seconds() > 7 * 86400,
                        "History snapshot is over seven days older than the requested cutoff",
                    ),
                ]
                if condition
            ],
        }


def predict_components(
    bundle: dict, X: pd.DataFrame, cutoffs: pd.Series, kind: str = "ml"
) -> pd.DataFrame:
    if kind not in ["ml", "resident", "global", "champion"]:
        raise ValueError("Unknown prediction strategy")
    if len(X) != len(cutoffs) or not X.index.equals(cutoffs.index):
        raise ValueError("Feature and cutoff rows must align")
    if X.empty:
        return assemble_predictions(
            cutoffs, {key: [] for key in ["facility", "days", "minutes", "lead_hours"]}
        )
    return assemble_predictions(cutoffs, raw_components(bundle, X, kind))


def raw_components(bundle: dict, X: pd.DataFrame, kind: str = "ml") -> dict:
    baseline = bundle["baseline"].predict(X, "resident" if kind == "ml" else kind)
    if kind in ["resident", "global"]:
        return baseline
    models = bundle["models"]
    sin = models["usage_sin"].predict(X)
    cos = models["usage_cos"].predict(X)
    minutes = np.mod(np.arctan2(sin, cos), 2 * np.pi) * 1440 / (2 * np.pi)
    # Near-zero resultant vectors are ambiguous; prefer the historical fallback.
    minutes = np.where(np.hypot(sin, cos) < 0.05, baseline["minutes"], minutes)
    raw = {
        "facility": models["facility"].predict(X),
        "days": models["usage_days"].predict(X),
        "minutes": minutes,
        "lead_hours": models["lead_hours"].predict(X),
    }
    cold = X.total_bookings.to_numpy() < 2
    for component in raw:
        raw[component] = np.where(cold, baseline[component], raw[component])
    if kind == "champion":
        for component, route in bundle["metadata"].get("routes", {}).items():
            if route != "ml":
                raw[component] = bundle["baseline"].predict(X, route)[component]
    return raw
