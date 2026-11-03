"""The only feature builder used by both historical training and live inference.

All scheduled usage/lead values come from bookings already created by the cutoff.
No population aggregate, target row, resident identity or latent persona is a feature.
"""

import numpy as np
import pandas as pd
from src.config import FACILITIES
from src.utils import timestamp, minutes_of_day


class FeatureBuilder:
    def __init__(self, bookings: pd.DataFrame):
        self.histories = {
            resident: group.sort_values(
                ["booking_timestamp", "booking_id"], kind="stable"
            )
            .reset_index(drop=True)
            .copy()
            for resident, group in bookings.groupby("resident_reference", sort=False)
        }

    def history(self, resident: str, cutoff: pd.Timestamp) -> pd.DataFrame:
        group = self.histories.get(resident)
        if group is None:
            return pd.DataFrame()
        boundary = group.booking_timestamp.searchsorted(timestamp(cutoff), side="right")
        return group.iloc[:boundary].copy()

    def build(self, resident: str, cutoff: pd.Timestamp) -> dict:
        cutoff = timestamp(cutoff)
        history = self.history(resident, cutoff)
        n = len(history)
        features: dict = {
            "total_bookings": n,
            "prediction_weekday": cutoff.day_of_week,
            "prediction_hour": cutoff.hour,
            "prediction_month": cutoff.month,
            "prediction_season": (cutoff.month % 12) // 3,
            "prediction_weekend": int(cutoff.day_of_week >= 5),
            "cold_start": int(n < 3),
            "days_since_last": 0.0,
            "mean_interval": 14.0,
            "median_interval": 14.0,
            "previous_booking_gap_days": 14.0,
            "frequency_per_30d": 0.0,
            "regularity_cv": 1.0,
            "facility_diversity": 0,
            "most_frequent_facility": "Unknown",
            "most_recent_facility": "Unknown",
            "mean_lead_hours": 24.0,
            "median_lead_hours": 24.0,
            "previous_usage_weekday": -1,
            "previous_usage_hour": -1,
            "preferred_weekday": -1,
            "preferred_hour": 12,
            "usage_sin": 0.0,
            "usage_cos": -1.0,
            "last_usage_offset_days": 0.0,
        }
        for window in [7, 30, 90]:
            features[f"bookings_{window}d"] = 0
        for idx in range(6):
            features.update(
                {
                    f"facility_{idx}_count": 0,
                    f"facility_{idx}_preference": 1 / 6,
                    f"facility_{idx}_recent_preference": 1 / 6,
                    f"facility_{idx}_lead_hours": 24.0,
                    f"facility_{idx}_usage_sin": 0.0,
                    f"facility_{idx}_usage_cos": -1.0,
                }
            )
        for weekday in range(7):
            features[f"weekday_{weekday}_preference"] = 1 / 7
        if not n:
            return features
        created = history.booking_timestamp
        usage = history.usage_timestamp
        intervals = created.diff().dt.total_seconds().dropna().to_numpy() / 86400
        elapsed = max((cutoff - created.iloc[0]).total_seconds() / 86400, 1)
        lead = (usage - created).dt.total_seconds().to_numpy() / 3600
        age = (cutoff - created).dt.total_seconds().to_numpy() / 86400
        counts = history.facility_reference.value_counts()
        minute = minutes_of_day(usage)
        angles = minute * 2 * np.pi / 1440
        features.update(
            {
                "days_since_last": float(age[-1]),
                "mean_interval": float(intervals.mean()) if len(intervals) else 14.0,
                "median_interval": float(np.median(intervals))
                if len(intervals)
                else 14.0,
                "previous_booking_gap_days": float(intervals[-1])
                if len(intervals)
                else 14.0,
                "frequency_per_30d": n / max(30.0, elapsed) * 30,
                "regularity_cv": float(intervals.std() / max(intervals.mean(), 0.01))
                if len(intervals)
                else 1.0,
                "facility_diversity": len(counts),
                "most_frequent_facility": max(
                    FACILITIES, key=lambda f: counts.get(f, 0)
                ),
                "most_recent_facility": history.facility_reference.iloc[-1],
                "mean_lead_hours": float(lead.mean()),
                "median_lead_hours": float(np.median(lead)),
                "previous_usage_weekday": int(usage.iloc[-1].day_of_week),
                "previous_usage_hour": int(usage.iloc[-1].hour),
                "preferred_weekday": int(usage.dt.day_of_week.mode().iloc[0]),
                "preferred_hour": int(usage.dt.hour.mode().iloc[0]),
                "usage_sin": float(np.sin(angles).mean()),
                "usage_cos": float(np.cos(angles).mean()),
                "last_usage_offset_days": (usage.iloc[-1] - cutoff).total_seconds()
                / 86400,
            }
        )
        recent_weights = np.exp(-age / 60)
        for window in [7, 30, 90]:
            features[f"bookings_{window}d"] = int((age <= window).sum())
        for idx, facility in enumerate(FACILITIES):
            mask = (history.facility_reference == facility).to_numpy()
            features[f"facility_{idx}_count"] = int(mask.sum())
            features[f"facility_{idx}_preference"] = (mask.sum() + 1) / (n + 6)
            features[f"facility_{idx}_recent_preference"] = (
                recent_weights[mask].sum() + 1
            ) / (recent_weights.sum() + 6)
            if mask.any():
                features[f"facility_{idx}_lead_hours"] = float(np.median(lead[mask]))
                features[f"facility_{idx}_usage_sin"] = float(
                    np.sin(angles[mask]).mean()
                )
                features[f"facility_{idx}_usage_cos"] = float(
                    np.cos(angles[mask]).mean()
                )
        for weekday in range(7):
            features[f"weekday_{weekday}_preference"] = (
                (usage.dt.day_of_week == weekday).sum() + 1
            ) / (n + 7)
        return features
